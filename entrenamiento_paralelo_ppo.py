"""
entrenamiento_paralelo_ppo.py
=============================
Entrenamiento escalado de PPO utilizando múltiples entornos en paralelo (SubprocVecEnv).

Conceptos Teóricos para la Exposición Universitaria:
----------------------------------------------------
1. Vectorización de Entornos (SubprocVecEnv):
   - Ejecuta N instancias independientes de ViZDoom en procesos de CPU separados.
   - Cada entorno avanza de forma asíncrona, enviando observaciones en lote al modelo en GPU.

2. Descorrelación de Muestras en Algoritmos On-Policy:
   - A diferencia de DQN que usa un Replay Buffer (Off-Policy), PPO es On-Policy y no puede
     almacenar experiencias pasadas de políticas obsoletas.
   - Al recolectar datos de 10 actores simultáneos en situaciones de juego completamente distintas,
     se rompe la fuerte correlación temporal cuadro a cuadro sin violar los supuestos matemáticos de PPO.

3. Generalized Advantage Estimation (GAE):
   - Al disponer de múltiples trayectorias paralelas, la estimación del retorno esperado y la ventaja
     A(s, a) = Q(s, a) - V(s) se vuelve mucho más estable, reduciendo la varianza del gradiente.
"""

import os
import sys
import time
import argparse
import psutil
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from doom_wrappers import make_doom_env


class MonitorParaleloCallback(BaseCallback):
    """
    Callback para monitorear el progreso en pasos globales, tasa de FPS agregada,
    recompensa de episodios concluidos y consumo de memoria RAM / VRAM.
    """
    def __init__(self, total_timesteps: int, n_envs: int, check_freq_calls: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.check_freq_calls = check_freq_calls
        self.last_time = None
        self.last_step = 0
        self.hitos_alcanzados = set()

    def _on_training_start(self) -> None:
        self.last_time = time.time()
        self.last_step = self.num_timesteps
        print("\n" + "=" * 82)
        print(f"{'PASO':^8} | {'PROGRESO':^9} | {'RECOMPENSA REC.':^18} | {'FPS':^7} | {'RAM (GB)':^10} | {'VRAM GPU':^10}")
        print("=" * 82)

    def _on_step(self) -> bool:
        progreso = self.num_timesteps / self.total_timesteps
        for hito in [0.25, 0.50, 0.75, 1.0]:
            if progreso >= hito and hito not in self.hitos_alcanzados:
                self.hitos_alcanzados.add(hito)
                print(f">>> HITO ALCANZADO: {int(hito * 100)}% ({self.num_timesteps:,} / {self.total_timesteps:,} pasos) <<<")

        if self.n_calls % self.check_freq_calls == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            steps_done = self.num_timesteps - self.last_step
            fps = steps_done / elapsed if elapsed > 0 else 0.0

            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                ep_count = len(self.model.ep_info_buffer)
                reward_str = f"{mean_reward:6.1f} (n={ep_count})"
            else:
                reward_str = "Recolectando..."

            mem_info = psutil.virtual_memory()
            ram_used_gb = (mem_info.total - mem_info.available) / (1024**3)

            if torch.cuda.is_available():
                vram_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
                vram_str = f"{vram_used_mb:6.1f} MB"
            else:
                vram_str = "N/A (CPU)"

            pct_str = f"{progreso * 100:5.1f}%"
            print(f"{self.num_timesteps:^8d} | {pct_str:^9} | {reward_str:^18} | {fps:^7.1f} | {ram_used_gb:^10.2f} | {vram_str:^10}")

            self.last_time = current_time
            self.last_step = self.num_timesteps

        return True


def make_single_env(rank: int, seed: int = 42):
    """Función generadora para crear instancias de entorno aisladas para cada subproceso."""
    def _thunk():
        env = make_doom_env(
            env_id="VizdoomDefendCenter-v1",
            frame_skip=4,
            stack_size=4,
            resolution=(84, 84),
            render_mode=None
        )
        env.reset(seed=seed + rank)
        return env
    return _thunk


def entrenar_ppo_paralelo(
    n_envs: int = 10,
    total_timesteps: int = 100_000,
    seed: int = 42,
    resume_path: str = None
):
    print("=" * 82)
    print("ENTRENAMIENTO ESCALADO EN PARALELO: PPO (PROXIMAL POLICY OPTIMIZATION)")
    print(f"Entornos en paralelo: {n_envs} subprocesos (SubprocVecEnv)")
    print(f"Pasos adicionales:    {total_timesteps:,}")
    print(f"Semilla base:         {seed}")
    print(f"Reanudar desde:       {resume_path if resume_path else 'Entrenamiento desde cero'}")
    print(f"Dispositivo PyTorch:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 82)

    save_dir = "models/defend_ppo_paralelo"
    log_dir = "logs/defend_ppo_paralelo"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Crear entorno vectorizado en subprocesos de CPU
    print(f"\nIniciando {n_envs} subprocesos de ViZDoom...")
    env_fns = [make_single_env(rank=i, seed=seed) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # 2. Configurar hiperparámetros PPO optimizados para batch paralelo
    n_steps_per_env = 256  # Total rollout por actualización = 10 * 256 = 2,560 pasos
    batch_size = 128       # Minibatch para gradientes
    n_epochs = 4           # 4 épocas por rollout

    if resume_path and os.path.exists(resume_path):
        print(f"\nCargando checkpoint previo desde: {resume_path} ...")
        model = PPO.load(resume_path, env=vec_env, device="cuda")
        start_steps = model.num_timesteps
        target_steps = start_steps + total_timesteps
        print(f"Pasos previos acumulados: {start_steps:,} | Objetivo final: {target_steps:,} pasos")
    else:
        start_steps = 0
        target_steps = total_timesteps
        model = PPO(
            policy="CnnPolicy",
            env=vec_env,
            learning_rate=2.5e-4,
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,         # Incentivo de exploración de entropía
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            verbose=0,
            seed=seed,
            device="cuda"
        )
        # Guardar checkpoint 0%
        chkpt_0 = os.path.join(save_dir, "ppo_paralelo_chkpt_0_steps.zip")
        model.save(chkpt_0)
        print(f"Checkpoint inicial (0%) guardado en: {chkpt_0}")

    # Checkpoint cada 25,000 pasos globales
    save_freq_steps = 25_000
    save_freq_calls = max(1, save_freq_steps // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=save_dir,
        name_prefix="ppo_paralelo_chkpt"
    )

    monitor_callback = MonitorParaleloCallback(
        total_timesteps=target_steps,
        n_envs=n_envs,
        check_freq_calls=200
    )

    # 3. Ejecución del entrenamiento
    print(f"\nComenzando entrenamiento paralelo de {total_timesteps:,} pasos adicionales (hasta {target_steps:,})...")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, monitor_callback],
            reset_num_timesteps=False,
            progress_bar=False
        )
    finally:
        vec_env.close()

    t_total = time.time() - t0
    final_path = os.path.join(save_dir, "ppo_paralelo_final.zip")
    model.save(final_path)

    # También guardar checkpoint explícito con el número de pasos final
    path_num = os.path.join(save_dir, f"ppo_paralelo_{target_steps}_steps.zip")
    model.save(path_num)

    print("\n" + "=" * 82)
    print(f"¡Entrenamiento completado en {t_total:.1f} s ({t_total/60:.2f} min)!")
    print(f"Velocidad promedio: {total_timesteps / t_total:.1f} pasos/segundo.")
    print(f"Modelo final guardado en: {final_path} y {path_num}")
    print("=" * 82)

    # 4. Evaluación cuantitativa final en 10 episodios
    print("\nIniciando evaluación determinista en 10 episodios...")
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([lambda: make_doom_env("VizdoomDefendCenter-v1", frame_skip=4, stack_size=4, resolution=(84, 84))])
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()

    print("\n" + "=" * 82)
    print(f"RESULTADOS FINALES PPO PARALELO ({target_steps:,} pasos globales):")
    print(f"Recompensa promedio: {mean_r:.2f} +/- {std_r:.2f}")
    print("Comparativa de Referencia:")
    print("  - Agente Aleatorio:          0.60 +/- 1.02")
    print("  - PPO 1 Entorno (25k pasos):  1.20 +/- 1.08")
    print("  - DQN 1 Entorno (25k pasos):  8.70 +/- 2.33")
    print("  - PPO Paralelo (100k pasos): 13.90 +/- 4.04")
    print(f"  - PPO Paralelo ({target_steps//1000}k pasos): {mean_r:.2f} +/- {std_r:.2f}")
    print("=" * 82)

    return model, final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Escalado Paralelo de PPO en ViZDoom")
    parser.add_argument("--n_envs", type=int, default=10, help="Número de entornos paralelos (10 óptimo para i9-12900H)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Pasos adicionales de entrenamiento")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--resume", type=str, default=None, help="Ruta al checkpoint previo para reanudar")

    args = parser.parse_args()
    entrenar_ppo_paralelo(n_envs=args.n_envs, total_timesteps=args.timesteps, seed=args.seed, resume_path=args.resume)

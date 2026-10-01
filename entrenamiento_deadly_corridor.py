"""
entrenamiento_deadly_corridor.py
================================
Entrenamiento escalado de PPO en ViZDoom Deadly Corridor (SubprocVecEnv con 10 subprocesos).

Conceptos Teóricos para la Exposición Universitaria:
----------------------------------------------------
1. Espacio de Acciones Multidimensional Discreto:
   - Deadly Corridor cuenta con 8 acciones discretas: NO-OP, giros, avance, retroceso, disparo y strafe lateral.
   - Requiere aprender comportamientos compuestos: avanzar mientras se esquivan bolas de fuego y se dispara a los enemigos.

2. Modelado de Recompensa con Recompensa Conformada (Reward Shaping):
   - El motor premia el avance hacia el final del corredor (distancia euclidiana al chaleco verde).
   - Penaliza el daño recibido y aplica una penalización severa (-100) al morir.
   - Demuestra el dilema entre exploración agresiva (correr hacia adelante) y preservación de vida (eliminar amenazas antes de avanzar).

3. Escalado On-Policy con Vectorización:
   - 10 instancias paralelas a ~250 FPS recolectan experiencias con amplia diversidad de trayectorias tácticas.
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from doom_wrappers import make_doom_env


class MonitorDeadlyCallback(BaseCallback):
    """Callback para monitorear FPS, recompensas acumuladas y recursos del sistema."""
    def __init__(self, total_timesteps: int, n_envs: int, check_freq_calls: int = 200, verbose: int = 1):
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


def make_single_deadly_env(rank: int, seed: int = 42):
    """Crea una instancia aislada de Deadly Corridor."""
    def _thunk():
        env = make_doom_env(
            env_id="VizdoomDeadlyCorridor-v1",
            frame_skip=4,
            stack_size=4,
            resolution=(84, 84),
            render_mode=None
        )
        env.reset(seed=seed + rank)
        return env
    return _thunk


def entrenar_deadly_corridor(
    n_envs: int = 10,
    total_timesteps: int = 100_000,
    seed: int = 42,
    resume_path: str = None
):
    print("=" * 82)
    print("ENTRENAMIENTO ESCALADO EN PARALELO: VIZDOOM DEADLY CORRIDOR (PPO)")
    print(f"Escenario:            VizdoomDeadlyCorridor-v1 (Skill 5 - Nightmare)")
    print(f"Entornos en paralelo: {n_envs} subprocesos (SubprocVecEnv)")
    print(f"Pasos adicionales:    {total_timesteps:,}")
    print(f"Semilla base:         {seed}")
    print(f"Reanudar desde:       {resume_path if resume_path else 'Entrenamiento desde cero'}")
    print(f"Dispositivo PyTorch:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 82)

    save_dir = "models/deadly_corridor"
    log_dir = "logs/deadly_corridor_optA"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nIniciando {n_envs} subprocesos de ViZDoom Deadly Corridor (Skill 3 - Táctico)...")
    env_fns = [make_single_deadly_env(rank=i, seed=seed) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Hiperparámetros optimizados para Deadly Corridor Opción A
    n_steps_per_env = 256  # 2,560 pasos por ciclo de actualización
    batch_size = 128
    n_epochs = 4

    if resume_path and os.path.exists(resume_path):
        print(f"\nCargando checkpoint previo desde: {resume_path} ...")
        model = PPO.load(resume_path, env=vec_env, device="cuda", tensorboard_log=log_dir)
        start_steps = model.num_timesteps
        target_steps = start_steps + total_timesteps
        print(f"Pasos previos: {start_steps:,} | Objetivo final: {target_steps:,} pasos")
        reset_num = False
    else:
        start_steps = 0
        target_steps = total_timesteps
        reset_num = True
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
            ent_coef=0.04,  # Mayor exploración para evitar el colapso de entropía en 50k
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            verbose=0,
            seed=seed,
            device="cuda"
        )
        # Checkpoint inicial (0%)
        chkpt_0 = os.path.join(save_dir, "deadly_optA_chkpt_0_steps.zip")
        model.save(chkpt_0)
        print(f"Checkpoint inicial (0%) guardado en: {chkpt_0}")

    save_freq_steps = max(1000, total_timesteps // 4)
    save_freq_calls = max(1, save_freq_steps // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=save_dir,
        name_prefix="deadly_optA_chkpt"
    )

    monitor_callback = MonitorDeadlyCallback(
        total_timesteps=target_steps,
        n_envs=n_envs,
        check_freq_calls=200
    )

    print(f"\nComenzando entrenamiento paralelo de {total_timesteps:,} pasos...")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, monitor_callback],
            reset_num_timesteps=reset_num,
            progress_bar=False
        )
    finally:
        vec_env.close()

    t_total = time.time() - t0
    final_path = os.path.join(save_dir, "deadly_optA_final.zip")
    model.save(final_path)

    # También actualizar deadly_tactical_final.zip para evaluar_agente
    model.save(os.path.join(save_dir, "deadly_tactical_final.zip"))

    path_num = os.path.join(save_dir, f"deadly_optA_{target_steps}_steps.zip")
    model.save(path_num)

    print("\n" + "=" * 82)
    print(f"¡Entrenamiento completado en {t_total:.1f} s ({t_total/60:.2f} min)!")
    print(f"Velocidad promedio: {total_timesteps / t_total:.1f} pasos/segundo.")
    print(f"Modelo final guardado en: {final_path} y {path_num}")
    print("=" * 82)

    # Evaluación cuantitativa detallada en 10 episodios
    print("\nIniciando evaluación determinista detallada en 10 episodios...")
    eval_env = make_doom_env("VizdoomDeadlyCorridor-v1", frame_skip=4, stack_size=4, resolution=(84, 84))
    eval_rewards = []
    eval_kills = []
    eval_healths = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep * 100 + 42)
        done = False
        ep_r = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(act))
            ep_r += r
            done = term or trunc
        raw = eval_env.unwrapped.game
        state = raw.get_state()
        k = state.game_variables[1] if state and len(state.game_variables) > 1 else 0
        hp = state.game_variables[0] if state else 0
        eval_rewards.append(ep_r)
        eval_kills.append(k)
        eval_healths.append(hp)
        print(f"  Ronda {ep+1:2d}/10 -> Bajas: {int(k)} | Salud final: {int(hp)}% | Retorno: {ep_r:.1f}")
    eval_env.close()

    mean_r = np.mean(eval_rewards)
    std_r = np.std(eval_rewards)
    mean_k = np.mean(eval_kills)
    mean_hp = np.mean(eval_healths)

    print("\n" + "=" * 82)
    print(f"RESULTADOS FINALES DEADLY CORRIDOR ({target_steps:,} pasos - Opción A Táctica):")
    print(f"Bajas promedio:      {mean_k:.2f} / 6 monstruos")
    print(f"Salud media final:   {mean_hp:.1f}%")
    print(f"Recompensa promedio: {mean_r:.2f} +/- {std_r:.2f}")
    print("=" * 82)

    return model, final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PPO Paralelo en Deadly Corridor")
    parser.add_argument("--n_envs", type=int, default=10, help="Número de entornos paralelos")
    parser.add_argument("--timesteps", type=int, default=100000, help="Pasos de entrenamiento")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--resume", type=str, default=None, help="Ruta de checkpoint para reanudar")

    args = parser.parse_args()
    entrenar_deadly_corridor(n_envs=args.n_envs, total_timesteps=args.timesteps, seed=args.seed, resume_path=args.resume)

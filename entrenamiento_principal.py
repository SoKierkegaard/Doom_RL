"""
entrenamiento_principal.py
==========================
Fase 3: Entrenamiento principal en ViZDoom DefendCenter.

Conceptos teóricos demostrados en este script:
1. Q-Learning vs Policy Gradient / Actor-Critic:
   - DQN (Off-Policy, Value-Based): Aprende la función Q(s, a; theta) mediante el error TD
     de Bellman, usando Replay Buffer y Target Network.
   - PPO (On-Policy, Policy Gradient / Actor-Critic): Optimiza directamente la política
     pi_theta(a|s) mediante ascenso de gradiente estocástico en la función objetivo recortada
     (Clipped Surrogate Objective) asistido por una red de valor V(s; phi) como crítico.

2. Experience Replay Buffer (DQN):
   - Capacidad: 12,000 transiciones.
   - Cálculo de RAM: 12,000 * (4 * 84 * 84) * 2 bytes obs ~= 645 MB de RAM.
   - Rompe la correlación temporal y permite reutilizar experiencias pasadas.

3. Red Objetivo (Target Network, DQN):
   - Se congela durante N pasos y se sincroniza para evitar la inestabilidad del objetivo móvil.

4. Hitos de Checkpoints:
   - Guarda el modelo al inicio (0%), ~25%, ~50%, ~75% y al final (100%).
   - Estos checkpoints se usarán en la Fase 4 para generar los videos comparativos de la evolución.
"""

import os
import sys
import time
import argparse
import psutil
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from doom_wrappers import make_doom_env


class MonitorRecursosHitosCallback(BaseCallback):
    """
    Callback para reportar FPS, recompensa reciente, uso de RAM y VRAM en GPU,
    y notificar el alcance de cada hito (25%, 50%, 75%, 100%).
    """
    def __init__(self, total_timesteps: int, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
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
        # Notificación de hitos (25%, 50%, 75%)
        progreso = self.num_timesteps / self.total_timesteps
        for hito in [0.25, 0.50, 0.75, 1.0]:
            if progreso >= hito and hito not in self.hitos_alcanzados:
                self.hitos_alcanzados.add(hito)
                print(f">>> HITO ALCANZADO: {int(hito * 100)}% ({self.num_timesteps:,} / {self.total_timesteps:,} pasos) <<<")

        if self.n_calls % self.check_freq == 0:
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


def crear_entorno_defend(render_mode=None):
    """Crea y envuelve el entorno VizdoomDefendCenter-v1."""
    env = make_doom_env(
        env_id="VizdoomDefendCenter-v1",
        frame_skip=4,
        stack_size=4,
        resolution=(84, 84),
        render_mode=render_mode
    )
    env = Monitor(env)
    return env


def entrenar(
    algoritmo: str = "DQN",
    total_timesteps: int = 25_000,
    seed: int = 42,
    resume_path: str = None
):
    print("=" * 82)
    print(f"FASE 3: ENTRENAMIENTO PRINCIPAL EN VIZDOOM DEFEND CENTER")
    print(f"Algoritmo:        {algoritmo.upper()}")
    print(f"Pasos totales:    {total_timesteps:,}")
    print(f"Semilla:          {seed}")
    print(f"GPU:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}")
    print("=" * 82)

    save_dir = f"models/defend_{algoritmo.lower()}"
    log_dir = f"logs/defend_{algoritmo.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Entorno preprocesado
    env = crear_entorno_defend()

    # Frecuencia de checkpoints para capturar hitos (cada 25% de los pasos)
    save_freq = max(1000, total_timesteps // 4)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix=f"{algoritmo.lower()}_defend_chkpt"
    )
    monitor_callback = MonitorRecursosHitosCallback(
        total_timesteps=total_timesteps,
        check_freq=1000
    )

    if resume_path and os.path.exists(resume_path):
        print(f"Reanudando entrenamiento desde checkpoint: {resume_path}")
        if algoritmo.upper() == "DQN":
            model = DQN.load(resume_path, env=env, device="cuda")
        else:
            model = PPO.load(resume_path, env=env, device="cuda")
    else:
        if algoritmo.upper() == "DQN":
            BUFFER_SIZE = 12_000  # ~645 MB de RAM
            model = DQN(
                policy="CnnPolicy",            # Red convolucional NatureCNN
                env=env,
                learning_rate=2.5e-4,          # Tasa de aprendizaje estándar
                buffer_size=BUFFER_SIZE,       # Buffer acotado para estabilidad de RAM
                learning_starts=1000,          # 1,000 pasos de exploración pura inicial
                batch_size=32,                 # Minibatch uniforme i.i.d.
                tau=1.0,                       # Copia dura periódica a la red objetivo
                target_update_interval=1000,   # Sincronización de Q_target cada 1,000 pasos
                gamma=0.99,                    # Factor de descuento
                train_freq=4,                  # Actualización cada 4 pasos de entorno
                gradient_steps=1,
                exploration_fraction=0.5,      # Epsilon decae durante el 50% inicial
                exploration_initial_eps=1.0,   # 100% aleatorio al inicio
                exploration_final_eps=0.05,    # 5% exploración residual
                tensorboard_log=log_dir,
                verbose=0,
                seed=seed,
                device="cuda"
            )
        elif algoritmo.upper() == "PPO":
            model = PPO(
                policy="CnnPolicy",
                env=env,
                learning_rate=2.5e-4,
                n_steps=128,                   # Longitud del rollout por ciclo
                batch_size=64,                 # Tamaño del minibatch de optimización
                n_epochs=4,                    # Épocas de optimización sobre el rollout
                gamma=0.99,                    # Descuento
                gae_lambda=0.95,               # Parámetro GAE (Generalized Advantage Estimation)
                clip_range=0.2,                # Rango de recorte epsilon del PPO surrogate
                ent_coef=0.01,                 # Coeficiente de entropía para exploración
                tensorboard_log=log_dir,
                verbose=0,
                seed=seed,
                device="cuda"
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algoritmo}. Usa 'DQN' o 'PPO'.")

    # Guardar checkpoint inicial (0% - sin entrenar o inicio)
    chkpt_0 = os.path.join(save_dir, f"{algoritmo.lower()}_defend_chkpt_0_steps.zip")
    model.save(chkpt_0)
    print(f"Checkpoint 0% (Inicio) guardado en: {chkpt_0}")

    # Entrenamiento
    print(f"\nIniciando entrenamiento de {total_timesteps:,} pasos...")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, monitor_callback],
        progress_bar=False
    )
    t_total = time.time() - t0

    # Guardar modelo final
    final_path = os.path.join(save_dir, f"{algoritmo.lower()}_defend_final.zip")
    model.save(final_path)

    print("\n" + "=" * 82)
    print(f"¡Entrenamiento completado en {t_total:.1f} s ({t_total/60:.2f} min)!")
    print(f"Velocidad promedio: {total_timesteps / t_total:.1f} pasos/segundo.")
    print(f"Modelo final guardado en: {final_path}")

    # Evaluación cuantitativa final en 10 episodios
    print("\n" + "=" * 82)
    print(f"EVALUACIÓN DEL MODELO FINAL ({algoritmo.upper()} - 10 Episodios)")
    print("=" * 82)

    eval_env = crear_entorno_defend()
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()
    env.close()

    print(f"Recompensa promedio lograda: {mean_r:.2f} +/- {std_r:.2f}")
    print(f"Línea base aleatoria:        0.60 +/- 1.02")
    print(f"Mejora sobre azar:           +{mean_r - 0.60:.2f} puntos")
    print("=" * 82)
    return model, final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Principal en ViZDoom DefendCenter")
    parser.add_argument("--algoritmo", type=str, default="DQN", choices=["DQN", "PPO"], help="Algoritmo a usar (DQN o PPO)")
    parser.add_argument("--timesteps", type=int, default=25000, help="Pasos totales de entrenamiento")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--resume", type=str, default=None, help="Ruta al checkpoint para reanudar")

    args = parser.parse_args()
    entrenar(algoritmo=args.algoritmo, total_timesteps=args.timesteps, seed=args.seed, resume_path=args.resume)

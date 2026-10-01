"""
fase2_humo_dqn.py
=================
Fase 2: Prueba de humo con Deep Q-Network (DQN) en VizdoomBasic-v1.

Conceptos teóricos de Deep RL demostrados:
1. Q-Learning con Redes Neuronales (DQN - Mnih et al., 2015):
   Aproximamos la función de valor de acción Q*(s, a) mediante una red neuronal convolucional
   (NatureCNN) con parámetros theta: Q(s, a; theta).

2. Buffer de Repetición de Experiencias (Experience Replay Buffer):
   - Almacena transiciones e_t = (s_t, a_t, r_t, s_{t+1}, done_t).
   - Rompe la correlación temporal entre pasos consecutivos; al muestrear minibatches uniformes,
     las muestras son aproximadamente independientes e idénticamente distribuidas (i.i.d.).
   - Cálculo de Memoria RAM:
     * Obs shape: (4, 84, 84) uint8 = 28,224 bytes por estado.
     * Para N = 10,000 transiciones:
       obs + next_obs = 10,000 * 28,224 * 2 = 564.48 MB
       acciones + recompensas + dones = 10,000 * (8 + 4 + 4) = 0.16 MB
       TOTAL ESTIMADO DE RAM = ~538 a 565 MB (completamente seguro para los 1.6 GB libres).

3. Red Objetivo (Target Network, theta^-):
   - Estabiliza el aprendizaje resolviendo el problema del objetivo móvil:
     y_t = r_t + gamma * max_a' Q(s_{t+1}, a'; theta^-)
   - Los pesos theta^- se mantienen congelados y se actualizan periódicamente cada N pasos.

4. Espacio de Acciones y Mínimos Locales en RL:
   - En ViZDoom Basic, un agente que tiene la acción 'NO-OP' (quedarse quieto) descubre que
     fallar un tiro penaliza más (-410) que quedarse quieto (-300).
   - Para evitar este mínimo local pasivo y forzar al agente a aprender a apuntar y disparar,
     usamos DiscardNoOpWrapper (acciones activas: 0=Disparar, 1=Moverse Derecha, 2=Moverse Izquierda).
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from doom_wrappers import make_doom_env


class DiscardNoOpWrapper(gym.ActionWrapper):
    """
    Wrapper para eliminar la acción pasiva 'NO-OP' (no presionar ningún botón).
    Acciones resultantes:
    - 0: ATTACK (disparar el arma)
    - 1: MOVE_RIGHT (desplazarse a la derecha)
    - 2: MOVE_LEFT (desplazarse a la izquierda)
    
    Explicación teórica:
    En entornos con penalizaciones de tiempo y munición, una acción pasiva permite que
    el agente caiga en el mínimo local de 'no hacer nada' para evitar el costo de fallar disparos.
    Al forzar acciones activas, el agente aprende a coordinar movimiento y disparo.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Discrete(3)

    def action(self, act: int) -> int:
        # En ViZDoom Basic: 0=NO-OP, 1=ATTACK, 2=MOVE_RIGHT, 3=MOVE_LEFT
        # Mapeamos 0->1, 1->2, 2->3
        return act + 1


def crear_entorno_fase2(render_mode=None):
    """Crea el entorno de ViZDoom Basic con pipeline completo y wrapper activo."""
    env = make_doom_env("VizdoomBasic-v1", frame_skip=4, stack_size=4, resolution=(84, 84), render_mode=render_mode)
    env = DiscardNoOpWrapper(env)
    env = Monitor(env)
    return env


class MonitorRecursosCallback(BaseCallback):
    """
    Callback para monitorear el progreso del entrenamiento, recompensa reciente,
    pasos/segundo (FPS) y consumo de memoria RAM y VRAM en GPU.
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_time = None
        self.last_step = 0

    def _on_training_start(self) -> None:
        self.last_time = time.time()
        self.last_step = self.num_timesteps
        print("\n" + "=" * 78)
        print(f"{'PASO':^8} | {'RECOMPENSA REC.':^17} | {'EPSILON':^8} | {'FPS':^7} | {'RAM (GB)':^10} | {'VRAM GPU':^10}")
        print("=" * 78)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            steps_done = self.num_timesteps - self.last_step
            fps = steps_done / elapsed if elapsed > 0 else 0.0

            # Recompensa media de episodios recientes
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                ep_count = len(self.model.ep_info_buffer)
                reward_str = f"{mean_reward:6.1f} (n={ep_count})"
            else:
                reward_str = "Recolectando..."

            epsilon = self.model.exploration_rate
            mem_info = psutil.virtual_memory()
            ram_used_gb = (mem_info.total - mem_info.available) / (1024**3)

            if torch.cuda.is_available():
                vram_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
                vram_str = f"{vram_used_mb:6.1f} MB"
            else:
                vram_str = "N/A (CPU)"

            print(f"{self.num_timesteps:^8d} | {reward_str:^17} | {epsilon:^8.3f} | {fps:^7.1f} | {ram_used_gb:^10.2f} | {vram_str:^10}")
            self.last_time = current_time
            self.last_step = self.num_timesteps

        return True


def ejecutar_prueba_humo_dqn():
    print("=" * 78)
    print("FASE 2: PRUEBA DE HUMO CON DQN EN VIZDOOM BASIC")
    print("=" * 78)

    BUFFER_SIZE = 10_000
    TOTAL_TIMESTEPS = 6_000
    OBS_BYTES = 4 * 84 * 84  # 28,224 bytes
    ESTIMATED_BUFFER_MB = (BUFFER_SIZE * OBS_BYTES * 2) / (1024**2)

    print(f"Capacidad del Replay Buffer:     {BUFFER_SIZE:,} transiciones")
    print(f"Uso de RAM estimado del Buffer:  ~{ESTIMATED_BUFFER_MB:.1f} MB")
    print(f"Pasos totales de entrenamiento:  {TOTAL_TIMESTEPS:,} pasos (~2 minutos)")
    print(f"Dispositivo de entrenamiento:    CUDA (GPU: {torch.cuda.get_device_name(0)})")

    os.makedirs("models/dqn_basic", exist_ok=True)
    os.makedirs("logs/dqn_basic", exist_ok=True)

    # Entorno de entrenamiento
    env = crear_entorno_fase2()

    # Configuración de DQN con parámetros ajustados para aprendizaje rápido
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=BUFFER_SIZE,
        learning_starts=300,
        batch_size=32,
        tau=1.0,
        target_update_interval=500,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="logs/dqn_basic",
        verbose=0,
        seed=42,
        device="cuda"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=3000,
        save_path="models/dqn_basic",
        name_prefix="dqn_basic_chkpt"
    )
    monitor_callback = MonitorRecursosCallback(check_freq=1000)

    print("\nIniciando entrenamiento corto...")
    t_start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, monitor_callback],
        progress_bar=False
    )
    t_total = time.time() - t_start

    print("=" * 78)
    print(f"Entrenamiento completado en {t_total:.1f} s ({t_total/60:.2f} min).")
    print(f"Velocidad promedio global: {TOTAL_TIMESTEPS / t_total:.1f} pasos/segundo.")

    # Guardar modelo final
    model_path = "models/dqn_basic_final.zip"
    model.save(model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

    # Evaluación cuantitativa en 10 episodios
    print("\n" + "=" * 78)
    print("EVALUACIÓN DEL AGENTE DQN ENTRENADO (10 Episodios - Determinista)")
    print("=" * 78)

    eval_env = crear_entorno_fase2()
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )

    # Detalle de algunos episodios
    print("Muestra de episodios de evaluación individual:")
    for i in range(5):
        obs, _ = eval_env.reset(seed=100 + i)
        done = False
        ep_r = 0.0
        steps = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(act)
            ep_r += r
            steps += 1
            done = term or trunc
        resultado = "¡Monstruo Eliminado!" if ep_r > 0 else "Tiempo agotado / Falló"
        print(f"  Episodio {i + 1}: Retorno = {ep_r:7.1f} | Pasos = {steps:2d} -> {resultado}")

    eval_env.close()
    env.close()

    print("-" * 78)
    print(f"Recompensa promedio Agente DQN:  {mean_reward:7.2f} +/- {std_reward:.2f}")
    print(f"Línea base Fase 0 (Aleatorio): -184.00 +/- 225.77")
    delta_r = mean_reward - (-184.00)
    print(f"Mejora sobre política aleatoria: +{delta_r:.2f} puntos")
    print("=" * 78)


if __name__ == "__main__":
    ejecutar_prueba_humo_dqn()

"""
fase0_reconocimiento.py
=======================
Fase 0: Reconocimiento del entorno y línea base aleatoria (sin entrenar).

Conceptos teóricos demostrados aquí:
- Entorno (Environment): El mundo de ViZDoom donde interactúa el agente.
- Agente (Agent): La entidad que toma decisiones (aquí, totalmente aleatorias: pi(a|s) = 1/|A|).
- Estado/Observación (State/Observation): Lo que el agente percibe (píxeles de la pantalla del juego).
- Acción (Action): Los comandos disponibles (moverse, disparar, etc.). En espacio discreto Discrete(n).
- Recompensa (Reward): La señal escalar que el entorno devuelve en cada paso temporal t.
- Retorno/Recompensa acumulada: G_t = sum(r_k). Mide el desempeño global de la política aleatoria.
"""

import sys
import os
import torch
import gymnasium as gym
import vizdoom
import vizdoom.gymnasium_wrapper
import stable_baselines3
import imageio
import numpy as np


def verificar_sistema():
    """Verifica versiones instaladas y disponibilidad de GPU."""
    print("=" * 60)
    print("1. VERIFICACIÓN DE VERSIONES Y HARDWARE")
    print("=" * 60)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA activo:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("ADVERTENCIA: CUDA no está disponible. PyTorch usará la CPU.")
    print(f"Gymnasium:    {gym.__version__}")
    print(f"ViZDoom:      {vizdoom.__version__}")
    print(f"SB3:          {stable_baselines3.__version__}")
    print()


def inspeccionar_escenarios():
    """Inspecciona los espacios de observación y acción de los escenarios principales."""
    print("=" * 60)
    print("2. INSPECCIÓN DE ESCENARIOS DISPONIBLES EN GIMNASIUM")
    print("=" * 60)

    escenarios = [
        ("VizdoomBasic-v1", "Escenario básico: disparar a un monstruo que se mueve enfrente"),
        ("VizdoomDefendCenter-v1", "Defensa central: sobrevivir disparando en 360 grados"),
        ("VizdoomDeadlyCorridor-v1", "Corredor mortal: avanzar por un pasillo esquivando y eliminando enemigos"),
        ("VizdoomHealthGathering-v1", "Recolección de vida: sobrevivir en un suelo tóxico recolectando medikits")
    ]

    for env_id, descripcion in escenarios:
        print(f"\n--- {env_id} ---")
        print(f"Descripción: {descripcion}")
        env = gym.make(env_id)
        obs, info = env.reset(seed=42)

        print(f"Espacio de acciones (Action Space): {env.action_space}")
        print(f"Espacio de observaciones (Observation Space): {env.observation_space}")

        if isinstance(obs, dict):
            print("Estructura de la observación (Diccionario):")
            for k, v in obs.items():
                if hasattr(v, "shape"):
                    print(f"  - '{k}': forma={v.shape}, tipo={v.dtype}, rango=[{v.min()}, {v.max()}]")
                else:
                    print(f"  - '{k}': {v}")
        else:
            print(f"Observación: forma={obs.shape}, tipo={obs.dtype}, rango=[{obs.min()}, {obs.max()}]")

        env.close()
    print()


def ejecutar_agente_aleatorio(env_id="VizdoomBasic-v1", num_episodios=5, video_path="videos/agente_aleatorio_basic.mp4"):
    """
    Ejecuta un agente con política aleatoria pi(a|s) = uniforme(A).
    Guarda los frames para generar un video de referencia 'antes de entrenar'.
    """
    print("=" * 60)
    print(f"3. LÍNEA BASE: AGENTE ALEATORIO EN {env_id}")
    print("=" * 60)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # render_mode='rgb_array' nos permite extraer los frames visuales completos
    env = gym.make(env_id, render_mode="rgb_array")
    
    frames = []
    recompensas_episodios = []
    longitudes_episodios = []
    MAX_VIDEO_FRAMES = 150  # ~5 segundos a 30 FPS para una demostración clara en la presentación

    for ep in range(num_episodios):
        obs, info = env.reset(seed=42 + ep)
        terminado = False
        truncado = False
        recompensa_total = 0.0
        pasos = 0

        # Guardamos el frame inicial
        if len(frames) < MAX_VIDEO_FRAMES:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        while not (terminado or truncado):
            # Acción aleatoria tomada del espacio de acciones del entorno
            accion = env.action_space.sample()

            # El entorno procesa la acción y devuelve:
            # - nuevo estado (obs)
            # - recompensa inmediata (reward)
            # - banderas de fin de episodio (terminated / truncated)
            obs, recompensa, terminado, truncado, info = env.step(accion)
            recompensa_total += recompensa
            pasos += 1

            if len(frames) < MAX_VIDEO_FRAMES:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        recompensas_episodios.append(recompensa_total)
        longitudes_episodios.append(pasos)
        print(f"Episodio {ep + 1}/{num_episodios} - Retorno total: {recompensa_total:8.2f} | Pasos: {pasos:4d}")

    env.close()

    # Guardar video con imageio
    if frames:
        print(f"\nGuardando video con {len(frames)} frames en: {video_path} ...")
        # ViZDoom corre típicamente a ~35 fps por defecto
        imageio.mimsave(video_path, frames, fps=30)
        print("¡Video guardado exitosamente!")

    media_r = np.mean(recompensas_episodios)
    std_r = np.std(recompensas_episodios)
    print(f"\nResumen Agente Aleatorio ({num_episodios} episodios):")
    print(f"Recompensa promedio: {media_r:.2f} +/- {std_r:.2f}")
    print(f"Duración promedio de episodio: {np.mean(longitudes_episodios):.1f} pasos")
    print("=" * 60)


if __name__ == "__main__":
    verificar_sistema()
    inspeccionar_escenarios()
    ejecutar_agente_aleatorio()

"""
evaluar_agente.py
=================
Fase 4 y 5: Evaluación y grabación de video con HUD/Telemetría en pantalla.

Superpone en cada fotograma del video:
- Nombre del modelo / algoritmo / hito (DQN vs PPO, 0% vs 50% vs 100%)
- Salud del jugador (HEALTH) con barra visual de color
- Munición restante (AMMO2)
- Monstruos eliminados (Kills)
- Acción ejecutada por la red (ATTACK, TURN_LEFT, TURN_RIGHT, NO-OP)
- Paso actual y tiempo de simulación en tics de Doom
"""

import os
import sys
import argparse
import time
import cv2
import torch
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from doom_wrappers import make_doom_env


NOMBRES_ACCIONES_DEFEND = {
    0: "NO-OP (Inactivo)",
    1: "ATTACK (Disparo)",
    2: "TURN_RIGHT (Giro Der)",
    3: "TURN_LEFT (Giro Izq)"
}

NOMBRES_ACCIONES_DEADLY = {
    0: "NO-OP",
    1: "MOVE_FWD (Avanzar)",
    2: "FWD+ATTACK (Avanzar Disparando)",
    3: "TURN_L+ATK (Giro Izq + Disparo)",
    4: "TURN_R+ATK (Giro Der + Disparo)",
    5: "TURN_LEFT (Giro Izquierda)",
    6: "TURN_RIGHT (Giro Derecha)",
    7: "MOVE_LEFT (Strafe Izq)",
    8: "MOVE_RIGHT (Strafe Der)",
    9: "ATTACK (Disparo)"
}


def superponer_hud(frame, etiqueta_modelo, ep_num, total_eps, paso, salud, municion, bajas, accion_id, scenario="defend", ep_retorno=0.0):
    """
    Superpone una barra de telemetría superior e inferior con texto nítido
    escalando la resolución a 640x480 para proyección en diapositivas.
    """
    frame_hd = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

    # 1. Banda superior translúcida (Información del Modelo y Estado)
    overlay = frame_hd.copy()
    cv2.rectangle(overlay, (0, 0), (640, 48), (15, 17, 23), -1)
    # Banda inferior translúcida (Controles y Bajas)
    cv2.rectangle(overlay, (0, 432), (640, 480), (15, 17, 23), -1)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame_hd, 1 - alpha, 0, frame_hd)

    # 2. Texto superior: Nombre del Modelo, Episodio y Pasos
    cv2.putText(frame_hd, f"MODELO: {etiqueta_modelo}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 166, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"RONDA {ep_num}/{total_eps} | PASO {paso:03d}", (410, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (201, 209, 217), 1, cv2.LINE_AA)

    # Línea divisoria superior e inferior
    cv2.line(frame_hd, (0, 48), (640, 48), (68, 76, 86), 1)
    cv2.line(frame_hd, (0, 432), (640, 432), (68, 76, 86), 1)

    # 3. Datos de supervivencia (Salud y Munición)
    color_hp = (63, 185, 80) if salud > 50 else ((210, 153, 34) if salud > 25 else (248, 81, 73))
    cv2.putText(frame_hd, f"SALUD: {int(salud)}%", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_hp, 1, cv2.LINE_AA)
    
    # Barra de salud gráfica
    barra_x, barra_y, barra_w, barra_h = 130, 31, 100, 12
    cv2.rectangle(frame_hd, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (50, 50, 50), -1)
    hp_ancho = max(0, min(barra_w, int((salud / 100.0) * barra_w)))
    cv2.rectangle(frame_hd, (barra_x, barra_y), (barra_x + hp_ancho, barra_y + barra_h), color_hp, -1)
    cv2.rectangle(frame_hd, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (139, 148, 158), 1)

    # Munición (si existe en el escenario)
    if municion is not None:
        color_ammo = (248, 81, 73) if municion <= 5 else (201, 209, 217)
        cv2.putText(frame_hd, f"MUNICION: {int(municion)} balas", (250, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_ammo, 1, cv2.LINE_AA)

    # 4. Texto inferior: Bajas y Acción de la Red
    acc_map = NOMBRES_ACCIONES_DEADLY if "deadly" in scenario.lower() else NOMBRES_ACCIONES_DEFEND
    accion_str = acc_map.get(accion_id, f"Accion {accion_id}")
    color_act = (255, 255, 0) if "ATTACK" in accion_str else ((88, 166, 255) if "FWD" in accion_str else (163, 113, 247))
    
    if "deadly" in scenario.lower():
        cv2.putText(frame_hd, f"KILLS: {int(bajas)} | RET: {ep_retorno:+.1f}", (12, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 215, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame_hd, f"SCORE: {bajas:.1f}", (12, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"ACCION RED: {accion_str}", (260, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_act, 1, cv2.LINE_AA)

    return frame_hd


def evaluar_modelo(
    checkpoint_path: str,
    scenario: str = "VizdoomDefendCenter-v1",
    num_episodios: int = 5,
    num_video_episodes: int = 3,
    deterministic: bool = True,
    record_video: bool = False,
    video_path: str = None,
    seed: int = 20,
    max_video_frames: int = 1200,
    fps_video: int = 20,
    etiqueta_custom: str = None
):
    print("=" * 78)
    print("EVALUACIÓN DE AGENTE DEEP RL EN VIZDOOM CON TELEMETRÍA")
    print(f"Escenario:       {scenario}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Episodios eval:  {num_episodios}")
    print(f"Episodios video: {num_video_episodes if record_video else 0}")
    print(f"Determinista:    {deterministic}")
    print(f"Grabar video:    {record_video} ({video_path if record_video else 'N/A'})")
    print("=" * 78)

    render_mode = "rgb_array" if record_video else None
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make_eval_env():
        return make_doom_env(scenario, frame_skip=4, stack_size=4, resolution=(84, 84), render_mode=render_mode)

    env = DummyVecEnv([_make_eval_env])

    # Cargar modelo (DQN o PPO)
    try:
        model = DQN.load(checkpoint_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        algo_name = "DQN"
    except Exception:
        model = PPO.load(checkpoint_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        algo_name = "PPO"

    etiqueta_modelo = etiqueta_custom if etiqueta_custom else f"{algo_name} ({os.path.basename(checkpoint_path)})"
    print(f"Modelo cargado: {etiqueta_modelo}")

    recompensas = []
    duraciones = []
    video_frames = []

    for ep in range(num_episodios):
        obs = env.reset()
        terminado = False
        ep_recompensa = 0.0
        ep_pasos = 0
        bajas_acumuladas = 0

        grabar_este_ep = record_video and (ep < num_video_episodes) and (len(video_frames) < max_video_frames)

        raw_game = env.envs[0].unwrapped.game
        state = raw_game.get_state()
        is_deadly = "deadly" in scenario.lower()
        if is_deadly:
            salud = state.game_variables[0] if state else 100.0
            municion = None
            bajas_acumuladas = state.game_variables[1] if (state and len(state.game_variables) > 1) else 0
        else:
            if state and len(state.game_variables) > 1:
                salud = state.game_variables[1]
                municion = state.game_variables[0]
            elif state and len(state.game_variables) == 1:
                salud = state.game_variables[0]
                municion = None
            else:
                salud = 100.0
                municion = None

        if grabar_este_ep:
            frame = env.envs[0].render()
            if frame is not None:
                frame_hud = superponer_hud(frame, etiqueta_modelo, ep + 1, num_video_episodes, ep_pasos, salud, municion, bajas_acumuladas, 0, scenario=scenario, ep_retorno=ep_recompensa)
                video_frames.append(frame_hud)

        while not terminado:
            accion, _ = model.predict(obs, deterministic=deterministic)
            obs, recompensa_vec, dones, infos = env.step(accion)
            recompensa = recompensa_vec[0]
            terminado = dones[0]
            ep_recompensa += recompensa
            ep_pasos += 1

            state = raw_game.get_state()
            if is_deadly:
                salud = state.game_variables[0] if state else 0.0
                municion = None
                if state and len(state.game_variables) > 1:
                    bajas_acumuladas = state.game_variables[1]
            else:
                if state:
                    if len(state.game_variables) > 1:
                        salud = state.game_variables[1]
                        municion = state.game_variables[0]
                    elif len(state.game_variables) == 1:
                        salud = state.game_variables[0]
                        municion = None
                else:
                    salud = 0.0
                if recompensa > 0.0:
                    bajas_acumuladas += recompensa

            if grabar_este_ep and len(video_frames) < max_video_frames:
                frame = env.envs[0].render()
                if frame is not None:
                    act_int = int(accion[0]) if hasattr(accion, "__len__") else int(accion)
                    frame_hud = superponer_hud(frame, etiqueta_modelo, ep + 1, num_video_episodes, ep_pasos, salud, municion, bajas_acumuladas, act_int, scenario=scenario, ep_retorno=ep_recompensa)
                    video_frames.append(frame_hud)

        recompensas.append(ep_recompensa)
        duraciones.append(ep_pasos)
        info_kills = f"Kills: {int(bajas_acumuladas):2d}" if is_deadly else f"Score Positivo: {bajas_acumuladas:6.1f}"
        print(f"  Episodio {ep + 1:2d}/{num_episodios:2d} -> {info_kills} | Retorno: {ep_recompensa:6.1f} | Pasos: {ep_pasos:3d} ({ep_pasos*4} tics)")

    env.close()

    if record_video and video_frames and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        segundos = len(video_frames) / fps_video
        print(f"\nExportando video ({len(video_frames)} frames a {fps_video} FPS = {segundos:.1f} s) a: {video_path} ...")
        imageio.mimsave(video_path, video_frames, fps=fps_video)
        print("¡Video con telemetría guardado con éxito!")

    mean_r = np.mean(recompensas)
    std_r = np.std(recompensas)
    mean_l = np.mean(duraciones)

    print("\n" + "=" * 78)
    print("RESUMEN ESTADÍSTICO DE EVALUACIÓN")
    print(f"Recompensa promedio: {mean_r:7.2f} +/- {std_r:.2f}")
    print(f"Supervivencia media: {mean_l:7.1f} pasos ({mean_l * 4:.1f} tics de Doom)")
    print("=" * 78)

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_length": mean_l
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar checkpoint y grabar video con HUD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo .zip del modelo")
    parser.add_argument("--scenario", type=str, default="VizdoomDefendCenter-v1", help="ID del escenario de ViZDoom")
    parser.add_argument("--episodes", type=int, default=5, help="Número de episodios a evaluar")
    parser.add_argument("--stochastic", action="store_true", help="Usar política estocástica")
    parser.add_argument("--record_video", action="store_true", help="Habilitar grabación de video")
    parser.add_argument("--video_path", type=str, default=None, help="Ruta de salida para el video .mp4")
    parser.add_argument("--seed", type=int, default=20, help="Semilla inicial")
    parser.add_argument("--label", type=str, default=None, help="Etiqueta para mostrar en el HUD")

    args = parser.parse_args()
    evaluar_modelo(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodios=args.episodes,
        deterministic=not args.stochastic,
        record_video=args.record_video,
        video_path=args.video_path,
        seed=args.seed,
        etiqueta_custom=args.label
    )

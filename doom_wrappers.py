"""
doom_wrappers.py
================
Módulo de wrappers de Gymnasium para el preprocesamiento de ViZDoom.

Conceptos teóricos de Deep Reinforcement Learning demostrados:
1. Observabilidad Parcial vs Propiedad de Markov (POMDP -> MDP):
   Una sola imagen fija no contiene información de velocidad ni dirección de proyectiles
   o enemigos (el estado actual s_t no resumiría toda la historia pasada).
   Al apilar k=4 fotogramas consecutivos (Frame Stacking), el agente percibe la trayectoria,
   velocidad y aceleración en el espacio de estados, restaurando la propiedad de Markov:
   P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...).

2. Reducción de Dimensionalidad y Eficiencia de Memoria:
   - Entrada original: 240 x 320 x 3 canales = 230,400 valores uint8 por fotograma.
   - Preprocesado: 84 x 84 x 1 canal = 7,056 valores uint8 por fotograma.
   - ¡Reducción de dimensionalidad de ~32.6 veces por frame!
   - Esto hace factible almacenar cientos de miles de transiciones en el Replay Buffer (DQN)
     sin saturar la memoria RAM del sistema.

3. Abstracción Temporal (Frame Skip, k=4):
   En Doom (~35 FPS nativos), un solo tick dura ~28 ms, tiempo en el cual un jugador humano
   no cambia de decisión. Repetir la misma acción durante k=4 tics:
   - Acelera el entrenamiento 4 veces en tics de simulación por inferencia de la red.
   - Reduce el horizonte temporal efectivo para la asignación de crédito (Credit Assignment).
   - Acumula las recompensas obtenidas durante los k tics para no perder señal.
"""

import os
import io
import zipfile
from typing import Optional, Tuple
import psutil
import torch as th
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import vizdoom.gymnasium_wrapper  # Registra los entornos de ViZDoom en Gymnasium
import stable_baselines3.common.save_util as save_util
import stable_baselines3.common.base_class as base_class


# ---------------------------------------------------------------------------
# PARCHE DE COMPATIBILIDAD: PyTorch 2.14 + Stable-Baselines3 en Windows
# En PyTorch 2.14, th.load() sobre ZipExtFile genera un error interno de miniz
# al buscar streams anidados. Envolver los bytes en io.BytesIO() lo soluciona.
# ---------------------------------------------------------------------------
def _patched_load_from_zip_file(load_path, load_data=True, custom_objects=None, device="auto", verbose=0, print_system_info=False):
    file = save_util.open_path(load_path, "r", verbose=verbose, suffix="zip")
    device = save_util.get_device(device=device)
    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            data, pytorch_variables, params = None, None, {}
            if "data" in namelist and load_data:
                data = save_util.json_to_data(archive.read("data").decode(), custom_objects=custom_objects)
            for f in [x for x in namelist if x.endswith(".pth")]:
                buf = io.BytesIO(archive.read(f))
                obj = th.load(buf, map_location=device, weights_only=True)
                if f in ("pytorch_variables.pth", "tensors.pth"):
                    pytorch_variables = obj
                else:
                    params[os.path.splitext(f)[0]] = obj
    finally:
        if hasattr(file, "close"):
            file.close()
    return data, params, pytorch_variables

save_util.load_from_zip_file = _patched_load_from_zip_file
base_class.load_from_zip_file = _patched_load_from_zip_file


class ExtractScreenWrapper(gym.ObservationWrapper):
    """
    Wrapper que extrae únicamente la pantalla visual ('screen') del diccionario de ViZDoom.
    
    Por defecto, ViZDoom entrega un diccionario con 'screen' y 'gamevariables' (salud, munición).
    Para garantizar que el agente aprenda exclusivamente a partir de la visión cruda
    (visión por computadora de extremo a extremo, estilo DQN Nature 2015),
    descartamos las variables auxiliares.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict), "Se esperaba un Dict como observation_space"
        assert "screen" in env.observation_space.spaces, "El espacio de observaciones no contiene la clave 'screen'"
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs: dict):
        return obs["screen"]


class DeadlyCorridorTacticalWrapper(gym.Wrapper):
    """
    Wrapper táctico para ViZDoom Deadly Corridor que:
    1. Fija la dificultad en Skill 3 (Hurt Me Plenty - estándar justa para combate táctico).
    2. Habilita 10 acciones compuestas incluyendo giro angular con disparo (TURN_LEFT+ATTACK, TURN_RIGHT+ATTACK).
    3. Rebalancea la recompensa (+150 por baja, atenúa avance ciego a 0.25x, penaliza daño -0.50 y bonifica llegar vivo con +200).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        import vizdoom as vzd
        game = self.env.unwrapped.game
        game.set_doom_skill(3)
        game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)

        # 0: MOVE_LEFT, 1: MOVE_RIGHT, 2: ATTACK, 3: MOVE_FORWARD, 4: MOVE_BACKWARD, 5: TURN_LEFT, 6: TURN_RIGHT
        custom_actions = [
            [0, 0, 0, 0, 0, 0, 0], # 0: NO-OP
            [0, 0, 0, 1, 0, 0, 0], # 1: MOVE_FORWARD
            [0, 0, 1, 1, 0, 0, 0], # 2: MOVE_FORWARD + ATTACK
            [0, 0, 1, 0, 0, 1, 0], # 3: TURN_LEFT + ATTACK (apuntar y disparar al nicho izq)
            [0, 0, 1, 0, 0, 0, 1], # 4: TURN_RIGHT + ATTACK (apuntar y disparar al nicho der)
            [0, 0, 0, 0, 0, 1, 0], # 5: TURN_LEFT
            [0, 0, 0, 0, 0, 0, 1], # 6: TURN_RIGHT
            [1, 0, 0, 0, 0, 0, 0], # 7: MOVE_LEFT (strafe)
            [0, 1, 0, 0, 0, 0, 0], # 8: MOVE_RIGHT (strafe)
            [0, 0, 1, 0, 0, 0, 0], # 9: ATTACK
        ]
        self.env.unwrapped.button_map = [np.array(a, dtype=np.int32) for a in custom_actions]
        self.action_space = gym.spaces.Discrete(len(custom_actions))
        curr = self.env
        while curr is not None:
            curr.action_space = self.action_space
            curr = getattr(curr, "env", None)
        self.last_health = 100.0
        self.last_kills = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_health = 100.0
        self.last_kills = 0.0
        return obs, info

    def step(self, action):
        if hasattr(action, "item"):
            action = action.item()
        action = int(action)

        obs, reward, term, trunc, info = self.env.step(action)
        raw_game = self.env.unwrapped.game
        state = raw_game.get_state()

        # 1. Escalar avance por distancia hacia la meta (0.40x para guiarlo al fondo)
        shaped_reward = reward * 0.40 if reward > 0 else reward

        # 2. Pequeño costo por paso (urgencia para no quedarse campeando)
        shaped_reward -= 0.10

        if state:
            curr_health = state.game_variables[0]
            curr_kills = state.game_variables[1]

            # 3. Gran bonificación por eliminar amenazas (+150 pts por baja)
            kill_diff = curr_kills - self.last_kills
            if kill_diff > 0:
                shaped_reward += kill_diff * 150.0

            # 4. Penalización por daño (-0.50 por punto de salud perdido)
            health_diff = self.last_health - curr_health
            if health_diff > 0:
                shaped_reward -= health_diff * 0.50

            self.last_health = curr_health
            self.last_kills = curr_kills
        elif term:
            # Penalización severa al morir (-50 extra a los -100 del motor)
            shaped_reward -= 50.0

        # 5. Bonificación EXCLUSIVA por llegar al final y tocar el chaleco verde vivo
        if term and not trunc and self.last_health > 0:
            shaped_reward += 400.0

        # 6. Penalización si se le acaba el tiempo sin llegar al chaleco (fracaso por campear)
        if trunc and not term:
            shaped_reward -= 100.0

        return obs, shaped_reward, term, trunc, info


def make_doom_env(
    env_id: str = "VizdoomBasic-v1",
    frame_skip: int = 4,
    stack_size: int = 4,
    resolution: Tuple[int, int] = (84, 84),
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Construye y envuelve el entorno de ViZDoom con la canalización estándar de Deep RL:
    1. Instanciación con frame_skip nativo en el motor C++ de ViZDoom.
    2. Wrapper táctico especializado si es DeadlyCorridor.
    3. Extracción de pantalla 'screen' (ignora variables internas).
    4. Conversión a escala de grises (1 canal de intensidad de luminancia).
    5. Redimensionamiento a resolución 84x84 (estándar Nature CNN).
    6. Apilamiento de fotogramas (FrameStack de k fotogramas en formato Canales x Alto x Ancho).

    Retorna un entorno listo para usar con Stable-Baselines3 (DQN, PPO, A2C).
    """
    # 1. Entorno base con frame_skip integrado
    env = gym.make(env_id, frame_skip=frame_skip, render_mode=render_mode)

    # 2. Wrapper táctico para Deadly Corridor (acciones compuestas y reward shaping)
    if "deadlycorridor" in env_id.lower() or "deadly" in env_id.lower():
        env = DeadlyCorridorTacticalWrapper(env)

    # 3. Extraer píxeles de pantalla (elimina el Dict)
    env = ExtractScreenWrapper(env)

    # 4. Escala de grises: (H, W, 3) -> (H, W)
    env = GrayscaleObservation(env, keep_dim=False)

    # 5. Redimensionamiento: (H, W) -> (84, 84)
    env = ResizeObservation(env, shape=resolution)

    # 6. Apilado temporal: (84, 84) con stack=4 -> (4, 84, 84)
    # PyTorch y SB3 esperan formato Channel-First: (Canales, Alto, Ancho)
    env = FrameStackObservation(env, stack_size=stack_size)

    return env


if __name__ == "__main__":
    # Prueba de sanidad directa
    print("Probando instanciación de entorno preprocesado...")
    test_env = make_doom_env("VizdoomBasic-v1")
    obs, info = test_env.reset(seed=42)
    print(f"Espacio de observación final: {test_env.observation_space}")
    print(f"Forma de la observación:      {obs.shape}")
    print(f"Tipo de datos (dtype):         {obs.dtype}")
    print(f"Espacio de acciones:          {test_env.action_space}")
    test_env.close()
    print("¡Entorno configurado correctamente!")

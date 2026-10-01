"""
fase1_preprocesamiento.py
=========================
Fase 1: Verificación visual del pipeline de preprocesamiento.

Este script comprueba la transformación de la observación visual que recibe
la red neuronal convolucional (CNN), guardando una figura ilustrativa en:
'videos/verificacion_observacion_red.png'.

Conexión teórica para la presentación:
1. De POMDP a MDP:
   - Un solo fotograma estático no permite saber si el enemigo va a la izquierda
     o a la derecha, ni si una bola de fuego se acerca o se aleja.
   - Los 4 fotogramas apilados [t-3, t-2, t-1, t] proporcionan la derivada temporal (velocidad).
2. Eficiencia de memoria y cómputo:
   - Imagen original: (240, 320, 3) = 230,400 bytes (~225 KB) por estado.
   - Imagen procesada: (4, 84, 84) = 28,224 bytes (~27.5 KB) por estado.
   - Reducción del 87.7% en peso por paso, crucial para el Replay Buffer de DQN.
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import vizdoom.gymnasium_wrapper
from doom_wrappers import make_doom_env


def verificar_preprocesamiento(output_image_path="videos/verificacion_observacion_red.png"):
    print("=" * 65)
    print("FASE 1: VERIFICACIÓN VISUAL DEL PREPROCESAMIENTO")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Creamos dos entornos:
    # 1. El entorno crudo para capturar la imagen RGB original tal como se ve en pantalla
    # 2. El entorno preprocesado para obtener lo que ingresará a la CNN
    raw_env = gym.make("VizdoomBasic-v1", frame_skip=4, render_mode="rgb_array")
    proc_env = make_doom_env("VizdoomBasic-v1", frame_skip=4, stack_size=4, resolution=(84, 84))

    raw_obs, _ = raw_env.reset(seed=123)
    proc_obs, _ = proc_env.reset(seed=123)

    # Realizamos acciones de movimiento y disparo para capturar movimiento real
    # Acción 2: MOVE_RIGHT, Acción 1: ATTACK (disparo con fogonazo)
    for a in [2, 2, 1]:
        raw_obs, _, _, _, _ = raw_env.step(a)
        proc_obs, _, _, _, _ = proc_env.step(a)

    # Frame original RGB
    rgb_frame = raw_obs["screen"]  # forma: (240, 320, 3)

    # Observación procesada que entra a la CNN
    # proc_obs tiene forma: (4, 84, 84), dtype=uint8
    print(f"Forma original de la imagen:       {rgb_frame.shape} (RGB uint8)")
    print(f"Forma de la observación a la CNN:  {proc_obs.shape} (4 frames en escala de grises)")
    print(f"Rango de valores de la red:        min={proc_obs.min()}, max={proc_obs.max()}")

    # Calculamos la diferencia de movimiento entre el frame t y el frame t-1
    diff_frame = np.abs(proc_obs[3].astype(np.int16) - proc_obs[2].astype(np.int16)).astype(np.uint8)

    # =========================================================================
    # Creación de la figura visual para la presentación
    # =========================================================================
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1e1e1e")

    titulos_stack = ["t - 3", "t - 2", "t - 1", "t (Actual)"]

    # Panel izquierdo: Original (grande)
    ax_orig = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax_orig.imshow(rgb_frame)
    ax_orig.set_title(
        f"Pantalla del Juego (Original)\n{rgb_frame.shape[0]}x{rgb_frame.shape[1]} RGB | 3 canales\nPeso: {rgb_frame.nbytes/1024:.1f} KB",
        color="white",
        fontsize=11,
        pad=8
    )
    ax_orig.axis("off")

    # 4 Frames apilados en la parte superior derecha
    for i in range(4):
        ax_f = plt.subplot2grid((2, 6), (0, 2 + i))
        ax_f.imshow(proc_obs[i], cmap="gray", vmin=0, vmax=255)
        ax_f.set_title(
            f"Frame {titulos_stack[i]}\n(84x84)",
            color="#58a6ff" if i == 3 else "#8b949e",
            fontsize=10,
            pad=6
        )
        ax_f.axis("off")

    # Fila inferior: Demostración de movimiento y tensor de entrada
    # Diferencia entre frames (velocidad)
    ax_diff = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    im_diff = ax_diff.imshow(diff_frame, cmap="hot")
    ax_diff.set_title(
        "Detección de Movimiento (|t - (t-1)|)\nLa red detecta velocidad y trayectoria",
        color="#f0883e",
        fontsize=10,
        pad=6
    )
    ax_diff.axis("off")
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

    # Resumen del Tensor que entra a la CNN
    ax_info = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax_info.axis("off")
    texto_resumen = (
        "TENSOR DE ENTRADA A LA CNN:\n"
        f"- Forma: [4, 84, 84] (Canales x Alto x Ancho)\n"
        f"- Tipo: uint8 [0 - 255] (Normalizado a [0, 1] en GPU)\n"
        f"- Peso por estado: {proc_obs.nbytes/1024:.1f} KB\n"
        f"- Reducción de datos: {rgb_frame.nbytes / proc_obs.nbytes:.1f}x más ligero\n\n"
        "BENEFICIOS CLAVE PARA TEORÍA:\n"
        "• Resuelve POMDP: añade noción de velocidad\n"
        "• Replay Buffer: 100k pasos caben en ~2.7 GB de RAM\n"
        "• Frame Skip (k=4): acelera 4x la simulación"
    )
    ax_info.text(
        0.05, 0.5,
        texto_resumen,
        color="white",
        fontsize=10,
        va="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#2d333b", edgecolor="#444c56")
    )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    raw_env.close()
    proc_env.close()

    print(f"\n¡Imagen de verificación guardada con éxito en: {output_image_path}!")
    print("=" * 65)


if __name__ == "__main__":
    verificar_preprocesamiento()

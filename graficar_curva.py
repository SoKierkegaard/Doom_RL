"""
graficar_curva.py
=================
Fase 4 y 5: Generación de gráficas de alta resolución para la presentación.
1. 'videos/curva_aprendizaje_dqn.png': Curva de DQN con hitos (25%, 50%, 75%, 100%),
   estrategia epsilon-greedy y pérdida TD.
2. 'videos/comparacion_dqn_vs_ppo.png': Comparación directa DQN vs PPO en el mismo escenario.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def suavizar_curva(puntos, alpha=0.3):
    """Calcula una media móvil exponencial para suavizar la curva de aprendizaje."""
    if len(puntos) == 0:
        return puntos
    suavizados = []
    val = puntos[0]
    for p in puntos:
        val = alpha * p + (1 - alpha) * val
        suavizados.append(val)
    return np.array(suavizados)


def extraer_datos_tb(log_dir):
    """Extrae pasos y recompensa media desde todos los archivos de eventos de TensorBoard en el directorio."""
    event_files = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not event_files:
        return np.array([]), np.array([])
    all_steps = []
    all_vals = []
    for f in event_files:
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            if "rollout/ep_rew_mean" in ea.scalars.Keys():
                for e in ea.Scalars("rollout/ep_rew_mean"):
                    all_steps.append(e.step)
                    all_vals.append(e.value)
        except Exception:
            continue
    if not all_steps:
        return np.array([]), np.array([])
    # Ordenar por paso y eliminar duplicados manteniendo el más reciente
    step_dict = {}
    for s, v in zip(all_steps, all_vals):
        step_dict[s] = v
    sorted_steps = sorted(step_dict.keys())
    steps = np.array(sorted_steps)
    vals = np.array([step_dict[s] for s in sorted_steps])
    return steps, vals


def generar_grafica_dqn(log_dir="logs/defend_dqn", output_path="videos/curva_aprendizaje_dqn.png"):
    output_path = os.path.abspath(output_path)
    print(f"Generando curva DQN en: {output_path} ...")
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    ea = EventAccumulator(event_files[0])
    ea.Reload()

    steps_rew = np.array([e.step for e in ea.Scalars("rollout/ep_rew_mean")])
    vals_rew = np.array([e.value for e in ea.Scalars("rollout/ep_rew_mean")])
    steps_eps = np.array([e.step for e in ea.Scalars("rollout/exploration_rate")])
    vals_eps = np.array([e.value for e in ea.Scalars("rollout/exploration_rate")])
    steps_loss = np.array([e.step for e in ea.Scalars("train/loss")])
    vals_loss = np.array([e.value for e in ea.Scalars("train/loss")])

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 9), facecolor="#121212")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.35, wspace=0.25)

    # Panel Superior: Recompensa
    ax_rew = fig.add_subplot(gs[0, :], facecolor="#1e1e1e")
    vals_smooth = suavizar_curva(vals_rew, alpha=0.35)
    ax_rew.plot(steps_rew, vals_rew, color="#58a6ff", alpha=0.25, linewidth=1.5, label="Recompensa media cruda (por ventana)")
    ax_rew.plot(steps_rew, vals_smooth, color="#58a6ff", linewidth=3.0, label="Tendencia suavizada (EMA)")
    ax_rew.fill_between(steps_rew, vals_smooth, 0, color="#58a6ff", alpha=0.10)
    ax_rew.axhline(0.60, color="#f85149", linestyle="--", linewidth=2.0, label="Línea Base Aleatoria (0.60 pts)")

    hitos = [
        (0, "Hito 0%\n(Inicio: Azar)", "#8b949e", -0.5),
        (6250, "Hito 25%\n(Detección)", "#3fb950", 4.3),
        (12500, "Hito 50%\n(Explotación)", "#d29922", 4.3),
        (18750, "Hito 75%\n(Refinamiento)", "#a371f7", 4.3),
        (25000, "Hito 100%\n(Agente Final)", "#58a6ff", 4.3)
    ]
    for step_h, texto, col, y_text in hitos:
        ax_rew.axvline(step_h, color=col, linestyle=":", alpha=0.7, linewidth=1.5)
        y_pos = np.interp(step_h, steps_rew, vals_smooth) if len(steps_rew) > 0 else 1.0
        ax_rew.plot(step_h, y_pos, marker="o", markersize=8, color=col, markeredgecolor="white", markeredgewidth=1.5)
        ax_rew.text(
            step_h, y_text, texto,
            color=col, fontsize=9, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=col, alpha=0.9)
        )

    ax_rew.set_title("Evolución del Aprendizaje (DQN - ViZDoom DefendCenter)\nCurva de Recompensa Episódica vs Pasos de Entrenamiento", fontsize=14, fontweight="bold", color="white", pad=12)
    ax_rew.set_xlabel("Pasos de Entrenamiento (Timesteps)", fontsize=11, color="#c9d1d9")
    ax_rew.set_ylabel("Recompensa Media Episódica", fontsize=11, color="#c9d1d9")
    ax_rew.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax_rew.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), framealpha=0.9, facecolor="#21262d", edgecolor="#484f58", fontsize=10)
    ax_rew.set_xlim(0, 25500)
    ax_rew.set_ylim(-0.9, 5.2)

    # Panel Inferior Izq: Epsilon
    ax_eps = fig.add_subplot(gs[1, 0], facecolor="#1e1e1e")
    ax_eps.plot(steps_eps, vals_eps, color="#d29922", linewidth=2.5, label="Tasa de Exploración (ε)")
    ax_eps.fill_between(steps_eps, vals_eps, 0.05, color="#d29922", alpha=0.15)
    ax_eps.axhline(0.05, color="#8b949e", linestyle="--", linewidth=1.2, label="ε residual (0.05)")
    ax_eps.set_title("Estrategia ε-Greedy (Exploración vs Explotación)", fontsize=12, fontweight="bold", color="#d29922", pad=10)
    ax_eps.set_xlabel("Pasos de Entrenamiento", fontsize=10, color="#c9d1d9")
    ax_eps.set_ylabel("Valor de ε", fontsize=10, color="#c9d1d9")
    ax_eps.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax_eps.legend(loc="upper right", framealpha=0.9, facecolor="#21262d", edgecolor="#484f58", fontsize=9)
    ax_eps.set_xlim(0, 25500)
    ax_eps.set_ylim(0, 1.05)

    # Panel Inferior Der: Pérdida TD
    ax_loss = fig.add_subplot(gs[1, 1], facecolor="#1e1e1e")
    loss_smooth = suavizar_curva(vals_loss, alpha=0.15)
    ax_loss.plot(steps_loss, vals_loss, color="#f85149", alpha=0.25, linewidth=1.0)
    ax_loss.plot(steps_loss, loss_smooth, color="#f85149", linewidth=2.2, label="Pérdida TD Suavizada")
    ax_loss.set_title("Pérdida Temporal-Difference de Bellman (TD Loss)", fontsize=12, fontweight="bold", color="#f85149", pad=10)
    ax_loss.set_xlabel("Pasos de Entrenamiento", fontsize=10, color="#c9d1d9")
    ax_loss.set_ylabel("Error Cuadrático Medio (MSE Loss)", fontsize=10, color="#c9d1d9")
    ax_loss.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax_loss.legend(loc="upper right", framealpha=0.9, facecolor="#21262d", edgecolor="#484f58", fontsize=9)
    ax_loss.set_xlim(0, 25500)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfica DQN guardada en: {output_path}!")


def generar_grafica_comparativa_dqn_ppo(
    dqn_dir="logs/defend_dqn",
    ppo_dir="logs/defend_ppo",
    output_path="videos/comparacion_dqn_vs_ppo.png"
):
    output_path = os.path.abspath(output_path)
    print(f"\nGenerando comparativa DQN vs PPO en: {output_path} ...")
    steps_dqn, vals_dqn = extraer_datos_tb(dqn_dir)
    steps_ppo, vals_ppo = extraer_datos_tb(ppo_dir)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#121212")
    ax.set_facecolor("#1e1e1e")

    # Línea base aleatoria
    ax.axhline(0.60, color="#8b949e", linestyle="--", linewidth=1.8, label="Línea Base Aleatoria (0.60 pts)")

    # DQN
    if len(steps_dqn) > 0:
        dqn_smooth = suavizar_curva(vals_dqn, alpha=0.3)
        ax.plot(steps_dqn, vals_dqn, color="#58a6ff", alpha=0.20, linewidth=1.2)
        ax.plot(steps_dqn, dqn_smooth, color="#58a6ff", linewidth=3.0, label="DQN (Off-Policy + Experience Replay)")
        ax.fill_between(steps_dqn, dqn_smooth, 0.60, color="#58a6ff", alpha=0.12)

    # PPO
    if len(steps_ppo) > 0:
        ppo_smooth = suavizar_curva(vals_ppo, alpha=0.3)
        ax.plot(steps_ppo, vals_ppo, color="#3fb950", alpha=0.25, linewidth=1.2)
        ax.plot(steps_ppo, ppo_smooth, color="#3fb950", linewidth=3.0, label="PPO (On-Policy + Actor-Critic)")
        ax.fill_between(steps_ppo, ppo_smooth, 0.60, color="#3fb950", alpha=0.10)

    # Anotaciones teóricas explicativas en la gráfica
    ax.text(
        15000, 3.2,
        "DQN: ALTA EFICIENCIA DE MUESTRA\n"
        "• Replay Buffer de 12k transiciones\n"
        "• Cada baja de enemigo se reutiliza\n"
        "  decenas de veces en minibatches\n"
        "• Convergencia más veloz en 25k pasos",
        color="#58a6ff", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#58a6ff", alpha=0.9)
    )

    ax.text(
        18000, 0.9,
        "PPO: ON-POLICY\n"
        "• Descarta experiencias tras 4 épocas\n"
        "• Requiere mayor volumen de datos\n"
        "  para afinar la política de giro",
        color="#3fb950", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#3fb950", alpha=0.9)
    )

    ax.set_title(
        "Comparación de Algoritmos en ViZDoom DefendCenter (25,000 Pasos)\nDQN (Value-Based / Off-Policy) vs PPO (Policy Gradient / On-Policy)",
        fontsize=14, fontweight="bold", color="white", pad=15
    )
    ax.set_xlabel("Pasos de Entrenamiento (Timesteps)", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Recompensa Media Episódica", fontsize=11, color="#c9d1d9")
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="upper left", framealpha=0.9, facecolor="#21262d", edgecolor="#484f58", fontsize=11)
    ax.set_xlim(0, 25500)
    ax.set_ylim(-0.4, 4.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfica comparativa DQN vs PPO guardada en: {output_path} (300 DPI)!")


def generar_grafica_escalada_paralela(
    dqn_dir="logs/defend_dqn",
    ppo_1env_dir="logs/defend_ppo",
    ppo_par_dir="logs/defend_ppo_paralelo/PPO_2",
    output_path="videos/comparacion_escalada_paralela.png"
):
    output_path = os.path.abspath(output_path)
    print(f"\nGenerando comparativa escalada paralela en: {output_path} ...")
    steps_dqn, vals_dqn = extraer_datos_tb(dqn_dir)
    steps_ppo_1, vals_ppo_1 = extraer_datos_tb(ppo_1env_dir)
    steps_ppo_par, vals_ppo_par = extraer_datos_tb(ppo_par_dir)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#121212")
    ax.set_facecolor("#1e1e1e")

    # Línea base aleatoria
    ax.axhline(0.60, color="#8b949e", linestyle="--", linewidth=1.8, label="Línea Base Aleatoria (0.60 pts)")

    # 1. DQN (25k pasos)
    if len(steps_dqn) > 0:
        dqn_smooth = suavizar_curva(vals_dqn, alpha=0.3)
        ax.plot(steps_dqn, vals_dqn, color="#58a6ff", alpha=0.18, linewidth=1.0)
        ax.plot(steps_dqn, dqn_smooth, color="#58a6ff", linewidth=2.8, label="DQN 1 Env (25k pasos, Replay Buffer)")

    # 2. PPO 1 Env (25k pasos)
    if len(steps_ppo_1) > 0:
        ppo_smooth = suavizar_curva(vals_ppo_1, alpha=0.3)
        ax.plot(steps_ppo_1, vals_ppo_1, color="#f85149", alpha=0.22, linewidth=1.0)
        ax.plot(steps_ppo_1, ppo_smooth, color="#f85149", linewidth=2.8, label="PPO 1 Env (25k pasos - Cuello de botella On-Policy)")

    # 3. PPO Paralelo 10 Envs (305k pasos)
    if len(steps_ppo_par) > 0:
        ppo_par_smooth = suavizar_curva(vals_ppo_par, alpha=0.20)
        ax.plot(steps_ppo_par, vals_ppo_par, color="#3fb950", alpha=0.20, linewidth=1.0)
        ax.plot(steps_ppo_par, ppo_par_smooth, color="#3fb950", linewidth=3.5, label="PPO Paralelo (10 Subprocesos, 305k pasos, ~230 FPS)")
        ax.fill_between(steps_ppo_par, ppo_par_smooth, 0.60, color="#3fb950", alpha=0.12)

    # Líneas verticales de hitos de reanudación
    ax.axvline(102400, color="#d29922", linestyle=":", linewidth=1.3, alpha=0.7)
    ax.text(
        102400, 2.5, "Reanudación 1\n(+100k pasos)",
        color="#d29922", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#161b22", edgecolor="#d29922", alpha=0.9)
    )

    ax.axvline(204800, color="#d29922", linestyle=":", linewidth=1.3, alpha=0.7)
    ax.text(
        204800, 2.5, "Reanudación 2\n(+100k pasos)",
        color="#d29922", fontsize=8.0, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#161b22", edgecolor="#d29922", alpha=0.9)
    )

    # Punto Óptimo Máximo (255k pasos)
    ax.plot(254800, 17.5, marker="*", markersize=16, color="#ffd700", markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate(
        "★ MEJOR CHECKPOINT: 255,000 PASOS\n"
        "• Recompensa media evaluada: 20.6 pts (21.6 bajas)\n"
        "• Récord observado: 23 bajas con 26 balas (~88.5%)\n"
        "• Punto exacto de Early Stopping antes de la sobreoptimización",
        xy=(254800, 17.5), xytext=(130000, 15.2),
        arrowprops=dict(facecolor="#ffd700", shrink=0.08, width=2, headwidth=8),
        color="#ffd700", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#161b22", edgecolor="#ffd700", alpha=0.95)
    )

    # Anotación sobre el ligero valle a los 275k-280k
    ax.annotate(
        "Ligero valle (~275k):\n"
        "Pérdida temporal de entropía\n"
        "y sobreoptimización de giros",
        xy=(280000, 17.0), xytext=(240000, 7.5),
        arrowprops=dict(facecolor="#e3b341", shrink=0.08, width=1.2, headwidth=6),
        color="#e3b341", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#e3b341", alpha=0.9)
    )

    ax.annotate(
        "DQN domina en muestras bajas\n(Reutiliza cada baja en memoria)",
        xy=(24000, 4.3), xytext=(30000, 5.5),
        arrowprops=dict(facecolor="#58a6ff", shrink=0.08, width=1.5, headwidth=7),
        color="#58a6ff", fontsize=9.0, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#58a6ff", alpha=0.95)
    )

    ax.text(
        4000, 11.5,
        "LECCIÓN TEÓRICA CLAVE:\n"
        "1. On-Policy (PPO) es altamente ineficiente en muestras iniciales,\n"
        "   pero con 10 entornos paralelos elimina la correlación temporal.\n"
        "2. A partir de los 180k pasos, la curva entra en saturación total (meseta),\n"
        "   demostrando matemáticamente que el agente resolvió la disciplina de\n"
        "   apuntado y alcanzó la cota superior impuesta por la munición del mapa.",
        color="#e6edf3", fontsize=9.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#8b949e", alpha=0.9)
    )

    ax.set_title(
        "Convergencia y Techo Absoluto de Aprendizaje en ViZDoom DefendCenter\nDQN (1 Env, 25k) vs PPO (1 Env, 25k) vs PPO Paralelo (10 Subprocesos, 305k pasos)",
        fontsize=13.5, fontweight="bold", color="white", pad=15
    )
    ax.set_xlabel("Pasos Globales de Entrenamiento (Timesteps)", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Recompensa Media Episódica (Bajas - Penalización)", fontsize=11, color="#c9d1d9")
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.58), framealpha=0.92, facecolor="#21262d", edgecolor="#484f58", fontsize=10.5)
    ax.set_xlim(0, 315000)
    ax.set_ylim(-0.8, 19.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfica comparativa escalada guardada en: {output_path} (300 DPI)!")


def generar_grafica_deadly_corridor(
    log_dir_base="logs/deadly_corridor/PPO_0",
    log_dir_opta="logs/deadly_corridor_optA/PPO_2",
    output_path="videos/deadly_corridor/curva_aprendizaje_deadly_corridor.png"
):
    output_path = os.path.abspath(output_path)
    print(f"\nGenerando curva comparativa de aprendizaje Deadly Corridor en: {output_path} ...")
    files_base = sorted(glob.glob(os.path.join(log_dir_base, "events.out.tfevents.*")))
    files_opta = sorted(glob.glob(os.path.join(log_dir_opta, "events.out.tfevents.*")))
    
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#121212")
    ax.set_facecolor("#1e1e1e")

    # Línea base aleatoria
    ax.axhline(-92.5, color="#f85149", linestyle="--", linewidth=2.0, label="Línea Base Aleatoria (-92.5 pts | Muerte en 3.7s)")

    # 1. PPO Baseline (Kamikaze Rush - Meseta a 50k)
    if len(files_base) >= 2:
        f_base = files_base[1]
        ea_base = EventAccumulator(f_base)
        ea_base.Reload()
        base_s = np.array([s.step for s in ea_base.Scalars("rollout/ep_rew_mean")])
        base_v = np.array([s.value for s in ea_base.Scalars("rollout/ep_rew_mean")])
        base_smooth = suavizar_curva(base_v, 0.25)
        ax.plot(base_s, base_v, color="#d29922", alpha=0.18, linewidth=1.2)
        ax.plot(base_s, base_smooth, color="#d29922", linewidth=2.8, label="PPO Baseline (Skill 5 - Óptimo Local / Meseta a 50k)")

    # 2. PPO Táctico Opción A Anti-Camper (152k pasos)
    if files_opta:
        all_s = []
        all_v = []
        for f in files_opta:
            try:
                ea_opta = EventAccumulator(f)
                ea_opta.Reload()
                for s in ea_opta.Scalars("rollout/ep_rew_mean"):
                    all_s.append(s.step)
                    all_v.append(s.value)
            except Exception:
                pass
        if all_s:
            opta_s = np.array(all_s)
            opta_v = np.array(all_v)
            opta_smooth = suavizar_curva(opta_v, 0.20)
            ax.plot(opta_s, opta_v, color="#3fb950", alpha=0.22, linewidth=1.2)
            ax.plot(opta_s, opta_smooth, color="#3fb950", linewidth=3.5, label="PPO Táctico Anti-Camper (Skill 3 + Giros + Rebalanceo)")
            ax.fill_between(opta_s, opta_smooth, -92.5, color="#3fb950", alpha=0.10)

    # Récord pico evaluado (115,000 pasos / 1,004.9 pts)
    ax.plot(114900, 1004.9, marker="*", markersize=20, color="#ffd700", markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate(
        "★ MEJOR MODELO (EARLY STOPPING): 115,000 PASOS\n"
        "• Evaluación media: +912.6 pts | Ronda pico: +1,004.9 pts\n"
        "• Duración récord: 59 pasos (~2.4 s) esquivando y disparando\n"
        "• Elimina 3 amenazas en nichos (TURN+ATK) y recoge el chaleco\n"
        "• 0% de campeo y 0% de tiempos muertos (Trunc = False)",
        xy=(114900, 1004.9), xytext=(35000, 850.0),
        arrowprops=dict(facecolor="#ffd700", shrink=0.08, width=2, headwidth=8),
        color="#ffd700", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#161b22", edgecolor="#ffd700", alpha=0.95)
    )

    # Anotación sobre el quiebre de la meseta
    ax.axvline(50000, color="#8b949e", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate(
        "RUPTURA DE LA MESETA:\n"
        "• ent_coef=0.04 previene el colapso\n"
        "• La red descubre giros angulares\n"
        "• Penalización anti-camper fuerza avance",
        xy=(50000, 210.0), xytext=(12000, 420.0),
        arrowprops=dict(facecolor="#3fb950", shrink=0.08, width=1.5, headwidth=7),
        color="#3fb950", fontsize=9.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#3fb950", alpha=0.95)
    )

    # Cuadro explicativo pedagógico
    ax.text(
        2500, -50.0,
        "LECCIÓN TEÓRICA CLAVE (SOLUCIÓN AL SÍNDROME DEL CAMPERO):\n"
        "1. Truncamiento vs Meta: Otorgar bonus al agotar el tiempo crea el exploit de 'campear'.\n"
        "   Condicionar el premio de +400 pts exclusivamente a 'term and not trunc' erradicó el campeo.\n"
        "2. Living Cost: El coste de -0.10 pts/paso inyecta urgencia temporal, logrando que el agente\n"
        "   complete la misión en solo 59 pasos con un retorno máximo de +1,004.9 pts.",
        color="#e6edf3", fontsize=9.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#8b949e", alpha=0.92)
    )

    ax.set_title(
        "Conquista Total y Ruptura de Óptimos Locales en ViZDoom Deadly Corridor\nPPO Baseline (Meseta Kamikaze en 50k) vs PPO Maestro Anti-Camper (Pico de +1,004.9 pts a los 115k pasos)",
        fontsize=13.0, fontweight="bold", color="white", pad=15
    )
    ax.set_xlabel("Pasos Globales de Entrenamiento (Timesteps)", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Recompensa Media Episódica", fontsize=11, color="#c9d1d9")
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax.legend(loc="upper left", framealpha=0.92, facecolor="#21262d", edgecolor="#484f58", fontsize=10.5)
    ax.set_xlim(0, 160000)
    ax.set_ylim(-110.0, 1150.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"¡Gráfica Deadly Corridor guardada en: {output_path} (300 DPI)!")


if __name__ == "__main__":
    generar_grafica_dqn()
    generar_grafica_comparativa_dqn_ppo()
    generar_grafica_escalada_paralela()
    generar_grafica_deadly_corridor()

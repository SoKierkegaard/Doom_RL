# Deep Reinforcement Learning en ViZDoom (Pixel-to-Action)

Proyecto de Deep Reinforcement Learning para entrenamiento de agentes autónomos en **ViZDoom** a partir exclusivamente de los píxeles crudos de la pantalla. Desarrollado como material práctico y demostración conceptual para presentación universitaria de Deep RL.

---

## 1. Conceptos Teóricos Clave Representados

1. **Ciclo Agente - Entorno - Recompensa:**
   * El agente percibe una observación visual $s_t$, elige una acción $a_t \in \mathcal{A}$ y el motor de ViZDoom retorna una recompensa escalar $r_{t+1}$ y el nuevo estado $s_{t+1}$.
2. **De POMDP a MDP (Frame Stacking):**
   * Una sola imagen estática no contiene velocidad ni trayectoria (Proceso de Decisión de Markov Parcialmente Observable).
   * Al apilar $4$ fotogramas consecutivos ($84 \times 84$), el tensor resultante `(4, 84, 84)` codifica dirección, velocidad y aceleración, restaurando la propiedad de Markov:
     $$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)$$
3. **Deep Q-Networks (DQN):**
   * Aproximación de la función de valor óptima $Q^*(s, a)$ con una red convolucional (*NatureCNN*):
     $$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta) \right)^2 \right]$$
4. **Experience Replay Buffer:**
   * Almacena transiciones pasadas $\langle s_t, a_t, r_t, s_{t+1}, d_t \rangle$ y muestrea minibatches uniformes i.i.d., rompiendo la correlación temporal que desestabiliza a las redes neuronales.
   * *Control de RAM:* $12,000$ transiciones ocupan $\sim 645\text{ MB}$, evitando saturar la memoria del equipo.
5. **Target Network ($\theta^-$):**
   * Congela los pesos del objetivo de Bellman y los sincroniza periódicamente, eliminando el problema de optimizar contra un "objetivo móvil".
6. **Exploración vs Explotación ($\epsilon$-Greedy):**
   * $\epsilon$ decae de $1.0$ (100% azar) a $0.05$ (explotación voraz con 5% de ruido residual).

---

## 2. Estructura de Scripts y Archivos

```text
doom-rl/
├── doom_wrappers.py             # Pipeline modular de Gymnasium + Parche PyTorch 2.14
├── fase0_reconocimiento.py      # Fase 0: Inspección de hardware, entornos y línea base azar
├── fase1_preprocesamiento.py    # Fase 1: Diagnóstico visual del tensor que ingresa a la CNN
├── fase2_humo_dqn.py            # Fase 2: Prueba de humo rápida con DQN en VizdoomBasic-v1
├── entrenamiento_principal.py   # Fase 3: Entrenamiento principal (DQN/PPO) en DefendCenter
├── evaluar_agente.py            # Fase 4: Evaluación cuantitativa de checkpoints y exportador de video
├── graficar_curva.py            # Fase 4: Gráfica multipanel 300 DPI de la curva de aprendizaje
├── models/
│   ├── dqn_basic/               # Checkpoints de la prueba de humo
│   └── defend_dqn/              # Checkpoints en hitos (0%, 25%, 50%, 75%, 100%)
├── logs/                        # Telemetría de TensorBoard
└── videos/                      # Videos MP4 y figuras diagnósticas para diapositivas
```

---

## 3. Instrucciones de Reproducción Paso a Paso

Asegúrate de tener el entorno virtual activo en PowerShell (`.\.venv\Scripts\Activate.ps1`):

### Paso 0: Reconocimiento y Línea Base
```powershell
python fase0_reconocimiento.py
```
*Verifica GPU RTX 3070 Ti, paquetes y genera el video inicial del agente aleatorio.*

### Paso 1: Verificación Visual del Preprocesamiento
```powershell
python fase1_preprocesamiento.py
```
*Genera la figura `videos/verificacion_observacion_red.png` comparando imagen cruda, 4 frames apilados y mapa térmico diferencial de movimiento.*

### Paso 2: Prueba de Humo DQN en Escenario Básico
```powershell
python fase2_humo_dqn.py
```
*Entrenamiento de ~1.8 min en `VizdoomBasic-v1`. Demuestra la subida de recompensa de -184 a -56.5 pts.*

### Paso 3: Entrenamiento Principal en DefendCenter
```powershell
python entrenamiento_principal.py --algoritmo DQN --timesteps 25000 --seed 42
```
*Entrenamiento de ~6.9 min en GPU. Guarda automáticamente los hitos en `models/defend_dqn/`:*
* `dqn_defend_chkpt_0_steps.zip` (Inicio / 0%)
* `dqn_defend_chkpt_6250_steps.zip` (25%)
* `dqn_defend_chkpt_12500_steps.zip` (50%)
* `dqn_defend_chkpt_18750_steps.zip` (75%)
* `dqn_defend_final.zip` (100% Final)

### Paso 4: Evaluación y Generación de Videos
Para evaluar cualquier checkpoint individual o exportar video:
```powershell
python evaluar_agente.py --checkpoint models/defend_dqn/dqn_defend_final.zip --episodes 10 --record_video --video_path videos/demo_final.mp4
```

### Paso 5: Generar Gráfica de Aprendizaje para Diapositivas
```powershell
python graficar_curva.py
```
*Lee los eventos de TensorBoard y genera `videos/curva_aprendizaje_dqn.png` a 300 DPI en modo oscuro.*

---

| Métrica | Agente Aleatorio (Inicio) | DQN 1 Env (25k pasos) | PPO 1 Env (25k pasos) | PPO Paralelo 10 Envs (100k) | PPO Paralelo Maestro (200k) | PPO Paralelo Óptimo (305k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paradigma** | Sin entrenamiento | Off-Policy (Value-Based) | On-Policy (Actor-Critic) | On-Policy Paralelo | On-Policy Paralelo Saturado | On-Policy Techo Absoluto |
| **Entornos Simultáneos** | 1 | 1 | 1 | 10 subprocesos | 10 subprocesos | **10 subprocesos (SubprocVecEnv)** |
| **Tiempo de Entrenamiento** | 0 min | 6.88 min | 5.75 min | 7.35 min | 14.6 min (2 fases) | **21.9 min (3 fases de ~7.3 min)** |
| **Rendimiento (FPS)** | ~65 FPS | 60.6 FPS | 72.4 FPS | 226.8 FPS | 229.5 FPS | **228.6 FPS** |
| **Recompensa / Bajas** | $0.60 \pm 1.02$ | $8.70 \pm 2.33$ | $1.20 \pm 1.08$ | $13.90 \pm 4.04$ | $20.30 \pm 1.42$ | **$19.67 \pm 0.47$ (Consistente 20-21 bajas)** |
| **Supervivencia Promedio**| ~3 segundos | ~12 segundos | ~4 segundos | ~27 segundos | ~38 - 40 segundos | **~38 - 40 segundos** |
| **Eficiencia de Munición**| 0% (tiros al aire) | ~35% | ~5% | ~60% | ~81% | **~81% (21 bajas con 26 balas)** |

---

## 5. Entrenamiento Escalado en Paralelo (PPO Multi-Actor)

Para entrenar o reanudar PPO con vectorización masiva sobre los 14 núcleos del CPU:
```powershell
# Entrenamiento inicial a 100,000 pasos
python entrenamiento_paralelo_ppo.py --n_envs 10 --timesteps 100000

# Reanudación continua hasta saturación (añade 100,000 pasos adicionales por llamada)
python entrenamiento_paralelo_ppo.py --n_envs 10 --timesteps 100000 --resume models/defend_ppo_paralelo/ppo_paralelo_final.zip
```

### Gráficas y Videos del Experimento 1 (Defend Center):
* Ubicación: [`videos/defend_center/`](videos/defend_center)
* [`videos/defend_center/1_defend_aleatorio_0steps.mp4`](videos/defend_center/1_defend_aleatorio_0steps.mp4): Agente aleatorio (muere en ~3s).
* [`videos/defend_center/2_defend_intermedio_12500steps.mp4`](videos/defend_center/2_defend_intermedio_12500steps.mp4): DQN a mitad de camino (12.5k pasos).
* [`videos/defend_center/3_defend_final_25000steps.mp4`](videos/defend_center/3_defend_final_25000steps.mp4): DQN final (8.7 bajas por ronda).
* [`videos/defend_center/4_defend_ppo_final_25000steps.mp4`](videos/defend_center/4_defend_ppo_final_25000steps.mp4): PPO 1 solo entorno (1.2 bajas).
* [`videos/defend_center/5_defend_ppo_paralelo_100k.mp4`](videos/defend_center/5_defend_ppo_paralelo_100k.mp4): PPO Paralelo 100k pasos (13.9 bajas).
* [`videos/defend_center/6_defend_ppo_paralelo_200k.mp4`](videos/defend_center/6_defend_ppo_paralelo_200k.mp4): PPO Paralelo Maestro 200k pasos (20.3 bajas).
* [`videos/defend_center/7_defend_ppo_paralelo_300k.mp4`](videos/defend_center/7_defend_ppo_paralelo_300k.mp4): PPO Paralelo 305k pasos (Saturación y varianza $\pm 0.47$).
* [`videos/defend_center/8_defend_ppo_paralelo_mejor_255k.mp4`](videos/defend_center/8_defend_ppo_paralelo_mejor_255k.mp4): **PPO Checkpoint TOP 255k pasos** (¡Récord absoluto de 23 bajas con 26 balas!).
* [`videos/defend_center/comparacion_escalada_paralela.png`](videos/defend_center/comparacion_escalada_paralela.png): **Gráfica de saturación completa (300 DPI)**.

---

## 6. Experimento 2: ViZDoom Deadly Corridor

Navegación y combate táctico en un pasillo estrecho con 6 enemigos de alta cadencia de fuego (ShotgunGuys, ChaingunGuys, Zombiemen) apostados en nichos laterales ($y = \pm 64$), con el objetivo de alcanzar el chaleco verde (*GreenArmor*) al final del corredor ($x = 1312$).

### Tabla Comparativa de Modelos:

| Métrica | Agente Aleatorio (0 pasos) | PPO Baseline (100k pasos) | PPO Intermedio (100k pasos) | **PPO Maestro Anti-Camper (115k pasos)** |
| :--- | :---: | :---: | :---: | :---: |
| **Dificultad de ViZDoom** | Skill 5 (Nightmare) | Skill 5 (Nightmare) | Skill 3 (Hurt Me Plenty) | **Skill 3 (Hurt Me Plenty - Táctico Justo)** |
| **Espacio de Acciones** | 8 acciones sin giro con tiro | 8 acciones (sin giro lateral) | 10 acciones (Giro Angular + Tiro) | **10 acciones (Giro Angular + Tiro + Avance)** |
| **Comportamiento Aprendido** | Giro errático y muerte pasiva | Carrera Suicida (*Kamikaze Rush*) | Despeje inicial con campeo en esquina | **¡Conquista Total Fluida y Despeje Rápido!** |
| **Acciones Predominantes** | Azar uniforme | `MOVE_FORWARD` (100%) | `TURN_L+ATK` (50%), `TURN_R` (35%) | `MOVE_FWD` (40%), `FWD+ATK` (37%), `TURN_L+ATK` (15%) |
| **Bajas de Enemigos** | $0$ bajas | $0$ bajas | $3.0$ bajas confirmadas | **$3.0$ bajas confirmadas (elimina amenazas)** |
| **Salud Conservada** | $0\%$ (muerto) | $0\%$ (muerto en ~1.3s) | $82.0\%$ de salud restante | **$>80\%$ de salud restante** |
| **Recompensa Media** | $-92.50 \pm 3.02$ | $+163.39 \pm 18.2$ | $+488.42 \pm 173.20$ | **$+912.59 \pm 184.54$ (Récord: $+1,004.9$)** |
| **Duración del Recorrido**| $\sim 3.7\text{ s}$ ($14$ tics) | $\sim 1.3\text{ s}$ ($52$ tics) | $60.0\text{ s}$ (525 pasos por timeout) | **$59.6$ pasos promedio (~$2.4\text{ s}$ / $238$ tics)** |
| **Tasa de Éxito / Chaleco**| 0% | Semillas favorables (~20%) | $80\%$ (con espera en timeout) | **$100\%$ (5 de 5 rondas tocan el chaleco vivas)** |

### Lecciones Teóricas Clave para la Exposición:
1. **Ruptura del Óptimo Local (*Reward Hacking* por Distancia):**
   * El escenario base premia el avance ciego, provocando que los agentes tradicionales colapsen a los 50,000 pasos en una carrera suicida. Al rebalancear (+150 por baja), la red prioriza eliminar las amenazas antes de cruzar.
2. **El "Síndrome del Campero" (*Camping Exploit*):**
   * Premiar la supervivencia en el truncamiento de tiempo (`trunc == True`) provocó que el agente matara a los primeros enemigos y se quedara quieto 500 pasos esperando que el reloj expirara para cobrar el bonus gratis.
   * La solución técnica consistió en **condicionar el bonus de $+400$ pts exclusivamente a tocar físicamente la meta (`term and not trunc`)**, penalizar el tiempo agotado ($-100$ pts) y aplicar un costo temporal (*Living Cost* de $-0.10$ pts/paso).
3. **Punto Óptimo y *Early Stopping* (115,000 pasos):**
   * Al igual que en Defend Center, el agente alcanzó su máxima eficiencia en el paso 115,000 (+912.6 pts de media y rounds de +1,004.9 pts), demostrando nuevamente el valor práctico del *Early Stopping* para evitar la sobreoptimización.

### Gráficas y Videos del Experimento 2 (Deadly Corridor):
* Ubicación: [`videos/deadly_corridor/`](videos/deadly_corridor)
* [`videos/deadly_corridor/1_deadly_aleatorio_0steps.mp4`](videos/deadly_corridor/1_deadly_aleatorio_0steps.mp4): Agente aleatorio (muere en 3.7s bajo fuego cruzado).
* [`videos/deadly_corridor/2_deadly_intermedio_50k.mp4`](videos/deadly_corridor/2_deadly_intermedio_50k.mp4): Checkpoint intermedio a 50k pasos (comienza la carrera veloz).
* [`videos/deadly_corridor/3_deadly_final_100k.mp4`](videos/deadly_corridor/3_deadly_final_100k.mp4): Modelo baseline a 100k pasos (carrera kamikaze explotando la recompensa por distancia).
* [`videos/deadly_corridor/4_deadly_tactical_final_100k.mp4`](videos/deadly_corridor/4_deadly_tactical_final_100k.mp4): Modelo táctico intermedio (muestra la ronda pico con $+484.9$ pts).
* [`videos/deadly_corridor/5_deadly_optA_conquista_final.mp4`](videos/deadly_corridor/5_deadly_optA_conquista_final.mp4): Modelo con eliminación de enemigos pero esperando en timeout (60 seg).
* [`videos/deadly_corridor/6_deadly_conquista_total_115k.mp4`](videos/deadly_corridor/6_deadly_conquista_total_115k.mp4): **¡LA OBRA MAESTRA DEFINITIVA!** (Video de 8.6 segundos con telemetría HUD completa: avanza disparando `FWD+ATK`, gira hacia los nichos con `TURN_L+ATK`, destruye las amenazas, conserva su salud y conquista el chaleco verde con récord de **$+1,004.9$ puntos** y $100\%$ de éxito).
* [`videos/deadly_corridor/curva_aprendizaje_deadly_corridor.png`](videos/deadly_corridor/curva_aprendizaje_deadly_corridor.png): **Curva comparativa completa a 300 DPI** detallando la meseta de los 50k pasos, el quiebre de la política y el pico máximo en los 115k pasos.


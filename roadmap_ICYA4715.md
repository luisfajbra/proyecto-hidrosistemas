# 🌊 Hoja de Ruta — Proyecto Final ICYA 4715
### Modelación de Hidrosistemas · Universidad de los Andes · 2026-1

> **Sistema elegido:** Ecuaciones de Saint-Venant 1D — Tránsito hidráulico en canal abierto  
> **Datos:** Sintéticos (parámetros verdaderos conocidos)  
> **Entrega:** 1 al 3 de junio de 2026 · Cita previa por correo electrónico  
> **Grupo:** 4 personas

---

## 📋 Resumen de Entregables

| Entregable | Descripción | Límite |
|---|---|---|
| 📄 Artículo técnico | Dos columnas, incluye abstract, métodos, resultados, conclusiones | Máx. 15 páginas (sin anexos) |
| 💻 Repositorio Git | README, `environment.yml`, semillas fijas, notebooks reproducibles | Todas las figuras y tablas |
| 🎤 Presentación oral | Figuras y bullet points, todos hablan | Máx. 15 min + 10 min preguntas |

---

## 🏗️ Estructura del Repositorio

```
proyecto_sv/
├── README.md
├── environment.yml
├── data/
│   └── synthetic/              # datos generados con parámetros verdaderos
├── src/
│   ├── model.py                # solver Saint-Venant (MacCormack)
│   ├── sensitivity.py          # análisis Sobol con SALib
│   ├── calibration.py          # CMA-ES + KGE
│   ├── uncertainty.py          # GLUE / Monte Carlo
│   └── pinn.py                 # PINN con PyTorch
├── notebooks/
│   ├── 01_model_verification.ipynb
│   ├── 02_sensitivity.ipynb
│   ├── 03_calibration.ipynb
│   └── 04_pinn.ipynb
└── figures/
```

---

## ⚙️ El Sistema: Saint-Venant 1D

### Ecuaciones gobernantes

Canal rectangular, flujo 1D no permanente:

```
∂h/∂t  +  ∂(hu)/∂x  =  0                              (continuidad)

∂(hu)/∂t  +  ∂(hu² + gh²/2)/∂x  =  -gh(∂B/∂x + Sf)  (momentum)
```

Donde la fricción de Manning es:

```
Sf = n² · u|u| / h^(4/3)
```

### Parámetros calibrables

| Parámetro | Símbolo | Rango de búsqueda | Valor "verdadero" |
|---|---|---|---|
| Coeficiente de Manning | `n` | 0.010 – 0.060 | **0.035** |
| Pendiente de fondo | `S₀` | 0.0001 – 0.005 | **0.001** |
| Caudal base upstream | `Q₀` | 10 – 100 m³/s | **50 m³/s** |
| Amplitud del hidrograma | `A_hyd` | 20 – 200 m³/s | **100 m³/s** |
| Ancho del canal | `B_w` | 20 – 80 m | **50 m** |

### Esquema numérico: MacCormack (predictor-corrector)

- 2do orden en espacio y tiempo
- Condición de estabilidad CFL: `dt ≤ 0.9 · dx / (|u| + √(gh))`
- Dominio: L = 5000 m, nx = 100 celdas, dx = 50 m
<<<<<<< HEAD
- Condición upstream: hidrograma externo (`Q_upstream_m3s`) leído desde CSV
=======
- Condición upstream: hidrograma triangular sintético
>>>>>>> 6b4577c (Sensibiidad local y global)
- Condición downstream: tirante normal (Manning)

### Verificación del modelo ⚠️

Antes de calibrar, verificar con:
- [ ] Solución analítica de onda cinemática en régimen subcrítico
- [ ] Conservación de masa en cada paso de tiempo
- [ ] Caso límite: flujo uniforme permanente → `h` constante

> **Nota:** Es muy común calibrar un modelo mal implementado y no enterarse. ¡La verificación no es opcional!

---

## 📊 Parte 1 — Modelo Numérico *(15 pts)*

**Responsable principal:** Persona 1 + Persona 2  
**Duración estimada:** Semana 1

### Checklist

- [x] Implementar solver `saint_venant_1d(params, q_upstream, time_seconds)` en `src/model.py`
- [x] Generación de datos sintéticos con ruido gaussiano (`σ = 5% · Q_max`)
- [x] Modo batch funcional (sin GUI, ejecutable desde terminal)
- [x] Paralelización con `joblib` para Monte Carlo masivo
- [x] Verificación contra solución analítica y conservación de masa
- [x] Figura: hidrograma simulado vs. "observado" sintético
- [x] Tabla: parámetros verdaderos y condiciones iniciales/frontera

```python
# Llamada mínima esperada
Q_sim = saint_venant_1d(
    params=[n, S0, B_w],
    q_upstream=Q_upstream,
    time_seconds=t_seconds,
)
```

---

## 🔍 Parte 2 — Análisis de Sensibilidad Global *(15 pts)*

**Responsable principal:** Persona 1  
**Duración estimada:** Semana 2

**Método:** Sobol (varianza total y primer orden) con **SALib**

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 5,
    'names': ['n', 'S0', 'Q0', 'A_hyd', 'B_w'],
    'bounds': [[0.01, 0.06], [0.0001, 0.005],
               [10, 100], [20, 200], [20, 80]]
}

# ~57,000 simulaciones (8192 × 7)
param_values = saltelli.sample(problem, 8192)
Y = np.array([run_model(p) for p in param_values])  # joblib paralelo
Si = sobol.analyze(problem, Y)
# Si['S1'], Si['ST'], Si['S1_conf'], Si['ST_conf']
```

### Checklist

- [ ] Justificar rangos de variación de cada parámetro
- [ ] Reportar índices S1 y ST **con intervalos de confianza** (bootstrap incluido en SALib)
- [ ] Figura: gráfico de barras S1 / ST con barras de error
- [ ] Discutir: ¿qué parámetros son influyentes? ¿cuáles fijar en calibración?
- [ ] Total simulaciones: ~57,000 (documentar en el reporte)

> **Resultado esperado:** `n` y `S₀` dominarán la sensibilidad; `B_w` probablemente será poco sensible.

---

## 🎯 Parte 3 — Calibración, Validación e Incertidumbre *(30 pts)*

**Responsables:** Persona 2 (calibración) + Persona 3 (incertidumbre)  
**Duración estimada:** Semanas 2–3

### 3a. Calibración con CMA-ES

**Función objetivo:** KGE (Kling-Gupta Efficiency) — más balanceada que NSE para caudales altos y bajos.

```
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]

donde: r = correlación, α = σ_sim/σ_obs, β = μ_sim/μ_obs
```

```python
import cma

def objective(params):
    Q_sim = saint_venant_1d(params)
    return -KGE(Q_obs_cal, Q_sim[warmup:])

es = cma.CMAEvolutionStrategy(x0, sigma0=0.3, {'seed': 42, 'maxiter': 1000})
es.optimize(objective)
```

**Partición de datos:**

```
|-- Calentamiento (10%) --|-- Calibración (60%) --|-- Validación (30%) --|
```

### 3b. Propagación de incertidumbre: GLUE

```python
# 50,000 simulaciones Monte Carlo
samples = latin_hypercube_sample(problem, 50_000)
kge_scores = [KGE(Q_obs, run_model(p)) for p in samples]

# Umbral de aceptación: KGE > 0.6
behavioral = samples[kge_scores > 0.6]

# Bandas 5-95% de la envolvente
lower = np.percentile(predictions_behavioral, 5, axis=0)
upper = np.percentile(predictions_behavioral, 95, axis=0)
```

### Checklist

- [ ] Definir y justificar función objetivo KGE
- [ ] Ejecutar CMA-ES con ≥ 5 semillas distintas (evaluar repetibilidad)
- [ ] Figura: curvas de convergencia del optimizador
- [ ] Tabla: métricas KGE en calibración **y** validación
- [ ] Discutir equifinalidad: ¿hay múltiples óptimos igualmente buenos?
- [ ] GLUE con 50,000 simulaciones → bandas 5–95%
- [ ] Figura: hidrograma observado + banda de incertidumbre
- [ ] Reportar: ¿qué % de observaciones cae dentro de la banda? (meta: ~90%)
- [ ] Comparar parámetros calibrados con valores verdaderos sintéticos

---

## 🧠 Parte 4 — PINN para Inversión de Parámetros *(30 pts)*

**Responsable principal:** Persona 4  
**Duración estimada:** Semana 3

### Arquitectura

| Elemento | Valor |
|---|---|
| Entradas | `(x, t)` |
| Salidas | `(h, q=hu)` |
| Capas ocultas | 5 capas × 100 neuronas |
| Activación | `tanh` |
| Inicialización | Glorot uniforme |
| Parámetros entrenables | `log_n`, `log_S₀` (en escala log para positividad) |

### Función de pérdida

```python
L_total = w1 * L_physics   # residuo de Saint-Venant (autodiferenciación)
        + w2 * L_data       # ajuste a datos observados
        + w3 * L_bc         # condiciones de frontera/inicial

# Pesos iniciales sugeridos: w1=1.0, w2=10.0, w3=5.0
# Rebalancear si L_physics >> L_data durante entrenamiento
```

```python
import torch
import torch.nn as nn

class SaintVenantPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(in=2, hidden=[100]*5, out=2, act=nn.Tanh())
        # Parámetros físicos como variables entrenables
        self.log_n  = nn.Parameter(torch.tensor(-3.35))  # ln(0.035)
        self.log_S0 = nn.Parameter(torch.tensor(-6.91))  # ln(0.001)

def pde_residual(model, x, t):
    h, q = model(x, t)  # autodiferenciación para ∂h/∂t, ∂q/∂x, etc.
    n  = torch.exp(model.log_n)
    S0 = torch.exp(model.log_S0)
    # Calcular residuos de continuidad y momentum
    ...
```

### Puntos de colocación

- **Interior:** 5000 puntos aleatorios en `(x,t) ∈ [0,L] × [0,T]`
- **Condición inicial:** 500 puntos en `t=0`
- **Condición de frontera:** 500 puntos en `x=0` y `x=L`
- **Datos observados:** serie temporal en sección de control (ruidosa)

### Checklist

- [ ] Implementar residuos de Saint-Venant con `torch.autograd`
- [ ] Entrenar con Adam (lr=1e-3) + L-BFGS para refinamiento final
- [ ] Figura: curva de pérdida total y por componente durante entrenamiento
- [ ] Tabla: comparación `n` y `S₀` — PINN vs. CMA-ES vs. valor verdadero
- [ ] Probar ≥ 3 semillas aleatorias distintas para evaluar sensibilidad
- [ ] Figura: hidrograma PINN vs. modelo numérico vs. observaciones
- [ ] **Discusión honesta de limitaciones** (problemas de convergencia, desbalance de pérdidas, etc.)
- [ ] Documentar estrategias probadas si la PINN no converge (curriculum learning, normalización, rebalanceo)

> **Nota importante:** La inversión con PINN **no reemplaza** al optimizador clásico. Ambas vías deben ejecutarse y compararse críticamente. Los fracasos parciales bien diagnosticados **suman puntos**.

---

## 📝 Parte 5 — Reporte, Código y Presentación *(10 pts)*

**Todos los integrantes**  
**Duración estimada:** Semana 4

### Artículo técnico (dos columnas, máx. 15 páginas)

Estructura obligatoria:

1. **Abstract** — qué se hizo, qué se encontró (≤ 250 palabras)
2. **Introducción** — motivación, objetivos, descripción del sistema
3. **Materiales y métodos** — ecuaciones, datos, algoritmos
4. **Resultados y discusión** — todas las partes 1–4 integradas
5. **Conclusiones**
6. **Referencias**

> ⚠️ Todas las figuras y tablas deben tener *caption* autoexplicativo, ejes legibles y ser citadas en el texto.

### Repositorio Git

- [ ] `README.md` con instrucciones claras de instalación y ejecución
- [ ] `environment.yml` con todas las dependencias y versiones fijadas
- [ ] Semillas fijas documentadas en todos los scripts
- [ ] Notebooks que reproducen **cada figura y tabla** del reporte
- [ ] Datos sintéticos incluidos en el repo (son livianos)

### Presentación oral (máx. 15 min + 10 min preguntas)

- Principalmente figuras y bullet points — poco texto corrido
- **Todos los integrantes deben hablar** y demostrar dominio de todo el trabajo
- Se evalúa calidad visual de las diapositivas

Estructura sugerida:

| Sección | Tiempo | Quién |
|---|---|---|
| Intro + sistema + modelo | 3 min | Persona 1 |
| Sensibilidad global | 2 min | Persona 1 |
| Calibración + incertidumbre | 5 min | Personas 2 y 3 |
| PINN + comparación | 4 min | Persona 4 |
| Conclusiones | 1 min | Todos |

---

## 📅 Cronograma

```
Semana 1 (13–17 may)   Semana 2 (20–24 may)   Semana 3 (27–31 may)   Semana 4 (1–3 jun)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ P1: Modelo SV   │    │ P1: Sobol       │    │ P3: GLUE/MC     │    │ Integración  │
│ P2: Datos sint. │    │ P2: CMA-ES      │    │ P4: PINN        │    │ Reporte      │
│ Verificación    │    │ Partición datos │    │ Comparaciones   │    │ Presentación │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └──────────────┘
```

---

## 🛠️ Stack Tecnológico

```yaml
# environment.yml
name: icya4715
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - jupyter
  - joblib           # paralelización Monte Carlo
  - pip:
    - SALib          # análisis de sensibilidad Sobol
    - cma            # optimizador CMA-ES
    - torch          # PINN
    - deepxde        # alternativa PINN (opcional)
```

---

## ⚠️ Checklist Final (antes de entregar)

### Código
- [ ] Todas las semillas fijas (`np.random.seed(42)`, `torch.manual_seed(42)`)
- [ ] `environment.yml` reproduce el entorno desde cero
- [ ] Cada notebook corre de principio a fin sin errores
- [ ] Total de simulaciones documentadas (meta: 10,000–100,000)

### Reporte
- [ ] Máximo 15 páginas sin contar anexos
- [ ] Todas las figuras citadas en el texto con caption
- [ ] Índices de Sobol reportados **con barras de incertidumbre**
- [ ] Métricas KGE reportadas en calibración **y** validación por separado
- [ ] Banda de incertidumbre 5–95% con cobertura discutida
- [ ] Comparación explícita PINN vs. CMA-ES vs. valor verdadero

### Presentación
- [ ] Calidad visual cuidada (sin diapositivas de texto denso)
- [ ] Los 4 integrantes hablan
- [ ] Respuestas preparadas para preguntas de las partes ajenas

---

## 📚 Referencias Clave

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Tartakovsky, A. M. et al. (2020). Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. *Water Resources Research*, 56.
- Lu, L. et al. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.
- Beven, K. & Binley, A. (1992). The future of distributed models: model calibration and uncertainty prediction. *Hydrological Processes*, 6, 279–298. *(GLUE)*
- Saltelli, A. et al. (2010). *Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.* Computer Physics Communications.

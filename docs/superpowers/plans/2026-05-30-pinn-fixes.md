# PINN Fixes — tanh, normalización Q, curriculum

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir el colapso a solución trivial de la PINN aplicando tres fixes quirúrgicos: activación tanh, normalización de Q en L_data, y curriculum de entrenamiento.

**Architecture:** Primera ejecución de la PINN convergió a flujo uniforme constante (Q≈100 m³/s) porque L_data ≈ 10⁴ dominaba la escala pero el gradiente de L_pde era más fácil de minimizar trivialmente. Tres fixes coordinados en `src/pinn.py` más actualización del notebook y documentación en README.

**Tech Stack:** PyTorch, Jupyter notebook (JSON). Trabajo directo en `main` (fixes pequeños y quirúrgicos).

**Archivos a tocar:**
- Modify: `src/pinn.py` — tanh, q_scale, n_epochs_warmup
- Modify: `notebooks/04_pinn.ipynb` — celdas 2 y 4 (config + train call)
- Create: `README.md` — documentación del proyecto y decisiones PINN

---

### Task 1: Fixes en src/pinn.py

**Files:**
- Modify: `src/pinn.py`

Los tres cambios son independientes entre sí pero se aplican en un solo commit.

- [ ] **Step 1: Verificar estado inicial**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from src.pinn import SVPINN
m = SVPINN(hidden_size=4, n_layers=2, S0=0.001, beta=1.0, n_init=0.03, Bw_init=45.0)
# Verificar que hoy usa Sigmoid
found_sigmoid = any(isinstance(layer, torch.nn.Sigmoid) for layer in m.net)
print('Sigmoid activo:', found_sigmoid)
"
```

Output esperado: `Sigmoid activo: True`

- [ ] **Step 2: Fix A — cambiar Sigmoid → Tanh en SVPINN.__init__**

En `src/pinn.py`, líneas 55-57, reemplazar las dos ocurrencias de `nn.Sigmoid()` por `nn.Tanh()`:

```python
        # MLP: 2 → [hidden_size]*n_layers → 2
        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)
```

También actualizar el docstring del módulo (líneas 1-8) para reflejar tanh:

```python
"""PINN para estimación de parámetros del modelo Saint-Venant 1D.

Red MLP f(x_norm, t_norm) -> (A, Q) con activación tanh en capas ocultas
y softplus en la capa de salida (garantiza A > 0, Q > 0).
Los parámetros físicos n y B_w se almacenan en log-espacio como nn.Parameter.

Para cambiar qué parámetros se estiman, modificar ESTIMATE_PARAMS.
"""
```

- [ ] **Step 3: Fix B — añadir n_epochs_warmup a train() y normalización q_scale**

Reemplazar la función `train()` completa (líneas 157-273) con la versión corregida:

```python
def train(
    model: SVPINN,
    *,
    x0_data: torch.Tensor,
    xL_data: torch.Tensor,
    t_data: torch.Tensor,
    L: float,
    T: float,
    lambda_data: float = 1.0,
    lambda_pde: float = 0.05,
    n_epochs_adam: int = 5000,
    n_epochs_warmup: int = 0,
    n_iter_lbfgs: int = 500,
    n_colloc: int = 2000,
    resample_every: int = 1000,
    t_warmup: float = 3600.0,
    verbose_every: int = 500,
) -> TrainResult:
    """Entrena la PINN con Adam (exploración) y L-BFGS (convergencia fina).

    x0_data: Q observado en x=0 [m³/s], shape (nt,)
    xL_data: Q observado en x=L [m³/s], shape (nt,)
    t_data:  tiempos [s],              shape (nt,)
    L, T:    longitud del canal [m] y duración [s]
    n_epochs_warmup: épocas iniciales con lambda_pde=0 (solo ajuste de datos).
                     Permite que la red aprenda la variación temporal antes de
                     introducir la restricción física.
    """
    x0_data = x0_data.float()
    xL_data = xL_data.float()
    t_data  = t_data.float()

    post_warm = t_data >= t_warmup
    t_post    = t_data[post_warm]
    T_total   = float(t_data[-1]) if T <= 0 else float(T)

    # Escala de Q para normalizar L_data (evita desbalance con L_pde)
    q_scale = max(x0_data.max().item(), xL_data.max().item(), 1.0)

    loss_history: list[dict[str, Any]] = []

    def _sample_colloc(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        x_c = torch.rand(n_colloc) * L
        t_c = torch.rand(n_colloc) * (T_total - max(t_warmup, 0.0)) + max(t_warmup, 0.0)
        return x_c, t_c

    def _compute_loss(
        x_c: torch.Tensor, t_c: torch.Tensor, lp: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Datos normalizados en los dos bordes (x=0 y x=L)
        tn = t_post / T_total
        _, Q0_pred = model(torch.zeros_like(tn), tn)
        _, QL_pred = model(torch.ones_like(tn),  tn)
        Q0_obs = x0_data[post_warm]
        QL_obs = xL_data[post_warm]
        l_data = (
            torch.mean(((Q0_pred - Q0_obs) / q_scale) ** 2)
            + torch.mean(((QL_pred - QL_obs) / q_scale) ** 2)
        )

        # Física en puntos de colocación interiores
        R_mass, R_mom = pde_residuals(model, x_c, t_c, L=L, T=T_total)
        l_pde = torch.mean(R_mass ** 2) + torch.mean(R_mom ** 2)

        l_total = lambda_data * l_data + lp * l_pde
        return l_total, l_data, l_pde

    # ── Fase 1: Adam ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_c, t_c = _sample_colloc(seed=0)

    for epoch in range(n_epochs_adam):
        if epoch > 0 and epoch % resample_every == 0:
            x_c, t_c = _sample_colloc(seed=epoch)

        # Curriculum: primeras n_epochs_warmup épocas sin física
        lp_current = 0.0 if epoch < n_epochs_warmup else lambda_pde
        if epoch == n_epochs_warmup and n_epochs_warmup > 0:
            print(f"[Adam {epoch:5d}] ** Activando L_pde (lambda={lambda_pde}) **")

        optimizer.zero_grad()
        l_total, l_data, l_pde = _compute_loss(x_c, t_c, lp=lp_current)
        l_total.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == n_epochs_adam - 1:
            entry: dict[str, Any] = {
                "epoch": epoch,
                "total": l_total.detach().item(),
                "data":  l_data.detach().item(),
                "pde":   l_pde.detach().item(),
                "n":     model.n.detach().item(),
                "Bw":    model.Bw.detach().item(),
            }
            loss_history.append(entry)
            print(
                f"[Adam {epoch:5d}] "
                f"total={l_total:.4e}  data={l_data:.4e}  pde={l_pde:.4e}  "
                f"n={model.n:.5f}  Bw={model.Bw:.3f}"
            )

    # ── Fase 2: L-BFGS ───────────────────────────────────────────────────────
    x_c, t_c = _sample_colloc(seed=9999)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=n_iter_lbfgs, history_size=50
    )

    def _closure() -> torch.Tensor:
        lbfgs.zero_grad()
        l_total, _, _ = _compute_loss(x_c, t_c, lp=lambda_pde)
        l_total.backward()
        return l_total

    l_final = lbfgs.step(_closure)
    loss_history.append({
        "epoch": n_epochs_adam,
        "total": l_final.detach().item() if l_final is not None else float("nan"),
        "n":  model.n.detach().item(),
        "Bw": model.Bw.detach().item(),
        "phase": "lbfgs",
    })
    print(f"[L-BFGS] n={model.n:.5f}  Bw={model.Bw:.3f}")

    return TrainResult(
        model=model,
        loss_history=loss_history,
        n_estimate=model.n.detach().item(),
        Bw_estimate=model.Bw.detach().item(),
    )
```

- [ ] **Step 4: Verificar que los cambios compilan y los checks pasan**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from src.pinn import SVPINN, pde_residuals, train, TrainResult

# Fix A: verificar tanh
m = SVPINN(hidden_size=4, n_layers=2, S0=0.001, beta=1.0, n_init=0.03, Bw_init=45.0)
found_tanh = any(isinstance(layer, torch.nn.Tanh) for layer in m.net)
found_sig  = any(isinstance(layer, torch.nn.Sigmoid) for layer in m.net)
assert found_tanh and not found_sig, f'tanh={found_tanh} sigmoid={found_sig}'

# Fix B: verificar n_epochs_warmup en firma
import inspect
sig = inspect.signature(train)
assert 'n_epochs_warmup' in sig.parameters, 'falta n_epochs_warmup'
assert sig.parameters['n_epochs_warmup'].default == 0

# Fix C: run corto con warmup
t_data = torch.linspace(0, 3600.0, 20)
q_up   = torch.full((20,), 80.0)
q_dn   = torch.full((20,), 70.0)
result = train(m, x0_data=q_up, xL_data=q_dn, t_data=t_data,
               L=5000.0, T=3600.0, n_epochs_adam=10, n_epochs_warmup=5,
               n_iter_lbfgs=2, n_colloc=20, resample_every=5, t_warmup=0.0,
               verbose_every=10)
assert result.n_estimate > 0 and result.Bw_estimate > 0
print('ALL FIXES VERIFIED')
"
```

Output esperado:
```
[Adam     0] total=...  ...
[Adam     5] ** Activando L_pde (lambda=0.05) **
[Adam     5] ...
[L-BFGS] n=...  Bw=...
ALL FIXES VERIFIED
```

- [ ] **Step 5: Commit**

```bash
git add src/pinn.py
git commit -m "fix(pinn): tanh, normalizacion Q, curriculum n_epochs_warmup"
```

---

### Task 2: Actualizar notebook 04_pinn.ipynb

**Files:**
- Modify: `notebooks/04_pinn.ipynb` (celdas 1 y 3 — config y train call)

- [ ] **Step 1: Verificar ids de celdas**

```bash
python -c "
import json
nb = json.load(open('notebooks/04_pinn.ipynb'))
for i,c in enumerate(nb['cells']):
    print(i, c['cell_type'], repr(''.join(c['source'])[:60]))
"
```

Expected: celdas 0=markdown, 1=config, 2=data, 3=train, 4=loss+params, 5=hydro.

- [ ] **Step 2: Crear script de parche para el notebook**

Crear `_patch_nb04.py` en la raíz del proyecto:

```python
# _patch_nb04.py
import json
from pathlib import Path

nb_path = Path("notebooks/04_pinn.ipynb")
nb = json.load(nb_path.open(encoding="utf-8"))

# ── Celda 1 (índice 1): Configuración ────────────────────────────────────────
new_config = """\
# ── Configuración ─────────────────────────────────────────────────────────────
# Cambiar True/False para estimar otros parámetros
ESTIMATE_PARAMS = {"n": True, "Bw": True}

# Pesos de la función de pérdida
LAMBDA_DATA = 1.0    # peso de L_data (ajuste a observaciones)
LAMBDA_PDE  = 0.05   # peso de L_pde  (cumplimiento de la física)

# Hiperparámetros de entrenamiento
N_EPOCHS_ADAM    = 6_000   # total épocas Fase 1 — Adam
N_EPOCHS_WARMUP  = 2_000   # primeras N épocas sin física (solo datos)
N_ITER_LBFGS     = 500     # iteraciones Fase 2 — L-BFGS
N_COLLOC         = 2_000   # puntos de colocación interiores
RESAMPLE_EVERY   = 1_000   # re-muestrear colloc cada N épocas Adam

# Conjeturas iniciales (lejos del valor verdadero para mostrar convergencia)
N_INIT  = 0.030   # verdadero: 0.035
BW_INIT = 45.0    # verdadero: 50.0 m

# Archivo de observaciones (mismo que en calibración, notebook 03)
CSV_NAME    = "series_corta_shift.csv"
WARMUP_SEC  = 3600.0
"""
nb["cells"][1]["source"] = new_config.splitlines(keepends=True)
nb["cells"][1]["outputs"] = []
nb["cells"][1]["execution_count"] = None

# ── Celda 3 (índice 3): Construcción y entrenamiento ─────────────────────────
new_train = """\
# ── Construcción y entrenamiento ──────────────────────────────────────────────
import importlib
import src.pinn as pinn_mod
importlib.reload(pinn_mod)

model = pinn_mod.SVPINN(
    hidden_size=64,
    n_layers=4,
    S0=sinteticos.S0,
    beta=1.0,
    n_init=N_INIT,
    Bw_init=BW_INIT,
    estimate_params=ESTIMATE_PARAMS,
)

# Fase 1 — Adam (curriculum: primero datos, luego datos+física)
# Fase 2 — L-BFGS: convergencia cuadrática hacia el mínimo final
result = pinn_mod.train(
    model,
    x0_data=qu_tensor,
    xL_data=qd_tensor,
    t_data=t_tensor,
    L=sinteticos.L,
    T=T_total,
    lambda_data=LAMBDA_DATA,
    lambda_pde=LAMBDA_PDE,
    n_epochs_adam=N_EPOCHS_ADAM,
    n_epochs_warmup=N_EPOCHS_WARMUP,
    n_iter_lbfgs=N_ITER_LBFGS,
    n_colloc=N_COLLOC,
    resample_every=RESAMPLE_EVERY,
    t_warmup=WARMUP_SEC,
    verbose_every=500,
)
print(f"\\nEstimado PINN: n={result.n_estimate:.5f}  Bw={result.Bw_estimate:.3f} m")
"""
nb["cells"][3]["source"] = new_train.splitlines(keepends=True)
nb["cells"][3]["outputs"] = []
nb["cells"][3]["execution_count"] = None

# Limpiar outputs de todas las celdas de código
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook actualizado OK")
```

- [ ] **Step 3: Ejecutar el script**

```bash
python _patch_nb04.py
```

Output esperado: `Notebook actualizado OK`

- [ ] **Step 4: Verificar que el notebook es JSON válido y tiene los cambios**

```bash
python -c "
import json
nb = json.load(open('notebooks/04_pinn.ipynb'))
cfg = ''.join(nb['cells'][1]['source'])
trn = ''.join(nb['cells'][3]['source'])
assert 'N_EPOCHS_WARMUP' in cfg, 'falta N_EPOCHS_WARMUP en config'
assert 'N_EPOCHS_ADAM    = 6_000' in cfg, 'N_EPOCHS_ADAM no actualizado'
assert 'n_epochs_warmup=N_EPOCHS_WARMUP' in trn, 'falta argumento en train()'
assert all(c['outputs'] == [] for c in nb['cells'] if c['cell_type']=='code'), 'outputs no limpios'
print('NOTEBOOK OK')
"
```

Output esperado: `NOTEBOOK OK`

- [ ] **Step 5: Borrar script auxiliar y commitear**

```bash
del _patch_nb04.py
git add notebooks/04_pinn.ipynb
git commit -m "fix(pinn): notebook 04 — curriculum y N_EPOCHS_WARMUP en config"
```

(En Linux/Mac: `rm _patch_nb04.py`)

---

### Task 3: Crear README.md

**Files:**
- Create: `README.md`

El README documenta el proyecto, su estructura, y — en la sección PINN — el historial de decisiones: qué corrimos, qué falló, por qué, y qué hicimos.

- [ ] **Step 1: Verificar que no existe README**

```bash
python -c "from pathlib import Path; print('existe' if Path('README.md').exists() else 'no existe')"
```

Output esperado: `no existe`

- [ ] **Step 2: Crear README.md**

Crear `README.md` en la raíz del proyecto con el siguiente contenido:

```markdown
# Proyecto Saint-Venant 1D — ICYA 4715

Modelación hidráulica 1D del canal rectangular usando las ecuaciones de Saint-Venant,
implementada para el curso ICYA 4715 (Hidrosistemas). El proyecto cubre verificación
del solver, análisis de sensibilidad, calibración y estimación de parámetros con PINN.

## Estructura

| Notebook | Descripción |
|----------|-------------|
| `01_model_verification.ipynb` | Verificación del solver MacCormack: propagación de hidrograma, conservación de masa, demo de paralelismo |
| `02_sensitivity.ipynb` | Sensibilidad global (Sobol) y OLS para los parámetros n, S0, B_w, η_Q |
| `03_calibration.ipynb` | Calibración Monte Carlo / GLUE del modelo Saint-Venant |
| `04_pinn.ipynb` | Estimación de parámetros con PINN (Physics-Informed Neural Network) |

## Datos sintéticos

Los datos de entrada se generan con `data/sinteticos.py`. Parámetros verdaderos del canal:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| n | 0.035 | Coeficiente de Manning |
| S₀ | 0.001 | Pendiente de fondo |
| Q₀ | 50.0 m³/s | Caudal base |
| B_w | 50.0 m | Ancho del canal |
| L | 5 000 m | Longitud del canal |

## Solver

`src/model.py` implementa Saint-Venant 1D conservativo con esquema MacCormack
predictor-corrector. El sub-stepping CFL garantiza estabilidad numérica: cada
intervalo de salida (dt=15 min) se avanza en múltiples pasos internos con
dt_CFL ≈ dx / (|u| + c).

## PINN — Decisiones de diseño y lecciones aprendidas

### Primera implementación (fallida)

La primera versión usaba:
- Activación **sigmoid** en capas ocultas
- Función de pérdida: L_total = L_data + 0.05 · L_pde
- Sin normalización de Q
- Sin curriculum (datos y física desde el inicio)

**Resultados:** La PINN convergió a flujo uniforme constante (Q ≈ 100 m³/s,
línea horizontal), estimando n = 0.011 y B_w = 18 m (valores verdaderos: 0.035 y 50 m).

### Diagnóstico

Tres causas apiladas:

1. **Desbalance de escala en la pérdida.** Q_obs ∈ [40, 370] m³/s → L_data inicial ≈ 10⁴.
   L_pde inicial ≈ 1. El gradiente de L_pde minimizaba trivialmente: Q = constante
   satisface ∂Q/∂x = 0 y ∂A/∂t = 0 (flujo uniforme), haciendo L_pde → 0 sin que la
   red aprenda la dinámica temporal. L_data nunca bajaba porque la solución constante
   no ajusta los bordes.

2. **Saturación por sigmoid.** Con 4 capas ocultas y activación sigmoid, las capas
   intermedias saturaban hacia valores constantes antes de aprender variaciones
   espacio-temporales. Tanh es zero-centered y tiene gradiente más robusto para PINNs.

3. **Sin curriculum.** Introducir la restricción física desde el primer epoch empuja
   al optimizador hacia la solución trivial (flujo uniforme) antes de que la red haya
   aprendido la señal temporal de los datos.

### Fixes aplicados

| Fix | Cambio | Razón |
|-----|--------|-------|
| Activación | sigmoid → tanh en capas ocultas | Evita saturación; gradiente más robusto |
| Normalización | MSE((Q_pred - Q_obs) / Q_max)² | Escala L_data de 10⁴ a ~1 |
| Curriculum | n_epochs_warmup=2000: primeras 2000 épocas sin L_pde | Red aprende señal temporal antes de aplicar física |

Configuración final del notebook: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.

## Instalación

```bash
pip install -r requirements.txt   # incluye torch>=2.0
```
```

- [ ] **Step 3: Verificar que el archivo es Markdown válido**

```bash
python -c "
from pathlib import Path
txt = Path('README.md').read_text(encoding='utf-8')
assert '## PINN' in txt, 'falta sección PINN'
assert 'Diagnóstico' in txt, 'falta diagnóstico'
assert 'Fixes aplicados' in txt, 'falta tabla de fixes'
assert len(txt) > 1000, f'README demasiado corto: {len(txt)} chars'
print(f'README OK ({len(txt)} chars)')
"
```

Output esperado: `README OK (XXXX chars)`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README con estructura del proyecto y lecciones PINN"
```

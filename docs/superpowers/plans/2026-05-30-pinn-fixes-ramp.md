# PINN Fixes — Gradient Clipping y Ramp Gradual de L_pde

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir el spike catastrófico de L_pde al activar la física en la época 2000 mediante gradient clipping y un ramp lineal de lambda_pde, y documentar el diagnóstico en README.

**Architecture:** Dos fixes quirúrgicos en `train()`: (1) `clip_grad_norm_` después de `backward()` limita el paso máximo del optimizador; (2) nuevo parámetro `n_epochs_ramp` hace subir lambda_pde linealmente de 0 a su valor final en lugar de activarlo abruptamente. El README registra la segunda ronda de fallos y fixes para trazabilidad.

**Tech Stack:** PyTorch, Jupyter notebook JSON, Markdown. Trabajo directo en `main`.

**Archivos a tocar:**
- Modify: `src/pinn.py` (líneas 168, 181-183, 233-241)
- Modify: `notebooks/04_pinn.ipynb` (celdas 1 y 3)
- Modify: `README.md` (añadir sección después de línea 75)

---

### Task 1: Gradient clipping + ramp en src/pinn.py

**Files:**
- Modify: `src/pinn.py`

Tres cambios precisos dentro de `train()`. No tocar nada fuera de esa función.

- [ ] **Step 1: Verificar estado actual (curriculum binario sin clipping)**

```bash
python -c "
import sys, inspect; sys.path.insert(0, '.')
from src.pinn import train
src = inspect.getsource(train)
print('tiene ramp:', 'n_epochs_ramp' in src)
print('tiene clipping:', 'clip_grad_norm_' in src)
"
```

Output esperado: `tiene ramp: False` y `tiene clipping: False`

- [ ] **Step 2: Cambio A — añadir n_epochs_ramp a la firma de train()**

En `src/pinn.py`, localizar la línea:

```python
    n_epochs_warmup: int = 0,
    n_iter_lbfgs: int = 500,
```

Reemplazar con:

```python
    n_epochs_warmup: int = 0,
    n_epochs_ramp: int = 0,
    n_iter_lbfgs: int = 500,
```

- [ ] **Step 3: Cambio B — actualizar docstring de train()**

Localizar el bloque de docstring:

```python
    n_epochs_warmup: épocas iniciales con lambda_pde=0 (solo ajuste de datos).
                     Permite que la red aprenda la variación temporal antes de
                     introducir la restricción física.
```

Reemplazar con:

```python
    n_epochs_warmup: épocas iniciales con lambda_pde=0 (solo ajuste de datos).
    n_epochs_ramp:   épocas para subir lambda_pde linealmente de 0 al valor final.
                     Previene el spike de gradiente al activar la física abruptamente.
```

- [ ] **Step 4: Cambio C — reemplazar curriculum binario por 3 fases + gradient clipping**

Localizar el bloque del bucle Adam que contiene:

```python
        # Curriculum: primeras n_epochs_warmup épocas sin física
        lp_current = 0.0 if epoch < n_epochs_warmup else lambda_pde
        if epoch == n_epochs_warmup and n_epochs_warmup > 0:
            print(f"[Adam {epoch:5d}] ** Activando L_pde (lambda={lambda_pde}) **")

        optimizer.zero_grad()
        l_total, l_data, l_pde = _compute_loss(x_c, t_c, lp=lp_current)
        l_total.backward()
        optimizer.step()
```

Reemplazar con:

```python
        # Curriculum 3 fases: sin física → ramp lineal → física completa
        if epoch < n_epochs_warmup:
            lp_current = 0.0
        elif epoch < n_epochs_warmup + n_epochs_ramp:
            progress = (epoch - n_epochs_warmup) / max(n_epochs_ramp, 1)
            lp_current = lambda_pde * progress
            if epoch == n_epochs_warmup:
                print(f"[Adam {epoch:5d}] ** Ramp L_pde: 0 → {lambda_pde} en {n_epochs_ramp} épocas **")
        else:
            lp_current = lambda_pde
            if epoch == n_epochs_warmup + n_epochs_ramp and n_epochs_ramp > 0:
                print(f"[Adam {epoch:5d}] ** L_pde completo (lambda={lambda_pde}) **")

        optimizer.zero_grad()
        l_total, l_data, l_pde = _compute_loss(x_c, t_c, lp=lp_current)
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

- [ ] **Step 5: Verificar que los cambios son correctos**

```bash
python -c "
import sys, inspect, torch; sys.path.insert(0, '.')
from src.pinn import SVPINN, train

# n_epochs_ramp en firma con default 0
sig = inspect.signature(train)
assert 'n_epochs_ramp' in sig.parameters, 'falta n_epochs_ramp'
assert sig.parameters['n_epochs_ramp'].default == 0, 'default debe ser 0'

# clip_grad_norm_ presente en source
src = inspect.getsource(train)
assert 'clip_grad_norm_' in src, 'falta clip_grad_norm_'
assert 'n_epochs_ramp' in src, 'falta ramp en cuerpo'

# Ejecución corta con ramp
m = SVPINN(hidden_size=8, n_layers=2, S0=0.001, beta=1.0, n_init=0.03, Bw_init=45.0)
t = torch.linspace(0, 3600.0, 20)
q = torch.full((20,), 80.0)
result = train(m, x0_data=q, xL_data=q*0.9, t_data=t,
               L=5000.0, T=3600.0, n_epochs_adam=15, n_epochs_warmup=5,
               n_epochs_ramp=5, n_iter_lbfgs=2, n_colloc=20,
               resample_every=10, t_warmup=0.0, verbose_every=15)
assert result.n_estimate > 0 and result.Bw_estimate > 0
print('VERIFICACION OK')
"
```

Output esperado incluye `** Ramp L_pde **` en época 5, `** L_pde completo **` en época 10, y termina en `VERIFICACION OK`.

- [ ] **Step 6: Commit**

```bash
git add src/pinn.py
git commit -m "fix(pinn): gradient clipping + ramp lineal lambda_pde (n_epochs_ramp)"
```

---

### Task 2: Actualizar notebook 04_pinn.ipynb

**Files:**
- Modify: `notebooks/04_pinn.ipynb` (celdas 1 y 3)

- [ ] **Step 1: Crear script de parche**

Crear `_patch_nb04_ramp.py` en la raíz del proyecto:

```python
# _patch_nb04_ramp.py
import json
from pathlib import Path

nb_path = Path("notebooks/04_pinn.ipynb")
nb = json.load(nb_path.open(encoding="utf-8"))

# ── Celda 1 (índice 1): añadir N_EPOCHS_RAMP ─────────────────────────────────
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
N_EPOCHS_RAMP    = 1_000   # épocas de ramp lineal 0 → LAMBDA_PDE
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

# ── Celda 3 (índice 3): añadir n_epochs_ramp al call de train() ───────────────
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

# Curriculum 3 fases:
#   Fase 1a: épocas 0 – N_EPOCHS_WARMUP       → solo L_data (sin física)
#   Fase 1b: épocas WARMUP – WARMUP+RAMP      → ramp lineal lambda_pde 0→0.05
#   Fase 1c: épocas WARMUP+RAMP – N_EPOCHS_ADAM → L_data + L_pde completa
#   Fase 2:  L-BFGS convergencia fina
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
    n_epochs_ramp=N_EPOCHS_RAMP,
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

- [ ] **Step 2: Ejecutar el script**

```bash
python _patch_nb04_ramp.py
```

Output esperado: `Notebook actualizado OK`

- [ ] **Step 3: Verificar cambios**

```bash
python -c "
import json
nb = json.load(open('notebooks/04_pinn.ipynb'))
cfg = ''.join(nb['cells'][1]['source'])
trn = ''.join(nb['cells'][3]['source'])
assert 'N_EPOCHS_RAMP    = 1_000' in cfg, 'falta N_EPOCHS_RAMP'
assert 'n_epochs_ramp=N_EPOCHS_RAMP' in trn, 'falta argumento ramp en train()'
assert len(nb['cells']) == 6, 'debe tener 6 celdas'
assert all(c['outputs'] == [] for c in nb['cells'] if c['cell_type'] == 'code')
print('NOTEBOOK OK')
"
```

Output esperado: `NOTEBOOK OK`

- [ ] **Step 4: Borrar script y commitear**

```bash
del _patch_nb04_ramp.py
git add notebooks/04_pinn.ipynb
git commit -m "fix(pinn): notebook 04 — N_EPOCHS_RAMP=1000 en config y train call"
```

(En Linux/Mac: `rm _patch_nb04_ramp.py`)

---

### Task 3: Actualizar README.md con segunda ronda de diagnóstico

**Files:**
- Modify: `README.md`

Añadir dos subsecciones nuevas después de la tabla "Fixes aplicados" existente y antes de "## Instalación". El README actual termina en la línea:

```
Configuración final del notebook: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.
```

- [ ] **Step 1: Verificar contenido actual**

```bash
python -c "
from pathlib import Path
txt = Path('README.md').read_text(encoding='utf-8')
print('tiene segunda ronda:', 'Segunda implementación' in txt)
print('tiene clipping:', 'clip' in txt.lower() or 'clipping' in txt.lower())
"
```

Output esperado: `tiene segunda ronda: False` y `tiene clipping: False`

- [ ] **Step 2: Añadir las nuevas secciones al README**

Localizar en `README.md` el bloque exacto:

```
Configuración final del notebook: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.

## Instalación
```

Reemplazar con:

```
Configuración final del notebook: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.

### Segunda implementación (fallida) — spike de L_pde

**Resultado:** L_pde creció de 1 a 10⁸ durante las 2000 épocas de warmup.
Al activar la física abruptamente en la época 2000, el gradiente de L_pde fue ≈10⁷
vs el de L_data ≈0.1. El optimizador destruyó todo lo aprendido en un solo paso:
Q_PINN colapsó a ≈0 m³/s, n = 0.0003, B_w = 1.7 m.

**Causa raíz:** Durante el warmup (solo L_data), la red aprende los bordes pero en los
puntos interiores sin restricción produce valores no físicos con gradientes espaciales
enormes. Al activar L_pde, esos residuos catastrófico generan un spike de gradiente
que destruye los pesos de la red.

### Fixes segunda ronda

| Fix | Cambio | Razón |
|-----|--------|-------|
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` en cada paso Adam | Limita el tamaño del paso; previene el spike catastrófico |
| Ramp gradual | n_epochs_ramp=1000: lambda_pde sube linealmente 0 → 0.05 | Transición suave; red adapta gradualmente sin choque |

Configuración final: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000, N_EPOCHS_RAMP=1000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.

## Instalación
```

- [ ] **Step 3: Verificar README**

```bash
python -c "
from pathlib import Path
txt = Path('README.md').read_text(encoding='utf-8')
assert 'Segunda implementación (fallida)' in txt, 'falta sección segunda ronda'
assert 'clip_grad_norm_' in txt, 'falta gradient clipping'
assert 'n_epochs_ramp' in txt, 'falta ramp'
assert 'spike' in txt.lower(), 'falta descripción del spike'
assert txt.count('## Instalación') == 1, 'Instalación duplicada'
print(f'README OK ({len(txt)} chars)')
"
```

Output esperado: `README OK (XXXX chars)` sin duplicados.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README documenta segunda ronda PINN (spike L_pde, clipping, ramp)"
```

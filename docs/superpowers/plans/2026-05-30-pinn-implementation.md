# PINN — Estimación de parámetros Saint-Venant 1D

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar una PINN en `src/pinn.py` que estime n y B_w a partir de observaciones sintéticas de Saint-Venant 1D, y mostrar resultados en `notebooks/04_pinn.ipynb`.

**Architecture:** MLP sigmoid f(x_norm, t_norm) → (A, Q) aprende el campo espacio-temporal. n y B_w son `nn.Parameter` en log-espacio. La función de pérdida combina MSE en bordes observados (L_data) con residuos SV en 2 000 puntos de colocación interiores calculados por autograd (L_pde). Entrenamiento: Adam 5 000 épocas → L-BFGS 500 iteraciones.

**Tech Stack:** PyTorch ≥ 2.0, NumPy, Pandas, Matplotlib. Worktree: `.worktrees/feature-pinn` (rama `feature/pinn`). Todos los comandos se ejecutan desde `C:/Users/Luis/Desktop/proyecto-sv-mh/.worktrees/feature-pinn`.

**Archivos a tocar:**
- Modify: `src/pinn.py` (actualmente vacío — Tasks 2–5)
- Modify: `notebooks/04_pinn.ipynb` (actualmente vacío — Task 6)

**NO tocar:** `src/model.py`, `src/sensitivity.py`, `data/sinteticos.py` ni ningún otro archivo existente.

---

### Task 1: Instalar PyTorch

**Files:**
- Run: `pip install torch` en el entorno virtual del proyecto
- Modify: `requirements.txt` (añadir `torch>=2.0`)

- [ ] **Step 1: Instalar PyTorch CPU-only**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Output esperado (última línea): `Successfully installed torch-X.X.X`

- [ ] **Step 2: Verificar autograd**

```bash
python -c "import torch; x = torch.tensor([1.0], requires_grad=True); y = x**2; y.backward(); print('autograd OK grad=', x.grad.item())"
```

Output esperado: `autograd OK grad= 2.0`

- [ ] **Step 3: Añadir torch a requirements.txt**

Abrir `requirements.txt` en la raíz del worktree y añadir al final:

```
torch>=2.0
```

El archivo completo debe quedar:

```
numpy>=1.24
scipy>=1.11
pandas>=2.1
matplotlib>=3.8
jupyter>=1.0
joblib>=1.3
SALib>=1.4.5
ydata-profiling>=4.6.0
torch>=2.0
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add torch>=2.0 dependency for PINN"
```

---

### Task 2: Clase SVPINN

**Files:**
- Modify: `src/pinn.py`

- [ ] **Step 1: Crear script de prueba `_test_svpinn.py`**

Crear en la raíz del worktree (`C:/Users/Luis/Desktop/proyecto-sv-mh/.worktrees/feature-pinn/`):

```python
# _test_svpinn.py
import sys; sys.path.insert(0, ".")
import torch
from src.pinn import SVPINN

# Test 1: forward pass — formas correctas y positividad
model = SVPINN(hidden_size=16, n_layers=2, S0=0.001, beta=1.0,
               n_init=0.030, Bw_init=45.0)
xn = torch.linspace(0, 1, 10)
tn = torch.linspace(0, 1, 10)
A, Q = model(xn, tn)
assert A.shape == (10,), f"A shape: {A.shape}"
assert Q.shape == (10,), f"Q shape: {Q.shape}"
assert (A > 0).all(), "A debe ser positivo"
assert (Q > 0).all(), "Q debe ser positivo"

# Test 2: propiedades n y Bw accesibles y positivas
assert model.n.item() > 0
assert model.Bw.item() > 0

# Test 3: estimate_params=False fija el parámetro (sin gradiente)
m2 = SVPINN(hidden_size=16, n_layers=2, S0=0.001, beta=1.0,
            n_init=0.030, Bw_init=45.0,
            estimate_params={"n": False, "Bw": True})
assert not m2._log_n.requires_grad,  "n debe estar fijo"
assert     m2._log_Bw.requires_grad, "Bw debe ser entrenable"

print("SVPINN TEST PASSED")
```

- [ ] **Step 2: Ejecutar — debe fallar con ImportError**

```bash
python _test_svpinn.py
```

Output esperado: `ImportError` o `ModuleNotFoundError` porque `src/pinn.py` está vacío.

- [ ] **Step 3: Implementar SVPINN en `src/pinn.py`**

Escribir el contenido completo de `src/pinn.py`:

```python
"""PINN para estimación de parámetros del modelo Saint-Venant 1D.

Red MLP f(x_norm, t_norm) -> (A, Q) con activación sigmoid en capas ocultas
y softplus en la capa de salida (garantiza A > 0, Q > 0).
Los parámetros físicos n y B_w se almacenan en log-espacio como nn.Parameter.

Para cambiar qué parámetros se estiman, modificar ESTIMATE_PARAMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

G = 9.81
MIN_AREA = 1e-4

# ── Configuración de estimación ───────────────────────────────────────────────
# Cambiar True/False para incluir o excluir parámetros del entrenamiento.
ESTIMATE_PARAMS: dict[str, bool] = {
    "n": True,
    "Bw": True,
}


# ── Modelo ────────────────────────────────────────────────────────────────────

class SVPINN(nn.Module):
    """MLP espacio-temporal + parámetros físicos estimables.

    Entradas: (x_norm, t_norm) ∈ [0,1]².
    Salidas:  (A [m²], Q [m³/s]) en cada punto.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        n_layers: int = 4,
        S0: float = 0.001,
        beta: float = 1.0,
        n_init: float = 0.030,
        Bw_init: float = 45.0,
        estimate_params: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        ep = ESTIMATE_PARAMS if estimate_params is None else estimate_params

        # MLP: 2 → [hidden_size]*n_layers → 2
        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.Sigmoid()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

        # Parámetros estimables en log-espacio (garantiza positividad)
        self._log_n = nn.Parameter(
            torch.log(torch.tensor(float(n_init))),
            requires_grad=bool(ep.get("n", False)),
        )
        self._log_Bw = nn.Parameter(
            torch.log(torch.tensor(float(Bw_init))),
            requires_grad=bool(ep.get("Bw", False)),
        )

        # Parámetros fijos (sin gradiente)
        self.register_buffer("S0",   torch.tensor(float(S0)))
        self.register_buffer("beta", torch.tensor(float(beta)))

    @property
    def n(self) -> torch.Tensor:
        return torch.exp(self._log_n)

    @property
    def Bw(self) -> torch.Tensor:
        return torch.exp(self._log_Bw)

    def forward(
        self, x_norm: torch.Tensor, t_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.stack([x_norm, t_norm], dim=-1)
        out = self.net(inp)
        # softplus garantiza A > MIN_AREA y Q > MIN_AREA en todo momento
        A = nn.functional.softplus(out[..., 0]) + MIN_AREA
        Q = nn.functional.softplus(out[..., 1]) + MIN_AREA
        return A, Q
```

- [ ] **Step 4: Ejecutar prueba**

```bash
python _test_svpinn.py
```

Output esperado: `SVPINN TEST PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): SVPINN MLP con activación sigmoid y parámetros estimables"
```

---

### Task 3: Residuos de la PDE (física por autograd)

**Files:**
- Modify: `src/pinn.py` (añadir `pde_residuals()`)

- [ ] **Step 1: Crear `_test_pde.py`**

```python
# _test_pde.py
import sys; sys.path.insert(0, ".")
import torch
from src.pinn import SVPINN, pde_residuals

model = SVPINN(hidden_size=16, n_layers=2, S0=0.001, beta=1.0,
               n_init=0.030, Bw_init=45.0)

N = 40
x_col = torch.rand(N) * 5000.0        # coordenadas físicas [m]
t_col = torch.rand(N) * 7200.0 + 3600.0  # [s], fuera del warmup

R_mass, R_mom = pde_residuals(model, x_col, t_col, L=5000.0, T=7200.0)

assert R_mass.shape == (N,), f"R_mass: {R_mass.shape}"
assert R_mom.shape  == (N,), f"R_mom:  {R_mom.shape}"
assert not torch.isnan(R_mass).any(), "NaN en R_mass"
assert not torch.isnan(R_mom).any(),  "NaN en R_mom"

# Los residuos deben tener gradiente respecto de los parámetros del modelo
loss = (R_mass**2 + R_mom**2).mean()
loss.backward()
assert model._log_n.grad  is not None, "grad(_log_n) es None"
assert model._log_Bw.grad is not None, "grad(_log_Bw) es None"

print("PDE RESIDUALS TEST PASSED")
```

- [ ] **Step 2: Ejecutar — debe fallar con AttributeError**

```bash
python _test_pde.py
```

Output esperado: `AttributeError: module 'src.pinn' has no attribute 'pde_residuals'`

- [ ] **Step 3: Añadir `pde_residuals()` a `src/pinn.py`**

Añadir después de la clase SVPINN (antes de cualquier otra función):

```python
# ── Residuos de la PDE ────────────────────────────────────────────────────────

def pde_residuals(
    model: SVPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    L: float,
    T: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuos de Saint-Venant en puntos de colocación (x_col [m], t_col [s]).

    Usa torch.autograd.grad con create_graph=True para que los gradientes
    fluyan hasta n y Bw durante el entrenamiento.

    Returns: (R_mass, R_mom) — shape (N,) cada uno.
    """
    # Normalizar y crear hojas de autograd para derivadas espaciales/temporales
    x_norm = (x_col / L).clone().detach().requires_grad_(True)
    t_norm = (t_col / T).clone().detach().requires_grad_(True)

    A, Q = model(x_norm, t_norm)

    Bw   = model.Bw
    n    = model.n
    S0   = model.S0
    beta = model.beta

    # Geometría hidráulica
    h     = torch.clamp(A / Bw, min=1e-4)
    h_c   = 0.5 * h
    per   = Bw + 2.0 * h
    R_hyd = A / per                          # radio hidráulico [m]

    # Flujo de momentum: F_Q = β Q²/A + g h_c A
    F_Q = beta * Q**2 / (A + MIN_AREA) + G * h_c * A

    # Pendiente de fricción de Manning: Sf = n² Q|Q| / (A² R^(4/3))
    Sf = n**2 * Q * torch.abs(Q) / ((A + MIN_AREA)**2 * R_hyd**(4.0 / 3.0))

    # Derivadas por autograd (regla de la cadena: ∂A/∂t = (∂A/∂t_norm)/T)
    ones     = torch.ones_like(A)
    dA_dtn   = torch.autograd.grad(A,   t_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dxn   = torch.autograd.grad(Q,   x_norm, grad_outputs=ones, create_graph=True)[0]
    dFQ_dxn  = torch.autograd.grad(F_Q, x_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dtn   = torch.autograd.grad(Q,   t_norm, grad_outputs=ones, create_graph=True)[0]

    # Residuos de continuidad y momentum
    R_mass = dA_dtn / T + dQ_dxn / L
    R_mom  = dQ_dtn / T + dFQ_dxn / L - G * A * (S0 - Sf)

    return R_mass, R_mom
```

- [ ] **Step 4: Ejecutar prueba**

```bash
python _test_pde.py
```

Output esperado: `PDE RESIDUALS TEST PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): pde_residuals() con autograd SV continuidad y momentum"
```

---

### Task 4: Entrenamiento Adam + L-BFGS

**Files:**
- Modify: `src/pinn.py` (añadir `TrainResult` y `train()`)

- [ ] **Step 1: Crear `_test_train.py`**

```python
# _test_train.py
import sys; sys.path.insert(0, ".")
import torch
from src.pinn import SVPINN, train

t_data  = torch.linspace(0.0, 3600.0, 20)
q_up    = torch.full((20,), 50.0) + torch.randn(20) * 1.0
q_dn    = torch.full((20,), 48.0) + torch.randn(20) * 1.0

model = SVPINN(hidden_size=16, n_layers=2, S0=0.001, beta=1.0,
               n_init=0.030, Bw_init=45.0)

result = train(
    model,
    x0_data=q_up,
    xL_data=q_dn,
    t_data=t_data,
    L=5000.0,
    T=3600.0,
    lambda_data=1.0,
    lambda_pde=0.05,
    n_epochs_adam=20,
    n_iter_lbfgs=5,
    n_colloc=30,
    resample_every=10,
    t_warmup=0.0,
    verbose_every=10,
)

assert hasattr(result, "n_estimate"),    "falta n_estimate"
assert hasattr(result, "Bw_estimate"),   "falta Bw_estimate"
assert hasattr(result, "loss_history"),  "falta loss_history"
assert result.n_estimate  > 0, "n debe ser positivo"
assert result.Bw_estimate > 0, "Bw debe ser positivo"
assert len(result.loss_history) > 0, "loss_history vacío"

print(f"TRAIN TEST PASSED — n={result.n_estimate:.4f}  Bw={result.Bw_estimate:.2f}")
```

- [ ] **Step 2: Ejecutar — debe fallar con AttributeError**

```bash
python _test_train.py
```

Output esperado: `AttributeError: module 'src.pinn' has no attribute 'train'`

- [ ] **Step 3: Añadir `TrainResult` y `train()` a `src/pinn.py`**

Añadir después de `pde_residuals()`:

```python
# ── Entrenamiento ─────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model: SVPINN
    loss_history: list[dict[str, Any]]
    n_estimate: float
    Bw_estimate: float


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
    """
    x0_data = x0_data.float()
    xL_data = xL_data.float()
    t_data  = t_data.float()

    post_warm  = t_data >= t_warmup
    t_post     = t_data[post_warm]
    T_total    = float(t_data[-1]) if T <= 0 else float(T)

    loss_history: list[dict[str, Any]] = []

    def _sample_colloc(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        x_c = torch.rand(n_colloc) * L
        t_c = torch.rand(n_colloc) * (T_total - max(t_warmup, 0.0)) + max(t_warmup, 0.0)
        return x_c, t_c

    def _compute_loss(
        x_c: torch.Tensor, t_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pérdida de datos en los dos bordes (x=0 y x=L)
        tn = t_post / T_total
        _, Q0_pred = model(torch.zeros_like(tn), tn)
        _, QL_pred = model(torch.ones_like(tn),  tn)
        l_data = (
            torch.mean((Q0_pred - x0_data[post_warm]) ** 2)
            + torch.mean((QL_pred - xL_data[post_warm]) ** 2)
        )

        # Pérdida de física en puntos de colocación interiores
        R_mass, R_mom = pde_residuals(model, x_c, t_c, L=L, T=T_total)
        l_pde = torch.mean(R_mass ** 2) + torch.mean(R_mom ** 2)

        l_total = lambda_data * l_data + lambda_pde * l_pde
        return l_total, l_data, l_pde

    # ── Fase 1: Adam ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_c, t_c = _sample_colloc(seed=0)

    for epoch in range(n_epochs_adam):
        if epoch > 0 and epoch % resample_every == 0:
            x_c, t_c = _sample_colloc(seed=epoch)
        optimizer.zero_grad()
        l_total, l_data, l_pde = _compute_loss(x_c, t_c)
        l_total.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == n_epochs_adam - 1:
            entry: dict[str, Any] = {
                "epoch": epoch,
                "total": float(l_total),
                "data":  float(l_data),
                "pde":   float(l_pde),
                "n":     float(model.n),
                "Bw":    float(model.Bw),
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
        l_total, _, _ = _compute_loss(x_c, t_c)
        l_total.backward()
        return l_total

    l_final = lbfgs.step(_closure)
    loss_history.append({
        "epoch": n_epochs_adam,
        "total": float(l_final) if l_final is not None else float("nan"),
        "n":  float(model.n),
        "Bw": float(model.Bw),
        "phase": "lbfgs",
    })
    print(f"[L-BFGS] n={model.n:.5f}  Bw={model.Bw:.3f}")

    return TrainResult(
        model=model,
        loss_history=loss_history,
        n_estimate=float(model.n),
        Bw_estimate=float(model.Bw),
    )
```

- [ ] **Step 4: Ejecutar prueba**

```bash
python _test_train.py
```

Output esperado (ejemplo):
```
[Adam     0] total=5.2e+03  data=5.1e+03  pde=8.4e+00  n=0.03000  Bw=45.000
[Adam    10] ...
[L-BFGS] n=0.03...  Bw=4...
TRAIN TEST PASSED — n=0.030x  Bw=4x.xx
```

- [ ] **Step 5: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): train() con Adam y L-BFGS, TrainResult dataclass"
```

---

### Task 5: Tabla de comparación

**Files:**
- Modify: `src/pinn.py` (añadir `build_comparison_table()`)

- [ ] **Step 1: Crear `_test_compare.py`**

```python
# _test_compare.py
import sys; sys.path.insert(0, ".")
import pandas as pd
from src.pinn import build_comparison_table

# Sin archivo GLUE (opcional)
table = build_comparison_table(
    n_pinn=0.032, Bw_pinn=48.5,
    n_true=0.035, Bw_true=50.0,
    glue_csv_path=None,
)
assert isinstance(table, pd.DataFrame),   "debe devolver DataFrame"
assert "PINN"      in table.columns,      "falta columna PINN"
assert "verdadero" in table.columns,      "falta columna verdadero"
assert len(table) == 2,                   f"debe tener 2 filas, tiene {len(table)}"
assert "calibracion_GLUE" not in table.columns, "no debe haber GLUE sin CSV"

# Con ruta inexistente — devuelve NaN en columna GLUE
table2 = build_comparison_table(
    n_pinn=0.032, Bw_pinn=48.5,
    n_true=0.035, Bw_true=50.0,
    glue_csv_path="no_existe.csv",
)
assert "calibracion_GLUE" in table2.columns, "debe añadir columna aunque no exista el archivo"
import math
assert math.isnan(table2["calibracion_GLUE"].iloc[0]), "debe ser NaN si el archivo no existe"

print("COMPARE TABLE TEST PASSED")
print(table.to_string(index=False))
```

- [ ] **Step 2: Ejecutar — debe fallar con AttributeError**

```bash
python _test_compare.py
```

Output esperado: `AttributeError: module 'src.pinn' has no attribute 'build_comparison_table'`

- [ ] **Step 3: Añadir `build_comparison_table()` a `src/pinn.py`**

Añadir después de `train()`:

```python
# ── Comparación de estimaciones ───────────────────────────────────────────────

def build_comparison_table(
    *,
    n_pinn: float,
    Bw_pinn: float,
    n_true: float,
    Bw_true: float,
    glue_csv_path: str | None = None,
) -> pd.DataFrame:
    """Tabla parámetro | verdadero | PINN | calibración GLUE (si existe).

    glue_csv_path: ruta a glue_parametros_aceptados.csv de notebook 03.
                   Si es None, la columna GLUE no se incluye.
                   Si el archivo no existe, la columna GLUE contiene NaN.
    """
    row_n  = {"parametro": "n (Manning)", "verdadero": n_true,  "PINN": n_pinn}
    row_bw = {"parametro": "B_w [m]",    "verdadero": Bw_true, "PINN": Bw_pinn}

    if glue_csv_path is not None:
        glue_path = Path(glue_csv_path)
        if glue_path.is_file():
            glue_df = pd.read_csv(glue_path)
            row_n["calibracion_GLUE"]  = float(glue_df["n"].median())
            row_bw["calibracion_GLUE"] = float(glue_df["B_W"].median())
        else:
            row_n["calibracion_GLUE"]  = float("nan")
            row_bw["calibracion_GLUE"] = float("nan")

    return pd.DataFrame([row_n, row_bw])
```

- [ ] **Step 4: Ejecutar prueba**

```bash
python _test_compare.py
```

Output esperado:
```
COMPARE TABLE TEST PASSED
   parametro  verdadero   PINN
 n (Manning)      0.035  0.032
     B_w [m]     50.000  48.500
```

- [ ] **Step 5: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): build_comparison_table() PINN vs GLUE vs verdadero"
```

---

### Task 6: Notebook 04_pinn.ipynb

**Files:**
- Modify: `notebooks/04_pinn.ipynb`

El notebook actualmente tiene una sola celda vacía. Reemplazar su contenido con las 7 celdas del workflow completo (1 markdown + 6 code).

- [ ] **Step 1: Verificar estado actual**

```bash
python -c "import json; nb=json.load(open('notebooks/04_pinn.ipynb')); print('celdas actuales:', len(nb['cells']))"
```

Output esperado: `celdas actuales: 1`

- [ ] **Step 2: Crear script que construye el notebook**

Crear `_build_nb04.py` en la raíz del worktree:

```python
# _build_nb04.py — genera notebooks/04_pinn.ipynb y se borra solo
import json, textwrap
from pathlib import Path

def cell_md(src): return {"cell_type":"markdown","id":f"md-{hash(src)&0xffff:04x}","metadata":{},"source":src.splitlines(keepends=True)}
def cell_code(src): return {"cell_type":"code","id":f"cd-{hash(src)&0xffff:04x}","metadata":{},"source":src.splitlines(keepends=True),"outputs":[],"execution_count":None}

cells = []

# ── Celda 1: Título (markdown) ─────────────────────────────────────────────
cells.append(cell_md(textwrap.dedent("""\
# Parte 4 — PINN: Estimación de parámetros Saint-Venant 1D

**Physics-Informed Neural Network** que aprende A(x,t) y Q(x,t) como funciones
continuas y estima los parámetros físicos n y B_w minimizando:

- **L_data** = MSE entre Q predicho en x=0 y x=L vs observaciones del CSV  
  *(ancla la solución a los datos reales en los bordes)*
- **L_pde** = residuos de las ecuaciones de Saint-Venant calculados por `torch.autograd`
  en 2 000 puntos de colocación interiores  
  *(hace que la red respete la física en todo el dominio, no solo en los bordes)*

**Por qué se necesitan los dos términos:** con solo L_data, el problema es no
identificable — infinitas combinaciones de (n, B_w, A, Q) ajustan los bordes.
L_pde resuelve la identifiabilidad porque n aparece en Sf y B_w en la geometría
hidráulica: ambos deben hacer que los residuos SV sean ≈ 0 en el interior.

**Entrenamiento:** Fase 1 Adam (exploración amplia) → Fase 2 L-BFGS (convergencia fina).
""")))

# ── Celda 2: Configuración ────────────────────────────────────────────────────
cells.append(cell_code(textwrap.dedent("""\
# ── Configuración ─────────────────────────────────────────────────────────────
# Cambiar True/False para estimar otros parámetros
ESTIMATE_PARAMS = {"n": True, "Bw": True}

# Pesos de la función de pérdida
LAMBDA_DATA = 1.0    # peso de L_data (ajuste a observaciones)
LAMBDA_PDE  = 0.05   # peso de L_pde  (cumplimiento de la física)

# Hiperparámetros de entrenamiento
N_EPOCHS_ADAM  = 5_000   # épocas Fase 1 — Adam
N_ITER_LBFGS   = 500     # iteraciones Fase 2 — L-BFGS
N_COLLOC       = 2_000   # puntos de colocación interiores
RESAMPLE_EVERY = 1_000   # re-muestrear colloc cada N épocas Adam

# Conjeturas iniciales (lejos del valor verdadero para mostrar convergencia)
N_INIT  = 0.030   # verdadero: 0.035
BW_INIT = 45.0    # verdadero: 50.0 m

# Archivo de observaciones (mismo que en calibración, notebook 03)
CSV_NAME    = "series_corta_shift.csv"
WARMUP_SEC  = 3600.0
""")))

# ── Celda 3: Carga de datos ───────────────────────────────────────────────────
cells.append(cell_code(textwrap.dedent("""\
# ── Carga de datos ─────────────────────────────────────────────────────────────
import importlib, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# Parámetros verdaderos desde sinteticos.py
_spec = importlib.util.spec_from_file_location("sinteticos", ROOT / "data" / "sinteticos.py")
sinteticos = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sinteticos)
TRUE_N, TRUE_BW = sinteticos.N_MANN, sinteticos.B_W

csv_path = ROOT / "data" / "synthetic" / CSV_NAME
df   = pd.read_csv(csv_path, parse_dates=["datetime"])
t_sec = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
q_up  = df["Q_upstream_m3s"].to_numpy(dtype=float)
q_dn  = df["Q_downstream_m3s"].to_numpy(dtype=float)

t_tensor  = torch.tensor(t_sec, dtype=torch.float32)
qu_tensor = torch.tensor(q_up,  dtype=torch.float32)
qd_tensor = torch.tensor(q_dn,  dtype=torch.float32)
T_total   = float(t_sec[-1])

print(f"CSV: {CSV_NAME}  |  pasos: {len(df)}  |  T={T_total/3600:.1f} h")
print(f"Valores verdaderos: n={TRUE_N}  B_w={TRUE_BW} m")
""")))

# ── Celda 4: Construcción y entrenamiento ─────────────────────────────────────
cells.append(cell_code(textwrap.dedent("""\
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

# Fase 1 — Adam: barrido amplio del espacio de parámetros
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
    n_iter_lbfgs=N_ITER_LBFGS,
    n_colloc=N_COLLOC,
    resample_every=RESAMPLE_EVERY,
    t_warmup=WARMUP_SEC,
    verbose_every=500,
)
print(f"\\nEstimado PINN: n={result.n_estimate:.5f}  Bw={result.Bw_estimate:.3f} m")
""")))

# ── Celda 5: Curvas de pérdida + tabla de parámetros ─────────────────────────
cells.append(cell_code(textwrap.dedent("""\
# ── Curvas de pérdida + estimación de parámetros ──────────────────────────────
import matplotlib.pyplot as plt
from IPython.display import display

history = result.loss_history
adam_entries = [h for h in history if "data" in h]
epochs_a = [h["epoch"] for h in adam_entries]
totals_a = [h["total"] for h in adam_entries]
datas_a  = [h["data"]  for h in adam_entries]
pdes_a   = [h["pde"]   for h in adam_entries]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Curvas de pérdida (fase Adam, eje log)
ax = axes[0]
ax.semilogy(epochs_a, totals_a, label="L_total", color="k")
ax.semilogy(epochs_a, datas_a,  label="L_data",  color="steelblue")
ax.semilogy(epochs_a, pdes_a,   label=f"L_pde (×{LAMBDA_PDE})", color="coral")
ax.set_xlabel("Época (Adam)")
ax.set_ylabel("Pérdida (escala log)")
ax.set_title("Curvas de pérdida — fase Adam")
ax.legend(); ax.grid(True, alpha=0.3)

# Tabla y gráfico de parámetros
glue_path = ROOT / "data" / "synthetic" / "glue_parametros_aceptados.csv"
table = pinn_mod.build_comparison_table(
    n_pinn=result.n_estimate, Bw_pinn=result.Bw_estimate,
    n_true=TRUE_N, Bw_true=TRUE_BW,
    glue_csv_path=str(glue_path),
)
display(table.style.format({
    "verdadero": "{:.4f}", "PINN": "{:.4f}",
    **({} if "calibracion_GLUE" not in table.columns else {"calibracion_GLUE": "{:.4f}"}),
}).set_caption("Estimaciones de parámetros"))

params     = ["n (Manning)", "B_w [m]"]
verdaderos = [TRUE_N, TRUE_BW]
pinns      = [result.n_estimate, result.Bw_estimate]
use_glue   = "calibracion_GLUE" in table.columns
calibs     = list(table["calibracion_GLUE"]) if use_glue else [np.nan, np.nan]

ax2 = axes[1]
x = np.arange(len(params)); w = 0.25
ax2.bar(x - w, verdaderos, w, label="Verdadero",         color="0.70")
ax2.bar(x,     pinns,      w, label="PINN",               color="steelblue")
if use_glue and not any(np.isnan(calibs)):
    ax2.bar(x + w, calibs, w, label="Calibración (GLUE)", color="coral")
ax2.set_xticks(x); ax2.set_xticklabels(params)
ax2.set_title("Parámetros estimados")
ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "pinn_parametros.png", dpi=150)
plt.show()
""")))

# ── Celda 6: Hidrograma de salida ─────────────────────────────────────────────
cells.append(cell_code(textwrap.dedent("""\
# ── Hidrograma: PINN vs observado en x=L ─────────────────────────────────────
model.eval()
with torch.no_grad():
    t_norm_all = t_tensor / T_total
    _, Q_pinn  = model(torch.ones_like(t_norm_all), t_norm_all)
Q_pinn_np = Q_pinn.numpy()

t_h       = t_sec / 3600.0
mask_post = t_sec >= WARMUP_SEC

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_h[mask_post], q_dn[mask_post],    "r.",        ms=3,  alpha=0.6, label="Q_obs  (x=L)")
ax.plot(t_h[mask_post], Q_pinn_np[mask_post], "steelblue", lw=1.5, label="Q_PINN (x=L)")
ax.axvspan(0, WARMUP_SEC / 3600, color="gray", alpha=0.12, label="Warm-up")
ax.set_xlabel("Tiempo (h)")
ax.set_ylabel("Q (m³/s)")
ax.set_title(
    f"Hidrograma salida — PINN vs observado\\n"
    f"n={result.n_estimate:.4f} (verd. {TRUE_N})  "
    f"Bw={result.Bw_estimate:.2f} m (verd. {TRUE_BW} m)"
)
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "pinn_hidrograma.png", dpi=150)
plt.show()
""")))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path("notebooks/04_pinn.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook escrito: {out}  ({len(cells)} celdas)")
```

- [ ] **Step 3: Ejecutar script para generar el notebook**

```bash
python _build_nb04.py
```

Output esperado: `Notebook escrito: notebooks/04_pinn.ipynb  (6 celdas)`

- [ ] **Step 4: Verificar JSON válido**

```bash
python -c "import json; nb=json.load(open('notebooks/04_pinn.ipynb')); print('celdas:', len(nb['cells'])); print('tipos:', [c['cell_type'] for c in nb['cells']])"
```

Output esperado:
```
celdas: 6
tipos: ['markdown', 'code', 'code', 'code', 'code', 'code']
```

- [ ] **Step 5: Borrar script auxiliar**

```bash
del _build_nb04.py _test_svpinn.py _test_pde.py _test_train.py _test_compare.py 2>nul
```

(En Linux/Mac: `rm -f _build_nb04.py _test_svpinn.py _test_pde.py _test_train.py _test_compare.py`)

- [ ] **Step 6: Verificación de integración final**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.pinn import SVPINN, pde_residuals, train, build_comparison_table, TrainResult
import torch
m = SVPINN(hidden_size=8, n_layers=2, S0=0.001, beta=1.0, n_init=0.03, Bw_init=45.0)
A, Q = m(torch.tensor([0.0, 0.5, 1.0]), torch.tensor([0.0, 0.5, 1.0]))
assert (A > 0).all() and (Q > 0).all()
import json; nb = json.load(open('notebooks/04_pinn.ipynb'))
assert len(nb['cells']) == 6
print('INTEGRATION OK')
"
```

Output esperado: `INTEGRATION OK`

- [ ] **Step 7: Commit final**

```bash
git add src/pinn.py notebooks/04_pinn.ipynb
git commit -m "feat(pinn): notebook 04 completo con 6 celdas PINN workflow"
```

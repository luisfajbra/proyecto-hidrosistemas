# PINN — h-network + L_h + L_steady Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mejorar la identificabilidad de n y B_w en la PINN mediante tres cambios coordinados: (1) red que predice h en lugar de A con A = B_w·h estricto, (2) pérdida de profundidad L_h usando h_outlet_m del CSV, y (3) residuos PDE en t=0 como restricción de flujo uniforme inicial.

**Architecture:** La red neuronal SVPINN pasa de predecir (A, Q) a predecir (h, Q). La identidad geométrica A = B_w·h se impone en `pde_residuals()`, acoplando B_w a toda la geometría hidráulica. `train()` recibe hL_data como nueva observación y calcula dos nuevas pérdidas: L_h (MSE de profundidad en x=L) y L_steady (residuos Saint-Venant en t=0, activos desde época 0).

**Tech Stack:** Python 3, PyTorch, dataclasses; archivos `src/config.py`, `src/pinn.py`, `notebooks/04_pinn.ipynb`, `README.md`.

> **Working directory para todos los comandos:** `c:\Users\Luis\Desktop\proyecto-sv-mh`

---

## File Map

| Acción | Archivo | Qué cambia |
|---|---|---|
| Modify | `src/config.py` | Añadir `lambda_h`, `lambda_steady`, `n_colloc_steady` a `PinnDefaults` |
| Modify | `src/pinn.py` | `MIN_AREA→MIN_VAL`; `forward()` devuelve `(h,Q)`; `pde_residuals()` usa `A=Bw*h`; nueva `_sample_steady_colloc()`; `train()` con nuevos args y pérdidas |
| Modify | `notebooks/04_pinn.ipynb` | Celda config, carga de h, llamada a `train()` |
| Modify | `README.md` | Sección tercera ronda PINN |

---

## Task 1: Nuevos defaults en config.py

**Files:**
- Modify: `src/config.py` líneas 62–78 (clase `PinnDefaults`)

- [ ] **Step 1: Añadir tres campos al final de PinnDefaults**

Abrir `src/config.py`. Localizar la clase `PinnDefaults`. Añadir los tres campos nuevos **después de** `gradient_clip_max_norm`:

```python
@dataclass(frozen=True)
class PinnDefaults:
    """Architecture and training defaults used by the PINN."""

    hidden_size: int = 64
    n_layers: int = 4
    beta: float = 1.0
    n_init: float = 0.030
    bw_init: float = 45.0
    lambda_data: float = 1.0
    lambda_pde: float = 0.05
    n_epochs_adam: int = 6000
    n_epochs_warmup: int = 2000
    n_epochs_ramp: int = 1000
    n_iter_lbfgs: int = 500
    n_colloc: int = 2000
    resample_every: int = 1000
    gradient_clip_max_norm: float = 1.0
    # Nuevos: términos de pérdida adicionales
    lambda_h: float        = 1.0   # peso L_h (profundidad observada en x=L)
    lambda_steady: float   = 0.1   # peso L_steady (residuos PDE en t=0)
    n_colloc_steady: int   = 200   # puntos de colocación en t=0
```

- [ ] **Step 2: Verificar que el módulo importa correctamente**

```bash
python -c "from src.config import DEFAULT_PINN; print(DEFAULT_PINN.lambda_h, DEFAULT_PINN.lambda_steady, DEFAULT_PINN.n_colloc_steady)"
```

Salida esperada: `1.0 0.1 200`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat(config): añadir lambda_h, lambda_steady, n_colloc_steady a PinnDefaults"
```

---

## Task 2: Renombrar MIN_AREA → MIN_VAL y reestructurar forward()

**Files:**
- Modify: `src/pinn.py` línea 23 y método `forward()` (líneas 85–93)

El cambio en `forward()` es semántico: la primera salida pasa de ser A (área) a h (profundidad). El código de la capa de salida es idéntico — solo cambia el nombre de la variable y el comentario.

- [ ] **Step 1: Renombrar MIN_AREA a MIN_VAL**

Localizar la línea 23 de `src/pinn.py`:
```python
MIN_AREA = 1e-4
```
Reemplazar por:
```python
MIN_VAL = 1e-4
```

- [ ] **Step 2: Reestructurar forward()**

Localizar el método `forward()` (líneas 85–93 aproximadamente). Reemplazar el cuerpo completo:

```python
def forward(
    self, x_norm: torch.Tensor, t_norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = torch.stack([x_norm, t_norm], dim=-1)
    out = self.net(inp)
    # softplus garantiza h > MIN_VAL y Q > MIN_VAL en todo momento
    h = nn.functional.softplus(out[..., 0]) + MIN_VAL  # profundidad [m]
    Q = nn.functional.softplus(out[..., 1]) + MIN_VAL  # caudal [m³/s]
    return h, Q
```

- [ ] **Step 3: Verificar que importa y la forma de salida es correcta**

```bash
python -c "
import torch
from src.pinn import SVPINN
m = SVPINN(S0=0.001, beta=1.0, n_init=0.030, Bw_init=45.0)
h, Q = m(torch.rand(5), torch.rand(5))
print('h shape:', h.shape, '  Q shape:', Q.shape)
print('h > 0:', (h > 0).all().item(), '  Q > 0:', (Q > 0).all().item())
"
```

Salida esperada:
```
h shape: torch.Size([5])   Q shape: torch.Size([5])
h > 0: True   Q > 0: True
```

- [ ] **Step 4: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): forward() devuelve (h,Q) en lugar de (A,Q); MIN_AREA->MIN_VAL"
```

---

## Task 3: Actualizar pde_residuals() — A = Bw·h

**Files:**
- Modify: `src/pinn.py` — función `pde_residuals()` (líneas 98–146 aproximadamente)

El cambio central: `A` deja de ser salida libre de la red y pasa a ser `Bw * h`. Esto acopla B_w a la continuidad, al radio hidráulico y a la fricción de Manning.

- [ ] **Step 1: Reemplazar el cuerpo completo de pde_residuals()**

Localizar `def pde_residuals(` en `src/pinn.py`. Reemplazar la función entera:

```python
def pde_residuals(
    model: SVPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    L: float,
    T: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuos de Saint-Venant en puntos de colocación (x_col [m], t_col [s]).

    La red predice (h, Q). A se calcula como A = Bw·h (identidad geométrica
    para canal rectangular), acoplando Bw a continuidad, radio hidráulico y Sf.
    """
    x_norm = (x_col / L).clone().detach().requires_grad_(True)
    t_norm = (t_col / T).clone().detach().requires_grad_(True)

    h, Q = model(x_norm, t_norm)

    Bw   = model.Bw
    n    = model.n
    S0   = model.S0
    beta = model.beta

    # Geometría hidráulica rectangular: A = Bw·h (identidad estricta)
    A     = Bw * h
    h_c   = 0.5 * h
    per   = Bw + 2.0 * h
    R_hyd = A / per                           # R = Bw·h / (Bw + 2h)

    # Flujo de momentum: F_Q = β Q²/A + g h_c A
    F_Q = beta * Q**2 / (A + MIN_VAL) + G * h_c * A

    # Pendiente de fricción de Manning: Sf = n² Q|Q| / (A² R^(4/3))
    Sf = n**2 * Q * torch.abs(Q) / ((A + MIN_VAL)**2 * R_hyd**(4.0 / 3.0))

    # Derivadas por autograd
    ones    = torch.ones_like(A)
    dA_dtn  = torch.autograd.grad(A,   t_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dxn  = torch.autograd.grad(Q,   x_norm, grad_outputs=ones, create_graph=True)[0]
    dFQ_dxn = torch.autograd.grad(F_Q, x_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dtn  = torch.autograd.grad(Q,   t_norm, grad_outputs=ones, create_graph=True)[0]

    # Residuos de continuidad y momentum
    R_mass = dA_dtn / T + dQ_dxn / L
    R_mom  = dQ_dtn / T + dFQ_dxn / L - G * A * (S0 - Sf)

    return R_mass, R_mom
```

- [ ] **Step 2: Smoke test de pde_residuals()**

```bash
python -c "
import torch
from src.pinn import SVPINN, pde_residuals
m = SVPINN(S0=0.001, beta=1.0, n_init=0.030, Bw_init=45.0)
x = torch.rand(10) * 5000
t = torch.rand(10) * 7200
R_mass, R_mom = pde_residuals(m, x, t, L=5000, T=7200)
print('R_mass shape:', R_mass.shape)
print('R_mom shape:', R_mom.shape)
print('finite:', R_mass.isfinite().all().item(), R_mom.isfinite().all().item())
"
```

Salida esperada:
```
R_mass shape: torch.Size([10])
R_mom shape: torch.Size([10])
finite: True True
```

- [ ] **Step 3: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): pde_residuals() usa A=Bw*h — Bw acoplado a geometria hidraulica"
```

---

## Task 4: Añadir _sample_steady_colloc()

**Files:**
- Modify: `src/pinn.py` — insertar función nueva antes de `def train(`

- [ ] **Step 1: Insertar función _sample_steady_colloc justo antes de train()**

Localizar la línea `def train(` en `src/pinn.py`. Insertar la función nueva inmediatamente antes:

```python
def _sample_steady_colloc(
    n: int, L: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Puntos aleatorios en x a t=0 para la condición inicial de flujo uniforme.

    Estos puntos se generan una sola vez antes del bucle de entrenamiento.
    Los residuos PDE en t=0 deben ser cero si la red y los parámetros son
    consistentes con el flujo uniforme estacionario inicial.
    """
    x_s = torch.rand(n) * L
    t_s = torch.zeros(n)
    return x_s, t_s
```

- [ ] **Step 2: Verificar que la función es importable y devuelve shapes correctas**

```bash
python -c "
from src.pinn import _sample_steady_colloc
x_s, t_s = _sample_steady_colloc(200, 5000.0)
print('x_s shape:', x_s.shape, '  t_s shape:', t_s.shape)
print('t_s all zero:', (t_s == 0).all().item())
print('x_s in [0, 5000]:', (x_s >= 0).all().item() and (x_s <= 5000).all().item())
"
```

Salida esperada:
```
x_s shape: torch.Size([200])   t_s shape: torch.Size([200])
t_s all zero: True
x_s in [0, 5000]: True
```

- [ ] **Step 3: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): añadir _sample_steady_colloc() para puntos de coloc en t=0"
```

---

## Task 5: Actualizar train() — nuevos args, L_h, L_steady, loss_history

**Files:**
- Modify: `src/pinn.py` — función `train()` completa (líneas 159–301 aproximadamente)

Este es el cambio más extenso. Se modifica la firma, la inicialización, `_compute_loss()`, el bucle Adam y el closure de L-BFGS.

- [ ] **Step 1: Actualizar la firma de train()**

Reemplazar las líneas de la firma de `train()`. La nueva firma añade `hL_data`, `lambda_h`, `lambda_steady` y `n_colloc_steady`:

```python
def train(
    model: SVPINN,
    *,
    x0_data: torch.Tensor,
    xL_data: torch.Tensor,
    hL_data: torch.Tensor,
    t_data: torch.Tensor,
    L: float,
    T: float,
    lambda_data: float   = DEFAULT_PINN.lambda_data,
    lambda_h: float      = DEFAULT_PINN.lambda_h,
    lambda_pde: float    = DEFAULT_PINN.lambda_pde,
    lambda_steady: float = DEFAULT_PINN.lambda_steady,
    n_colloc_steady: int = DEFAULT_PINN.n_colloc_steady,
    n_epochs_adam: int   = DEFAULT_PINN.n_epochs_adam,
    n_epochs_warmup: int = DEFAULT_PINN.n_epochs_warmup,
    n_epochs_ramp: int   = DEFAULT_PINN.n_epochs_ramp,
    n_iter_lbfgs: int    = DEFAULT_PINN.n_iter_lbfgs,
    n_colloc: int        = DEFAULT_PINN.n_colloc,
    resample_every: int  = DEFAULT_PINN.resample_every,
    t_warmup: float      = DEFAULT_NUMERICAL.warmup_seconds,
    verbose_every: int   = 500,
) -> TrainResult:
    """Entrena la PINN con Adam y L-BFGS.

    x0_data: Q observado en x=0 [m³/s], shape (nt,)
    xL_data: Q observado en x=L [m³/s], shape (nt,)
    hL_data: h observado en x=L [m],    shape (nt,)
    t_data:  tiempos [s],               shape (nt,)
    L, T:    longitud del canal [m] y duración [s]
    lambda_h:      peso de L_h (profundidad en x=L); activo desde época 0.
    lambda_steady: peso de L_steady (residuos PDE en t=0); activo desde época 0.
    n_colloc_steady: número de puntos de colocación en t=0.
    """
```

- [ ] **Step 2: Añadir conversión de hL_data y escalas después de las conversiones existentes**

Localizar el bloque donde se convierten x0_data, xL_data, t_data a float. Añadir `hL_data` y `h_scale` justo después:

```python
    x0_data = x0_data.float()
    xL_data = xL_data.float()
    hL_data = hL_data.float()      # NUEVO
    t_data  = t_data.float()

    post_warm = t_data >= t_warmup
    t_post    = t_data[post_warm]
    T_total   = float(t_data[-1]) if T <= 0 else float(T)

    q_scale = max(x0_data.max().item(), xL_data.max().item(), 1.0)
    h_scale = max(hL_data.max().item(), 1e-2)              # NUEVO
```

- [ ] **Step 3: Añadir muestreo de puntos estacionarios antes del bucle Adam**

Localizar la línea `optimizer = torch.optim.Adam(...)`. Insertar antes:

```python
    # Puntos de colocación en t=0 (fijos durante todo el entrenamiento)
    x_steady, t_steady = _sample_steady_colloc(n_colloc_steady, L)
```

- [ ] **Step 4: Reemplazar _compute_loss() con la nueva versión**

Localizar y reemplazar `def _compute_loss(` completa dentro de `train()`:

```python
    def _compute_loss(
        x_c: torch.Tensor, t_c: torch.Tensor, lp: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── L_data: caudales en x=0 y x=L ───────────────────────────────────
        tn = t_post / T_total
        _, Q0_pred = model(torch.zeros_like(tn), tn)
        _, QL_pred = model(torch.ones_like(tn),  tn)
        Q0_obs = x0_data[post_warm]
        QL_obs = xL_data[post_warm]
        l_data = (
            torch.mean(((Q0_pred - Q0_obs) / q_scale) ** 2)
            + torch.mean(((QL_pred - QL_obs) / q_scale) ** 2)
        )

        # ── L_h: profundidad observada en x=L ────────────────────────────────
        hL_pred, _ = model(torch.ones_like(tn), tn)
        hL_obs = hL_data[post_warm]
        l_h = torch.mean(((hL_pred - hL_obs) / h_scale) ** 2)

        # ── L_pde: residuos Saint-Venant en puntos interiores ────────────────
        R_mass, R_mom = pde_residuals(model, x_c, t_c, L=L, T=T_total)
        l_pde = torch.mean(R_mass ** 2) + torch.mean(R_mom ** 2)

        # ── L_steady: residuos en t=0 (flujo uniforme estacionario inicial) ──
        R_mass_s, R_mom_s = pde_residuals(model, x_steady, t_steady, L=L, T=T_total)
        l_steady = torch.mean(R_mass_s ** 2) + torch.mean(R_mom_s ** 2)

        l_total = (
            lambda_data    * l_data
            + lambda_h     * l_h
            + lp           * l_pde
            + lambda_steady * l_steady
        )
        return l_total, l_data, l_h, l_pde, l_steady
```

- [ ] **Step 5: Actualizar el bucle Adam — desempaquetado y loss_history**

Localizar las líneas del bucle Adam que llaman `_compute_loss` y registran el log. Reemplazar:

```python
        optimizer.zero_grad()
        l_total, l_data, l_h, l_pde, l_steady = _compute_loss(x_c, t_c, lp=lp_current)
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=DEFAULT_PINN.gradient_clip_max_norm,
        )
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == n_epochs_adam - 1:
            entry: dict[str, Any] = {
                "epoch":  epoch,
                "total":  l_total.detach().item(),
                "data":   l_data.detach().item(),
                "h":      l_h.detach().item(),
                "pde":    l_pde.detach().item(),
                "steady": l_steady.detach().item(),
                "n":      model.n.detach().item(),
                "Bw":     model.Bw.detach().item(),
            }
            loss_history.append(entry)
            print(
                f"[Adam {epoch:5d}] "
                f"total={l_total:.4e}  data={l_data:.4e}  h={l_h:.4e}  "
                f"pde={l_pde:.4e}  steady={l_steady:.4e}  "
                f"n={model.n:.5f}  Bw={model.Bw:.3f}"
            )
```

- [ ] **Step 6: Actualizar el closure de L-BFGS**

Localizar `def _closure()` dentro de `train()`. Reemplazar:

```python
    def _closure() -> torch.Tensor:
        lbfgs.zero_grad()
        l_total, _, _, _, _ = _compute_loss(x_c, t_c, lp=lambda_pde)
        l_total.backward()
        return l_total
```

- [ ] **Step 7: Smoke test completo de train()**

```bash
python -c "
import torch, importlib
import src.pinn as pinn_mod
importlib.reload(pinn_mod)

nt = 50
t  = torch.linspace(0, 7200, nt)
Q0 = torch.full((nt,), 50.0)
QL = torch.full((nt,), 50.0)
hL = torch.full((nt,), 0.9)

model = pinn_mod.SVPINN(S0=0.001, beta=1.0, n_init=0.030, Bw_init=45.0)
result = pinn_mod.train(
    model, x0_data=Q0, xL_data=QL, hL_data=hL, t_data=t,
    L=5000, T=7200,
    n_epochs_adam=5, n_epochs_warmup=2, n_epochs_ramp=1, n_iter_lbfgs=2,
    verbose_every=1,
)
print('n_estimate:', result.n_estimate)
print('Bw_estimate:', result.Bw_estimate)
print('loss_history keys:', list(result.loss_history[0].keys()))
"
```

Salida esperada (valores numéricos variarán):
```
[Adam     0] total=...  data=...  h=...  pde=...  steady=...  n=...  Bw=...
...
n_estimate: <float>
Bw_estimate: <float>
loss_history keys: ['epoch', 'total', 'data', 'h', 'pde', 'steady', 'n', 'Bw']
```

- [ ] **Step 8: Commit**

```bash
git add src/pinn.py
git commit -m "feat(pinn): train() con L_h (profundidad) y L_steady (residuos PDE en t=0)"
```

---

## Task 6: Actualizar notebook 04_pinn.ipynb

**Files:**
- Modify: `notebooks/04_pinn.ipynb` — celdas de configuración, carga de datos y entrenamiento

> Los cambios se aplican con la herramienta NotebookEdit. Cada step identifica la celda por su contenido actual.

- [ ] **Step 1: Añadir constantes nuevas a la celda de configuración**

Localizar la celda que contiene `LAMBDA_DATA` y `LAMBDA_PDE`. Añadir al final de esa celda:

```python
# Nuevos términos de pérdida — tercera ronda PINN
LAMBDA_H        = DEFAULT_PINN.lambda_h        # 1.0  — peso L_h (profundidad)
LAMBDA_STEADY   = DEFAULT_PINN.lambda_steady    # 0.1  — peso L_steady (t=0)
N_COLLOC_STEADY = DEFAULT_PINN.n_colloc_steady  # 200  — puntos en t=0
```

- [ ] **Step 2: Añadir carga de h_outlet_m en la celda de carga de datos**

Localizar la celda que contiene `t_tensor = torch.tensor(t_sec, ...)`. Añadir justo después de `qd_tensor`:

```python
h_outlet = df["h_outlet_m"].to_numpy(dtype=float)
hL_tensor = torch.tensor(h_outlet, dtype=torch.float32)
print(f"h_outlet: min={h_outlet.min():.3f} m  max={h_outlet.max():.3f} m  media={h_outlet.mean():.3f} m")
```

- [ ] **Step 3: Actualizar la llamada a pinn_mod.train()**

Localizar la celda que contiene `result = pinn_mod.train(`. Añadir los tres argumentos nuevos:

```python
result = pinn_mod.train(
    model,
    x0_data=qu_tensor,
    xL_data=qd_tensor,
    hL_data=hL_tensor,           # NUEVO: profundidad observada en x=L
    t_data=t_tensor,
    L=sinteticos.L,
    T=T_total,
    lambda_data=LAMBDA_DATA,
    lambda_h=LAMBDA_H,           # NUEVO
    lambda_pde=LAMBDA_PDE,
    lambda_steady=LAMBDA_STEADY, # NUEVO
    n_colloc_steady=N_COLLOC_STEADY, # NUEVO
    n_epochs_adam=N_EPOCHS_ADAM,
    n_epochs_warmup=N_EPOCHS_WARMUP,
    n_epochs_ramp=N_EPOCHS_RAMP,
    n_iter_lbfgs=N_ITER_LBFGS,
    n_colloc=N_COLLOC,
    t_warmup=WARMUP_SEC,
    verbose_every=500,
)
```

- [ ] **Step 4: Actualizar la celda de curvas de pérdida para incluir l_h y l_steady**

Localizar la celda que contiene `adam_entries = [h for h in history if "data" in h]`. Añadir extracción de los nuevos campos:

```python
adam_entries = [h for h in history if "data" in h]
epochs_a  = [h["epoch"]  for h in adam_entries]
totals_a  = [h["total"]  for h in adam_entries]
datas_a   = [h["data"]   for h in adam_entries]
hs_a      = [h["h"]      for h in adam_entries]   # NUEVO
pdes_a    = [h["pde"]    for h in adam_entries]
steadys_a = [h["steady"] for h in adam_entries]   # NUEVO
```

Y en el gráfico, añadir las dos trazas nuevas al subplot de pérdidas:

```python
ax.semilogy(epochs_a, hs_a,      label="L_h",              color="seagreen")
ax.semilogy(epochs_a, steadys_a, label="L_steady",          color="mediumpurple")
```

- [ ] **Step 5: Commit del notebook**

```bash
git add notebooks/04_pinn.ipynb
git commit -m "feat(nb04): añadir hL_data, LAMBDA_H, LAMBDA_STEADY — tercera ronda PINN"
```

---

## Task 7: Documentar tercera ronda PINN en README

**Files:**
- Modify: `README.md` — añadir nueva sección después de "### Fixes segunda ronda"

- [ ] **Step 1: Añadir sección de tercera ronda al README**

Localizar la sección `### Fixes segunda ronda` en `README.md`. Añadir inmediatamente después del bloque de la tabla de fixes y antes de `## Instalación`:

```markdown
### Tercera implementación — identificabilidad (h-network + L_h + L_steady)

**Problema diagnosticado:** La PINN convergía a soluciones espurias con valores
incorrectos de n y B_w (ej. n → 0.012, B_w → 19.5 en lugar de 0.035 y 50 m).
Las pérdidas L_data y L_pde bajaban, pero los parámetros divergían de los valores
verdaderos.

**Causa raíz:** dos problemas acoplados:

1. **Compensación interna de la red.** La red predecía A(x,t) libremente. Al disminuir B_w,
   la red ajustaba A internamente para mantener h = A/B_w razonable, desacoplando B_w
   de la geometría hidráulica real. B_w solo aparecía en la pendiente de fricción Sf,
   con influencia débil sobre L_pde.

2. **Identificabilidad insuficiente.** Estimar n y B_w desde solo Q en x=0 y x=L es un
   problema sub-determinado: múltiples pares (n, B_w) producen hidrogramas similares en
   los bordes. La columna h_outlet_m del CSV no se estaba usando, perdiendo información
   que rompe esa degeneración.

**Fixes aplicados:**

| Fix | Cambio | Razón |
|-----|--------|-------|
| h-network | Red predice (h, Q); A = B_w·h estricto en pde_residuals() | B_w aparece en continuidad, radio hidráulico y Sf simultáneamente; la red no puede compensarlo internamente |
| L_h | MSE(h_pred en x=L − h_obs) usando h_outlet_m del CSV | La profundidad junto con el caudal proporciona la curva de aforo en x=L, que identifica n y B_w independientemente |
| L_steady | Residuos Saint-Venant en t=0 via autograd | A t=0 el canal está en flujo uniforme estacionario; los residuos → 0 implica Manning implícitamente; activo desde época 0, ancla los parámetros antes de que L_pde intervenga |

**Función de pérdida final:**

```
L_total = λ_data · L_data  +  λ_h · L_h  +  λ_pde(t) · L_pde  +  λ_steady · L_steady
```

Donde:
- **L_data** = MSE(Q_pred/Q_max − Q_obs/Q_max) en x=0 y x=L, post-warmup
- **L_h** = MSE(h_pred/h_max − h_obs/h_max) en x=L, post-warmup
- **L_pde** = residuos Saint-Venant en 2000 puntos interiores (con ramp 0→λ_pde)
- **L_steady** = residuos Saint-Venant en 200 puntos x aleatorios a t=0

**Configuración:**

```python
LAMBDA_DATA    = 1.0     # peso caudales observados
LAMBDA_H       = 1.0     # peso profundidades observadas
LAMBDA_PDE     = 0.05    # peso física interior (con warmup + ramp)
LAMBDA_STEADY  = 0.1     # peso condición inicial estacionaria
N_COLLOC_STEADY = 200    # puntos de colocación en t=0
```
```

- [ ] **Step 2: Commit del README**

```bash
git add README.md
git commit -m "docs: README tercera ronda PINN — h-network, L_h, L_steady"
```

---

## Self-Review

**Spec coverage:**
- ✅ h-network: `forward()` devuelve `(h,Q)` — Task 2
- ✅ `pde_residuals()` usa `A = Bw*h` — Task 3
- ✅ `MIN_AREA → MIN_VAL` — Task 2
- ✅ `_sample_steady_colloc()` — Task 4
- ✅ `train()` con `hL_data`, `lambda_h`, `lambda_steady`, `n_colloc_steady` — Task 5
- ✅ `l_h` y `l_steady` en `loss_history` — Task 5
- ✅ Loss total = `lambda_data*l_data + lambda_h*l_h + lp*l_pde + lambda_steady*l_steady` — Task 5
- ✅ Notebook: config, hL_tensor, train() call, curvas de pérdida — Task 6
- ✅ README documenta motivación + tres fixes + tabla — Task 7

**Type consistency:**
- `forward()` devuelve `(h, Q)` — Task 2 ✓
- `pde_residuals()` llama `h, Q = model(...)` — Task 3 ✓
- `_compute_loss()` llama `hL_pred, _ = model(...)` — Task 5 ✓
- `_compute_loss()` retorna 5-tupla; L-BFGS desempaqueta con `_, _, _, _, _` — Task 5 ✓
- `h_scale` definido antes de `_compute_loss()` — Task 5 ✓
- `x_steady`, `t_steady` definidos antes del bucle Adam — Task 5 ✓
- `DEFAULT_PINN.lambda_h`, `.lambda_steady`, `.n_colloc_steady` definidos en Task 1 y usados en Task 5 ✓

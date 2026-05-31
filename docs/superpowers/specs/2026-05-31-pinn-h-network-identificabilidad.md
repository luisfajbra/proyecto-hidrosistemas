# Spec: PINN — h-network + L_h + L_steady

**Fecha:** 2026-05-31
**Archivos afectados:** `src/pinn.py`, `src/config.py`, `notebooks/04_pinn.ipynb`

---

## Objetivo

Mejorar la identificabilidad de n y B_w en la PINN mediante tres cambios coordinados:

1. **h-network:** reestructurar la red para predecir `(h, Q)` en lugar de `(A, Q)`, con `A = B_w · h` como identidad geométrica estricta.
2. **L_h:** añadir pérdida de profundidad usando las observaciones `h_outlet_m` del CSV, que actualmente no se usan.
3. **L_steady:** añadir residuos Saint-Venant evaluados en `t=0` mediante autograd — restricción PINN-nativa que ancla n y B_w a la condición inicial de flujo uniforme.

---

## Motivación física

Estimar n y B_w desde solo Q en x=0 y x=L es sub-determinado: múltiples pares (n, B_w) producen hidrogramas similares en los bordes. Las tres mejoras atacan esto desde distintos ángulos:

- **h-network:** B_w deja de poder ser compensado internamente por la red. Con `A = B_w · h`, el parámetro aparece en la geometría hidráulica completa (área, perímetro, radio hidráulico, flujo de momentum) y no solo en la pendiente de fricción.
- **L_h:** la profundidad h en x=L junto con Q en x=L proporciona la curva de aforo (rating curve) en el punto de salida. h y Q responden de forma diferente a n y B_w, rompiendo la degeneración.
- **L_steady:** a t=0 el canal está en flujo uniforme estacionario. Los residuos Saint-Venant son cero solo si la red y los parámetros son consistentes con esa condición inicial. La ecuación de Manning emerge automáticamente del término de momentum sin escribir ninguna fórmula a mano.

---

## Cambio 1: h-network (`src/pinn.py` — clase `SVPINN`)

### `forward()`

El código de la capa de salida no cambia. Solo cambia la interpretación y el nombre de la variable:

```python
# ANTES
A = nn.functional.softplus(out[..., 0]) + MIN_AREA
Q = nn.functional.softplus(out[..., 1]) + MIN_AREA
return A, Q

# DESPUÉS
h = nn.functional.softplus(out[..., 0]) + MIN_AREA   # profundidad [m]
Q = nn.functional.softplus(out[..., 1]) + MIN_AREA   # caudal [m³/s]
return h, Q
```

`MIN_AREA = 1e-4` se renombra a `MIN_VAL = 1e-4` para reflejar que ahora es el piso numérico de h y Q, no de A.

### `pde_residuals()`

```python
# ANTES
A, Q = model(x_norm, t_norm)
h     = torch.clamp(A / Bw, min=1e-4)
per   = Bw + 2.0 * h
R_hyd = A / per

# DESPUÉS
h, Q = model(x_norm, t_norm)          # h predicha directamente
A     = Bw * h                         # A = B_w · h  ← identidad geométrica
per   = Bw + 2.0 * h
R_hyd = A / per                        # R = B_w·h / (B_w + 2h)
```

Impacto: Bw aparece en A (continuidad), en per (radio hidráulico), en Sf (fricción Manning) — tres veces en lugar de una. La red no puede mantener Bw incorrecto sin aumentar los residuos PDE.

---

## Cambio 2: L_h — pérdida de profundidad (`src/pinn.py` — `train()`)

`h_outlet_m` está disponible en todos los CSVs de series. Se pasa como argumento a `train()` junto con los datos de caudal existentes.

### Nueva signature de `train()`

```python
def train(
    model: SVPINN,
    *,
    x0_data: torch.Tensor,       # Q en x=0 [m³/s]
    xL_data: torch.Tensor,       # Q en x=L [m³/s]
    hL_data: torch.Tensor,       # h en x=L [m]  ← NUEVO
    t_data: torch.Tensor,
    L: float,
    T: float,
    lambda_data: float = DEFAULT_PINN.lambda_data,
    lambda_h: float    = DEFAULT_PINN.lambda_h,      # NUEVO
    lambda_pde: float  = DEFAULT_PINN.lambda_pde,
    lambda_steady: float = DEFAULT_PINN.lambda_steady, # NUEVO
    n_colloc_steady: int = DEFAULT_PINN.n_colloc_steady, # NUEVO
    ...
)
```

### Cálculo de L_h dentro de `_compute_loss()`

```python
h_scale = max(hL_data.max().item(), 1e-2)   # escala de profundidad

# Predicción de h en x=L, tiempos post-warmup
hL_pred, _ = model(torch.ones_like(tn), tn)
hL_obs     = hL_data[post_warm]
l_h = torch.mean(((hL_pred - hL_obs) / h_scale) ** 2)
```

### Loss total en `_compute_loss()`

```python
l_total = (lambda_data  * l_data
         + lambda_h     * l_h
         + lp           * l_pde
         + lambda_steady * l_steady)
```

**Schedule de L_h:** activo desde época 0, peso constante `lambda_h`. No hace ramp — la profundidad es una observación directa, no una restricción de física.

---

## Cambio 3: L_steady — residuos PDE en t=0 (`src/pinn.py`)

### Generación de puntos de colocación estacionarios

```python
def _sample_steady_colloc(n: int, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Puntos aleatorios en x a t=0 (condición inicial de flujo uniforme)."""
    x_s = torch.rand(n) * L
    t_s = torch.zeros(n)
    return x_s, t_s
```

Estos puntos se generan una sola vez antes del bucle de entrenamiento (la condición inicial es fija, no se re-muestrea).

### Cálculo de L_steady en `_compute_loss()`

```python
R_mass_s, R_mom_s = pde_residuals(model, x_steady, t_steady, L=L, T=T_total)
l_steady = torch.mean(R_mass_s ** 2) + torch.mean(R_mom_s ** 2)
```

Reutiliza `pde_residuals()` sin modificación — la única diferencia es que los tiempos son todos cero.

**Qué impone via autograd a t=0:**
- `∂Q/∂x → 0` : Q constante en x (flujo uniforme)
- `∂A/∂t → 0` : área estacionaria
- `gA(S₀ − Sf) → 0` : Sf = S₀ → ecuación de Manning (emerge automáticamente)

**Schedule de L_steady:** activo desde época 0, peso constante `lambda_steady`. Se activa antes que L_pde (que arranca en 0 durante warmup). Esto ancla n y Bw desde las primeras épocas.

---

## Cambio 4: `src/config.py` — nuevos defaults

```python
@dataclass(frozen=True)
class PinnDefaults:
    ...
    # Nuevos
    lambda_h: float        = 1.0    # peso L_h (observaciones de profundidad)
    lambda_steady: float   = 0.1    # peso L_steady (residuos PDE en t=0)
    n_colloc_steady: int   = 200    # puntos de colocación en t=0
```

**Justificación de defaults:**
- `lambda_h = 1.0`: mismo orden que `lambda_data` (ambos son MSE de observaciones normalizadas)
- `lambda_steady = 0.1`: conservador — los residuos PDE en t=0 pueden ser grandes al inicio; un peso bajo evita dominar el loss
- `n_colloc_steady = 200`: ~10% de `n_colloc` (2000); suficiente para cubrir el dominio espacial sin overhead computacional alto

---

## Cambio 5: `notebooks/04_pinn.ipynb` — celda de configuración y carga de datos

### Celda de configuración (añadir)

```python
LAMBDA_H       = DEFAULT_PINN.lambda_h       # 1.0
LAMBDA_STEADY  = DEFAULT_PINN.lambda_steady   # 0.1
N_COLLOC_STEADY = DEFAULT_PINN.n_colloc_steady # 200
```

### Celda de carga de datos (añadir vector h)

```python
h_outlet = df["h_outlet_m"].to_numpy(dtype=float)
hL_tensor = torch.tensor(h_outlet, dtype=torch.float32)
```

### Celda de entrenamiento (añadir argumentos)

```python
result = pinn_mod.train(
    model,
    x0_data=qu_tensor,
    xL_data=qd_tensor,
    hL_data=hL_tensor,        # NUEVO
    t_data=t_tensor,
    L=sinteticos.L,
    T=T_total,
    lambda_data=LAMBDA_DATA,
    lambda_h=LAMBDA_H,        # NUEVO
    lambda_pde=LAMBDA_PDE,
    lambda_steady=LAMBDA_STEADY, # NUEVO
    n_colloc_steady=N_COLLOC_STEADY, # NUEVO
    ...
)
```

### Curvas de pérdida (añadir l_h y l_steady al log)

El `loss_history` debe registrar `l_h` y `l_steady` en cada entrada para poder graficarlos.

---

## Impacto sobre el resto del código

| Componente | Impacto |
|---|---|
| `build_comparison_table()` | Sin cambio — usa solo n_estimate y Bw_estimate |
| `TrainResult` | Añadir `lh_history` y `steady_history` opcionales, o incluirlos en `loss_history` |
| Callers del notebook | Solo el notebook 04 usa `train()` directamente |
| `pde_residuals()` | Sin cambio de firma — solo se llama con t=0 para L_steady |

---

## Fuera de scope

- Modificar el schedule de ramp para L_steady o L_h (defaults son suficientes para experimentar)
- Añadir más puntos de observación (x=L/2 u otras estaciones)
- Cambiar la arquitectura de la red (capas, neuronas, activación)
- Fijar B_w y estimar solo n (se evalúa post-experimento según resultados)

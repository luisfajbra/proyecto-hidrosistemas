# Spec: Dashboard Saint-Venant / PINN

**Fecha:** 2026-05-30  
**Proyecto frontend:** `saint-venant-PINN/` (Vite + React 19 + TypeScript + shadcn/ui `radix-nova` + Tailwind v4)

---

## Objetivo

Dashboard técnico de una sola página para visualizar resultados del proyecto de modelación hidráulica 1D Saint-Venant y PINN. Primera pantalla activa: **Resumen**. Sin landing page.

---

## Estrategia de datos

Los CSVs viven en `proyecto-sv-mh/data/synthetic/` — fuera del root de Vite. Decisión: **copiar todos los archivos de datos a `saint-venant-PINN/public/data/`**.

- Vite sirve `public/` estáticamente; el frontend los carga en runtime con `fetch('/data/<archivo>')`.
- CSV parser: función inline sin dependencia adicional (CSVs limpios, sin escaping).
- Archivos copiados:
  - `series_corta_muskingum.csv`, `series_larga_muskingum.csv`
  - `series_corta_ruido.csv`, `series_larga_ruido.csv`
  - `series_corta_shift.csv`, `series_larga_shift.csv`
  - `sobol_indices.csv`, `sobol_ejecucion.csv`
  - `metricas_ols_cal_val.csv`, `parametros_ols_sensibilidad.csv`
  - `sensibilidad_conclusiones.txt`, `suposiciones_errores.txt`
- **Series largas** (200 000 filas ≈ 6–8 MB): carga on-demand al seleccionar, con `Skeleton` durante la carga.
- **CSVs PINN** (`pinn_loss_history.csv`, `pinn_hidrograma.csv`): aún no existen; los loaders los intentan cargar y, si fallan, los componentes muestran `Skeleton` con mensaje "pendiente de ejecutar notebook 04".

---

## Navegación

`Tabs` horizontal (shadcn) en el layout raíz. 5 pestañas en orden:

```
[Resumen] [Hidrogramas] [Sensibilidad] [Calibración] [PINN]
```

---

## Estructura de archivos

```
saint-venant-PINN/
  public/
    data/                        ← CSVs y TXTs copiados aquí
  src/
    lib/
      types.ts                   ← tipos de dominio
      data/
        loaders.ts               ← fetch + parse CSVs (async)
        static.ts                ← parámetros verdaderos y config PINN (hardcoded)
    components/
      ui/                        ← shadcn (existentes + nuevos por CLI)
      charts/
        PlotWrapper.tsx          ← createPlotlyComponent(Plotly) — instancia única
        HydrographChart.tsx
        SensitivityChart.tsx
        LossCurveChart.tsx       ← curva de pérdida PINN (log scale)
        HydrographPINNChart.tsx  ← Q_obs vs Q_PINN + banda warm-up
      dashboard/
        ResumenTab.tsx
        HidrogramasTab.tsx
        SensibilidadTab.tsx
        CalibracionTab.tsx
        PinnTab.tsx
      MetricCard.tsx             ← Card: label + valor + unidad
      DatasetSelector.tsx        ← Select serie (corta/larga × muskingum/ruido/shift)
    App.tsx                      ← layout raíz + Tabs
```

---

## Tipos de dominio (`src/lib/types.ts`)

```ts
interface HydrographPoint {
  datetime: string
  Q_upstream_m3s: number
  Q_downstream_m3s: number
  h_outlet_m: number
}

interface SobolIndex {
  parametro: string
  S1: number
  S1_conf: number
  ST: number
  ST_conf: number
}

interface CalibrationMetric {
  periodo: string   // "calibracion" | "validacion" | "post_warmup"
  NSE: number
  KGE: number
  RMSE_m3s: number
}

interface ParameterEstimate {
  parametro: string
  verdadero: number
  ols: number
  SE: number | null
}

interface PinnLossPoint {
  epoch: number
  total: number
  data: number
  pde: number
}

interface PinnHydrographPoint {
  t_h: number
  Q_obs: number
  Q_pinn: number
}

type SerieLength = "corta" | "larga"
type SerieType   = "muskingum" | "ruido" | "shift"
```

---

## Componentes shadcn requeridos

Ya instalados: `card`, `table`, `tabs`.  
Instalar con CLI antes de implementar: `button`, `select`, `badge`, `alert`, `skeleton`.

---

## Convención de diseño

- **Sin landing page** — primera pantalla es el dashboard (Resumen).
- **Sin cards anidadas** — charts y tablas van directamente en el layout de cada tab.
- **Gap** (`gap-*`) para espaciado; nunca `space-y-*` ni `space-x-*`.
- **Tokens semánticos** (`text-muted-foreground`, `text-destructive`, `bg-muted`, etc.) — sin colores crudos (`text-blue-500`).
- **Lucide React** para iconos.
- **Texto en español** en toda la UI.
- **Responsive**: desktop/laptop como objetivo principal; aceptable en mobile.

---

## Vistas — especificación detallada

### 1. Resumen

**Datos:** `metricas_ols_cal_val.csv`, `parametros_ols_sensibilidad.csv`, `series_corta_muskingum.csv` (carga inmediata).

**Layout:**
1. Fila de `MetricCard` con parámetros verdaderos del canal:
   - `n = 0.035`, `S₀ = 0.001`, `Q₀ = 50 m³/s`, `B_w = 50 m`, `L = 5000 m`
2. Fila de `MetricCard` con métricas OLS por periodo (calibración / validación / post-warmup):
   - NSE, KGE, RMSE (m³/s)
3. `HydrographChart` con serie corta muskingum — Q_upstream y Q_downstream.

### 2. Hidrogramas

**Datos:** las 6 series CSV cargadas on-demand.

**Componentes:**
- `DatasetSelector`: dos `Select` (longitud: corta/larga; tipo: muskingum/ruido/shift).
- `HydrographChart`: trazas `Q_upstream_m3s`, `Q_downstream_m3s`, opcionalmente `h_outlet_m` (eje Y secundario).
- `Skeleton` visible mientras se carga la serie larga.

**Tooltips:** habilitados via `hoverinfo` de Plotly con etiquetas claras.

### 3. Sensibilidad

**Datos:** `sobol_indices.csv`, `sobol_ejecucion.csv`, `sensibilidad_conclusiones.txt`.

**Componentes:**
- `SensitivityChart`: Plotly bar chart agrupado con trazas `S1` y `ST`, con barras de error (`S1_conf`, `ST_conf`). Parámetros en eje X: `n`, `S0`, `B_W`, `eta_Q`.
- `Table` shadcn con columnas: Parámetro | S1 | ±conf | ST | ±conf.
- `Badge` por fila: "Influyente" si `ST >= 0.0520` (umbral = 10% del max(ST)).
- Texto con conclusiones cargado desde `sensibilidad_conclusiones.txt`.

### 4. Calibración

**Datos:** `parametros_ols_sensibilidad.csv`, `metricas_ols_cal_val.csv`, `suposiciones_errores.txt`.

**Componentes:**
- `Table` shadcn: Parámetro | Verdadero | OLS | SE — SE muestra `—` si es nulo.
- Fila de `MetricCard` con NSE, KGE, RMSE por periodo.
- `Alert` (variant `destructive` o `warning`) por cada supuesto marcado como "REVISAR" en `suposiciones_errores.txt` (parsear las líneas con `-> REVISAR`).

### 5. PINN

**Datos:**
- Estáticos (hardcoded en `static.ts`): arquitectura y config de entrenamiento.
- On-demand (pueden no existir): `pinn_loss_history.csv`, `pinn_hidrograma.csv`.

**Config hardcoded mostrada (cards/tabla):**

| Campo | Valor |
|---|---|
| Activación | tanh |
| Normalización | Q / Q_max |
| hidden_size | 64 |
| n_layers | 4 |
| N_EPOCHS_ADAM | 6 000 |
| N_EPOCHS_WARMUP | 2 000 |
| N_EPOCHS_RAMP | 1 000 |
| N_ITER_LBFGS | 500 |
| LAMBDA_DATA | 1.0 |
| LAMBDA_PDE | 0.05 |
| gradient clipping max_norm | 1.0 |
| N_COLLOC | 2 000 |

**Placeholders preparados:**
- `LossCurveChart`: espera `epoch, total, data, pde`; eje Y log; 3 trazas (L_total, L_data, L_pde×λ). Si CSV ausente → `Skeleton` con label "Curva de pérdidas — pendiente de ejecutar notebook 04".
- `HydrographPINNChart`: espera `t_h, Q_obs, Q_pinn`; banda gris para warm-up period. Si CSV ausente → `Skeleton` con label "Q_obs vs Q_PINN — pendiente de ejecutar notebook 04".

**Estimados PINN** (cuando disponibles): `MetricCard` para `n_estimate` y `Bw_estimate` vs verdadero.

---

## Plotly — patrón de implementación

`PlotWrapper.tsx` es el único lugar donde se instancia el componente Plotly:

```tsx
import createPlotlyComponent from "react-plotly.js/factory"
import Plotly from "plotly.js-dist-min"

const Plot = createPlotlyComponent(Plotly)
```

Todos los charts (`HydrographChart`, `SensitivityChart`, etc.) importan `Plot` desde `PlotWrapper` — nunca instancian `createPlotlyComponent` directamente.

`PlotWrapper` acepta `data`, `layout` y `config` como props y aplica defaults de tema (fondo transparente, fuente coherente con el design system).

---

## Datos estáticos (`src/lib/data/static.ts`)

```ts
export const TRUE_PARAMS = {
  n: 0.035, S0: 0.001, Q0: 50, B_w: 50, L: 5000
} as const

export const PINN_CONFIG = {
  activation: "tanh",
  normalization: "Q / Q_max",
  hidden_size: 64,
  n_layers: 4,
  N_EPOCHS_ADAM: 6000,
  N_EPOCHS_WARMUP: 2000,
  N_EPOCHS_RAMP: 1000,
  N_ITER_LBFGS: 500,
  LAMBDA_DATA: 1.0,
  LAMBDA_PDE: 0.05,
  gradient_clip_max_norm: 1.0,
  N_COLLOC: 2000,
} as const
```

---

## Flujo de datos

```
public/data/*.csv
      │
      ▼
src/lib/data/loaders.ts   (fetch + parse + tipado)
      │
      ▼
Tab components            (useState / useEffect para carga lazy)
      │
      ▼
Chart / Table components  (props tipadas)
```

No hay estado global ni context — cada tab carga sus datos independientemente. Si en el futuro se necesita compartir datos entre tabs, se eleva el estado a `App.tsx`.

---

## Fuera de scope (v1)

- Autenticación o multiusuario.
- Persistencia de estado entre recargas (URL params, localStorage).
- Servidor de datos dinámico — todo es estático en `public/data/`.
- Tema claro/oscuro configurable por el usuario (el `theme-provider` existente gestiona el tema del sistema).
- Tests automatizados.

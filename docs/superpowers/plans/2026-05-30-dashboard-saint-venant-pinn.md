# Dashboard Saint-Venant / PINN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-tab technical dashboard in `saint-venant-PINN/` for visualizing hydraulic modeling results (hydrographs, Sobol sensitivity, OLS calibration, PINN).

**Architecture:** All data served statically from `public/data/` (copied from `../data/synthetic/`). No global state — each tab loads its own data via `useEffect`. Plotly instantiated once via factory pattern in `PlotWrapper`. shadcn/ui for all layout components.

**Tech Stack:** Vite 8, React 19, TypeScript 6, shadcn radix-nova, Tailwind v4, Plotly (react-plotly.js + plotly.js-dist-min), lucide-react.

> **Working directory for all commands:** `saint-venant-PINN/` unless stated otherwise.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `public/data/*.csv` + `*.txt` | Static data assets |
| Create | `src/lib/types.ts` | Domain type definitions |
| Create | `src/lib/data/static.ts` | Hardcoded true params + PINN config |
| Create | `src/lib/data/loaders.ts` | Async CSV fetch + parse |
| Create | `src/components/charts/PlotWrapper.tsx` | Plotly factory singleton + theme defaults |
| Create | `src/components/charts/HydrographChart.tsx` | Hydrograph line chart (scattergl) |
| Create | `src/components/charts/SensitivityChart.tsx` | Sobol grouped bar chart |
| Create | `src/components/charts/LossCurveChart.tsx` | PINN loss curve (log Y axis) |
| Create | `src/components/charts/HydrographPINNChart.tsx` | Q_obs vs Q_PINN + warm-up band |
| Create | `src/components/MetricCard.tsx` | Labeled value card |
| Create | `src/components/DatasetSelector.tsx` | Series length × type selectors |
| Create | `src/components/dashboard/ResumenTab.tsx` | Resumen view |
| Create | `src/components/dashboard/HidrogramasTab.tsx` | Hidrogramas view |
| Create | `src/components/dashboard/SensibilidadTab.tsx` | Sensibilidad view |
| Create | `src/components/dashboard/CalibracionTab.tsx` | Calibración view |
| Create | `src/components/dashboard/PinnTab.tsx` | PINN view |
| Modify | `src/App.tsx` | Root layout + Tabs wiring |

---

## Task 1: Copy data files to public/data

**Files:**
- Create: `saint-venant-PINN/public/data/` (directory + files)

Run from **project root** (`proyecto-sv-mh/`):

- [ ] **Step 1: Create directory and copy all data files**

```powershell
New-Item -ItemType Directory -Force saint-venant-PINN\public\data
Copy-Item data\synthetic\*.csv saint-venant-PINN\public\data\
Copy-Item data\synthetic\*.txt saint-venant-PINN\public\data\
```

- [ ] **Step 2: Verify files are present (still from project root)**

```powershell
Get-ChildItem saint-venant-PINN\public\data | Select-Object Name
```

Expected: 13 files — `series_corta_muskingum.csv`, `series_larga_muskingum.csv`, `series_corta_ruido.csv`, `series_larga_ruido.csv`, `series_corta_shift.csv`, `series_larga_shift.csv`, `sobol_indices.csv`, `sobol_ejecucion.csv`, `metricas_ols_cal_val.csv`, `parametros_ols_sensibilidad.csv`, `batimetria.csv`, `sensibilidad_conclusiones.txt`, `suposiciones_errores.txt`.

- [ ] **Step 3: Commit (from project root)**

```bash
git add saint-venant-PINN/public/data/
git commit -m "data: copy synthetic CSVs and TXTs to public/data for Vite dev server"
```

---

## Task 2: Install missing shadcn components

**Files:**
- Modify: `src/components/ui/` (files added by CLI)

- [ ] **Step 1: Add missing components via CLI**

```bash
npx shadcn add button select badge alert skeleton
```

Accept any prompts (overwrite if asked).

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/ui/
git commit -m "feat: add shadcn button, select, badge, alert, skeleton components"
```

---

## Task 3: Domain types

**Files:**
- Create: `src/lib/types.ts`

- [ ] **Step 1: Write types file**

Create `src/lib/types.ts`:

```ts
export interface HydrographPoint {
  datetime: string
  Q_upstream_m3s: number
  Q_downstream_m3s: number
  h_outlet_m: number
}

export interface SobolIndex {
  parametro: string
  S1: number
  S1_conf: number
  ST: number
  ST_conf: number
}

export interface CalibrationMetric {
  periodo: string
  NSE: number
  KGE: number
  RMSE_m3s: number
}

export interface ParameterEstimate {
  parametro: string
  verdadero: number
  ols: number
  SE: number | null
}

export interface PinnLossPoint {
  epoch: number
  total: number
  data: number
  pde: number
}

export interface PinnHydrographPoint {
  t_h: number
  Q_obs: number
  Q_pinn: number
}

export type SerieLength = "corta" | "larga"
export type SerieType = "muskingum" | "ruido" | "shift"
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/lib/types.ts
git commit -m "feat: add domain type definitions"
```

---

## Task 4: Static data

**Files:**
- Create: `src/lib/data/static.ts`

- [ ] **Step 1: Write static data file**

Create `src/lib/data/static.ts`:

```ts
export const TRUE_PARAMS = {
  n: 0.035,
  S0: 0.001,
  Q0: 50,
  B_w: 50,
  L: 5000,
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
  WARMUP_H: 1.0,
} as const
```

`WARMUP_H` is derived from the notebook's `WARMUP_SEC = 3600` (1 hour of simulation warm-up, used to draw the shaded band in `HydrographPINNChart`).

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/lib/data/static.ts
git commit -m "feat: add static true params and PINN training config"
```

---

## Task 5: CSV loaders

**Files:**
- Create: `src/lib/data/loaders.ts`

- [ ] **Step 1: Write loaders file**

Create `src/lib/data/loaders.ts`:

```ts
import type {
  HydrographPoint,
  SobolIndex,
  CalibrationMetric,
  ParameterEstimate,
  PinnLossPoint,
  PinnHydrographPoint,
  SerieLength,
  SerieType,
} from "@/lib/types"

function parseCsv<T>(
  text: string,
  transform: (row: Record<string, string>) => T,
): T[] {
  const lines = text.trim().split("\n")
  const headers = lines[0].split(",").map((h) => h.trim())
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim())
    const row: Record<string, string> = {}
    headers.forEach((h, i) => {
      row[h] = values[i] ?? ""
    })
    return transform(row)
  })
}

async function fetchText(path: string): Promise<string> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`fetch ${path} → ${res.status}`)
  return res.text()
}

async function fetchTextOrNull(path: string): Promise<string | null> {
  try {
    const res = await fetch(path)
    if (!res.ok) return null
    return res.text()
  } catch {
    return null
  }
}

export async function loadHydrograph(
  length: SerieLength,
  type: SerieType,
): Promise<HydrographPoint[]> {
  const text = await fetchText(`/data/series_${length}_${type}.csv`)
  return parseCsv(text, (row) => ({
    datetime: row.datetime,
    Q_upstream_m3s: parseFloat(row.Q_upstream_m3s),
    Q_downstream_m3s: parseFloat(row.Q_downstream_m3s),
    h_outlet_m: parseFloat(row.h_outlet_m),
  }))
}

export async function loadSobolIndices(): Promise<SobolIndex[]> {
  const text = await fetchText("/data/sobol_indices.csv")
  return parseCsv(text, (row) => ({
    parametro: row.parametro,
    S1: parseFloat(row.S1),
    S1_conf: parseFloat(row.S1_conf),
    ST: parseFloat(row.ST),
    ST_conf: parseFloat(row.ST_conf),
  }))
}

export async function loadCalibrationMetrics(): Promise<CalibrationMetric[]> {
  const text = await fetchText("/data/metricas_ols_cal_val.csv")
  return parseCsv(text, (row) => ({
    periodo: row.periodo,
    NSE: parseFloat(row.NSE),
    KGE: parseFloat(row.KGE),
    RMSE_m3s: parseFloat(row.RMSE_m3s),
  }))
}

export async function loadParameterEstimates(): Promise<ParameterEstimate[]> {
  const text = await fetchText("/data/parametros_ols_sensibilidad.csv")
  return parseCsv(text, (row) => ({
    parametro: row.parametro,
    verdadero: parseFloat(row.verdadero),
    ols: parseFloat(row.ols),
    SE: row.SE !== "" ? parseFloat(row.SE) : null,
  }))
}

export async function loadSensibilidadConclusiones(): Promise<string> {
  return fetchText("/data/sensibilidad_conclusiones.txt")
}

export async function loadSuposicionesErrores(): Promise<string> {
  return fetchText("/data/suposiciones_errores.txt")
}

export async function loadPinnLossHistory(): Promise<PinnLossPoint[] | null> {
  const text = await fetchTextOrNull("/data/pinn_loss_history.csv")
  if (!text) return null
  return parseCsv(text, (row) => ({
    epoch: parseInt(row.epoch, 10),
    total: parseFloat(row.total),
    data: parseFloat(row.data),
    pde: parseFloat(row.pde),
  }))
}

export async function loadPinnHydrograph(): Promise<PinnHydrographPoint[] | null> {
  const text = await fetchTextOrNull("/data/pinn_hidrograma.csv")
  if (!text) return null
  return parseCsv(text, (row) => ({
    t_h: parseFloat(row.t_h),
    Q_obs: parseFloat(row.Q_obs),
    Q_pinn: parseFloat(row.Q_pinn),
  }))
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/lib/data/loaders.ts
git commit -m "feat: add async CSV loaders with graceful null on missing PINN files"
```

---

## Task 6: PlotWrapper

**Files:**
- Create: `src/components/charts/PlotWrapper.tsx`

- [ ] **Step 1: Create charts directory and write PlotWrapper**

Create `src/components/charts/PlotWrapper.tsx`:

```tsx
import createPlotlyComponent from "react-plotly.js/factory"
import Plotly from "plotly.js-dist-min"
import type { Data, Layout, Config } from "plotly.js"

const PlotlyChart = createPlotlyComponent(Plotly)

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Geist Variable, sans-serif", size: 12 },
  margin: { t: 32, r: 20, b: 56, l: 64 },
}

const BASE_CONFIG: Partial<Config> = {
  responsive: true,
  displayModeBar: false,
}

interface PlotWrapperProps {
  data: Data[]
  layout?: Partial<Layout>
  config?: Partial<Config>
  style?: React.CSSProperties
  className?: string
}

export function PlotWrapper({
  data,
  layout = {},
  config = {},
  style,
  className,
}: PlotWrapperProps) {
  return (
    <PlotlyChart
      data={data}
      layout={{ ...BASE_LAYOUT, ...layout }}
      config={{ ...BASE_CONFIG, ...config }}
      style={{ width: "100%", height: "100%", ...style }}
      className={className}
      useResizeHandler
    />
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/charts/PlotWrapper.tsx
git commit -m "feat: add PlotWrapper with Plotly factory singleton and theme defaults"
```

---

## Task 7: MetricCard and DatasetSelector

**Files:**
- Create: `src/components/MetricCard.tsx`
- Create: `src/components/DatasetSelector.tsx`

- [ ] **Step 1: Write MetricCard**

Create `src/components/MetricCard.tsx`:

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  description?: string
}

export function MetricCard({ label, value, unit, description }: MetricCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-semibold tabular-nums">
          {value}
          {unit && (
            <span className="ml-1 text-sm font-normal text-muted-foreground">{unit}</span>
          )}
        </p>
        {description && (
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: Write DatasetSelector**

Create `src/components/DatasetSelector.tsx`:

```tsx
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { SerieLength, SerieType } from "@/lib/types"

interface DatasetSelectorProps {
  length: SerieLength
  type: SerieType
  onLengthChange: (v: SerieLength) => void
  onTypeChange: (v: SerieType) => void
}

export function DatasetSelector({
  length,
  type,
  onLengthChange,
  onTypeChange,
}: DatasetSelectorProps) {
  return (
    <div className="flex gap-3">
      <Select value={length} onValueChange={(v) => onLengthChange(v as SerieLength)}>
        <SelectTrigger className="w-36">
          <SelectValue placeholder="Longitud" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="corta">Corta (~2 h)</SelectItem>
          <SelectItem value="larga">Larga (~833 h)</SelectItem>
        </SelectContent>
      </Select>

      <Select value={type} onValueChange={(v) => onTypeChange(v as SerieType)}>
        <SelectTrigger className="w-44">
          <SelectValue placeholder="Tipo" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="muskingum">Muskingum</SelectItem>
          <SelectItem value="ruido">Con ruido</SelectItem>
          <SelectItem value="shift">Con shift</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/components/MetricCard.tsx src/components/DatasetSelector.tsx
git commit -m "feat: add MetricCard and DatasetSelector components"
```

---

## Task 8: HydrographChart

**Files:**
- Create: `src/components/charts/HydrographChart.tsx`

- [ ] **Step 1: Write HydrographChart**

Create `src/components/charts/HydrographChart.tsx`:

```tsx
import type { Data, Layout } from "plotly.js"
import type { HydrographPoint } from "@/lib/types"
import { PlotWrapper } from "./PlotWrapper"

interface HydrographChartProps {
  data: HydrographPoint[]
  showDepth?: boolean
  height?: number
}

export function HydrographChart({
  data,
  showDepth = false,
  height = 360,
}: HydrographChartProps) {
  const x = data.map((d) => d.datetime)

  const traces: Data[] = [
    {
      x,
      y: data.map((d) => d.Q_upstream_m3s),
      type: "scattergl",
      mode: "lines",
      name: "Q aguas arriba",
      hovertemplate: "%{y:.2f} m³/s<extra>Q upstream</extra>",
      line: { width: 1.5 },
    },
    {
      x,
      y: data.map((d) => d.Q_downstream_m3s),
      type: "scattergl",
      mode: "lines",
      name: "Q aguas abajo",
      hovertemplate: "%{y:.2f} m³/s<extra>Q downstream</extra>",
      line: { width: 1.5 },
    },
  ]

  if (showDepth) {
    traces.push({
      x,
      y: data.map((d) => d.h_outlet_m),
      type: "scattergl",
      mode: "lines",
      name: "h salida (m)",
      yaxis: "y2",
      hovertemplate: "%{y:.3f} m<extra>h outlet</extra>",
      line: { width: 1, dash: "dot" },
    })
  }

  const layout: Partial<Layout> = {
    height,
    xaxis: { title: "Tiempo", type: "date" },
    yaxis: { title: "Caudal (m³/s)" },
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.25 },
    ...(showDepth
      ? {
          yaxis2: {
            title: "Profundidad (m)",
            overlaying: "y" as const,
            side: "right" as const,
          },
        }
      : {}),
  }

  return <PlotWrapper data={traces} layout={layout} />
}
```

`scattergl` uses WebGL — handles the 200 000-row long series without performance issues.

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/charts/HydrographChart.tsx
git commit -m "feat: add HydrographChart with scattergl and optional depth Y2 axis"
```

---

## Task 9: SensitivityChart

**Files:**
- Create: `src/components/charts/SensitivityChart.tsx`

- [ ] **Step 1: Write SensitivityChart**

Create `src/components/charts/SensitivityChart.tsx`:

```tsx
import type { Data, Layout } from "plotly.js"
import type { SobolIndex } from "@/lib/types"
import { PlotWrapper } from "./PlotWrapper"

interface SensitivityChartProps {
  data: SobolIndex[]
  height?: number
}

export function SensitivityChart({ data, height = 360 }: SensitivityChartProps) {
  const x = data.map((d) => d.parametro)

  const traces: Data[] = [
    {
      x,
      y: data.map((d) => d.S1),
      error_y: {
        type: "data",
        array: data.map((d) => d.S1_conf),
        visible: true,
      },
      type: "bar",
      name: "S1 — primer orden",
    },
    {
      x,
      y: data.map((d) => d.ST),
      error_y: {
        type: "data",
        array: data.map((d) => d.ST_conf),
        visible: true,
      },
      type: "bar",
      name: "ST — total",
    },
  ]

  const layout: Partial<Layout> = {
    height,
    barmode: "group",
    xaxis: { title: "Parámetro" },
    yaxis: { title: "Índice de Sobol", range: [0, 1.1] },
    legend: { orientation: "h", y: -0.25 },
    hovermode: "closest",
  }

  return <PlotWrapper data={traces} layout={layout} />
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/charts/SensitivityChart.tsx
git commit -m "feat: add SensitivityChart grouped bar chart with error bars"
```

---

## Task 10: PINN chart components

**Files:**
- Create: `src/components/charts/LossCurveChart.tsx`
- Create: `src/components/charts/HydrographPINNChart.tsx`

- [ ] **Step 1: Write LossCurveChart**

Create `src/components/charts/LossCurveChart.tsx`:

```tsx
import type { Data, Layout } from "plotly.js"
import type { PinnLossPoint } from "@/lib/types"
import { PlotWrapper } from "./PlotWrapper"
import { PINN_CONFIG } from "@/lib/data/static"

interface LossCurveChartProps {
  data: PinnLossPoint[]
  height?: number
}

export function LossCurveChart({ data, height = 360 }: LossCurveChartProps) {
  const x = data.map((d) => d.epoch)

  const traces: Data[] = [
    {
      x,
      y: data.map((d) => d.total),
      type: "scatter",
      mode: "lines",
      name: "L_total",
      line: { width: 2 },
    },
    {
      x,
      y: data.map((d) => d.data),
      type: "scatter",
      mode: "lines",
      name: "L_data",
      line: { width: 1.5 },
    },
    {
      x,
      y: data.map((d) => d.pde),
      type: "scatter",
      mode: "lines",
      name: `L_pde (×${PINN_CONFIG.LAMBDA_PDE})`,
      line: { width: 1.5 },
    },
  ]

  const layout: Partial<Layout> = {
    height,
    xaxis: { title: "Época (Adam)" },
    yaxis: { title: "Pérdida", type: "log" },
    legend: { orientation: "h", y: -0.25 },
    hovermode: "x unified",
  }

  return <PlotWrapper data={traces} layout={layout} />
}
```

- [ ] **Step 2: Write HydrographPINNChart**

Create `src/components/charts/HydrographPINNChart.tsx`:

```tsx
import type { Data, Layout } from "plotly.js"
import type { PinnHydrographPoint } from "@/lib/types"
import { PlotWrapper } from "./PlotWrapper"
import { PINN_CONFIG } from "@/lib/data/static"

interface HydrographPINNChartProps {
  data: PinnHydrographPoint[]
  height?: number
}

export function HydrographPINNChart({ data, height = 360 }: HydrographPINNChartProps) {
  const x = data.map((d) => d.t_h)

  const traces: Data[] = [
    {
      x,
      y: data.map((d) => d.Q_obs),
      type: "scatter",
      mode: "markers",
      name: "Q_obs (m³/s)",
      marker: { size: 3, opacity: 0.6 },
      hovertemplate: "t=%{x:.2f} h  Q_obs=%{y:.2f} m³/s<extra></extra>",
    },
    {
      x,
      y: data.map((d) => d.Q_pinn),
      type: "scatter",
      mode: "lines",
      name: "Q_PINN (m³/s)",
      line: { width: 2 },
      hovertemplate: "t=%{x:.2f} h  Q_PINN=%{y:.2f} m³/s<extra></extra>",
    },
  ]

  const layout: Partial<Layout> = {
    height,
    xaxis: { title: "Tiempo (h)" },
    yaxis: { title: "Caudal (m³/s)" },
    legend: { orientation: "h", y: -0.25 },
    hovermode: "x unified",
    shapes: [
      {
        type: "rect" as const,
        x0: 0,
        x1: PINN_CONFIG.WARMUP_H,
        y0: 0,
        y1: 1,
        yref: "paper" as const,
        fillcolor: "rgba(128,128,128,0.12)",
        line: { width: 0 },
        layer: "below" as const,
      },
    ],
  }

  return <PlotWrapper data={traces} layout={layout} />
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/components/charts/LossCurveChart.tsx src/components/charts/HydrographPINNChart.tsx
git commit -m "feat: add LossCurveChart (log) and HydrographPINNChart (warm-up band)"
```

---

## Task 11: ResumenTab

**Files:**
- Create: `src/components/dashboard/ResumenTab.tsx`

- [ ] **Step 1: Write ResumenTab**

Create `src/components/dashboard/ResumenTab.tsx`:

```tsx
import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { MetricCard } from "@/components/MetricCard"
import { HydrographChart } from "@/components/charts/HydrographChart"
import { loadHydrograph, loadCalibrationMetrics } from "@/lib/data/loaders"
import { TRUE_PARAMS } from "@/lib/data/static"
import type { HydrographPoint, CalibrationMetric } from "@/lib/types"

export function ResumenTab() {
  const [series, setSeries] = useState<HydrographPoint[] | null>(null)
  const [metrics, setMetrics] = useState<CalibrationMetric[] | null>(null)

  useEffect(() => {
    loadHydrograph("corta", "muskingum").then(setSeries)
    loadCalibrationMetrics().then(setMetrics)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Parámetros verdaderos del canal
        </h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <MetricCard label="Manning n" value={TRUE_PARAMS.n} />
          <MetricCard label="Pendiente S₀" value={TRUE_PARAMS.S0} />
          <MetricCard label="Caudal base Q₀" value={TRUE_PARAMS.Q0} unit="m³/s" />
          <MetricCard label="Ancho B_w" value={TRUE_PARAMS.B_w} unit="m" />
          <MetricCard label="Longitud L" value={TRUE_PARAMS.L.toLocaleString()} unit="m" />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Métricas de calibración OLS
        </h2>
        {metrics ? (
          <div className="flex flex-col gap-4">
            {metrics.map((m) => (
              <div key={m.periodo} className="flex flex-col gap-2">
                <p className="text-xs font-medium capitalize text-muted-foreground">
                  {m.periodo}
                </p>
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="NSE" value={m.NSE.toFixed(3)} />
                  <MetricCard label="KGE" value={m.KGE.toFixed(3)} />
                  <MetricCard label="RMSE" value={m.RMSE_m3s.toFixed(2)} unit="m³/s" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-24 rounded-lg" />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Hidrograma — serie corta Muskingum
        </h2>
        {series ? (
          <HydrographChart data={series} />
        ) : (
          <Skeleton className="h-[360px] rounded-lg" />
        )}
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/ResumenTab.tsx
git commit -m "feat: add ResumenTab with parameter cards, OLS metrics, and overview hydrograph"
```

---

## Task 12: HidrogramasTab

**Files:**
- Create: `src/components/dashboard/HidrogramasTab.tsx`

- [ ] **Step 1: Write HidrogramasTab**

Create `src/components/dashboard/HidrogramasTab.tsx`:

```tsx
import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { DatasetSelector } from "@/components/DatasetSelector"
import { HydrographChart } from "@/components/charts/HydrographChart"
import { loadHydrograph } from "@/lib/data/loaders"
import type { HydrographPoint, SerieLength, SerieType } from "@/lib/types"

export function HidrogramasTab() {
  const [length, setLength] = useState<SerieLength>("corta")
  const [type, setType] = useState<SerieType>("muskingum")
  const [data, setData] = useState<HydrographPoint[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setData(null)
    setError(null)
    loadHydrograph(length, type)
      .then(setData)
      .catch(() => setError("No se pudo cargar la serie seleccionada."))
      .finally(() => setLoading(false))
  }, [length, type])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Hidrogramas sintéticos
        </h2>
        <DatasetSelector
          length={length}
          type={type}
          onLengthChange={setLength}
          onTypeChange={setType}
        />
      </div>

      {loading ? (
        <Skeleton className="h-[400px] rounded-lg" />
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : data ? (
        <HydrographChart data={data} showDepth height={400} />
      ) : null}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/HidrogramasTab.tsx
git commit -m "feat: add HidrogramasTab with lazy-loaded series selector"
```

---

## Task 13: SensibilidadTab

**Files:**
- Create: `src/components/dashboard/SensibilidadTab.tsx`

- [ ] **Step 1: Write SensibilidadTab**

Create `src/components/dashboard/SensibilidadTab.tsx`:

```tsx
import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { SensitivityChart } from "@/components/charts/SensitivityChart"
import { loadSobolIndices, loadSensibilidadConclusiones } from "@/lib/data/loaders"
import type { SobolIndex } from "@/lib/types"

const INFLUYENTE_THRESHOLD = 0.052

export function SensibilidadTab() {
  const [indices, setIndices] = useState<SobolIndex[] | null>(null)
  const [conclusiones, setConclusiones] = useState<string | null>(null)

  useEffect(() => {
    loadSobolIndices().then(setIndices)
    loadSensibilidadConclusiones().then(setConclusiones)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Índices de Sobol — análisis de sensibilidad global
        </h2>
        {indices ? (
          <SensitivityChart data={indices} />
        ) : (
          <Skeleton className="h-[360px] rounded-lg" />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Tabla de índices
        </h2>
        {indices ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parámetro</TableHead>
                <TableHead className="text-right">S1</TableHead>
                <TableHead className="text-right">±conf S1</TableHead>
                <TableHead className="text-right">ST</TableHead>
                <TableHead className="text-right">±conf ST</TableHead>
                <TableHead>Clasificación</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {indices.map((row) => (
                <TableRow key={row.parametro}>
                  <TableCell className="font-mono">{row.parametro}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.S1.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.S1_conf.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.ST.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.ST_conf.toFixed(4)}
                  </TableCell>
                  <TableCell>
                    {row.ST >= INFLUYENTE_THRESHOLD ? (
                      <Badge>Influyente</Badge>
                    ) : (
                      <Badge variant="secondary">Poco sensible</Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Skeleton className="h-40 rounded-lg" />
        )}
      </section>

      {conclusiones && (
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Conclusiones
          </h2>
          <pre className="rounded-md bg-muted p-4 font-mono text-xs text-muted-foreground whitespace-pre-wrap">
            {conclusiones}
          </pre>
        </section>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/SensibilidadTab.tsx
git commit -m "feat: add SensibilidadTab with Sobol chart, table with badges, and conclusions"
```

---

## Task 14: CalibracionTab

**Files:**
- Create: `src/components/dashboard/CalibracionTab.tsx`

- [ ] **Step 1: Write CalibracionTab**

Create `src/components/dashboard/CalibracionTab.tsx`:

```tsx
import { useEffect, useState } from "react"
import { AlertCircle } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { MetricCard } from "@/components/MetricCard"
import {
  loadParameterEstimates,
  loadCalibrationMetrics,
  loadSuposicionesErrores,
} from "@/lib/data/loaders"
import type { ParameterEstimate, CalibrationMetric } from "@/lib/types"

function extractRevisarLines(text: string): string[] {
  return text
    .split("\n")
    .filter((line) => line.includes("-> REVISAR"))
    .map((line) => line.trim())
}

export function CalibracionTab() {
  const [params, setParams] = useState<ParameterEstimate[] | null>(null)
  const [metrics, setMetrics] = useState<CalibrationMetric[] | null>(null)
  const [alertas, setAlertas] = useState<string[]>([])

  useEffect(() => {
    loadParameterEstimates().then(setParams)
    loadCalibrationMetrics().then(setMetrics)
    loadSuposicionesErrores().then((text) => setAlertas(extractRevisarLines(text)))
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      {alertas.length > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Supuestos a revisar
          </h2>
          {alertas.map((msg, i) => (
            <Alert key={i} variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Revisar</AlertTitle>
              <AlertDescription className="font-mono text-xs">{msg}</AlertDescription>
            </Alert>
          ))}
        </section>
      )}

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Parámetros estimados OLS
        </h2>
        {params ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parámetro</TableHead>
                <TableHead className="text-right">Verdadero</TableHead>
                <TableHead className="text-right">OLS</TableHead>
                <TableHead className="text-right">SE</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {params.map((row) => (
                <TableRow key={row.parametro}>
                  <TableCell className="font-mono">{row.parametro}</TableCell>
                  <TableCell className="text-right tabular-nums">{row.verdadero}</TableCell>
                  <TableCell className="text-right tabular-nums">{row.ols.toFixed(6)}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.SE !== null ? row.SE.toFixed(6) : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Skeleton className="h-36 rounded-lg" />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Métricas de ajuste
        </h2>
        {metrics ? (
          <div className="flex flex-col gap-4">
            {metrics.map((m) => (
              <div key={m.periodo} className="flex flex-col gap-2">
                <p className="text-xs font-medium capitalize text-muted-foreground">
                  {m.periodo}
                </p>
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="NSE" value={m.NSE.toFixed(3)} />
                  <MetricCard label="KGE" value={m.KGE.toFixed(3)} />
                  <MetricCard label="RMSE" value={m.RMSE_m3s.toFixed(2)} unit="m³/s" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Skeleton className="h-36 rounded-lg" />
        )}
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/CalibracionTab.tsx
git commit -m "feat: add CalibracionTab with REVISAR alerts, params table, and metrics"
```

---

## Task 15: PinnTab

**Files:**
- Create: `src/components/dashboard/PinnTab.tsx`

The state uses `undefined` (loading), `null` (file not found), or `T[]` (loaded). This three-way distinction allows precise placeholder rendering without an extra `loading` boolean.

- [ ] **Step 1: Write PinnTab**

Create `src/components/dashboard/PinnTab.tsx`:

```tsx
import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableRow,
} from "@/components/ui/table"
import { MetricCard } from "@/components/MetricCard"
import { LossCurveChart } from "@/components/charts/LossCurveChart"
import { HydrographPINNChart } from "@/components/charts/HydrographPINNChart"
import { loadPinnLossHistory, loadPinnHydrograph } from "@/lib/data/loaders"
import { PINN_CONFIG, TRUE_PARAMS } from "@/lib/data/static"
import type { PinnLossPoint, PinnHydrographPoint } from "@/lib/types"

const CONFIG_ROWS: { label: string; value: string | number }[] = [
  { label: "Activación", value: PINN_CONFIG.activation },
  { label: "Normalización", value: PINN_CONFIG.normalization },
  { label: "Arquitectura", value: `${PINN_CONFIG.n_layers} capas × ${PINN_CONFIG.hidden_size}` },
  { label: "Épocas Adam", value: PINN_CONFIG.N_EPOCHS_ADAM.toLocaleString("es") },
  { label: "Épocas warmup", value: PINN_CONFIG.N_EPOCHS_WARMUP.toLocaleString("es") },
  { label: "Épocas ramp λ_pde", value: PINN_CONFIG.N_EPOCHS_RAMP.toLocaleString("es") },
  { label: "Iter. L-BFGS", value: PINN_CONFIG.N_ITER_LBFGS.toLocaleString("es") },
  { label: "λ_data", value: PINN_CONFIG.LAMBDA_DATA },
  { label: "λ_pde", value: PINN_CONFIG.LAMBDA_PDE },
  { label: "Grad. clipping max_norm", value: PINN_CONFIG.gradient_clip_max_norm },
  { label: "Puntos de colocación", value: PINN_CONFIG.N_COLLOC.toLocaleString("es") },
]

const PENDING_LABEL = "Pendiente — ejecutar notebook 04"

function PendingPlaceholder() {
  return (
    <div className="flex h-[360px] items-center justify-center rounded-lg border border-dashed">
      <p className="text-sm text-muted-foreground">{PENDING_LABEL}</p>
    </div>
  )
}

export function PinnTab() {
  const [lossData, setLossData] = useState<PinnLossPoint[] | null | undefined>(undefined)
  const [hydroData, setHydroData] = useState<PinnHydrographPoint[] | null | undefined>(undefined)

  useEffect(() => {
    loadPinnLossHistory().then(setLossData)
    loadPinnHydrograph().then(setHydroData)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Configuración del modelo
        </h2>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-md border">
            <Table>
              <TableBody>
                {CONFIG_ROWS.map((r) => (
                  <TableRow key={r.label}>
                    <TableCell className="text-muted-foreground">{r.label}</TableCell>
                    <TableCell className="text-right font-mono font-medium">
                      {r.value}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div className="grid grid-cols-2 gap-4 content-start">
            <MetricCard label="n verdadero" value={TRUE_PARAMS.n} />
            <MetricCard label="B_w verdadero" value={TRUE_PARAMS.B_w} unit="m" />
            <MetricCard label="S₀ verdadero" value={TRUE_PARAMS.S0} />
            <MetricCard label="Longitud L" value={TRUE_PARAMS.L.toLocaleString("es")} unit="m" />
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Curva de pérdidas — fase Adam
        </h2>
        {lossData === undefined ? (
          <Skeleton className="h-[360px] rounded-lg" />
        ) : lossData === null ? (
          <PendingPlaceholder />
        ) : (
          <LossCurveChart data={lossData} />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Hidrograma — Q_obs vs Q_PINN
        </h2>
        {hydroData === undefined ? (
          <Skeleton className="h-[360px] rounded-lg" />
        ) : hydroData === null ? (
          <PendingPlaceholder />
        ) : (
          <HydrographPINNChart data={hydroData} />
        )}
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/components/dashboard/PinnTab.tsx
git commit -m "feat: add PinnTab with config table and pending chart placeholders"
```

---

## Task 16: App.tsx — final wiring and visual verification

**Files:**
- Modify: `src/App.tsx`

- [ ] **Step 1: Rewrite App.tsx**

Replace the entire contents of `src/App.tsx` with:

```tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResumenTab } from "@/components/dashboard/ResumenTab"
import { HidrogramasTab } from "@/components/dashboard/HidrogramasTab"
import { SensibilidadTab } from "@/components/dashboard/SensibilidadTab"
import { CalibracionTab } from "@/components/dashboard/CalibracionTab"
import { PinnTab } from "@/components/dashboard/PinnTab"

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b px-6 py-4">
        <h1 className="text-base font-semibold">Saint-Venant 1D / PINN — Resultados</h1>
        <p className="text-xs text-muted-foreground">
          Modelación hidráulica 1D · Sensibilidad global · Calibración OLS · Red neuronal física
        </p>
      </header>

      <main className="flex-1 px-6 pb-10">
        <Tabs defaultValue="resumen">
          <TabsList className="mt-4">
            <TabsTrigger value="resumen">Resumen</TabsTrigger>
            <TabsTrigger value="hidrogramas">Hidrogramas</TabsTrigger>
            <TabsTrigger value="sensibilidad">Sensibilidad</TabsTrigger>
            <TabsTrigger value="calibracion">Calibración</TabsTrigger>
            <TabsTrigger value="pinn">PINN</TabsTrigger>
          </TabsList>

          <TabsContent value="resumen">
            <ResumenTab />
          </TabsContent>
          <TabsContent value="hidrogramas">
            <HidrogramasTab />
          </TabsContent>
          <TabsContent value="sensibilidad">
            <SensibilidadTab />
          </TabsContent>
          <TabsContent value="calibracion">
            <CalibracionTab />
          </TabsContent>
          <TabsContent value="pinn">
            <PinnTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles clean**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Start dev server and do visual verification**

```bash
npm run dev
```

Open `http://localhost:5173` and verify each tab:

- [ ] Header visible with title and subtitle
- [ ] **Resumen**: 5 parameter cards, 2×3 metric cards (calibración + validación), hydrograph renders
- [ ] **Hidrogramas**: selector changes series, Skeleton shows while loading larga, chart renders with depth trace
- [ ] **Sensibilidad**: grouped bar chart renders, all 4 params show `Influyente` badge, conclusions text block appears
- [ ] **Calibración**: 3 REVISAR alerts appear at top, params table shows `—` in SE column, metrics render
- [ ] **PINN**: config table with 11 rows, 4 true-param cards, two dashed "Pendiente" placeholders

- [ ] **Step 4: Commit**

```bash
git add src/App.tsx
git commit -m "feat: wire App.tsx with 5-tab dashboard layout — v1 complete"
```

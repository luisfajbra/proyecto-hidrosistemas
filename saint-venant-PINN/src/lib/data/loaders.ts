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
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim() !== "")
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

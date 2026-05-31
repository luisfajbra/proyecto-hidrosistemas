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

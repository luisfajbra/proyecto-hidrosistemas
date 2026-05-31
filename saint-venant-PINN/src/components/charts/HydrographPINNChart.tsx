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
    xaxis: { title: { text: "Tiempo (h)" } },
    yaxis: { title: { text: "Caudal (m³/s)" } },
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

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
    xaxis: { title: { text: "Época (Adam)" } },
    yaxis: { title: { text: "Pérdida" }, type: "log" },
    legend: { orientation: "h", y: -0.25 },
    hovermode: "x unified",
  }

  return <PlotWrapper data={traces} layout={layout} />
}

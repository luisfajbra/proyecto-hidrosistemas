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
      type: "scatter",
      mode: "lines",
      name: "Q aguas arriba",
      hovertemplate: "%{y:.2f} m³/s<extra>Q upstream</extra>",
      line: { width: 1.5 },
    },
    {
      x,
      y: data.map((d) => d.Q_downstream_m3s),
      type: "scatter",
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
      type: "scatter",
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

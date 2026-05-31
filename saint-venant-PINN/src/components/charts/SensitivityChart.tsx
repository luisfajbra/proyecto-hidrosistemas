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
    xaxis: { title: { text: "Parámetro" } },
    yaxis: { title: { text: "Índice de Sobol" }, range: [0, 1.1] },
    legend: { orientation: "h", y: -0.25 },
    hovermode: "closest",
  }

  return <PlotWrapper data={traces} layout={layout} />
}

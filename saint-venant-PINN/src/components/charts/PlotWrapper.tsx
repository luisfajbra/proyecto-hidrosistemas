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

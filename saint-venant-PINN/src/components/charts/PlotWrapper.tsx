import { useEffect, useRef } from "react"
import Plotly from "plotly.js-dist-min"
import type { Data, Layout, Config } from "plotly.js"

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Geist Variable, sans-serif", size: 12 },
  margin: { t: 32, r: 20, b: 56, l: 64 },
  autosize: true,
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
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    Plotly.react(
      ref.current,
      data,
      { ...BASE_LAYOUT, ...layout },
      { ...BASE_CONFIG, ...config },
    )
  })

  useEffect(() => {
    const el = ref.current
    return () => {
      if (el) Plotly.purge(el)
    }
  }, [])

  return <div ref={ref} style={{ width: "100%", ...style }} className={className} />
}

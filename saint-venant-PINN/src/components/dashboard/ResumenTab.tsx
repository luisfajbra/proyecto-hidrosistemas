import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { MetricCard } from "@/components/MetricCard"
import { HydrographChart } from "@/components/charts/HydrographChart"
import { loadHydrograph, loadCalibrationMetrics } from "@/lib/data/loaders"
import { TRUE_PARAMS } from "@/lib/data/static"
import type { HydrographPoint, CalibrationMetric } from "@/lib/types"

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-4 border-b pb-1.5 text-[13px] font-normal text-muted-foreground">
      {children}
    </p>
  )
}

export function ResumenTab() {
  const [series, setSeries] = useState<HydrographPoint[] | null>(null)
  const [metrics, setMetrics] = useState<CalibrationMetric[] | null>(null)

  useEffect(() => {
    loadHydrograph("corta", "muskingum").then(setSeries).catch(console.error)
    loadCalibrationMetrics().then(setMetrics).catch(console.error)
  }, [])

  return (
    <div className="flex flex-col gap-8 pt-6">
      <section>
        <SectionLabel>Parámetros verdaderos del canal</SectionLabel>
        <div className="grid grid-cols-5 gap-8">
          <MetricCard label="Manning n" value={TRUE_PARAMS.n} />
          <MetricCard label="Pendiente S₀" value={TRUE_PARAMS.S0} />
          <MetricCard label="Caudal base Q₀" value={TRUE_PARAMS.Q0} unit="m³/s" />
          <MetricCard label="Ancho B_w" value={TRUE_PARAMS.B_w} unit="m" />
          <MetricCard label="Longitud L" value={TRUE_PARAMS.L.toLocaleString()} unit="m" />
        </div>
      </section>

      <section>
        <SectionLabel>Calibración OLS</SectionLabel>
        {metrics ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="py-1.5 pr-8 text-left text-[11px] font-normal text-muted-foreground">
                  Periodo
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  NSE
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  KGE
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  RMSE (m³/s)
                </th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.periodo} className="border-b last:border-0">
                  <td className="py-2 pr-8 capitalize text-muted-foreground">{m.periodo}</td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">{m.NSE.toFixed(3)}</td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">{m.KGE.toFixed(3)}</td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums">{m.RMSE_m3s.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Skeleton className="h-16" />
        )}
      </section>

      <section>
        <SectionLabel>Hidrograma — serie corta Muskingum</SectionLabel>
        {series ? (
          <HydrographChart data={series} />
        ) : (
          <Skeleton className="h-[360px]" />
        )}
      </section>
    </div>
  )
}

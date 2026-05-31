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
    loadHydrograph("corta", "muskingum").then(setSeries).catch(console.error)
    loadCalibrationMetrics().then(setMetrics).catch(console.error)
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

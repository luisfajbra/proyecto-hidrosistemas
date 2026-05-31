import { Fragment, useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { MetricCard } from "@/components/MetricCard"
import { LossCurveChart } from "@/components/charts/LossCurveChart"
import { HydrographPINNChart } from "@/components/charts/HydrographPINNChart"
import { loadPinnLossHistory, loadPinnHydrograph } from "@/lib/data/loaders"
import { PINN_CONFIG, TRUE_PARAMS } from "@/lib/data/static"
import type { PinnLossPoint, PinnHydrographPoint } from "@/lib/types"

const CONFIG_ROWS: { label: string; value: string | number }[] = [
  { label: "Activación", value: PINN_CONFIG.activation },
  { label: "Normalización", value: PINN_CONFIG.normalization },
  { label: "Arquitectura", value: `${PINN_CONFIG.n_layers} capas × ${PINN_CONFIG.hidden_size}` },
  { label: "Épocas Adam", value: PINN_CONFIG.N_EPOCHS_ADAM.toLocaleString("es") },
  { label: "Épocas warmup", value: PINN_CONFIG.N_EPOCHS_WARMUP.toLocaleString("es") },
  { label: "Épocas ramp λ_pde", value: PINN_CONFIG.N_EPOCHS_RAMP.toLocaleString("es") },
  { label: "Iter. L-BFGS", value: PINN_CONFIG.N_ITER_LBFGS.toLocaleString("es") },
  { label: "λ_data", value: PINN_CONFIG.LAMBDA_DATA },
  { label: "λ_pde", value: PINN_CONFIG.LAMBDA_PDE },
  { label: "Grad. clipping max_norm", value: PINN_CONFIG.gradient_clip_max_norm },
  { label: "Puntos de colocación", value: PINN_CONFIG.N_COLLOC.toLocaleString("es") },
]

const PENDING_LABEL = "Pendiente — ejecutar notebook 04"

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-4 border-b pb-1.5 text-[13px] font-normal text-muted-foreground">
      {children}
    </p>
  )
}

function PendingPlaceholder() {
  return (
    <div className="flex h-[360px] items-center justify-center border border-dashed">
      <p className="text-sm text-muted-foreground">{PENDING_LABEL}</p>
    </div>
  )
}

export function PinnTab() {
  const [lossData, setLossData] = useState<PinnLossPoint[] | null | undefined>(undefined)
  const [hydroData, setHydroData] = useState<PinnHydrographPoint[] | null | undefined>(undefined)

  useEffect(() => {
    loadPinnLossHistory().then(setLossData).catch(() => setLossData(null))
    loadPinnHydrograph().then(setHydroData).catch(() => setHydroData(null))
  }, [])

  return (
    <div className="flex flex-col gap-8 pt-6">
      <section>
        <SectionLabel>Configuración del modelo</SectionLabel>
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1fr_240px]">
          <dl className="grid grid-cols-[1fr_auto] gap-x-10 gap-y-1.5">
            {CONFIG_ROWS.map((r) => (
              <Fragment key={r.label}>
                <dt className="text-[13px] text-muted-foreground">{r.label}</dt>
                <dd className="text-right font-mono text-[13px]">{r.value}</dd>
              </Fragment>
            ))}
          </dl>
          <div className="flex flex-col gap-6">
            <MetricCard label="n verdadero" value={TRUE_PARAMS.n} />
            <MetricCard label="B_w verdadero" value={TRUE_PARAMS.B_w} unit="m" />
            <MetricCard label="S₀ verdadero" value={TRUE_PARAMS.S0} />
            <MetricCard label="Longitud L" value={TRUE_PARAMS.L.toLocaleString("es")} unit="m" />
          </div>
        </div>
      </section>

      <section>
        <SectionLabel>Curva de pérdidas — fase Adam</SectionLabel>
        {lossData === undefined ? (
          <Skeleton className="h-[360px]" />
        ) : lossData === null ? (
          <PendingPlaceholder />
        ) : (
          <LossCurveChart data={lossData} />
        )}
      </section>

      <section>
        <SectionLabel>Hidrograma — Q_obs vs Q_PINN</SectionLabel>
        {hydroData === undefined ? (
          <Skeleton className="h-[360px]" />
        ) : hydroData === null ? (
          <PendingPlaceholder />
        ) : (
          <HydrographPINNChart data={hydroData} />
        )}
      </section>
    </div>
  )
}

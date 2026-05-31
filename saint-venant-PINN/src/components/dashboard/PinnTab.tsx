import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableRow,
} from "@/components/ui/table"
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

function PendingPlaceholder() {
  return (
    <div className="flex h-[360px] items-center justify-center rounded-lg border border-dashed">
      <p className="text-sm text-muted-foreground">{PENDING_LABEL}</p>
    </div>
  )
}

export function PinnTab() {
  const [lossData, setLossData] = useState<PinnLossPoint[] | null | undefined>(undefined)
  const [hydroData, setHydroData] = useState<PinnHydrographPoint[] | null | undefined>(undefined)

  useEffect(() => {
    loadPinnLossHistory().then(setLossData)
    loadPinnHydrograph().then(setHydroData)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Configuración del modelo
        </h2>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-md border">
            <Table>
              <TableBody>
                {CONFIG_ROWS.map((r) => (
                  <TableRow key={r.label}>
                    <TableCell className="text-muted-foreground">{r.label}</TableCell>
                    <TableCell className="text-right font-mono font-medium">
                      {r.value}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div className="grid grid-cols-2 gap-4 content-start">
            <MetricCard label="n verdadero" value={TRUE_PARAMS.n} />
            <MetricCard label="B_w verdadero" value={TRUE_PARAMS.B_w} unit="m" />
            <MetricCard label="S₀ verdadero" value={TRUE_PARAMS.S0} />
            <MetricCard label="Longitud L" value={TRUE_PARAMS.L.toLocaleString("es")} unit="m" />
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Curva de pérdidas — fase Adam
        </h2>
        {lossData === undefined ? (
          <Skeleton className="h-[360px] rounded-lg" />
        ) : lossData === null ? (
          <PendingPlaceholder />
        ) : (
          <LossCurveChart data={lossData} />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Hidrograma — Q_obs vs Q_PINN
        </h2>
        {hydroData === undefined ? (
          <Skeleton className="h-[360px] rounded-lg" />
        ) : hydroData === null ? (
          <PendingPlaceholder />
        ) : (
          <HydrographPINNChart data={hydroData} />
        )}
      </section>
    </div>
  )
}

import { useEffect, useState } from "react"
import { AlertCircle } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { MetricCard } from "@/components/MetricCard"
import {
  loadParameterEstimates,
  loadCalibrationMetrics,
  loadSuposicionesErrores,
} from "@/lib/data/loaders"
import type { ParameterEstimate, CalibrationMetric } from "@/lib/types"

function extractRevisarLines(text: string): string[] {
  return text
    .split("\n")
    .filter((line) => line.includes("-> REVISAR"))
    .map((line) => line.trim())
}

export function CalibracionTab() {
  const [params, setParams] = useState<ParameterEstimate[] | null>(null)
  const [metrics, setMetrics] = useState<CalibrationMetric[] | null>(null)
  const [alertas, setAlertas] = useState<string[]>([])

  useEffect(() => {
    loadParameterEstimates().then(setParams).catch(console.error)
    loadCalibrationMetrics().then(setMetrics).catch(console.error)
    loadSuposicionesErrores()
      .then((text) => setAlertas(extractRevisarLines(text)))
      .catch(console.error)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      {alertas.length > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Supuestos a revisar
          </h2>
          {alertas.map((msg, i) => (
            <Alert key={i} variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Revisar</AlertTitle>
              <AlertDescription className="font-mono text-xs">{msg}</AlertDescription>
            </Alert>
          ))}
        </section>
      )}

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Parámetros estimados OLS
        </h2>
        {params ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parámetro</TableHead>
                <TableHead className="text-right">Verdadero</TableHead>
                <TableHead className="text-right">OLS</TableHead>
                <TableHead className="text-right">SE</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {params.map((row) => (
                <TableRow key={row.parametro}>
                  <TableCell className="font-mono">{row.parametro}</TableCell>
                  <TableCell className="text-right tabular-nums">{row.verdadero}</TableCell>
                  <TableCell className="text-right tabular-nums">{row.ols.toFixed(6)}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.SE !== null ? row.SE.toFixed(6) : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Skeleton className="h-36 rounded-lg" />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Métricas de ajuste
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
          <Skeleton className="h-36 rounded-lg" />
        )}
      </section>
    </div>
  )
}

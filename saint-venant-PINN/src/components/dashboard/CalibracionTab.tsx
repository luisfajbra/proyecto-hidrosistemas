import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  loadParameterEstimates,
  loadCalibrationMetrics,
  loadSuposicionesErrores,
} from "@/lib/data/loaders"
import type { ParameterEstimate, CalibrationMetric } from "@/lib/types"

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-4 border-b pb-1.5 text-[13px] font-normal text-muted-foreground">
      {children}
    </p>
  )
}

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
    <div className="flex flex-col gap-8 pt-6">
      {alertas.length > 0 && (
        <section>
          <SectionLabel>Supuestos a revisar</SectionLabel>
          <div className="flex flex-col gap-2">
            {alertas.map((msg, i) => (
              <p
                key={i}
                className="border-l-2 border-destructive/40 pl-3 font-mono text-xs text-destructive/70"
              >
                {msg}
              </p>
            ))}
          </div>
        </section>
      )}

      <section>
        <SectionLabel>Parámetros estimados OLS</SectionLabel>
        {params ? (
          <Table>
            <TableHeader>
              <TableRow className="border-b">
                <TableHead className="text-[11px] font-normal text-muted-foreground">
                  Parámetro
                </TableHead>
                <TableHead className="text-right text-[11px] font-normal text-muted-foreground">
                  Verdadero
                </TableHead>
                <TableHead className="text-right text-[11px] font-normal text-muted-foreground">
                  OLS
                </TableHead>
                <TableHead className="text-right text-[11px] font-normal text-muted-foreground">
                  SE
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {params.map((row) => (
                <TableRow key={row.parametro} className="border-b last:border-0">
                  <TableCell className="font-mono py-2">{row.parametro}</TableCell>
                  <TableCell className="py-2 text-right tabular-nums">{row.verdadero}</TableCell>
                  <TableCell className="py-2 text-right font-mono tabular-nums">
                    {row.ols.toFixed(6)}
                  </TableCell>
                  <TableCell className="py-2 text-right font-mono tabular-nums text-muted-foreground">
                    {row.SE !== null ? row.SE.toFixed(6) : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Skeleton className="h-24" />
        )}
      </section>

      <section>
        <SectionLabel>Métricas de ajuste</SectionLabel>
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
                  <td className="px-4 py-2 text-right font-mono tabular-nums">
                    {m.RMSE_m3s.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Skeleton className="h-16" />
        )}
      </section>
    </div>
  )
}

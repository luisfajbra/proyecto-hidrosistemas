import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { SensitivityChart } from "@/components/charts/SensitivityChart"
import { loadSobolIndices, loadSensibilidadConclusiones } from "@/lib/data/loaders"
import type { SobolIndex } from "@/lib/types"

const INFLUYENTE_THRESHOLD = 0.052

export function SensibilidadTab() {
  const [indices, setIndices] = useState<SobolIndex[] | null>(null)
  const [conclusiones, setConclusiones] = useState<string | null>(null)

  useEffect(() => {
    loadSobolIndices().then(setIndices)
    loadSensibilidadConclusiones().then(setConclusiones)
  }, [])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Índices de Sobol — análisis de sensibilidad global
        </h2>
        {indices ? (
          <SensitivityChart data={indices} />
        ) : (
          <Skeleton className="h-[360px] rounded-lg" />
        )}
      </section>

      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Tabla de índices
        </h2>
        {indices ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Parámetro</TableHead>
                <TableHead className="text-right">S1</TableHead>
                <TableHead className="text-right">±conf S1</TableHead>
                <TableHead className="text-right">ST</TableHead>
                <TableHead className="text-right">±conf ST</TableHead>
                <TableHead>Clasificación</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {indices.map((row) => (
                <TableRow key={row.parametro}>
                  <TableCell className="font-mono">{row.parametro}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.S1.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.S1_conf.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.ST.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {row.ST_conf.toFixed(4)}
                  </TableCell>
                  <TableCell>
                    {row.ST >= INFLUYENTE_THRESHOLD ? (
                      <Badge>Influyente</Badge>
                    ) : (
                      <Badge variant="secondary">Poco sensible</Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Skeleton className="h-40 rounded-lg" />
        )}
      </section>

      {conclusiones && (
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Conclusiones
          </h2>
          <pre className="rounded-md bg-muted p-4 font-mono text-xs text-muted-foreground whitespace-pre-wrap">
            {conclusiones}
          </pre>
        </section>
      )}
    </div>
  )
}

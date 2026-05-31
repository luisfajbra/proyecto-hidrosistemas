import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { SensitivityChart } from "@/components/charts/SensitivityChart"
import { loadSobolIndices, loadSensibilidadConclusiones } from "@/lib/data/loaders"
import type { SobolIndex } from "@/lib/types"

const INFLUYENTE_THRESHOLD = 0.052

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-4 border-b pb-1.5 text-[13px] font-normal text-muted-foreground">
      {children}
    </p>
  )
}

export function SensibilidadTab() {
  const [indices, setIndices] = useState<SobolIndex[] | null>(null)
  const [conclusiones, setConclusiones] = useState<string | null>(null)

  useEffect(() => {
    loadSobolIndices().then(setIndices).catch(console.error)
    loadSensibilidadConclusiones().then(setConclusiones).catch(console.error)
  }, [])

  return (
    <div className="flex flex-col gap-8 pt-6">
      <section>
        <SectionLabel>Índices de Sobol — sensibilidad global</SectionLabel>
        {indices ? (
          <SensitivityChart data={indices} />
        ) : (
          <Skeleton className="h-[360px]" />
        )}
      </section>

      <section>
        <SectionLabel>Índices por parámetro</SectionLabel>
        {indices ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="py-1.5 pr-8 text-left text-[11px] font-normal text-muted-foreground">
                  Parámetro
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  S1
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  ±conf
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  ST
                </th>
                <th className="px-4 py-1.5 text-right text-[11px] font-normal text-muted-foreground">
                  ±conf
                </th>
                <th className="py-1.5 pl-6 text-left text-[11px] font-normal text-muted-foreground">
                  Influencia
                </th>
              </tr>
            </thead>
            <tbody>
              {indices.map((row) => {
                const influyente = row.ST >= INFLUYENTE_THRESHOLD
                return (
                  <tr key={row.parametro} className="border-b last:border-0">
                    <td className="py-2 pr-8 font-mono">{row.parametro}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {row.S1.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {row.S1_conf.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {row.ST.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums text-muted-foreground">
                      {row.ST_conf.toFixed(4)}
                    </td>
                    <td className="py-2 pl-6">
                      <span
                        className={
                          influyente ? "text-xs font-medium" : "text-xs text-muted-foreground"
                        }
                      >
                        {influyente ? "influyente" : "—"}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : (
          <Skeleton className="h-32" />
        )}
      </section>

      {conclusiones && (
        <section>
          <SectionLabel>Notas</SectionLabel>
          <pre className="font-mono text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed">
            {conclusiones}
          </pre>
        </section>
      )}
    </div>
  )
}

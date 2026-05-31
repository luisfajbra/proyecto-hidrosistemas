import { useEffect, useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { DatasetSelector } from "@/components/DatasetSelector"
import { HydrographChart } from "@/components/charts/HydrographChart"
import { loadHydrograph } from "@/lib/data/loaders"
import type { HydrographPoint, SerieLength, SerieType } from "@/lib/types"

export function HidrogramasTab() {
  const [length, setLength] = useState<SerieLength>("corta")
  const [type, setType] = useState<SerieType>("muskingum")
  const [data, setData] = useState<HydrographPoint[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setData(null)
    setError(null)
    loadHydrograph(length, type)
      .then((d) => { if (!cancelled) setData(d) })
      .catch(() => { if (!cancelled) setError("No se pudo cargar la serie seleccionada.") })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [length, type])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-1.5">
        <p className="text-[13px] font-normal text-muted-foreground">Hidrogramas sintéticos</p>
        <DatasetSelector
          length={length}
          type={type}
          onLengthChange={setLength}
          onTypeChange={setType}
        />
      </div>

      {loading ? (
        <Skeleton className="h-[400px]" />
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : data ? (
        <HydrographChart data={data} showDepth height={400} />
      ) : null}
    </div>
  )
}

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
    setLoading(true)
    setData(null)
    setError(null)
    loadHydrograph(length, type)
      .then(setData)
      .catch(() => setError("No se pudo cargar la serie seleccionada."))
      .finally(() => setLoading(false))
  }, [length, type])

  return (
    <div className="flex flex-col gap-6 pt-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Hidrogramas sintéticos
        </h2>
        <DatasetSelector
          length={length}
          type={type}
          onLengthChange={setLength}
          onTypeChange={setType}
        />
      </div>

      {loading ? (
        <Skeleton className="h-[400px] rounded-lg" />
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : data ? (
        <HydrographChart data={data} showDepth height={400} />
      ) : null}
    </div>
  )
}

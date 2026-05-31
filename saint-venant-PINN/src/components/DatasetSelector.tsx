import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { SerieLength, SerieType } from "@/lib/types"

interface DatasetSelectorProps {
  length: SerieLength
  type: SerieType
  onLengthChange: (v: SerieLength) => void
  onTypeChange: (v: SerieType) => void
}

export function DatasetSelector({
  length,
  type,
  onLengthChange,
  onTypeChange,
}: DatasetSelectorProps) {
  return (
    <div className="flex gap-3">
      <Select value={length} onValueChange={(v) => onLengthChange(v as SerieLength)}>
        <SelectTrigger className="w-36">
          <SelectValue placeholder="Longitud" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="corta">Corta (~2 h)</SelectItem>
          <SelectItem value="larga">Larga (~833 h)</SelectItem>
        </SelectContent>
      </Select>

      <Select value={type} onValueChange={(v) => onTypeChange(v as SerieType)}>
        <SelectTrigger className="w-44">
          <SelectValue placeholder="Tipo" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="muskingum">Muskingum</SelectItem>
          <SelectItem value="ruido">Con ruido</SelectItem>
          <SelectItem value="shift">Con shift</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

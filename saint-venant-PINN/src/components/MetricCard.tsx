interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  description?: string
}

export function MetricCard({ label, value, unit, description }: MetricCardProps) {
  return (
    <div>
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <p className="mt-0.5 font-mono text-[22px] font-medium tabular-nums leading-none">
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-muted-foreground">{unit}</span>}
      </p>
      {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
    </div>
  )
}

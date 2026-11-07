interface Props {
  rate: number
}

export default function EngagementBadge({ rate }: Props) {
  return (
    <span className="bg-white/10 px-2 py-0.5 rounded text-white text-sm font-mono">
      {rate.toFixed(2)}%
    </span>
  )
}

import { Link } from 'react-router-dom'
import VideoThumbnail from './VideoThumbnail'
import EngagementBadge from './EngagementBadge'

interface Props {
  id: number
  label: string
  avgEngagement: number
  thumbnailUrl?: string | null
  videoCount: number
}

export default function FormatCard({ id, label, avgEngagement, thumbnailUrl, videoCount }: Props) {
  return (
    <Link to={`/format/${id}`} className="flex flex-col gap-2 hover:opacity-80 transition-opacity">
      <VideoThumbnail src={thumbnailUrl} alt={label} />
      <div>
        <p className="text-white font-medium text-sm">{label}</p>
        <div className="flex items-center gap-2 mt-1">
          <EngagementBadge rate={avgEngagement} />
          <span className="text-neutral-400 text-xs">{videoCount} videos</span>
        </div>
      </div>
    </Link>
  )
}

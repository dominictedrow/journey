import { VIDEO_SIZE } from '../../constants/styles'

interface Props {
  src?: string | null
  alt?: string
}

export default function VideoThumbnail({ src, alt = '' }: Props) {
  return (
    <div
      style={{ width: VIDEO_SIZE.width, height: VIDEO_SIZE.height }}
      className="bg-neutral-800 rounded-lg overflow-hidden flex-shrink-0"
    >
      {src ? (
        <img src={src} alt={alt} className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-neutral-600 text-xs">
          No thumbnail
        </div>
      )}
    </div>
  )
}

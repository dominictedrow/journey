import { Link } from 'react-router-dom'
import { IG_GRADIENT } from '../../constants/styles'

interface Props {
  open: boolean
  onClose: () => void
}

const links = [
  { to: '/home', label: 'Home' },
  { to: '/library', label: 'Library' },
  { to: '/scan', label: 'Scan' },
  { to: '/profile', label: 'Profile' },
]

export default function Sidebar({ open, onClose }: Props) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-20 flex">
      <div className="w-64 bg-neutral-900 h-full flex flex-col p-6 gap-4">
        <div className="flex justify-between items-center mb-4">
          <span
            className="text-xl font-bold bg-clip-text text-transparent"
            style={{ backgroundImage: IG_GRADIENT }}
          >
            Instana
          </span>
          <button onClick={onClose} className="text-white text-xl leading-none hover:opacity-70">
            ✕
          </button>
        </div>
        {links.map((l) => (
          <Link
            key={l.to}
            to={l.to}
            onClick={onClose}
            className="text-white text-lg hover:opacity-70 transition-opacity"
          >
            {l.label}
          </Link>
        ))}
      </div>
      <div className="flex-1" onClick={onClose} />
    </div>
  )
}

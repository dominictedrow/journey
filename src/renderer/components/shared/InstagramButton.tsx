import { ButtonHTMLAttributes } from 'react'
import { IG_GRADIENT } from '../../constants/styles'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
}

export default function InstagramButton({ children, className = '', style, ...props }: Props) {
  return (
    <button
      {...props}
      style={{ background: IG_GRADIENT, ...style }}
      className={`px-6 py-3 rounded-full text-white font-semibold disabled:opacity-50 ${className}`}
    >
      {children}
    </button>
  )
}

import { InputHTMLAttributes } from 'react'
import { TEXT_INPUT_CLASS } from '../../constants/styles'

export default function TextInput({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`${TEXT_INPUT_CLASS} outline-none w-full ${className}`}
    />
  )
}

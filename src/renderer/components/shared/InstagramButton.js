import { jsx as _jsx } from "react/jsx-runtime";
import { IG_GRADIENT } from '../../constants/styles';
export default function InstagramButton({ children, className = '', style, ...props }) {
    return (_jsx("button", { ...props, style: { background: IG_GRADIENT, ...style }, className: `px-6 py-3 rounded-full text-white font-semibold disabled:opacity-50 ${className}`, children: children }));
}

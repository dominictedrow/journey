import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link } from 'react-router-dom';
import { IG_GRADIENT } from '../../constants/styles';
const links = [
    { to: '/home', label: 'Home' },
    { to: '/library', label: 'Library' },
    { to: '/scan', label: 'Scan' },
    { to: '/profile', label: 'Profile' },
];
export default function Sidebar({ open, onClose }) {
    if (!open)
        return null;
    return (_jsxs("div", { className: "fixed inset-0 z-20 flex", children: [_jsxs("div", { className: "w-64 bg-neutral-900 h-full flex flex-col p-6 gap-4", children: [_jsxs("div", { className: "flex justify-between items-center mb-4", children: [_jsx("span", { className: "text-xl font-bold bg-clip-text text-transparent", style: { backgroundImage: IG_GRADIENT }, children: "Instana" }), _jsx("button", { onClick: onClose, className: "text-white text-xl leading-none hover:opacity-70", children: "\u2715" })] }), links.map((l) => (_jsx(Link, { to: l.to, onClick: onClose, className: "text-white text-lg hover:opacity-70 transition-opacity", children: l.label }, l.to)))] }), _jsx("div", { className: "flex-1", onClick: onClose })] }));
}

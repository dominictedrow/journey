import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function Modal({ open, onClose, children }) {
    if (!open)
        return null;
    return (_jsxs("div", { className: "fixed inset-0 z-50 flex items-center justify-center", children: [_jsx("div", { className: "absolute inset-0 bg-black/70", onClick: onClose }), _jsx("div", { className: "relative bg-neutral-900 rounded-xl p-6 max-w-md w-full mx-4", children: children })] }));
}

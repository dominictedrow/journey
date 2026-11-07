import { jsx as _jsx } from "react/jsx-runtime";
import { useNavigate } from 'react-router-dom';
export default function BackButton() {
    const navigate = useNavigate();
    return (_jsx("button", { onClick: () => navigate(-1), className: "text-white text-sm opacity-70 hover:opacity-100 transition-opacity flex items-center gap-1", children: "\u2190 Back" }));
}

import { jsxs as _jsxs } from "react/jsx-runtime";
export default function EngagementBadge({ rate }) {
    return (_jsxs("span", { className: "bg-white/10 px-2 py-0.5 rounded text-white text-sm font-mono", children: [rate.toFixed(2), "%"] }));
}

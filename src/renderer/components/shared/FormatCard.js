import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link } from 'react-router-dom';
import VideoThumbnail from './VideoThumbnail';
import EngagementBadge from './EngagementBadge';
export default function FormatCard({ id, label, avgEngagement, thumbnailUrl, videoCount }) {
    return (_jsxs(Link, { to: `/format/${id}`, className: "flex flex-col gap-2 hover:opacity-80 transition-opacity", children: [_jsx(VideoThumbnail, { src: thumbnailUrl, alt: label }), _jsxs("div", { children: [_jsx("p", { className: "text-white font-medium text-sm", children: label }), _jsxs("div", { className: "flex items-center gap-2 mt-1", children: [_jsx(EngagementBadge, { rate: avgEngagement }), _jsxs("span", { className: "text-neutral-400 text-xs", children: [videoCount, " videos"] })] })] })] }));
}

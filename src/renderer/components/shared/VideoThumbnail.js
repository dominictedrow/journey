import { jsx as _jsx } from "react/jsx-runtime";
import { VIDEO_SIZE } from '../../constants/styles';
export default function VideoThumbnail({ src, alt = '' }) {
    return (_jsx("div", { style: { width: VIDEO_SIZE.width, height: VIDEO_SIZE.height }, className: "bg-neutral-800 rounded-lg overflow-hidden flex-shrink-0", children: src ? (_jsx("img", { src: src, alt: alt, className: "w-full h-full object-cover" })) : (_jsx("div", { className: "w-full h-full flex items-center justify-center text-neutral-600 text-xs", children: "No thumbnail" })) }));
}

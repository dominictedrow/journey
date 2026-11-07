import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import BackButton from '../components/layout/BackButton';
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles';
import VideoThumbnail from '../components/shared/VideoThumbnail';
import EngagementBadge from '../components/shared/EngagementBadge';
import { apiFetch } from '../api/client';
const CHAR_LABELS = {
    footageType: 'Footage Type',
    backgroundType: 'Background',
    filmingStyle: 'Filming Style',
    lighting: 'Lighting',
    setting: 'Setting',
    audioType: 'Audio',
    hookStrength: 'Hook Strength',
    productionQuality: 'Production Quality',
    cutsPerSecond: 'Cuts / sec',
};
const HOOK_COLOR = {
    strong: 'text-green-400',
    medium: 'text-yellow-400',
    weak: 'text-red-400',
};
export default function FormatDescriptionPage() {
    const { id } = useParams();
    const formatId = parseInt(id ?? '');
    const [format, setFormat] = useState(null);
    const [insights, setInsights] = useState(null);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        if (!id)
            return;
        apiFetch(`/api/formats/${id}`)
            .then(setFormat)
            .catch(() => setFormat(null))
            .finally(() => setLoading(false));
        apiFetch('/api/analytics/insights').then(setInsights).catch(() => { });
    }, [id]);
    if (loading) {
        return (_jsxs("div", { className: "p-8 pt-16", children: [_jsx(BackButton, {}), _jsx("p", { className: "text-neutral-500 text-sm mt-8", children: "Loading\u2026" })] }));
    }
    if (!format) {
        return (_jsxs("div", { className: "p-8 pt-16", children: [_jsx(BackButton, {}), _jsx("p", { className: "text-neutral-500 text-sm mt-8", children: "Format not found." })] }));
    }
    const chars = format.characteristics;
    const isBest = insights?.bestFormat.id === formatId;
    const isWorst = insights?.worstFormat.id === formatId;
    const weaknesses = isWorst ? (insights?.worstFormatWeaknesses ?? []) : [];
    const recommendations = insights?.overallRecommendations ?? [];
    const hasAiContent = weaknesses.length > 0 || recommendations.length > 0;
    return (_jsxs("div", { className: "p-8 pt-16", children: [_jsx(BackButton, {}), _jsxs("div", { className: "grid grid-cols-3 gap-8 mt-4", children: [_jsxs("div", { className: "col-span-1 flex flex-col gap-4", children: [_jsx(VideoThumbnail, { src: format.thumbnailUrl, alt: format.label }), _jsxs("div", { className: "flex items-center gap-3 flex-wrap", children: [_jsx(EngagementBadge, { rate: format.avgEngagement }), _jsxs("span", { className: "text-neutral-400 text-sm", children: [format.videoCount, " videos"] }), isBest && (_jsx("span", { className: "text-xs px-2 py-0.5 rounded-full text-white font-medium", style: { background: IG_GRADIENT }, children: "Top performer" })), isWorst && (_jsx("span", { className: "text-xs px-2 py-0.5 rounded-full bg-red-900/40 text-red-400", children: "Needs work" }))] }), _jsx("div", { className: "h-px w-full rounded-full", style: { background: IG_GRADIENT } }), _jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("p", { className: "text-neutral-500 text-xs uppercase tracking-wider", children: "All videos" }), format.members.map((m) => (_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("img", { src: m.thumbnailUrl ?? undefined, alt: "", className: "w-10 h-[72px] object-cover rounded" }), _jsxs("div", { children: [m.engagementRate != null ? (_jsx(EngagementBadge, { rate: m.engagementRate })) : (_jsx("span", { className: "text-neutral-600 text-xs", children: "\u2014" })), m.timestamp && (_jsx("p", { className: "text-neutral-600 text-xs mt-1", children: new Date(m.timestamp).toLocaleDateString() }))] })] }, m.mediaId)))] })] }), _jsxs("div", { className: "col-span-2 flex flex-col gap-6", children: [_jsx("h1", { className: `${HEADING_CLASS} capitalize`, children: format.label }), _jsxs("div", { className: "bg-neutral-900 rounded-2xl p-6", children: [_jsx("p", { className: "text-white font-semibold mb-4", children: "Format Profile" }), _jsx("div", { className: "grid grid-cols-2 gap-x-8 gap-y-4", children: Object.entries(chars).map(([key, val]) => (_jsxs("div", { children: [_jsx("p", { className: "text-neutral-500 text-xs uppercase tracking-wider mb-1", children: CHAR_LABELS[key] ?? key }), _jsx("p", { className: `text-sm font-medium capitalize ${key === 'hookStrength' ? (HOOK_COLOR[String(val)] ?? 'text-white') : 'text-white'}`, children: typeof val === 'number' ? val.toFixed(1) : String(val).replace(/_/g, ' ') })] }, key))) })] }), _jsxs("div", { className: "bg-neutral-900 rounded-2xl p-6 flex flex-col gap-4", children: [_jsx("p", { className: "text-white font-semibold", children: "AI Analysis" }), !hasAiContent && (_jsx("p", { className: "text-neutral-500 text-sm", children: insights
                                            ? 'No AI analysis available — click Refresh Insights on the Home page'
                                            : 'Add a Grok API key in Profile to generate AI-powered analysis' })), weaknesses.length > 0 && (_jsxs("div", { children: [_jsx("p", { className: "text-neutral-400 text-xs uppercase tracking-wider mb-2", children: "Weaknesses" }), _jsx("div", { className: "flex flex-col gap-2", children: weaknesses.map((w, i) => (_jsxs(motion.div, { initial: { opacity: 0, x: -4 }, animate: { opacity: 1, x: 0 }, transition: { delay: i * 0.07 }, className: "flex gap-2 text-sm bg-red-950/30 border border-red-900/40 rounded-lg px-3 py-2", children: [_jsx("span", { className: "text-red-500 mt-0.5 shrink-0", children: "\u2715" }), _jsx("span", { className: "text-red-200", children: w })] }, i))) })] })), recommendations.length > 0 && (_jsxs("div", { children: [_jsx("p", { className: "text-neutral-400 text-xs uppercase tracking-wider mb-2", children: "Recommendations" }), _jsx("div", { className: "flex flex-col gap-2", children: recommendations.map((rec, i) => (_jsxs(motion.div, { initial: { opacity: 0, x: -4 }, animate: { opacity: 1, x: 0 }, transition: { delay: i * 0.07 }, className: "flex gap-2 text-sm text-neutral-300", children: [_jsx("span", { className: "text-neutral-500 mt-0.5 shrink-0", children: "\u203A" }), _jsx("span", { children: rec })] }, i))) })] }))] })] })] })] }));
}

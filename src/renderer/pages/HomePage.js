import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles';
import FormatCard from '../components/shared/FormatCard';
import EngagementBadge from '../components/shared/EngagementBadge';
import Modal from '../components/shared/Modal';
import { apiFetch } from '../api/client';
function useCountdown(targetIso) {
    const [remaining, setRemaining] = useState(null);
    useEffect(() => {
        if (!targetIso) {
            setRemaining(null);
            return;
        }
        const tick = () => setRemaining(Math.max(0, new Date(targetIso).getTime() - Date.now()));
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, [targetIso]);
    return remaining;
}
function formatCountdown(ms) {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
export default function HomePage() {
    const [formats, setFormats] = useState([]);
    const [insights, setInsights] = useState(null);
    const [scheduler, setScheduler] = useState(null);
    const [refreshing, setRefreshing] = useState(false);
    const [timerModal, setTimerModal] = useState(false);
    const [schedHour, setSchedHour] = useState(9);
    const [schedMinute, setSchedMinute] = useState(0);
    const [schedSaving, setSchedSaving] = useState(false);
    const remaining = useCountdown(scheduler?.nextRun ?? null);
    useEffect(() => {
        apiFetch('/api/formats').then(setFormats).catch(() => { });
        apiFetch('/api/analytics/insights').then(setInsights).catch(() => { });
        apiFetch('/api/scheduler/status')
            .then((s) => {
            setScheduler(s);
            setSchedHour(s.config.hour);
            setSchedMinute(s.config.minute);
        })
            .catch(() => { });
    }, []);
    async function handleRefreshInsights() {
        setRefreshing(true);
        try {
            const report = await apiFetch('/api/analytics/insights/refresh', { method: 'POST' });
            setInsights(report);
        }
        finally {
            setRefreshing(false);
        }
    }
    async function saveSchedule(enabled) {
        setSchedSaving(true);
        try {
            const status = await apiFetch('/api/scheduler/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled, hour: schedHour, minute: schedMinute }),
            });
            setScheduler(status);
            setTimerModal(false);
        }
        finally {
            setSchedSaving(false);
        }
    }
    const top5 = insights?.allFormats.slice(0, 5) ?? formats.slice(0, 5);
    const cards = formats.slice(0, 5);
    const timeValue = `${String(schedHour).padStart(2, '0')}:${String(schedMinute).padStart(2, '0')}`;
    return (_jsxs("div", { className: "p-8 pt-16", children: [_jsx("h1", { className: `${HEADING_CLASS} text-3xl mb-6`, children: "Analyze Insta!" }), _jsxs("button", { onClick: () => setTimerModal(true), className: "w-full rounded-2xl p-6 mb-8 flex flex-col items-center gap-1 hover:opacity-90 transition-opacity", style: { background: IG_GRADIENT }, children: [_jsx("p", { className: "text-white/70 text-xs uppercase tracking-widest", children: "Next scheduled scan" }), remaining != null ? (_jsx("p", { className: "text-white font-mono text-4xl font-bold", children: formatCountdown(remaining) })) : (_jsx("p", { className: "text-white font-mono text-2xl opacity-60", children: "-- : -- : --" })), _jsx("p", { className: "text-white/60 text-xs mt-0.5", children: scheduler?.config.enabled
                            ? `Daily at ${String(schedHour).padStart(2, '0')}:${String(schedMinute).padStart(2, '0')} — click to change`
                            : 'Click to set up daily scan' })] }), _jsxs("div", { className: "grid grid-cols-2 gap-8 mb-10", children: [_jsxs("div", { children: [_jsx("p", { className: "text-white font-semibold mb-4", children: "Top Formats" }), top5.length > 0 ? (_jsx("div", { className: "flex flex-col gap-1", children: top5.map((f, i) => (_jsxs(Link, { to: `/format/${f.id}`, className: "flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-white/5 transition-colors", children: [_jsx("span", { className: "text-neutral-500 font-mono text-sm w-5", children: i + 1 }), _jsx("span", { className: "text-white text-sm flex-1 capitalize", children: f.label }), _jsx(EngagementBadge, { rate: f.avgEngagement }), _jsxs("span", { className: "text-neutral-500 text-xs", children: [f.videoCount, "v"] })] }, f.id))) })) : (_jsx("p", { className: "text-neutral-500 text-sm", children: "No formats yet \u2014 run a scan first" }))] }), _jsxs("div", { children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsx("p", { className: "text-white font-semibold", children: "AI Insights" }), _jsx("button", { onClick: handleRefreshInsights, disabled: refreshing, className: "text-neutral-400 text-xs underline hover:text-white disabled:opacity-40", children: refreshing ? 'Generating…' : 'Refresh' })] }), insights ? (_jsxs("div", { className: "flex flex-col gap-3", children: [insights.overallRecommendations.map((rec, i) => (_jsxs(motion.div, { initial: { opacity: 0, y: 4 }, animate: { opacity: 1, y: 0 }, transition: { delay: i * 0.08 }, className: "flex gap-2 text-sm text-neutral-300", children: [_jsx("span", { className: "text-neutral-500 mt-0.5", children: "\u203A" }), _jsx("span", { children: rec })] }, i))), insights.overallRecommendations.length === 0 && (_jsx("p", { className: "text-neutral-500 text-sm", children: "Click Refresh to generate recommendations" })), _jsxs("p", { className: "text-neutral-600 text-xs mt-2", children: ["Generated ", new Date(insights.generatedAt).toLocaleDateString()] })] })) : (_jsx("p", { className: "text-neutral-500 text-sm", children: formats.length > 0 ? 'Loading insights…' : 'Run a scan to generate AI insights' }))] })] }), cards.length > 0 && (_jsxs(_Fragment, { children: [_jsx("p", { className: "text-white font-semibold mb-4", children: "Recent Formats" }), _jsx("div", { className: "grid grid-cols-5 gap-4", children: cards.map((f) => (_jsx(FormatCard, { id: f.id, label: f.label, avgEngagement: f.avgEngagement, thumbnailUrl: f.thumbnailUrl ?? null, videoCount: f.videoCount }, f.id))) })] })), _jsxs(Modal, { open: timerModal, onClose: () => setTimerModal(false), children: [_jsx("p", { className: "text-white font-semibold mb-5", children: "Daily Scan Schedule" }), _jsxs("div", { className: "flex flex-col gap-4", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("p", { className: "text-neutral-400 text-sm shrink-0", children: "Run daily at" }), _jsx("input", { type: "time", value: timeValue, onChange: (e) => {
                                            const [h, m] = e.target.value.split(':').map(Number);
                                            setSchedHour(h ?? 9);
                                            setSchedMinute(m ?? 0);
                                        }, className: "bg-neutral-800 text-white px-3 py-2 rounded border border-white/20 text-sm" })] }), _jsxs("div", { className: "flex gap-3 mt-2", children: [_jsx("button", { onClick: () => saveSchedule(true), disabled: schedSaving, className: "flex-1 py-2 rounded-lg text-white text-sm font-medium disabled:opacity-40", style: { background: IG_GRADIENT }, children: schedSaving ? 'Saving…' : 'Enable' }), scheduler?.config.enabled && (_jsx("button", { onClick: () => saveSchedule(false), disabled: schedSaving, className: "px-4 py-2 rounded-lg text-neutral-400 text-sm hover:text-white border border-white/10 hover:border-white/30 transition-colors disabled:opacity-40", children: "Disable" }))] })] })] })] }));
}

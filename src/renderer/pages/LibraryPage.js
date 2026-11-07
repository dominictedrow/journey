import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles';
import FormatCard from '../components/shared/FormatCard';
import TextInput from '../components/shared/TextInput';
import { apiFetch, apiBase } from '../api/client';
const STEP_LABELS = {
    fetching_media: 'Fetching media from Instagram',
    fetching_insights: 'Fetching engagement data',
    downloading: 'Downloading video',
    extracting_frames: 'Extracting frames',
    detecting_audio: 'Detecting audio',
    ocr: 'Reading text overlays',
    edit_patterns: 'Detecting edit cuts',
    grok_analysis: 'Analysing with AI',
    done: 'Video complete',
    skipped: 'Already analysed',
    error: 'Error — skipping',
};
export default function LibraryPage() {
    const [formats, setFormats] = useState([]);
    const [search, setSearch] = useState('');
    const [scanStatus, setScanStatus] = useState('idle');
    const [progress, setProgress] = useState(0);
    const [log, setLog] = useState([]);
    const loadFormats = useCallback(async () => {
        try {
            const data = await apiFetch('/api/formats');
            setFormats(data);
        }
        catch { /* ignore */ }
    }, []);
    useEffect(() => {
        loadFormats();
    }, [loadFormats]);
    function appendLog(msg) {
        setLog((prev) => [msg, ...prev].slice(0, 6));
    }
    async function handleScan() {
        setScanStatus('running');
        setProgress(0);
        setLog([]);
        try {
            const { jobId } = await apiFetch('/api/pipeline/run', { method: 'POST' });
            const source = new EventSource(`${apiBase}/api/pipeline/events/${jobId}`);
            source.onmessage = (e) => {
                const event = JSON.parse(e.data);
                const pct = event.total > 0 ? Math.round((event.current / event.total) * 100) : 0;
                setProgress(pct);
                const step = event.step ?? '';
                const label = step ? (STEP_LABELS[step] ?? step.replace(/_/g, ' ')) : '';
                if (step && step !== 'starting')
                    appendLog(label + (event.error ? ` — ${event.error}` : ''));
                if (event.status === 'done') {
                    source.close();
                    // Poll job status until the backend's rebuildFormats + generateInsights finish,
                    // then fetch the updated format list. activeJobs status becomes 'done' only
                    // after the full post-pipeline chain completes.
                    void pollJobDone(jobId);
                }
            };
            source.onerror = () => {
                source.close();
                setScanStatus('error');
            };
        }
        catch {
            setScanStatus('error');
        }
    }
    async function pollJobDone(jobId) {
        const deadline = Date.now() + 120_000;
        while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 1500));
            try {
                const job = await apiFetch(`/api/pipeline/status/${jobId}`);
                if (job.status === 'done' || job.status === 'error')
                    break;
            }
            catch {
                break;
            }
        }
        await loadFormats();
        setScanStatus('done');
    }
    const filtered = formats.filter((f) => f.label.toLowerCase().includes(search.toLowerCase()));
    return (_jsxs("div", { className: "p-8 pt-16", children: [_jsxs("div", { className: "flex items-center justify-between mb-6", children: [_jsx("h1", { className: HEADING_CLASS, children: "Library" }), scanStatus !== 'running' && (_jsx("button", { onClick: handleScan, className: "px-4 py-1.5 rounded-full text-white text-sm font-medium hover:opacity-80 transition-opacity", style: { background: IG_GRADIENT }, children: scanStatus === 'done' ? 'Scan Again' : 'Scan' }))] }), _jsx(AnimatePresence, { mode: "wait", children: scanStatus === 'running' ? (_jsxs(motion.div, { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, className: "flex flex-col gap-4 pt-4", children: [_jsxs("div", { className: "flex justify-between text-sm", children: [_jsx("span", { className: "text-neutral-400", children: "Analysing\u2026" }), _jsxs("span", { className: "text-white font-mono", children: [progress, "%"] })] }), _jsx("div", { className: "h-1 rounded-full bg-neutral-800 overflow-hidden", children: _jsx(motion.div, { className: "h-full rounded-full", style: { background: IG_GRADIENT }, animate: { width: `${progress}%` }, transition: { type: 'spring', stiffness: 60, damping: 20 } }) }), _jsx("div", { className: "flex flex-col gap-1 mt-2", children: _jsx(AnimatePresence, { initial: false, children: log.map((msg, i) => (_jsx(motion.p, { initial: { opacity: 0, y: -6 }, animate: { opacity: Math.max(0.2, 1 - i * 0.2), y: 0 }, exit: { opacity: 0, height: 0 }, className: "text-xs text-neutral-400", children: msg }, msg + i))) }) })] }, "scanning")) : (_jsxs(motion.div, { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, children: [scanStatus === 'done' && (_jsx(motion.p, { initial: { opacity: 0, y: -4 }, animate: { opacity: 1, y: 0 }, className: "text-green-400 text-sm mb-4", children: "\u2713 Scan complete" })), scanStatus === 'error' && (_jsx("p", { className: "text-red-400 text-sm mb-4", children: "Scan failed \u2014 check logs and try again" })), _jsx(TextInput, { value: search, onChange: (e) => setSearch(e.target.value), placeholder: "Search formats\u2026", className: "mb-6" }), filtered.length === 0 ? (_jsx("div", { className: "text-center py-20", children: _jsx("p", { className: "text-neutral-500 text-sm", children: formats.length === 0
                                    ? 'No formats yet — hit Scan to analyse your reels.'
                                    : 'No formats match your search.' }) })) : (_jsx("div", { className: "grid grid-cols-4 gap-4", children: filtered.map((f) => (_jsx(FormatCard, { id: f.id, label: f.label, avgEngagement: f.avgEngagement, thumbnailUrl: f.thumbnailUrl, videoCount: f.videoCount }, f.id))) }))] }, "library")) })] }));
}

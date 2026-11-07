import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import BackButton from '../components/layout/BackButton';
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles';
import InstagramButton from '../components/shared/InstagramButton';
import { apiFetch, apiBase } from '../api/client';
import { usePipelineStore } from '../store/pipelineStore';
const STEP_LABELS = {
    fetching_media: 'Fetching media from Instagram',
    fetching_insights: 'Fetching engagement data',
    downloading: 'Downloading video',
    extracting_frames: 'Extracting frames & detecting audio',
    ocr: 'Reading text overlays & edit cuts',
    grok_analysis: 'Analysing with AI',
    done: 'Video complete',
    skipped: 'Already analysed',
    error: 'Error — skipping',
};
export default function ScanningPage() {
    const { status, progress, setJob, setProgress, setStatus, reset } = usePipelineStore();
    const [log, setLog] = useState([]);
    function appendLog(msg) {
        setLog((prev) => [msg, ...prev].slice(0, 6));
    }
    async function handleScan() {
        reset();
        setLog([]);
        const { jobId } = await apiFetch('/api/pipeline/run', { method: 'POST' });
        setJob(jobId);
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
                setStatus('done');
            }
        };
        source.onerror = () => {
            source.close();
            setStatus('error');
        };
    }
    return (_jsxs("div", { className: "p-8 pt-16", children: [_jsx(BackButton, {}), _jsx("h1", { className: `${HEADING_CLASS} mt-4 mb-8`, children: "Scan" }), _jsx("div", { className: "w-full max-w-lg mx-auto bg-neutral-900 rounded-2xl p-8 flex flex-col items-center gap-6 min-h-64", children: _jsxs(AnimatePresence, { mode: "wait", children: [status === 'idle' && (_jsxs(motion.div, { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, className: "flex flex-col items-center gap-4 py-8", children: [_jsx("p", { className: "text-neutral-400 text-sm text-center", children: "Analyse your videos with AI \u2014 no Grok key needed to get started" }), _jsx(InstagramButton, { onClick: handleScan, children: "Start Scan" })] }, "idle")), status === 'running' && (_jsxs(motion.div, { initial: { opacity: 0 }, animate: { opacity: 1 }, className: "w-full flex flex-col gap-4", children: [_jsxs("div", { className: "flex justify-between text-sm", children: [_jsx("span", { className: "text-neutral-400", children: "Analysing\u2026" }), _jsxs("span", { className: "text-white font-mono", children: [progress, "%"] })] }), _jsx("div", { className: "h-1 rounded-full bg-neutral-800 overflow-hidden", children: _jsx(motion.div, { className: "h-full rounded-full", style: { background: IG_GRADIENT }, animate: { width: `${progress}%` }, transition: { type: 'spring', stiffness: 60, damping: 20 } }) }), _jsx("div", { className: "flex flex-col gap-1 mt-2", children: _jsx(AnimatePresence, { initial: false, children: log.map((msg, i) => (_jsx(motion.p, { initial: { opacity: 0, y: -6 }, animate: { opacity: Math.max(0.2, 1 - i * 0.2), y: 0 }, exit: { opacity: 0, height: 0 }, className: "text-xs text-neutral-400", children: msg }, msg + i))) }) })] }, "running")), status === 'done' && (_jsxs(motion.div, { initial: { opacity: 0, scale: 0.95 }, animate: { opacity: 1, scale: 1 }, className: "flex flex-col items-center gap-4 py-8 text-center", children: [_jsx(motion.p, { initial: { y: -20, opacity: 0 }, animate: { y: 0, opacity: 1 }, transition: { delay: 0.1 }, className: "text-green-400 text-lg", children: "\u2713 Scan complete" }), _jsx("p", { className: "text-neutral-400 text-sm", children: "Head to Library to view your formats" }), _jsx(InstagramButton, { onClick: handleScan, children: "Scan Again" })] }, "done")), status === 'error' && (_jsxs(motion.div, { initial: { opacity: 0 }, animate: { opacity: 1 }, className: "flex flex-col items-center gap-4 py-8", children: [_jsx("p", { className: "text-red-400", children: "Scan failed" }), _jsx("button", { onClick: handleScan, className: "text-neutral-400 text-sm underline hover:text-white", children: "Try again" })] }, "error"))] }) })] }));
}

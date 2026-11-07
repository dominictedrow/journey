import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import BackButton from '../components/layout/BackButton'
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles'
import InstagramButton from '../components/shared/InstagramButton'
import { apiFetch, apiBase } from '../api/client'
import { usePipelineStore } from '../store/pipelineStore'

const STEP_LABELS: Record<string, string> = {
  fetching_media: 'Fetching media from Instagram',
  fetching_insights: 'Fetching engagement data',
  downloading: 'Downloading video',
  extracting_frames: 'Extracting frames & detecting audio',
  ocr: 'Reading text overlays & edit cuts',
  grok_analysis: 'Analysing with AI',
  done: 'Video complete',
  skipped: 'Already analysed',
  error: 'Error — skipping',
}

export default function ScanningPage() {
  const { status, progress, setJob, setProgress, setStatus, reset } = usePipelineStore()
  const [log, setLog] = useState<string[]>([])

  function appendLog(msg: string) {
    setLog((prev) => [msg, ...prev].slice(0, 6))
  }

  async function handleScan() {
    reset()
    setLog([])

    const { jobId } = await apiFetch<{ jobId: string }>('/api/pipeline/run', { method: 'POST' })
    setJob(jobId)

    const source = new EventSource(`${apiBase}/api/pipeline/events/${jobId}`)

    source.onmessage = (e: MessageEvent<string>) => {
      const event = JSON.parse(e.data) as {
        step: string
        current: number
        total: number
        status: string
        error?: string
      }

      const pct = event.total > 0 ? Math.round((event.current / event.total) * 100) : 0
      setProgress(pct)

      const step = event.step ?? ''
      const label = step ? (STEP_LABELS[step] ?? step.replace(/_/g, ' ')) : ''
      if (step && step !== 'starting') appendLog(label + (event.error ? ` — ${event.error}` : ''))

      if (event.status === 'done') {
        source.close()
        setStatus('done')
      }
    }

    source.onerror = () => {
      source.close()
      setStatus('error')
    }
  }

  return (
    <div className="p-8 pt-16">
      <BackButton />
      <h1 className={`${HEADING_CLASS} mt-4 mb-8`}>Scan</h1>

      {/* Grey scanning box */}
      <div className="w-full max-w-lg mx-auto bg-neutral-900 rounded-2xl p-8 flex flex-col items-center gap-6 min-h-64">
        <AnimatePresence mode="wait">
          {status === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4 py-8"
            >
              <p className="text-neutral-400 text-sm text-center">
                Analyse your videos with AI — no Grok key needed to get started
              </p>
              <InstagramButton onClick={handleScan}>Start Scan</InstagramButton>
            </motion.div>
          )}

          {status === 'running' && (
            <motion.div
              key="running"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="w-full flex flex-col gap-4"
            >
              <div className="flex justify-between text-sm">
                <span className="text-neutral-400">Analysing…</span>
                <span className="text-white font-mono">{progress}%</span>
              </div>

              {/* Progress bar */}
              <div className="h-1 rounded-full bg-neutral-800 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: IG_GRADIENT }}
                  animate={{ width: `${progress}%` }}
                  transition={{ type: 'spring', stiffness: 60, damping: 20 }}
                />
              </div>

              {/* Step log */}
              <div className="flex flex-col gap-1 mt-2">
                <AnimatePresence initial={false}>
                  {log.map((msg, i) => (
                    <motion.p
                      key={msg + i}
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: Math.max(0.2, 1 - i * 0.2), y: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      className="text-xs text-neutral-400"
                    >
                      {msg}
                    </motion.p>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {status === 'done' && (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center gap-4 py-8 text-center"
            >
              <motion.p
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="text-green-400 text-lg"
              >
                ✓ Scan complete
              </motion.p>
              <p className="text-neutral-400 text-sm">Head to Library to view your formats</p>
              <InstagramButton onClick={handleScan}>Scan Again</InstagramButton>
            </motion.div>
          )}

          {status === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center gap-4 py-8"
            >
              <p className="text-red-400">Scan failed</p>
              <button
                onClick={handleScan}
                className="text-neutral-400 text-sm underline hover:text-white"
              >
                Try again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

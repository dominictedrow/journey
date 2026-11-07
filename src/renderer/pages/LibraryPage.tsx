import { useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles'
import FormatCard from '../components/shared/FormatCard'
import TextInput from '../components/shared/TextInput'
import { apiFetch, apiBase } from '../api/client'

interface FormatSummary {
  id: number
  label: string
  avgEngagement: number
  videoCount: number
  thumbnailUrl: string | null
}

const STEP_LABELS: Record<string, string> = {
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
}

export default function LibraryPage() {
  const [formats, setFormats] = useState<FormatSummary[]>([])
  const [search, setSearch] = useState('')
  const [scanStatus, setScanStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [log, setLog] = useState<string[]>([])

  const loadFormats = useCallback(async () => {
    try {
      const data = await apiFetch<FormatSummary[]>('/api/formats')
      setFormats(data)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    loadFormats()
  }, [loadFormats])

  function appendLog(msg: string) {
    setLog((prev) => [msg, ...prev].slice(0, 6))
  }

  async function handleScan() {
    setScanStatus('running')
    setProgress(0)
    setLog([])

    try {
      const { jobId } = await apiFetch<{ jobId: string }>('/api/pipeline/run', { method: 'POST' })
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
          // Poll job status until the backend's rebuildFormats + generateInsights finish,
          // then fetch the updated format list. activeJobs status becomes 'done' only
          // after the full post-pipeline chain completes.
          void pollJobDone(jobId)
        }
      }

      source.onerror = () => {
        source.close()
        setScanStatus('error')
      }
    } catch {
      setScanStatus('error')
    }
  }

  async function pollJobDone(jobId: string): Promise<void> {
    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      await new Promise<void>((r) => setTimeout(r, 1500))
      try {
        const job = await apiFetch<{ status: string }>(`/api/pipeline/status/${jobId}`)
        if (job.status === 'done' || job.status === 'error') break
      } catch {
        break
      }
    }
    await loadFormats()
    setScanStatus('done')
  }

  const filtered = formats.filter((f) =>
    f.label.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="p-8 pt-16">
      <div className="flex items-center justify-between mb-6">
        <h1 className={HEADING_CLASS}>Library</h1>
        {scanStatus !== 'running' && (
          <button
            onClick={handleScan}
            className="px-4 py-1.5 rounded-full text-white text-sm font-medium hover:opacity-80 transition-opacity"
            style={{ background: IG_GRADIENT }}
          >
            {scanStatus === 'done' ? 'Scan Again' : 'Scan'}
          </button>
        )}
      </div>

      <AnimatePresence mode="wait">
        {scanStatus === 'running' ? (
          <motion.div
            key="scanning"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col gap-4 pt-4"
          >
            <div className="flex justify-between text-sm">
              <span className="text-neutral-400">Analysing…</span>
              <span className="text-white font-mono">{progress}%</span>
            </div>
            <div className="h-1 rounded-full bg-neutral-800 overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: IG_GRADIENT }}
                animate={{ width: `${progress}%` }}
                transition={{ type: 'spring', stiffness: 60, damping: 20 }}
              />
            </div>
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
        ) : (
          <motion.div
            key="library"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {scanStatus === 'done' && (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-green-400 text-sm mb-4"
              >
                ✓ Scan complete
              </motion.p>
            )}
            {scanStatus === 'error' && (
              <p className="text-red-400 text-sm mb-4">Scan failed — check logs and try again</p>
            )}

            <TextInput
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search formats…"
              className="mb-6"
            />

            {filtered.length === 0 ? (
              <div className="text-center py-20">
                <p className="text-neutral-500 text-sm">
                  {formats.length === 0
                    ? 'No formats yet — hit Scan to analyse your reels.'
                    : 'No formats match your search.'}
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-4">
                {filtered.map((f) => (
                  <FormatCard
                    key={f.id}
                    id={f.id}
                    label={f.label}
                    avgEngagement={f.avgEngagement}
                    thumbnailUrl={f.thumbnailUrl}
                    videoCount={f.videoCount}
                  />
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

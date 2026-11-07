import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles'
import FormatCard from '../components/shared/FormatCard'
import EngagementBadge from '../components/shared/EngagementBadge'
import Modal from '../components/shared/Modal'
import { apiFetch } from '../api/client'

interface FormatSnapshot {
  id: number
  label: string
  avgEngagement: number
  videoCount: number
  thumbnailUrl?: string | null
}

interface InsightReport {
  bestFormat: FormatSnapshot
  worstFormat: FormatSnapshot
  worstFormatWeaknesses: string[]
  overallRecommendations: string[]
  generatedAt: string
  allFormats: FormatSnapshot[]
}

interface SchedulerStatus {
  running: boolean
  nextRun: string | null
  lastRun: string | null
  config: { enabled: boolean; hour: number; minute: number }
}

function useCountdown(targetIso: string | null) {
  const [remaining, setRemaining] = useState<number | null>(null)

  useEffect(() => {
    if (!targetIso) { setRemaining(null); return }
    const tick = () => setRemaining(Math.max(0, new Date(targetIso).getTime() - Date.now()))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [targetIso])

  return remaining
}

function formatCountdown(ms: number): string {
  const s = Math.floor(ms / 1000)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

export default function HomePage() {
  const [formats, setFormats] = useState<FormatSnapshot[]>([])
  const [insights, setInsights] = useState<InsightReport | null>(null)
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [timerModal, setTimerModal] = useState(false)
  const [schedHour, setSchedHour] = useState(9)
  const [schedMinute, setSchedMinute] = useState(0)
  const [schedSaving, setSchedSaving] = useState(false)

  const remaining = useCountdown(scheduler?.nextRun ?? null)

  useEffect(() => {
    apiFetch<FormatSnapshot[]>('/api/formats').then(setFormats).catch(() => {})
    apiFetch<InsightReport>('/api/analytics/insights').then(setInsights).catch(() => {})
    apiFetch<SchedulerStatus>('/api/scheduler/status')
      .then((s) => {
        setScheduler(s)
        setSchedHour(s.config.hour)
        setSchedMinute(s.config.minute)
      })
      .catch(() => {})
  }, [])

  async function handleRefreshInsights() {
    setRefreshing(true)
    try {
      const report = await apiFetch<InsightReport>('/api/analytics/insights/refresh', { method: 'POST' })
      setInsights(report)
    } finally {
      setRefreshing(false)
    }
  }

  async function saveSchedule(enabled: boolean) {
    setSchedSaving(true)
    try {
      const status = await apiFetch<SchedulerStatus>('/api/scheduler/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled, hour: schedHour, minute: schedMinute }),
      })
      setScheduler(status)
      setTimerModal(false)
    } finally {
      setSchedSaving(false)
    }
  }

  const top5 = insights?.allFormats.slice(0, 5) ?? formats.slice(0, 5)
  const cards = formats.slice(0, 5)
  const timeValue = `${String(schedHour).padStart(2, '0')}:${String(schedMinute).padStart(2, '0')}`

  return (
    <div className="p-8 pt-16">
      <h1 className={`${HEADING_CLASS} text-3xl mb-6`}>Analyze Insta!</h1>

      {/* Countdown timer — click to configure */}
      <button
        onClick={() => setTimerModal(true)}
        className="w-full rounded-2xl p-6 mb-8 flex flex-col items-center gap-1 hover:opacity-90 transition-opacity"
        style={{ background: IG_GRADIENT }}
      >
        <p className="text-white/70 text-xs uppercase tracking-widest">Next scheduled scan</p>
        {remaining != null ? (
          <p className="text-white font-mono text-4xl font-bold">{formatCountdown(remaining)}</p>
        ) : (
          <p className="text-white font-mono text-2xl opacity-60">-- : -- : --</p>
        )}
        <p className="text-white/60 text-xs mt-0.5">
          {scheduler?.config.enabled
            ? `Daily at ${String(schedHour).padStart(2,'0')}:${String(schedMinute).padStart(2,'0')} — click to change`
            : 'Click to set up daily scan'}
        </p>
      </button>

      <div className="grid grid-cols-2 gap-8 mb-10">
        {/* Top 5 ranked list */}
        <div>
          <p className="text-white font-semibold mb-4">Top Formats</p>
          {top5.length > 0 ? (
            <div className="flex flex-col gap-1">
              {top5.map((f, i) => (
                <Link
                  key={f.id}
                  to={`/format/${f.id}`}
                  className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-white/5 transition-colors"
                >
                  <span className="text-neutral-500 font-mono text-sm w-5">{i + 1}</span>
                  <span className="text-white text-sm flex-1 capitalize">{f.label}</span>
                  <EngagementBadge rate={f.avgEngagement} />
                  <span className="text-neutral-500 text-xs">{f.videoCount}v</span>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-neutral-500 text-sm">No formats yet — run a scan first</p>
          )}
        </div>

        {/* AI insights */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <p className="text-white font-semibold">AI Insights</p>
            <button
              onClick={handleRefreshInsights}
              disabled={refreshing}
              className="text-neutral-400 text-xs underline hover:text-white disabled:opacity-40"
            >
              {refreshing ? 'Generating…' : 'Refresh'}
            </button>
          </div>

          {insights ? (
            <div className="flex flex-col gap-3">
              {insights.overallRecommendations.map((rec, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="flex gap-2 text-sm text-neutral-300"
                >
                  <span className="text-neutral-500 mt-0.5">›</span>
                  <span>{rec}</span>
                </motion.div>
              ))}
              {insights.overallRecommendations.length === 0 && (
                <p className="text-neutral-500 text-sm">Click Refresh to generate recommendations</p>
              )}
              <p className="text-neutral-600 text-xs mt-2">
                Generated {new Date(insights.generatedAt).toLocaleDateString()}
              </p>
            </div>
          ) : (
            <p className="text-neutral-500 text-sm">
              {formats.length > 0 ? 'Loading insights…' : 'Run a scan to generate AI insights'}
            </p>
          )}
        </div>
      </div>

      {/* Recent FormatCards */}
      {cards.length > 0 && (
        <>
          <p className="text-white font-semibold mb-4">Recent Formats</p>
          <div className="grid grid-cols-5 gap-4">
            {cards.map((f) => (
              <FormatCard
                key={f.id}
                id={f.id}
                label={f.label}
                avgEngagement={f.avgEngagement}
                thumbnailUrl={f.thumbnailUrl ?? null}
                videoCount={f.videoCount}
              />
            ))}
          </div>
        </>
      )}

      {/* Schedule modal */}
      <Modal open={timerModal} onClose={() => setTimerModal(false)}>
        <p className="text-white font-semibold mb-5">Daily Scan Schedule</p>

        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <p className="text-neutral-400 text-sm shrink-0">Run daily at</p>
            <input
              type="time"
              value={timeValue}
              onChange={(e) => {
                const [h, m] = e.target.value.split(':').map(Number)
                setSchedHour(h ?? 9)
                setSchedMinute(m ?? 0)
              }}
              className="bg-neutral-800 text-white px-3 py-2 rounded border border-white/20 text-sm"
            />
          </div>

          <div className="flex gap-3 mt-2">
            <button
              onClick={() => saveSchedule(true)}
              disabled={schedSaving}
              className="flex-1 py-2 rounded-lg text-white text-sm font-medium disabled:opacity-40"
              style={{ background: IG_GRADIENT }}
            >
              {schedSaving ? 'Saving…' : 'Enable'}
            </button>
            {scheduler?.config.enabled && (
              <button
                onClick={() => saveSchedule(false)}
                disabled={schedSaving}
                className="px-4 py-2 rounded-lg text-neutral-400 text-sm hover:text-white border border-white/10 hover:border-white/30 transition-colors disabled:opacity-40"
              >
                Disable
              </button>
            )}
          </div>
        </div>
      </Modal>
    </div>
  )
}

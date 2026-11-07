import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import BackButton from '../components/layout/BackButton'
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles'
import TextInput from '../components/shared/TextInput'
import { apiFetch } from '../api/client'
import { useAppStore } from '../store/appStore'

interface SchedulerStatus {
  running: boolean
  nextRun: string | null
  lastRun: string | null
  config: { enabled: boolean; hour: number; minute: number }
}

interface Snapshot {
  scan: number
  avgEngagementRate: number
  formatsCount: number
  scannedAt: string | null
}

export default function ProfilePage() {
  const { profile, account, loadProfile } = useAppStore()
  const navigate = useNavigate()
  const [grokKey, setGrokKey] = useState('')
  const [keySaved, setKeySaved] = useState(false)
  const [keyError, setKeyError] = useState('')
  const [igToken, setIgToken] = useState('')
  const [tokenSaved, setTokenSaved] = useState(false)
  const [tokenError, setTokenError] = useState('')
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null)
  const [schedHour, setSchedHour] = useState(9)
  const [schedMinute, setSchedMinute] = useState(0)
  const [schedSaving, setSchedSaving] = useState(false)
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])

  useEffect(() => {
    apiFetch<SchedulerStatus>('/api/scheduler/status')
      .then((s) => {
        setScheduler(s)
        setSchedHour(s.config.hour)
        setSchedMinute(s.config.minute)
      })
      .catch(() => {})

    apiFetch<Snapshot[]>('/api/analytics/snapshots')
      .then(setSnapshots)
      .catch(() => {})
  }, [])

  async function saveGrokKey() {
    if (!grokKey.trim()) return
    setKeyError('')
    try {
      await apiFetch('/api/settings/grok-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: grokKey.trim() }),
      })
      setGrokKey('')
      setKeySaved(true)
      setTimeout(() => setKeySaved(false), 3000)
    } catch {
      setKeyError('Failed to save key')
    }
  }

  async function reconnectWithToken() {
    if (!igToken.trim()) return
    setTokenError('')
    try {
      await apiFetch('/api/auth/connect-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: igToken.trim() }),
      })
      setIgToken('')
      setTokenSaved(true)
      await loadProfile()
      setTimeout(() => setTokenSaved(false), 3000)
    } catch {
      setTokenError('Invalid or expired token')
    }
  }

  async function disconnect() {
    await apiFetch('/api/auth/disconnect', { method: 'DELETE' })
    await loadProfile()
    navigate('/onboarding', { replace: true })
  }

  async function toggleScheduler(enabled: boolean) {
    setSchedSaving(true)
    try {
      const status = await apiFetch<SchedulerStatus>('/api/scheduler/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled, hour: schedHour, minute: schedMinute }),
      })
      setScheduler(status)
    } finally {
      setSchedSaving(false)
    }
  }

  async function saveScheduleTime() {
    if (!scheduler) return
    await toggleScheduler(scheduler.config.enabled)
  }

  const timeValue = `${String(schedHour).padStart(2, '0')}:${String(schedMinute).padStart(2, '0')}`

  const totalScans = snapshots.length
  const latestFormats = snapshots.at(-1)?.formatsCount ?? 0
  const firstRate = snapshots[0]?.avgEngagementRate ?? 0
  const latestRate = snapshots.at(-1)?.avgEngagementRate ?? 0
  const pctIncrease =
    firstRate > 0 ? Math.round(((latestRate - firstRate) / firstRate) * 100) : 0

  return (
    <div className="p-8 pt-16 flex gap-8 min-h-screen">
      {/* Left column — settings */}
      <div className="w-80 flex-shrink-0">
        <BackButton />
        <h1 className={`${HEADING_CLASS} mt-4 mb-8`}>Profile</h1>

        {profile && (
          <section className="mb-8">
            <p className="text-neutral-400 text-xs uppercase tracking-widest mb-2">Name</p>
            <p className="text-white text-lg">{profile.username}</p>
          </section>
        )}

        {account && (
          <section className="mb-8">
            <p className="text-neutral-400 text-xs uppercase tracking-widest mb-2">Instagram</p>
            <p className="text-white text-lg">@{account.username}</p>
            {account.followersCount != null && (
              <p className="text-neutral-400 text-sm mt-0.5">
                {account.followersCount.toLocaleString()} followers
              </p>
            )}
            <button
              onClick={disconnect}
              className="text-red-400 text-sm mt-3 hover:opacity-70 transition-opacity"
            >
              Log out of account
            </button>
          </section>
        )}

        <section className="mb-8">
          <p className="text-neutral-400 text-xs uppercase tracking-widest mb-2">Update Access Token</p>
          <div className="flex gap-2">
            <TextInput
              type="password"
              value={igToken}
              onChange={(e) => setIgToken(e.target.value)}
              placeholder="IGAAS..."
              onKeyDown={(e) => e.key === 'Enter' && reconnectWithToken()}
            />
            <button
              onClick={reconnectWithToken}
              disabled={!igToken.trim()}
              className="px-4 py-2 bg-white/10 rounded text-white text-sm hover:bg-white/20 disabled:opacity-40 transition-colors flex-shrink-0"
            >
              Apply
            </button>
          </div>
          {tokenSaved && <p className="text-green-400 text-sm mt-1">Token updated ✓</p>}
          {tokenError && <p className="text-red-400 text-sm mt-1">{tokenError}</p>}
        </section>

        <section className="mb-8">
          <p className="text-neutral-400 text-xs uppercase tracking-widest mb-2">xAI Grok API Key</p>
          <div className="flex gap-2">
            <TextInput
              type="password"
              value={grokKey}
              onChange={(e) => setGrokKey(e.target.value)}
              placeholder="xai-..."
              onKeyDown={(e) => e.key === 'Enter' && saveGrokKey()}
            />
            <button
              onClick={saveGrokKey}
              disabled={!grokKey.trim()}
              className="px-4 py-2 bg-white/10 rounded text-white text-sm hover:bg-white/20 disabled:opacity-40 transition-colors flex-shrink-0"
            >
              Save
            </button>
          </div>
          {keySaved && <p className="text-green-400 text-sm mt-1">Saved ✓</p>}
          {keyError && <p className="text-red-400 text-sm mt-1">{keyError}</p>}
        </section>

        <section>
          <p className="text-neutral-400 text-xs uppercase tracking-widest mb-4">Daily Auto-Scan</p>
          <div className="bg-neutral-900 rounded-xl p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white text-sm font-medium">
                  {scheduler?.config.enabled ? 'Enabled' : 'Disabled'}
                </p>
                {scheduler?.nextRun && (
                  <p className="text-neutral-500 text-xs mt-0.5">
                    Next: {new Date(scheduler.nextRun).toLocaleString()}
                  </p>
                )}
                {scheduler?.lastRun && (
                  <p className="text-neutral-600 text-xs">
                    Last: {new Date(scheduler.lastRun).toLocaleString()}
                  </p>
                )}
              </div>
              <button
                onClick={() => toggleScheduler(!(scheduler?.config.enabled ?? false))}
                disabled={schedSaving}
                className={`relative w-12 h-6 rounded-full transition-colors disabled:opacity-50 ${
                  scheduler?.config.enabled ? 'bg-green-500' : 'bg-neutral-700'
                }`}
              >
                <span
                  className={`absolute top-1 left-0 w-4 h-4 bg-white rounded-full transition-transform ${
                    scheduler?.config.enabled ? 'translate-x-7' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
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
                className="bg-neutral-800 text-white px-3 py-1.5 rounded border border-white/20 text-sm"
              />
              <button
                onClick={saveScheduleTime}
                disabled={schedSaving}
                className="px-3 py-1.5 bg-white/10 rounded text-white text-sm hover:bg-white/20 disabled:opacity-40 transition-colors"
              >
                {schedSaving ? 'Saving…' : 'Set'}
              </button>
            </div>
          </div>
        </section>
      </div>

      {/* Right column — graph + stats */}
      <div className="flex-1 flex flex-col gap-4 pt-16">
        {/* Big graph box */}
        <div className="flex-1 bg-neutral-900 rounded-2xl p-6 flex flex-col min-h-0">
          <p className="text-neutral-400 text-xs uppercase tracking-widest mb-4">
            Avg Engagement Rate per Scan
          </p>
          {snapshots.length < 2 ? (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-neutral-600 text-sm text-center">
                Run at least 2 scans to see the trend.
              </p>
            </div>
          ) : (
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={snapshots} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="scan"
                    tick={{ fill: '#737373', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: 'Scan #', position: 'insideBottom', fill: '#525252', fontSize: 11, offset: -2 }}
                  />
                  <YAxis
                    tick={{ fill: '#737373', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                  />
                  <Tooltip
                    contentStyle={{ background: '#171717', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                    labelStyle={{ color: '#a3a3a3', fontSize: 12 }}
                    itemStyle={{ color: '#ffffff', fontSize: 13 }}
                    formatter={(v) => [`${(Number(v) || 0).toFixed(2)}%`, 'Avg engagement'] as [string, string]}
                    labelFormatter={(l) => `Scan ${String(l)}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgEngagementRate"
                    stroke="#ffffff"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#ffffff', strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: '#ffffff' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Small stats box with IG gradient */}
        <div
          className="rounded-2xl p-6 flex items-center justify-around h-32 flex-shrink-0"
          style={{ background: IG_GRADIENT }}
        >
          <div className="text-center">
            <p className="text-white text-2xl font-bold">{totalScans}</p>
            <p className="text-white/70 text-xs mt-0.5">Scans</p>
          </div>
          <div className="w-px h-10 bg-white/30" />
          <div className="text-center">
            <p className="text-white text-2xl font-bold">{latestFormats}</p>
            <p className="text-white/70 text-xs mt-0.5">Formats found</p>
          </div>
          <div className="w-px h-10 bg-white/30" />
          <div className="text-center">
            <p className="text-white text-2xl font-bold">
              {totalScans >= 2 ? `${pctIncrease > 0 ? '+' : ''}${pctIncrease}%` : '—'}
            </p>
            <p className="text-white/70 text-xs mt-0.5">Engagement increase</p>
          </div>
        </div>
      </div>
    </div>
  )
}

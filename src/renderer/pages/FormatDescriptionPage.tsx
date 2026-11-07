import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import BackButton from '../components/layout/BackButton'
import { HEADING_CLASS, IG_GRADIENT } from '../constants/styles'
import VideoThumbnail from '../components/shared/VideoThumbnail'
import EngagementBadge from '../components/shared/EngagementBadge'
import { apiFetch } from '../api/client'

interface FormatMember {
  mediaId: string
  thumbnailUrl: string | null
  engagementRate: number | null
  timestamp: string | null
}

interface FormatCharacteristics {
  footageType: string
  backgroundType: string
  filmingStyle: string
  lighting: string
  setting: string
  audioType: string
  hookStrength: string
  productionQuality: string
  cutsPerSecond: number
}

interface FormatDetail {
  id: number
  label: string
  videoCount: number
  avgEngagement: number
  topMediaId: string | null
  bottomMediaId: string | null
  thumbnailUrl: string | null
  members: FormatMember[]
  characteristics: FormatCharacteristics
}

interface InsightReport {
  bestFormat: { id: number }
  worstFormat: { id: number }
  worstFormatWeaknesses: string[]
  overallRecommendations: string[]
}

const CHAR_LABELS: Record<string, string> = {
  footageType: 'Footage Type',
  backgroundType: 'Background',
  filmingStyle: 'Filming Style',
  lighting: 'Lighting',
  setting: 'Setting',
  audioType: 'Audio',
  hookStrength: 'Hook Strength',
  productionQuality: 'Production Quality',
  cutsPerSecond: 'Cuts / sec',
}

const HOOK_COLOR: Record<string, string> = {
  strong: 'text-green-400',
  medium: 'text-yellow-400',
  weak: 'text-red-400',
}

export default function FormatDescriptionPage() {
  const { id } = useParams()
  const formatId = parseInt(id ?? '')
  const [format, setFormat] = useState<FormatDetail | null>(null)
  const [insights, setInsights] = useState<InsightReport | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    apiFetch<FormatDetail>(`/api/formats/${id}`)
      .then(setFormat)
      .catch(() => setFormat(null))
      .finally(() => setLoading(false))
    apiFetch<InsightReport>('/api/analytics/insights').then(setInsights).catch(() => {})
  }, [id])

  if (loading) {
    return (
      <div className="p-8 pt-16">
        <BackButton />
        <p className="text-neutral-500 text-sm mt-8">Loading…</p>
      </div>
    )
  }

  if (!format) {
    return (
      <div className="p-8 pt-16">
        <BackButton />
        <p className="text-neutral-500 text-sm mt-8">Format not found.</p>
      </div>
    )
  }

  const chars = format.characteristics
  const isBest = insights?.bestFormat.id === formatId
  const isWorst = insights?.worstFormat.id === formatId
  const weaknesses = isWorst ? (insights?.worstFormatWeaknesses ?? []) : []
  const recommendations = insights?.overallRecommendations ?? []
  const hasAiContent = weaknesses.length > 0 || recommendations.length > 0

  return (
    <div className="p-8 pt-16">
      <BackButton />

      <div className="grid grid-cols-3 gap-8 mt-4">
        {/* Left — thumbnail + member list */}
        <div className="col-span-1 flex flex-col gap-4">
          <VideoThumbnail src={format.thumbnailUrl} alt={format.label} />

          <div className="flex items-center gap-3 flex-wrap">
            <EngagementBadge rate={format.avgEngagement} />
            <span className="text-neutral-400 text-sm">{format.videoCount} videos</span>
            {isBest && (
              <span
                className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
                style={{ background: IG_GRADIENT }}
              >
                Top performer
              </span>
            )}
            {isWorst && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-red-900/40 text-red-400">
                Needs work
              </span>
            )}
          </div>

          <div className="h-px w-full rounded-full" style={{ background: IG_GRADIENT }} />

          <div className="flex flex-col gap-2">
            <p className="text-neutral-500 text-xs uppercase tracking-wider">All videos</p>
            {format.members.map((m) => (
              <div key={m.mediaId} className="flex items-center gap-3">
                <img
                  src={m.thumbnailUrl ?? undefined}
                  alt=""
                  className="w-10 h-[72px] object-cover rounded"
                />
                <div>
                  {m.engagementRate != null ? (
                    <EngagementBadge rate={m.engagementRate} />
                  ) : (
                    <span className="text-neutral-600 text-xs">—</span>
                  )}
                  {m.timestamp && (
                    <p className="text-neutral-600 text-xs mt-1">
                      {new Date(m.timestamp).toLocaleDateString()}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — format profile + AI analysis */}
        <div className="col-span-2 flex flex-col gap-6">
          <h1 className={`${HEADING_CLASS} capitalize`}>{format.label}</h1>

          {/* Format characteristics */}
          <div className="bg-neutral-900 rounded-2xl p-6">
            <p className="text-white font-semibold mb-4">Format Profile</p>
            <div className="grid grid-cols-2 gap-x-8 gap-y-4">
              {(Object.entries(chars) as [string, string | number][]).map(([key, val]) => (
                <div key={key}>
                  <p className="text-neutral-500 text-xs uppercase tracking-wider mb-1">
                    {CHAR_LABELS[key] ?? key}
                  </p>
                  <p
                    className={`text-sm font-medium capitalize ${
                      key === 'hookStrength' ? (HOOK_COLOR[String(val)] ?? 'text-white') : 'text-white'
                    }`}
                  >
                    {typeof val === 'number' ? val.toFixed(1) : String(val).replace(/_/g, ' ')}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* AI analysis */}
          <div className="bg-neutral-900 rounded-2xl p-6 flex flex-col gap-4">
            <p className="text-white font-semibold">AI Analysis</p>

            {!hasAiContent && (
              <p className="text-neutral-500 text-sm">
                {insights
                  ? 'No AI analysis available — click Refresh Insights on the Home page'
                  : 'Add a Grok API key in Profile to generate AI-powered analysis'}
              </p>
            )}

            {weaknesses.length > 0 && (
              <div>
                <p className="text-neutral-400 text-xs uppercase tracking-wider mb-2">Weaknesses</p>
                <div className="flex flex-col gap-2">
                  {weaknesses.map((w, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -4 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.07 }}
                      className="flex gap-2 text-sm bg-red-950/30 border border-red-900/40 rounded-lg px-3 py-2"
                    >
                      <span className="text-red-500 mt-0.5 shrink-0">✕</span>
                      <span className="text-red-200">{w}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {recommendations.length > 0 && (
              <div>
                <p className="text-neutral-400 text-xs uppercase tracking-wider mb-2">
                  Recommendations
                </p>
                <div className="flex flex-col gap-2">
                  {recommendations.map((rec, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -4 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.07 }}
                      className="flex gap-2 text-sm text-neutral-300"
                    >
                      <span className="text-neutral-500 mt-0.5 shrink-0">›</span>
                      <span>{rec}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

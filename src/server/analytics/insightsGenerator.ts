import OpenAI from 'openai'
import { eq } from 'drizzle-orm'
import { db } from '../db/index'
import { formats, insights } from '../db/schema'
import { tokenStore } from '../auth/tokenStore'
import { settings } from '../config/settings'
import { decodeCentroid } from '../clustering/featureEncoder'
import { getFormats } from '../clustering/formatRepository'
import { logger } from '../logger'

export interface FormatSnapshot {
  id: number
  label: string
  avgEngagement: number
  videoCount: number
}

export interface InsightReport {
  bestFormat: FormatSnapshot
  worstFormat: FormatSnapshot
  worstFormatWeaknesses: string[]
  overallRecommendations: string[]
  generatedAt: string
  allFormats: FormatSnapshot[]
}

async function callGrokText(
  prompt: string,
): Promise<{ worst_format_weaknesses: string[]; overall_recommendations: string[] }> {
  const keychainKey = await tokenStore.getGrokKey()
  const apiKey = keychainKey ?? process.env['GROK_API_KEY'] ?? null

  if (!apiKey) {
    return {
      worst_format_weaknesses: ['Add a Grok API key in Profile to generate AI-powered weaknesses'],
      overall_recommendations: ['Add a Grok API key in Profile to get AI-powered recommendations'],
    }
  }

  const client = new OpenAI({ apiKey, baseURL: 'https://api.x.ai/v1' })
  const response = await client.chat.completions.create({
    model: settings.grok.textModel,
    temperature: 0,
    max_tokens: 512,
    messages: [
      { role: 'system', content: 'Return ONLY valid JSON. No markdown, no explanation.' },
      { role: 'user', content: prompt },
    ],
  })

  const raw = response.choices[0]?.message?.content ?? '{}'
  try {
    return JSON.parse(raw) as {
      worst_format_weaknesses: string[]
      overall_recommendations: string[]
    }
  } catch {
    return { worst_format_weaknesses: [], overall_recommendations: [] }
  }
}

export async function generateInsights(accountId: string): Promise<InsightReport> {
  const allFormats = getFormats(accountId)

  const fallback: InsightReport = {
    bestFormat: { id: 0, label: 'None', avgEngagement: 0, videoCount: 0 },
    worstFormat: { id: 0, label: 'None', avgEngagement: 0, videoCount: 0 },
    worstFormatWeaknesses: ['Rebuild clusters after scanning to generate insights'],
    overallRecommendations: [],
    generatedAt: new Date().toISOString(),
    allFormats: [],
  }

  if (allFormats.length < 2) return fallback

  const sorted = [...allFormats].sort((a, b) => b.avgEngagement - a.avgEngagement)
  const best = sorted[0]!
  const worst = sorted[sorted.length - 1]!

  const bestRow = db.select().from(formats).where(eq(formats.id, best.id)).all()[0]
  const worstRow = db.select().from(formats).where(eq(formats.id, worst.id)).all()[0]

  const bestChars = bestRow
    ? decodeCentroid(new Float32Array(JSON.parse(bestRow.centroidJson) as number[]))
    : null
  const worstChars = worstRow
    ? decodeCentroid(new Float32Array(JSON.parse(worstRow.centroidJson) as number[]))
    : null

  const multiplier =
    best.avgEngagement > 0
      ? (best.avgEngagement / Math.max(worst.avgEngagement, 0.01)).toFixed(1)
      : '?'

  const fmtChars = (c: typeof bestChars) =>
    c
      ? `footage=${c.footageType}, category=${c.contentCategory}, structure=${c.narrativeStructure}, subject=${c.subjectType}, presenter=${c.presenterStyle}, background=${c.backgroundType}, filming=${c.filmingStyle}, lighting=${c.lighting}, setting=${c.setting}, color=${c.colorTone}, aesthetic=${c.aesthetic}, audio=${c.audioType}, voiceover=${c.voiceoverStyle}, hook=${c.hookStrength} (${c.hookType}), pacing=${c.pacing}, quality=${c.productionQuality}, cuts/sec=${c.cutsPerSecond}`
      : ''

  const prompt = `Instagram Reel format performance data for one account:

Best format: "${best.label}" — ${best.avgEngagement.toFixed(2)}% avg engagement (${best.videoCount} videos)
  ${fmtChars(bestChars)}

Worst format: "${worst.label}" — ${worst.avgEngagement.toFixed(2)}% avg engagement (${worst.videoCount} videos)
  ${fmtChars(worstChars)}

The best outperforms the worst by ${multiplier}x.

Return ONLY:
{
  "worst_format_weaknesses": ["2-3 short points explaining why the worst format underperforms"],
  "overall_recommendations": ["2-3 actionable tips to improve engagement across all formats"]
}`

  const grokResult = await callGrokText(prompt).catch((err: unknown) => {
    logger.error(`Grok text failed: ${String(err)}`)
    return { worst_format_weaknesses: [] as string[], overall_recommendations: [] as string[] }
  })

  const report: InsightReport = {
    bestFormat: {
      id: best.id,
      label: best.label,
      avgEngagement: best.avgEngagement,
      videoCount: best.videoCount,
    },
    worstFormat: {
      id: worst.id,
      label: worst.label,
      avgEngagement: worst.avgEngagement,
      videoCount: worst.videoCount,
    },
    worstFormatWeaknesses: grokResult.worst_format_weaknesses,
    overallRecommendations: grokResult.overall_recommendations,
    generatedAt: new Date().toISOString(),
    allFormats: sorted.map((f) => ({
      id: f.id,
      label: f.label,
      avgEngagement: f.avgEngagement,
      videoCount: f.videoCount,
    })),
  }

  db.insert(insights)
    .values({
      accountId,
      bestFormatId: best.id,
      worstFormatId: worst.id,
      reportJson: JSON.stringify(report),
    })
    .run()

  return report
}

export function getLatestInsights(accountId: string): InsightReport | null {
  const rows = db
    .select()
    .from(insights)
    .where(eq(insights.accountId, accountId))
    .all()
    .sort((a, b) => (b.generatedAt?.getTime() ?? 0) - (a.generatedAt?.getTime() ?? 0))

  const row = rows[0]
  if (!row) return null
  try {
    return JSON.parse(row.reportJson) as InsightReport
  } catch {
    return null
  }
}

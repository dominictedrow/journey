import OpenAI from 'openai'
import { inArray } from 'drizzle-orm'
import { db } from '../db/index'
import { videoAnalyses } from '../db/schema'
import { tokenStore } from '../auth/tokenStore'
import { settings } from '../config/settings'
import { logger } from '../logger'
import { FEATURE_DIM, DIM_WEIGHTS } from './featureEncoder'

function euclideanSq(a: Float32Array, b: Float32Array): number {
  let sum = 0
  for (let i = 0; i < FEATURE_DIM; i++) {
    const diff = (a[i]! - b[i]!) * DIM_WEIGHTS[i]!
    sum += diff * diff
  }
  return sum
}

export async function runCoherencePass(
  assignments: number[],
  vectors: Float32Array[],
  centroids: Float32Array[],
  mediaIds: string[],
): Promise<number[]> {
  const keychainKey = await tokenStore.getGrokKey()
  const apiKey = keychainKey ?? process.env['GROK_API_KEY'] ?? null
  if (!apiKey) return assignments

  const client = new OpenAI({ apiKey, baseURL: 'https://api.x.ai/v1' })
  const updated = [...assignments]
  const k = centroids.length

  for (let c = 0; c < k; c++) {
    const memberIndices: number[] = []
    for (let i = 0; i < assignments.length; i++) {
      if (assignments[i] === c) memberIndices.push(i)
    }
    if (memberIndices.length < 3) continue

    const memberMediaIds = memberIndices.map((i) => mediaIds[i]!)
    const rows = db
      .select({ mediaId: videoAnalyses.mediaId, grokRawJson: videoAnalyses.grokRawJson })
      .from(videoAnalyses)
      .where(inArray(videoAnalyses.mediaId, memberMediaIds))
      .all()

    const fingerprintMap = new Map<string, string[]>()
    for (const row of rows) {
      if (!row.grokRawJson) continue
      try {
        const parsed = JSON.parse(row.grokRawJson) as {
          structural_fingerprint?: string[]
          distinguishing_features?: string[]
        }
        const fp = parsed.structural_fingerprint ?? parsed.distinguishing_features ?? []
        if (fp.length > 0) fingerprintMap.set(row.mediaId, fp)
      } catch {
        // ignore malformed json
      }
    }

    if (fingerprintMap.size < 3) continue

    const lines: string[] = []
    const lineToGlobalIdx: number[] = []

    for (const globalIdx of memberIndices) {
      const mid = mediaIds[globalIdx]!
      const fp = fingerprintMap.get(mid)
      if (!fp) continue
      lines.push(`Video ${lines.length + 1}: [${fp.map((s) => `"${s}"`).join(', ')}]`)
      lineToGlobalIdx.push(globalIdx)
    }

    if (lines.length < 3) continue

    const prompt = `These ${lines.length} videos were automatically clustered as a single content format. Review their structural fingerprints and flag any that clearly belong to a different production template.

${lines.join('\n')}

Return ONLY valid JSON: {"outliers": []} — list the 1-based video numbers that do NOT share the same production template as the majority. Return [] if all belong together.`

    try {
      const response = await client.chat.completions.create({
        model: settings.grok.textModel,
        temperature: 0,
        max_tokens: 128,
        messages: [
          { role: 'system', content: 'Return ONLY valid JSON. No markdown, no explanation.' },
          { role: 'user', content: prompt },
        ],
      })

      const raw = response.choices[0]?.message?.content ?? '{}'
      const parsed = JSON.parse(raw) as { outliers: number[] }
      if (!Array.isArray(parsed.outliers) || parsed.outliers.length === 0) continue

      for (const oneBasedIdx of parsed.outliers) {
        const lineIdx = oneBasedIdx - 1
        if (lineIdx < 0 || lineIdx >= lineToGlobalIdx.length) continue
        const globalIdx = lineToGlobalIdx[lineIdx]!

        let bestCluster = c
        let bestDist = Infinity
        for (let other = 0; other < k; other++) {
          if (other === c) continue
          const d = euclideanSq(vectors[globalIdx]!, centroids[other]!)
          if (d < bestDist) {
            bestDist = d
            bestCluster = other
          }
        }

        if (bestCluster !== c) {
          updated[globalIdx] = bestCluster
          logger.info(
            `Coherence pass: reassigned ${mediaIds[globalIdx]} from cluster ${c} → ${bestCluster}`,
          )
        }
      }
    } catch (err) {
      logger.warn(
        `Coherence pass cluster ${c} skipped: ${err instanceof Error ? err.message : String(err)}`,
      )
    }
  }

  return updated
}

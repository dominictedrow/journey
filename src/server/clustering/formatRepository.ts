import { eq, inArray } from 'drizzle-orm'
import { db } from '../db/index'
import { featureVectors, formats, formatMemberships, insights, media, mediaInsights } from '../db/schema'
import { clusterVectors } from './clusterEngine'
import { decodeCentroid, FEATURE_DIM, type FormatCharacteristics } from './featureEncoder'
import { generateFormatLabels } from './labelGenerator'
import { runCoherencePass } from './coherencePass'

export interface FormatSummary {
  id: number
  accountId: string
  label: string
  videoCount: number
  avgEngagement: number
  topMediaId: string | null
  bottomMediaId: string | null
  thumbnailUrl: string | null
  createdAt: Date
}

export interface FormatMember {
  mediaId: string
  thumbnailUrl: string | null
  engagementRate: number | null
  timestamp: Date | null
}

export interface FormatDetail extends FormatSummary {
  members: FormatMember[]
  characteristics: FormatCharacteristics
}

export async function rebuildFormats(accountId: string): Promise<FormatSummary[]> {
  const vectorRows = db
    .select({ mediaId: featureVectors.mediaId, vector: featureVectors.vector })
    .from(featureVectors)
    .innerJoin(media, eq(featureVectors.mediaId, media.id))
    .where(eq(media.accountId, accountId))
    .all()

  if (vectorRows.length < 2) return []

  const mediaIds = vectorRows.map((r) => r.mediaId)
  const vectors = vectorRows.map((r) => {
    const buf = r.vector as Buffer
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
  })

  const { assignments: rawAssignments, centroids } = await clusterVectors(vectors, accountId)
  const assignments = await runCoherencePass(rawAssignments, vectors, centroids, mediaIds)

  const allChars = centroids.map((cen) => decodeCentroid(cen))
  const labels = await generateFormatLabels(allChars)

  // Clear existing formats for this account
  const existing = db.select({ id: formats.id }).from(formats).where(eq(formats.accountId, accountId)).all()
  if (existing.length > 0) {
    const ids = existing.map((f) => f.id)
    db.update(insights)
      .set({ bestFormatId: null, worstFormatId: null })
      .where(eq(insights.accountId, accountId))
      .run()
    db.delete(formatMemberships).where(inArray(formatMemberships.formatId, ids)).run()
    db.delete(formats).where(eq(formats.accountId, accountId)).run()
  }

  const result: FormatSummary[] = []

  for (let c = 0; c < centroids.length; c++) {
    const memberIndices: number[] = []
    for (let i = 0; i < assignments.length; i++) {
      if (assignments[i] === c) memberIndices.push(i)
    }
    if (memberIndices.length === 0) continue

    const memberMediaIds = memberIndices.map((i) => mediaIds[i]!)

    const insightRows = db
      .select()
      .from(mediaInsights)
      .where(inArray(mediaInsights.mediaId, memberMediaIds))
      .all()

    const rates = insightRows.map((ins) => ({
      mediaId: ins.mediaId,
      rate: ins.engagementRate ?? 0,
    }))

    const avgEngagement =
      rates.length > 0 ? rates.reduce((s, r) => s + r.rate, 0) / rates.length : 0

    const sorted = [...rates].sort((a, b) => b.rate - a.rate)
    const topMediaId = sorted[0]?.mediaId ?? memberMediaIds[0] ?? null
    const bottomMediaId = sorted[sorted.length - 1]?.mediaId ?? null

    const label = labels[c] ?? allChars[c]!.footageType
    const centroidJson = JSON.stringify(Array.from(centroids[c]!))

    const [format] = db
      .insert(formats)
      .values({ accountId, label, centroidJson, videoCount: memberMediaIds.length, avgEngagement, topMediaId, bottomMediaId })
      .returning()
      .all()

    if (!format) continue

    for (const idx of memberIndices) {
      const v = vectors[idx]!
      const cen = centroids[c]!
      let dist = 0
      for (let d = 0; d < FEATURE_DIM; d++) dist += (v[d]! - cen[d]!) ** 2
      db.insert(formatMemberships)
        .values({ mediaId: mediaIds[idx]!, formatId: format.id, distance: Math.sqrt(dist) })
        .run()
    }

    const topMedia = topMediaId
      ? db.select({ thumbnailUrl: media.thumbnailUrl }).from(media).where(eq(media.id, topMediaId)).all()[0]
      : null

    result.push({
      id: format.id,
      accountId,
      label: format.label,
      videoCount: format.videoCount,
      avgEngagement: format.avgEngagement ?? 0,
      topMediaId: format.topMediaId,
      bottomMediaId: format.bottomMediaId,
      thumbnailUrl: topMedia?.thumbnailUrl ?? null,
      createdAt: format.createdAt,
    })
  }

  return result
}

export function getFormats(accountId: string): FormatSummary[] {
  return db
    .select()
    .from(formats)
    .where(eq(formats.accountId, accountId))
    .all()
    .map((f) => {
      const topMedia = f.topMediaId
        ? db.select({ thumbnailUrl: media.thumbnailUrl }).from(media).where(eq(media.id, f.topMediaId)).all()[0]
        : null
      return {
        id: f.id,
        accountId: f.accountId,
        label: f.label,
        videoCount: f.videoCount,
        avgEngagement: f.avgEngagement ?? 0,
        topMediaId: f.topMediaId,
        bottomMediaId: f.bottomMediaId,
        thumbnailUrl: topMedia?.thumbnailUrl ?? null,
        createdAt: f.createdAt,
      }
    })
}

export function getFormatDetail(formatId: number): FormatDetail | null {
  const format = db.select().from(formats).where(eq(formats.id, formatId)).all()[0]
  if (!format) return null

  const memberships = db
    .select({ mediaId: formatMemberships.mediaId })
    .from(formatMemberships)
    .where(eq(formatMemberships.formatId, formatId))
    .all()

  const members: FormatMember[] = memberships.map(({ mediaId }) => {
    const m = db.select().from(media).where(eq(media.id, mediaId)).all()[0]
    const ins = db.select().from(mediaInsights).where(eq(mediaInsights.mediaId, mediaId)).all()[0]
    const engagementRate = ins?.engagementRate ?? null
    return { mediaId, thumbnailUrl: m?.thumbnailUrl ?? null, engagementRate, timestamp: m?.timestamp ?? null }
  })

  const topMedia = format.topMediaId
    ? db.select({ thumbnailUrl: media.thumbnailUrl }).from(media).where(eq(media.id, format.topMediaId)).all()[0]
    : null

  const centroidArr = JSON.parse(format.centroidJson) as number[]
  const centroid = new Float32Array(centroidArr)

  return {
    id: format.id,
    accountId: format.accountId,
    label: format.label,
    videoCount: format.videoCount,
    avgEngagement: format.avgEngagement ?? 0,
    topMediaId: format.topMediaId,
    bottomMediaId: format.bottomMediaId,
    thumbnailUrl: topMedia?.thumbnailUrl ?? null,
    createdAt: format.createdAt,
    members,
    characteristics: decodeCentroid(centroid),
  }
}

import { Router } from 'express'
import { eq } from 'drizzle-orm'
import { tokenStore } from '../auth/tokenStore'
import { generateInsights, getLatestInsights } from '../analytics/insightsGenerator'
import { getFormats } from '../clustering/formatRepository'
import { db } from '../db/index'
import { scanSnapshots } from '../db/schema'

const router = Router()

// Returns cached insights (fast). Use POST /refresh to regenerate with Grok.
router.get('/insights', async (_req, res) => {
  const userId = await tokenStore.getUserId()
  if (!userId) {
    res.status(401).json({ error: 'Not connected to Instagram' })
    return
  }
  const cached = getLatestInsights(userId)
  if (cached) {
    res.json(cached)
    return
  }
  // No cached report — generate now (first run)
  const report = await generateInsights(userId)
  res.json(report)
})

// Regenerates insights with a fresh Grok call and caches the result.
router.post('/insights/refresh', async (_req, res) => {
  const userId = await tokenStore.getUserId()
  if (!userId) {
    res.status(401).json({ error: 'Not connected to Instagram' })
    return
  }
  const report = await generateInsights(userId)
  res.json(report)
})

// Chart-ready engagement data sorted by avg engagement descending.
router.get('/engagement', async (_req, res) => {
  const userId = await tokenStore.getUserId()
  if (!userId) {
    res.status(401).json({ error: 'Not connected to Instagram' })
    return
  }
  const formats = getFormats(userId)
  const data = formats
    .sort((a, b) => b.avgEngagement - a.avgEngagement)
    .map((f) => ({
      id: f.id,
      label: f.label,
      avgEngagement: f.avgEngagement,
      videoCount: f.videoCount,
    }))
  res.json(data)
})

router.get('/snapshots', async (_req, res) => {
  const userId = await tokenStore.getUserId()
  if (!userId) {
    res.status(401).json({ error: 'Not connected to Instagram' })
    return
  }
  const snaps = db
    .select()
    .from(scanSnapshots)
    .where(eq(scanSnapshots.accountId, userId))
    .orderBy(scanSnapshots.scannedAt)
    .all()
  res.json(
    snaps.map((s, i) => ({
      scan: i + 1,
      avgEngagementRate: s.avgEngagementRate,
      formatsCount: s.formatsCount,
      scannedAt: s.scannedAt?.toISOString() ?? null,
    })),
  )
})

export default router

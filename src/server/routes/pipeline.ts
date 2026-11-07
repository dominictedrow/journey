import { Router } from 'express'
import { randomUUID } from 'crypto'
import { eq } from 'drizzle-orm'
import { db } from '../db/index'
import { syncJobs } from '../db/schema'
import { tokenStore } from '../auth/tokenStore'
import { runPipeline, getJobEmitter } from '../pipeline/pipelineOrchestrator'
import type { PipelineEvent } from '../pipeline/pipelineOrchestrator'
import { logger } from '../logger'
import { recordScanSnapshot } from '../analytics/snapshotRecorder'
import { rebuildFormats } from '../clustering/formatRepository'
import { generateInsights } from '../analytics/insightsGenerator'

const router = Router()

interface JobState {
  status: 'running' | 'done' | 'error'
  progress: number
  accountId: string
}

const activeJobs = new Map<string, JobState>()

router.post('/run', async (_req, res) => {
  const userId = await tokenStore.getUserId()
  if (!userId) {
    res.status(401).json({ error: 'Not connected to Instagram' })
    return
  }

  const jobId = randomUUID()
  activeJobs.set(jobId, { status: 'running', progress: 0, accountId: userId })

  const [job] = db
    .insert(syncJobs)
    .values({ accountId: userId, triggeredBy: 'manual', status: 'running', startedAt: new Date() })
    .returning()
    .all()

  res.json({ jobId })

  // Wire progress tracking before starting so no events are missed
  const onProgress = (e: PipelineEvent) => {
    const pct = e.total > 0 ? Math.round((e.current / e.total) * 100) : 0
    const current = activeJobs.get(jobId)
    if (current) activeJobs.set(jobId, { ...current, progress: pct })
  }

  runPipeline(jobId, userId, true)
    .then(() => rebuildFormats(userId))
    .then(() => generateInsights(userId))
    .then(() => {
      const current = activeJobs.get(jobId)
      if (current) activeJobs.set(jobId, { ...current, status: 'done', progress: 100 })
      db.update(syncJobs)
        .set({ status: 'done', finishedAt: new Date() })
        .where(eq(syncJobs.id, job!.id))
        .run()
      recordScanSnapshot(userId)
      setTimeout(() => activeJobs.delete(jobId), 5 * 60_000)
    })
    .catch((err: unknown) => {
      logger.error(`Pipeline ${jobId} failed: ${String(err)}`)
      const current = activeJobs.get(jobId)
      if (current) activeJobs.set(jobId, { ...current, status: 'error' })
      db.update(syncJobs)
        .set({ status: 'error', finishedAt: new Date() })
        .where(eq(syncJobs.id, job!.id))
        .run()
    })

  // Attach progress listener after emitter is created by runPipeline
  setImmediate(() => {
    getJobEmitter(jobId)?.on('progress', onProgress)
  })
})

router.get('/status/:jobId', (req, res) => {
  const job = activeJobs.get(req.params['jobId']!)
  if (!job) {
    res.status(404).json({ error: 'Job not found' })
    return
  }
  res.json(job)
})

router.get('/events/:jobId', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.flushHeaders()

  const jobId = req.params['jobId']!
  const emitter = getJobEmitter(jobId)

  if (!emitter) {
    const job = activeJobs.get(jobId)
    const terminal: PipelineEvent = {
      jobId,
      mediaId: '',
      step: 'complete',
      current: 1,
      total: 1,
      status: job?.status ?? 'done',
    }
    res.write(`data: ${JSON.stringify(terminal)}\n\n`)
    res.end()
    return
  }

  const send = (event: PipelineEvent) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`)
  }

  emitter.on('progress', send)
  req.on('close', () => emitter.off('progress', send))
})

export default router

import { Router } from 'express'
import { getSchedulerStatus, setScheduler, triggerNow } from '../scheduler/dailyScanner'

const router = Router()

router.get('/status', (_req, res) => {
  res.json(getSchedulerStatus())
})

router.post('/toggle', (req, res) => {
  const { enabled, hour, minute } = req.body as {
    enabled: boolean
    hour?: number
    minute?: number
  }
  const status = setScheduler(enabled, hour, minute)
  res.json(status)
})

router.post('/run-now', async (_req, res) => {
  res.json({ ok: true })
  await triggerNow()
})

export default router

import cron from 'node-cron'
import fs from 'fs'
import path from 'path'
import { eq } from 'drizzle-orm'
import { randomUUID } from 'crypto'
import { db } from '../db/index'
import { accounts, syncJobs } from '../db/schema'
import { runPipeline } from '../pipeline/pipelineOrchestrator'
import { logger } from '../logger'
import { recordScanSnapshot } from '../analytics/snapshotRecorder'
import { rebuildFormats } from '../clustering/formatRepository'
import { generateInsights } from '../analytics/insightsGenerator'

const CONFIG_PATH = path.join(process.env['INSTANA_USER_DATA']!, 'scheduler.json')

interface SchedulerConfig {
  enabled: boolean
  hour: number
  minute: number
}

export interface SchedulerStatus {
  running: boolean
  nextRun: string | null
  lastRun: string | null
  config: SchedulerConfig
}

const DEFAULT_CONFIG: SchedulerConfig = { enabled: false, hour: 9, minute: 0 }

let currentTask: ReturnType<typeof cron.schedule> | null = null

function loadConfig(): SchedulerConfig {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8')) as SchedulerConfig
    }
  } catch {}
  return { ...DEFAULT_CONFIG }
}

function saveConfig(config: SchedulerConfig): void {
  fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true })
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config))
}

function nextRunTime(hour: number, minute: number): Date {
  const now = new Date()
  const next = new Date()
  next.setHours(hour, minute, 0, 0)
  if (next <= now) next.setDate(next.getDate() + 1)
  return next
}

async function runScheduledScan(): Promise<void> {
  const allAccounts = db.select().from(accounts).all()
  if (allAccounts.length === 0) return

  for (const account of allAccounts) {
    const jobId = randomUUID()

    const [job] = db
      .insert(syncJobs)
      .values({
        accountId: account.id,
        triggeredBy: 'scheduler',
        status: 'running',
        startedAt: new Date(),
      })
      .returning()
      .all()

    logger.info(`Scheduled scan starting for account ${account.id}`)

    runPipeline(jobId, account.id)
      .then(() => rebuildFormats(account.id))
      .then(() => generateInsights(account.id))
      .then(() => {
        db.update(syncJobs)
          .set({ status: 'done', finishedAt: new Date() })
          .where(eq(syncJobs.id, job!.id))
          .run()
        recordScanSnapshot(account.id)
        logger.info(`Scheduled scan done for account ${account.id}`)
      })
      .catch((err: unknown) => {
        logger.error(`Scheduled scan failed for ${account.id}: ${String(err)}`)
        db.update(syncJobs)
          .set({ status: 'error', finishedAt: new Date() })
          .where(eq(syncJobs.id, job!.id))
          .run()
      })
  }
}

export function startScheduler(): void {
  const config = loadConfig()
  if (!config.enabled) return
  const expr = `${config.minute} ${config.hour} * * *`
  currentTask = cron.schedule(expr, () => void runScheduledScan())
  logger.info(`Scheduler started: ${expr}`)
}

export function getSchedulerStatus(): SchedulerStatus {
  const config = loadConfig()

  const lastJob = db
    .select()
    .from(syncJobs)
    .where(eq(syncJobs.triggeredBy, 'scheduler'))
    .all()
    .sort((a, b) => (b.startedAt?.getTime() ?? 0) - (a.startedAt?.getTime() ?? 0))[0]

  const lastRun = lastJob?.startedAt?.toISOString() ?? null
  const nextRun = config.enabled
    ? nextRunTime(config.hour, config.minute).toISOString()
    : null

  return { running: currentTask !== null, nextRun, lastRun, config }
}

export function setScheduler(enabled: boolean, hour?: number, minute?: number): SchedulerStatus {
  if (currentTask) {
    currentTask.stop()
    currentTask = null
  }

  const prev = loadConfig()
  const config: SchedulerConfig = {
    enabled,
    hour: hour ?? prev.hour,
    minute: minute ?? prev.minute,
  }
  saveConfig(config)

  if (enabled) {
    const expr = `${config.minute} ${config.hour} * * *`
    currentTask = cron.schedule(expr, () => void runScheduledScan())
    logger.info(`Scheduler updated: ${expr}`)
  }

  return getSchedulerStatus()
}

export async function triggerNow(): Promise<void> {
  await runScheduledScan()
}

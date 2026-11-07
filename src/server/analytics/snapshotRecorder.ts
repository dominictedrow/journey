import { avg, count, eq } from 'drizzle-orm'
import { db } from '../db/index'
import { formats, scanSnapshots } from '../db/schema'
import { logger } from '../logger'

export function recordScanSnapshot(accountId: string): void {
  const [result] = db
    .select({ avgEngagement: avg(formats.avgEngagement), formatsCount: count(formats.id) })
    .from(formats)
    .where(eq(formats.accountId, accountId))
    .all()

  const avgEngagementRate = parseFloat(result?.avgEngagement ?? '0') || 0
  const formatsCount = result?.formatsCount ?? 0

  db.insert(scanSnapshots).values({ accountId, avgEngagementRate, formatsCount }).run()
  logger.info(`Scan snapshot recorded for ${accountId}: avgEngagement=${avgEngagementRate}, formats=${formatsCount}`)
}

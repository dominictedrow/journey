import { Router } from 'express'
import { eq } from 'drizzle-orm'
import { db } from '../db/index'
import { profiles, accounts } from '../db/schema'
import { tokenStore } from '../auth/tokenStore'
import { connectWithToken } from '../auth/oauth'

const router = Router()

router.get('/', async (_req, res) => {
  const [profile] = db.select().from(profiles).limit(1).all()
  if (!profile) {
    res.json(null)
    return
  }

  const userId = await tokenStore.getUserId()
  const account = userId
    ? (db.select().from(accounts).where(eq(accounts.id, userId)).limit(1).all()[0] ?? null)
    : null

  res.json({ profile, account })
})

router.post('/', async (req, res) => {
  const { username, pfpUrl, igToken, grokKey } = req.body as {
    username: string
    pfpUrl?: string
    igToken?: string
    grokKey?: string
  }

  const resolvedIgToken = igToken?.trim() || process.env['IG_ACCESS_TOKEN']?.trim() || (await tokenStore.getToken()) || ''
  const resolvedGrokKey = grokKey?.trim() || process.env['GROK_API_KEY']?.trim() || ''

  const [existing] = db.select().from(profiles).limit(1).all()
  let profile
  if (existing) {
    db.update(profiles)
      .set({ username, pfpUrl: pfpUrl ?? null })
      .where(eq(profiles.id, existing.id))
      .run()
    profile = { ...existing, username, pfpUrl: pfpUrl ?? null }
  } else {
    const [created] = db.insert(profiles)
      .values({ username, pfpUrl: pfpUrl ?? null })
      .returning()
      .all()
    profile = created
  }

  if (resolvedGrokKey) {
    await tokenStore.saveGrokKey(resolvedGrokKey)
  }

  let account = null
  if (resolvedIgToken) {
    const { userId } = await connectWithToken(resolvedIgToken)
    account = db.select().from(accounts).where(eq(accounts.id, userId)).limit(1).all()[0] ?? null
  }

  res.json({ profile, account })
})

export default router

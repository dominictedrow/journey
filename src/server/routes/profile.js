import { Router } from 'express';
import { eq } from 'drizzle-orm';
import { db } from '../db/index';
import { profiles, accounts } from '../db/schema';
import { tokenStore } from '../auth/tokenStore';
import { connectWithToken } from '../auth/oauth';
const router = Router();
router.get('/', async (_req, res) => {
    const [profile] = db.select().from(profiles).limit(1).all();
    if (!profile) {
        res.json(null);
        return;
    }
    const userId = await tokenStore.getUserId();
    const account = userId
        ? (db.select().from(accounts).where(eq(accounts.id, userId)).limit(1).all()[0] ?? null)
        : null;
    res.json({ profile, account });
});
router.post('/', async (req, res) => {
    const { username, pfpUrl, igToken, grokKey } = req.body;
    const [existing] = db.select().from(profiles).limit(1).all();
    let profile;
    if (existing) {
        db.update(profiles)
            .set({ username, pfpUrl: pfpUrl ?? null })
            .where(eq(profiles.id, existing.id))
            .run();
        profile = { ...existing, username, pfpUrl: pfpUrl ?? null };
    }
    else {
        const [created] = db.insert(profiles)
            .values({ username, pfpUrl: pfpUrl ?? null })
            .returning()
            .all();
        profile = created;
    }
    if (grokKey?.trim()) {
        await tokenStore.saveGrokKey(grokKey.trim());
    }
    let account = null;
    if (igToken?.trim()) {
        const { userId } = await connectWithToken(igToken.trim());
        account = db.select().from(accounts).where(eq(accounts.id, userId)).limit(1).all()[0] ?? null;
    }
    res.json({ profile, account });
});
export default router;

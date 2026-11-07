import { Router } from 'express';
import { eq } from 'drizzle-orm';
import { db } from '../db/index';
import { media, mediaInsights, videoAnalyses } from '../db/schema';
import { tokenStore } from '../auth/tokenStore';
import { fetchAllMedia } from '../instagram/mediaFetcher';
import { fetchAllInsights } from '../instagram/insightsFetcher';
const router = Router();
router.post('/fetch', async (_req, res) => {
    const userId = await tokenStore.getUserId();
    if (!userId) {
        res.status(401).json({ error: 'Not connected to Instagram' });
        return;
    }
    const fetched = await fetchAllMedia(userId);
    await fetchAllInsights(userId);
    res.json({ fetched });
});
router.get('/', async (req, res) => {
    const userId = await tokenStore.getUserId();
    if (!userId) {
        res.status(401).json({ error: 'Not connected to Instagram' });
        return;
    }
    const limit = Math.min(parseInt(req.query['limit'] ?? '50'), 200);
    const offset = parseInt(req.query['offset'] ?? '0');
    const items = db.select().from(media)
        .where(eq(media.accountId, userId))
        .limit(limit)
        .offset(offset)
        .all();
    res.json(items);
});
router.get('/:id', (req, res) => {
    const [item] = db.select().from(media).where(eq(media.id, req.params['id'])).all();
    if (!item) {
        res.status(404).json({ error: 'Not found' });
        return;
    }
    const [insights] = db.select().from(mediaInsights).where(eq(mediaInsights.mediaId, item.id)).all();
    const [analysis] = db.select().from(videoAnalyses).where(eq(videoAnalyses.mediaId, item.id)).all();
    res.json({ ...item, insights: insights ?? null, analysis: analysis ?? null });
});
export default router;

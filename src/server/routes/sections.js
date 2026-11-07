import { Router } from 'express';
import { eq } from 'drizzle-orm';
import { db } from '../db/index';
import { sections, syncJobs } from '../db/schema';
import { tokenStore } from '../auth/tokenStore';
import { fetchAllMedia } from '../instagram/mediaFetcher';
import { fetchAllInsights } from '../instagram/insightsFetcher';
import { logger } from '../logger';
const router = Router();
router.get('/', (_req, res) => {
    res.json(db.select().from(sections).all());
});
router.post('/', async (req, res) => {
    const { name, color, igProfileUrl } = req.body;
    const userId = await tokenStore.getUserId();
    if (!userId) {
        res.status(401).json({ error: 'Not connected to Instagram' });
        return;
    }
    const [section] = db.insert(sections)
        .values({ name, color, igProfileUrl, accountId: userId })
        .returning()
        .all();
    // Respond immediately, trigger scan in background
    res.json(section);
    const [job] = db.insert(syncJobs)
        .values({ accountId: userId, triggeredBy: 'manual', status: 'running', startedAt: new Date() })
        .returning()
        .all();
    fetchAllMedia(userId)
        .then((fetched) => fetchAllInsights(userId).then(() => fetched))
        .then((fetched) => {
        db.update(syncJobs)
            .set({ status: 'done', mediaFetched: fetched, finishedAt: new Date() })
            .where(eq(syncJobs.id, job.id))
            .run();
        logger.info(`Background scan done — ${fetched} media items`);
    })
        .catch((err) => {
        logger.error(`Background scan failed: ${String(err)}`);
        db.update(syncJobs)
            .set({ status: 'error', finishedAt: new Date() })
            .where(eq(syncJobs.id, job.id))
            .run();
    });
});
router.patch('/:id', (req, res) => {
    const sectionId = parseInt(req.params['id']);
    const [existing] = db.select().from(sections).where(eq(sections.id, sectionId)).all();
    if (!existing) {
        res.status(404).json({ error: 'Section not found' });
        return;
    }
    const { name, color, igProfileUrl } = req.body;
    db.update(sections)
        .set({
        name: name ?? existing.name,
        color: color ?? existing.color,
        igProfileUrl: igProfileUrl ?? existing.igProfileUrl,
    })
        .where(eq(sections.id, sectionId))
        .run();
    res.json({ ...existing, name: name ?? existing.name, color: color ?? existing.color, igProfileUrl: igProfileUrl ?? existing.igProfileUrl });
});
router.delete('/:id', (req, res) => {
    const sectionId = parseInt(req.params['id']);
    db.delete(sections).where(eq(sections.id, sectionId)).run();
    res.json({ ok: true });
});
export default router;

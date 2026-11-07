import { Router } from 'express';
import { tokenStore } from '../auth/tokenStore';
import { rebuildFormats, getFormats, getFormatDetail } from '../clustering/formatRepository';
import { logger } from '../logger';
const router = Router();
router.post('/rebuild', async (_req, res) => {
    const userId = await tokenStore.getUserId();
    if (!userId) {
        res.status(401).json({ error: 'Not connected to Instagram' });
        return;
    }
    try {
        const result = await rebuildFormats(userId);
        res.json(result);
    }
    catch (err) {
        logger.error(`rebuildFormats failed: ${err instanceof Error ? err.stack : String(err)}`);
        res.status(500).json({ error: 'Format rebuild failed' });
    }
});
router.get('/', async (_req, res) => {
    const userId = await tokenStore.getUserId();
    if (!userId) {
        res.status(401).json({ error: 'Not connected to Instagram' });
        return;
    }
    res.json(getFormats(userId));
});
router.get('/:id', (req, res) => {
    const id = parseInt(req.params['id']);
    if (isNaN(id)) {
        res.status(400).json({ error: 'Invalid format id' });
        return;
    }
    const format = getFormatDetail(id);
    if (!format) {
        res.status(404).json({ error: 'Format not found' });
        return;
    }
    res.json(format);
});
export default router;

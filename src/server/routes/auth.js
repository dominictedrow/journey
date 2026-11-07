import { Router } from 'express';
import { tokenStore } from '../auth/tokenStore';
import { connectInstagram, connectWithToken, disconnectInstagram } from '../auth/oauth';
const router = Router();
router.get('/status', async (_req, res) => {
    const token = await tokenStore.getToken();
    const expiry = await tokenStore.getTokenExpiry();
    const userId = await tokenStore.getUserId();
    if (!token || !userId) {
        res.json({ connected: false });
        return;
    }
    res.json({ connected: true, userId, expiresAt: expiry?.toISOString() ?? null });
});
router.post('/connect', async (_req, res) => {
    const { userId, username } = await connectInstagram();
    res.json({ userId, username });
});
router.post('/connect-token', async (req, res) => {
    const { token } = req.body;
    if (!token?.trim()) {
        res.status(400).json({ error: 'token required' });
        return;
    }
    const { userId, username } = await connectWithToken(token.trim());
    res.json({ userId, username });
});
router.delete('/disconnect', async (_req, res) => {
    await disconnectInstagram();
    res.json({ ok: true });
});
export default router;

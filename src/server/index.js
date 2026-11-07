import 'dotenv/config';
import express from 'express';
import net from 'net';
import { logger } from './logger';
import authRouter from './routes/auth';
import profileRouter from './routes/profile';
import mediaRouter from './routes/media';
import sectionsRouter from './routes/sections';
import pipelineRouter from './routes/pipeline';
import formatsRouter from './routes/formats';
import analyticsRouter from './routes/analytics';
import schedulerRouter from './routes/scheduler';
import { startScheduler } from './scheduler/dailyScanner';
import { tokenStore } from './auth/tokenStore';
import { settings } from './config/settings';
function findAvailablePort(preferred) {
    return new Promise((resolve, reject) => {
        const srv = net.createServer();
        srv.listen(preferred, () => {
            const addr = srv.address();
            srv.close(() => resolve(addr.port));
        });
        srv.on('error', () => {
            const fallback = net.createServer();
            fallback.listen(0, () => {
                const addr = fallback.address();
                fallback.close(() => resolve(addr.port));
            });
            fallback.on('error', reject);
        });
    });
}
export async function startServer() {
    const app = express();
    app.use(express.json());
    // Allow the Electron renderer (different port in dev) to call the API
    app.use((_req, res, next) => {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PATCH,DELETE,OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
        next();
    });
    app.get('/api/health', (_req, res) => {
        res.json({ status: 'ok', timestamp: new Date().toISOString() });
    });
    app.use('/api/auth', authRouter);
    app.use('/api/profile', profileRouter);
    app.use('/api/media', mediaRouter);
    app.use('/api/sections', sectionsRouter);
    app.use('/api/pipeline', pipelineRouter);
    app.use('/api/formats', formatsRouter);
    app.use('/api/analytics', analyticsRouter);
    app.use('/api/scheduler', schedulerRouter);
    app.get('/api/settings', async (_req, res) => {
        const hasGrokKey = !!(await tokenStore.getGrokKey());
        res.json({ hasGrokKey, frames: settings.frames });
    });
    app.post('/api/settings/grok-key', async (req, res) => {
        const { key } = req.body;
        await tokenStore.saveGrokKey(key);
        res.json({ ok: true });
    });
    app.use((err, _req, res, _next) => {
        logger.error(err.message, { stack: err.stack });
        res.status(500).json({ error: err.message });
    });
    const preferredPort = parseInt(process.env['PORT'] ?? '3001');
    const port = await findAvailablePort(preferredPort);
    await new Promise((resolve) => {
        app.listen(port, '127.0.0.1', () => {
            logger.info(`Server started on port ${port}`);
            startScheduler();
            resolve();
        });
    });
    return port;
}

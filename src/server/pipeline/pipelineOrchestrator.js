import path from 'path';
import os from 'os';
import { EventEmitter } from 'events';
import { and, eq } from 'drizzle-orm';
import { db } from '../db/index';
import { media, videoAnalyses, featureVectors } from '../db/schema';
import { encodeAnalysis, FEATURE_DIM } from '../clustering/featureEncoder';
import { downloadVideo, cleanupMedia } from './videoDownloader';
import { extractFrames } from './frameExtractor';
import { detectAudio } from './audioExtractor';
import { detectText, initOcrWorker, terminateOcrWorker } from './ocrProcessor';
import { detectCutsPerSecond } from './editPatternDetector';
import { analyzeFrames, resetGrokClient } from './grokAnalyzer';
import { fetchAllMedia } from '../instagram/mediaFetcher';
import { fetchAllInsights } from '../instagram/insightsFetcher';
import { settings } from '../config/settings';
import { logger } from '../logger';
function concurrencyLimiter(limit) {
    let active = 0;
    const queue = [];
    return function (fn) {
        return new Promise((resolve, reject) => {
            function attempt() {
                if (active < limit) {
                    active++;
                    fn()
                        .then(resolve, reject)
                        .finally(() => {
                        active--;
                        queue.shift()?.();
                    });
                }
                else {
                    queue.push(attempt);
                }
            }
            attempt();
        });
    };
}
const emitters = new Map();
export function getJobEmitter(jobId) {
    return emitters.get(jobId);
}
export async function runPipeline(jobId, accountId, force = false) {
    const emitter = new EventEmitter();
    emitters.set(jobId, emitter);
    function emit(event) {
        emitter.emit('progress', { jobId, ...event });
    }
    emit({ mediaId: '', step: 'fetching_media', current: 0, total: 0, status: 'running' });
    await fetchAllMedia(accountId);
    emit({ mediaId: '', step: 'fetching_insights', current: 0, total: 0, status: 'running' });
    await fetchAllInsights(accountId);
    const videoMedia = db
        .select()
        .from(media)
        .where(eq(media.accountId, accountId))
        .all()
        .filter((m) => m.mediaType === 'REEL' || m.mediaType === 'VIDEO');
    // Load structural fingerprints from prior completed analyses for this account.
    // These serve as template anchors so Grok uses consistent phrasing across scans.
    // Loaded before any force-delete so a re-scan still benefits from previous classifications.
    const priorRows = db
        .select({ grokRawJson: videoAnalyses.grokRawJson })
        .from(videoAnalyses)
        .innerJoin(media, eq(videoAnalyses.mediaId, media.id))
        .where(and(eq(media.accountId, accountId), eq(videoAnalyses.status, 'done')))
        .all();
    const knownFingerprints = priorRows.flatMap((row) => {
        if (!row.grokRawJson)
            return [];
        try {
            const parsed = JSON.parse(row.grokRawJson);
            const fp = parsed.structural_fingerprint ?? parsed.distinguishing_features ?? [];
            return fp.length > 0 ? [fp] : [];
        }
        catch {
            return [];
        }
    });
    if (force && videoMedia.length > 0) {
        const { inArray } = await import('drizzle-orm');
        const ids = videoMedia.map((m) => m.id);
        db.delete(videoAnalyses).where(inArray(videoAnalyses.mediaId, ids)).run();
        logger.info(`Pipeline ${jobId}: force-reset ${ids.length} video analyses`);
    }
    const total = videoMedia.length;
    logger.info(`Pipeline ${jobId}: ${total} videos to process`);
    resetGrokClient();
    await initOcrWorker();
    try {
        const limit = concurrencyLimiter(3);
        await Promise.all(videoMedia.map((item, i) => limit(async () => {
            const [existing] = db
                .select()
                .from(videoAnalyses)
                .where(eq(videoAnalyses.mediaId, item.id))
                .all();
            if (!force && (existing?.status === 'done' || existing?.status === 'error')) {
                emit({ mediaId: item.id, step: 'skipped', current: i + 1, total, status: 'running' });
                return;
            }
            emit({ mediaId: item.id, step: 'starting', current: i, total, status: 'running' });
            if (existing) {
                db.update(videoAnalyses)
                    .set({ status: 'processing' })
                    .where(eq(videoAnalyses.id, existing.id))
                    .run();
            }
            else {
                db.insert(videoAnalyses).values({ mediaId: item.id, status: 'processing' }).run();
            }
            try {
                // ── INSTAGRAM GRAPH API NOTE ──
                // item.mediaUrl is populated by POST /api/media/fetch (Phase 3).
                // Signed URLs expire in a few hours. If you see download errors, re-run the fetch first.
                if (!item.mediaUrl) {
                    throw new Error('No media URL — run a fresh media fetch before scanning');
                }
                const tmpBase = path.join(os.tmpdir(), 'instana', item.id);
                emit({ mediaId: item.id, step: 'downloading', current: i, total, status: 'running' });
                const videoPath = await downloadVideo(item.id, item.mediaUrl);
                emit({ mediaId: item.id, step: 'extracting_frames', current: i, total, status: 'running' });
                const [{ paths: framePaths, durationSeconds }, hasAudio] = await Promise.all([
                    extractFrames(videoPath, tmpBase, settings.frames.totalExtracted),
                    detectAudio(videoPath),
                ]);
                emit({ mediaId: item.id, step: 'ocr', current: i, total, status: 'running' });
                const [detectedText, cutsPerSecond] = await Promise.all([
                    detectText(framePaths.slice(0, 3)),
                    detectCutsPerSecond(framePaths, durationSeconds),
                ]);
                emit({ mediaId: item.id, step: 'grok_analysis', current: i, total, status: 'running' });
                const analysis = await analyzeFrames(framePaths, durationSeconds, detectedText, hasAudio, cutsPerSecond, knownFingerprints);
                db.update(videoAnalyses)
                    .set({
                    detectedText,
                    cutsPerSecond,
                    footageType: analysis.footage_type,
                    backgroundType: analysis.scene.background,
                    sceneJson: JSON.stringify(analysis.scene),
                    textStyleJson: JSON.stringify(analysis.text_overlays),
                    audioType: analysis.audio_type,
                    hookStrength: analysis.hook_strength,
                    productionQuality: analysis.production_quality,
                    grokRawJson: JSON.stringify(analysis),
                    status: 'done',
                })
                    .where(eq(videoAnalyses.mediaId, item.id))
                    .run();
                const vec = encodeAnalysis(analysis, cutsPerSecond);
                const vecBuf = Buffer.from(vec.buffer);
                const existingVec = db.select().from(featureVectors).where(eq(featureVectors.mediaId, item.id)).all();
                if (existingVec.length > 0) {
                    db.update(featureVectors).set({ vector: vecBuf, dimension: FEATURE_DIM }).where(eq(featureVectors.mediaId, item.id)).run();
                }
                else {
                    db.insert(featureVectors).values({ mediaId: item.id, vector: vecBuf, dimension: FEATURE_DIM }).run();
                }
                cleanupMedia(item.id);
                emit({ mediaId: item.id, step: 'done', current: i + 1, total, status: 'running' });
            }
            catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                logger.error(`Pipeline error on ${item.id}: ${msg}`);
                db.update(videoAnalyses)
                    .set({ status: 'error' })
                    .where(eq(videoAnalyses.mediaId, item.id))
                    .run();
                cleanupMedia(item.id);
                emit({ mediaId: item.id, step: 'error', current: i + 1, total, status: 'running', error: msg });
            }
        })));
    }
    finally {
        await terminateOcrWorker();
    }
    emit({ mediaId: '', step: 'complete', current: total, total, status: 'done' });
    emitters.delete(jobId);
}

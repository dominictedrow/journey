import axios from 'axios';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { createWriteStream } from 'fs';
import { logger } from '../logger';
function tmpDir(mediaId) {
    return path.join(os.tmpdir(), 'instana', mediaId);
}
export async function downloadVideo(mediaId, videoUrl) {
    // ── INSTAGRAM GRAPH API NOTE ──
    // videoUrl comes from media.media_url stored during the Phase 3 media fetch.
    // Instagram signed URLs expire after a few hours. If download fails with 4xx,
    // run POST /api/media/fetch again to refresh URLs, then re-scan.
    const dir = tmpDir(mediaId);
    fs.mkdirSync(dir, { recursive: true });
    const dest = path.join(dir, 'video.mp4');
    if (fs.existsSync(dest))
        return dest;
    const response = await axios.get(videoUrl, {
        responseType: 'stream',
        timeout: 60_000,
    });
    await new Promise((resolve, reject) => {
        const writer = createWriteStream(dest);
        response.data.pipe(writer);
        writer.on('finish', resolve);
        writer.on('error', reject);
    });
    logger.info(`Downloaded video for ${mediaId} → ${dest}`);
    return dest;
}
export function cleanupMedia(mediaId) {
    const dir = tmpDir(mediaId);
    if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
        logger.info(`Cleaned up temp files for ${mediaId}`);
    }
}

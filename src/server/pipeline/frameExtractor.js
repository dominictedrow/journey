import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';
import { settings } from '../config/settings';
if (process.env['FFMPEG_PATH'])
    ffmpeg.setFfmpegPath(process.env['FFMPEG_PATH']);
if (process.env['FFPROBE_PATH'])
    ffmpeg.setFfprobePath(process.env['FFPROBE_PATH']);
function getVideoDuration(videoPath) {
    return new Promise((resolve, reject) => {
        ffmpeg.ffprobe(videoPath, (err, meta) => {
            if (err)
                reject(err);
            else
                resolve(meta.format.duration ?? 0);
        });
    });
}
function extractFrameAt(videoPath, timestampSecs, outPath) {
    return new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .seekInput(timestampSecs)
            .outputOptions(['-vframes', '1', '-q:v', '2'])
            .output(outPath)
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}
export async function extractFrames(videoPath, outputDir, count = settings.frames.totalExtracted) {
    const duration = await getVideoDuration(videoPath);
    const framesDir = path.join(outputDir, 'frames');
    fs.mkdirSync(framesDir, { recursive: true });
    // Sample at evenly spaced intervals, offset by half an interval to avoid black frames at t=0
    const interval = duration / count;
    const timestamps = Array.from({ length: count }, (_, i) => (i + 0.5) * interval);
    const framePaths = (await Promise.all(timestamps.map(async (ts, i) => {
        const outPath = path.join(framesDir, `frame_${String(i).padStart(3, '0')}.jpg`);
        await extractFrameAt(videoPath, ts, outPath);
        return fs.existsSync(outPath) ? outPath : null;
    }))).filter((p) => p !== null);
    return { paths: framePaths.sort(), durationSeconds: duration };
}

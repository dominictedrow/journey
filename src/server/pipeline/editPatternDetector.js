import sharp from 'sharp';
const CUT_THRESHOLD = 30; // mean absolute pixel difference per channel to count as a cut
export async function detectCutsPerSecond(framePaths, durationSeconds) {
    if (framePaths.length < 2 || durationSeconds <= 0)
        return 0;
    const results = await Promise.all(Array.from({ length: framePaths.length - 1 }, async (_, i) => {
        const [prev, curr] = await Promise.all([
            sharp(framePaths[i]).resize(64, 64).grayscale().raw().toBuffer(),
            sharp(framePaths[i + 1]).resize(64, 64).grayscale().raw().toBuffer(),
        ]);
        let diff = 0;
        for (let j = 0; j < prev.length; j++) {
            diff += Math.abs((prev[j] ?? 0) - (curr[j] ?? 0));
        }
        return diff / prev.length > CUT_THRESHOLD ? 1 : 0;
    }));
    return results.reduce((a, b) => a + b, 0) / durationSeconds;
}

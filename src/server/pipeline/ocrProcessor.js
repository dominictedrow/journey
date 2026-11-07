import { createWorker, PSM } from 'tesseract.js';
import { logger } from '../logger';
let sharedWorker = null;
function workerOptions() {
    const langPath = process.env['INSTANA_LANG_PATH'];
    return { logger: () => { }, ...(langPath ? { langPath } : {}) };
}
export async function initOcrWorker() {
    sharedWorker = await createWorker('eng', 1, workerOptions());
    await sharedWorker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT });
}
export async function terminateOcrWorker() {
    if (sharedWorker) {
        await sharedWorker.terminate();
        sharedWorker = null;
    }
}
export async function detectText(framePaths) {
    if (framePaths.length === 0)
        return '';
    let worker;
    let owned = false;
    if (sharedWorker) {
        worker = sharedWorker;
    }
    else {
        worker = await createWorker('eng', 1, workerOptions());
        await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT });
        owned = true;
    }
    const texts = [];
    for (const p of framePaths) {
        try {
            const { data } = await worker.recognize(p);
            const trimmed = data.text.trim();
            if (trimmed)
                texts.push(trimmed);
        }
        catch {
            logger.warn(`OCR failed for frame ${p}`);
        }
    }
    if (owned)
        await worker.terminate();
    return texts.join(' | ');
}

import { createWorker, PSM, type Worker } from 'tesseract.js'
import { logger } from '../logger'

let sharedWorker: Worker | null = null

function workerOptions() {
  const langPath = process.env['INSTANA_LANG_PATH']
  return { logger: () => {}, ...(langPath ? { langPath } : {}) }
}

export async function initOcrWorker(): Promise<void> {
  sharedWorker = await createWorker('eng', 1, workerOptions())
  await sharedWorker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT })
}

export async function terminateOcrWorker(): Promise<void> {
  if (sharedWorker) {
    await sharedWorker.terminate()
    sharedWorker = null
  }
}

export async function detectText(framePaths: string[]): Promise<string> {
  if (framePaths.length === 0) return ''

  let worker: Worker
  let owned = false

  if (sharedWorker) {
    worker = sharedWorker
  } else {
    worker = await createWorker('eng', 1, workerOptions())
    await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT })
    owned = true
  }

  const texts: string[] = []
  for (const p of framePaths) {
    try {
      const { data } = await worker.recognize(p)
      const trimmed = data.text.trim()
      if (trimmed) texts.push(trimmed)
    } catch {
      logger.warn(`OCR failed for frame ${p}`)
    }
  }

  if (owned) await worker.terminate()
  return texts.join(' | ')
}

import { app, BrowserWindow } from 'electron'
import path from 'path'

// Must be set before any server modules are required — static imports are hoisted
// and would run before these assignments, causing logger to fall back to process.cwd() = '/'
process.env['INSTANA_USER_DATA'] = app.getPath('userData')
process.env['INSTANA_APP_PATH'] = app.getAppPath()
process.env['META_APP_ID'] = '1138251245155979'
process.env['META_APP_SECRET'] = '597cf1f0e574343167ac5c714eadb571'

// eslint-disable-next-line @typescript-eslint/no-require-imports
const ffmpegStatic = require('ffmpeg-static') as string
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ffprobeStatic = (require('ffprobe-static') as { path: string }).path

// Replace app.asar with app.asar.unpacked so binaries run outside the archive
function unpackBin(p: string): string {
  return p.replace(/app\.asar(?!\.unpacked)/, 'app.asar.unpacked')
}

process.env['FFMPEG_PATH'] = app.isPackaged ? unpackBin(ffmpegStatic) : ffmpegStatic
process.env['FFPROBE_PATH'] = app.isPackaged ? unpackBin(ffprobeStatic) : ffprobeStatic
process.env['INSTANA_LANG_PATH'] = app.isPackaged
  ? path.join(process.resourcesPath, 'traineddata')
  : process.cwd()

const ICON = path.join(app.getAppPath(), 'assets', 'icon.png')

async function createWindow() {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { startServer } = require('../server/index') as typeof import('../server/index')
  const port = await startServer()

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: '#000000',
    icon: ICON,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      additionalArguments: [`--api-port=${port}`],
    },
  })

  if (process.env['VITE_DEV_SERVER_URL']) {
    win.loadURL(process.env['VITE_DEV_SERVER_URL'])
  } else {
    win.loadFile(path.join(__dirname, '../dist/index.html'))
  }
}

app.whenReady().then(() => {
  app.dock?.setIcon(ICON)
  createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

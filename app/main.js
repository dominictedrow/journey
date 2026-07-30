const { app, BrowserWindow, ipcMain, session } = require('electron');
const path = require('node:path');
const { spawn } = require('node:child_process');

const registry = require('./registry');
const db = require('./db');
const settings = require('./settings');

const isDev = !app.isPackaged;
const DEV_SERVER_URL = 'http://localhost:5173';

// Where `skills/` and `scripts/` actually live on disk: the repo root in
// dev, or the packaged app's Resources dir (see the `extraResources` entries
// in package.json's `build` config) once built by electron-builder. Needed
// because `conda run` spawns a real OS process that reads skills/*.py off
// disk directly — it can't reach into an asar archive the way Electron's
// own `require()`/`loadFile()` can.
const RESOURCES_ROOT = isDev ? path.join(__dirname, '..') : process.resourcesPath;

// Vite dev mode injects its React Fast Refresh preamble as an inline
// <script> tag, so script-src needs 'unsafe-inline' in dev only — the
// production build (loaded from disk, no dev server) has no inline scripts
// and keeps the strict policy.
const CSP =
  "default-src 'self'; " +
  `script-src 'self'${isDev ? " 'unsafe-inline'" : ''}; ` +
  "style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data:; " +
  `connect-src 'self'${isDev ? ' ws://localhost:5173 http://localhost:5173' : ''}; ` +
  "object-src 'none';";

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 820,
    minHeight: 560,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#07090B',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      devTools: isDev,
    },
  });

  win.webContents.on('will-navigate', (event, url) => {
    if (isDev && url.startsWith(DEV_SERVER_URL)) return;
    event.preventDefault();
  });

  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));

  if (isDev) {
    win.loadURL(DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'ui', 'index.html'));
  }

  return win;
}

app.whenReady().then(() => {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [CSP],
      },
    });
  });

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// --- skills ---

function findSkill(id) {
  const skill = registry.find((s) => s.id === id);
  if (!skill) throw new Error(`unknown skill id: ${id}`);
  return skill;
}

ipcMain.handle('skills:list', () => registry);

ipcMain.handle('skills:lastResults', () => {
  return db.listSkillRuns().map((row) => JSON.parse(row.envelope));
});

ipcMain.handle('skills:run', (event, id, input) => {
  const skill = findSkill(id);
  const args = ['run', '-n', 'c_work', 'python', '-m', `skills.${skill.id}`];
  if (input !== undefined && input !== null) {
    args.push('--input', JSON.stringify(input));
  }

  return new Promise((resolve) => {
    const child = spawn('conda', args, { shell: false, cwd: RESOURCES_ROOT });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk;
      for (const line of chunk.toString().split('\n')) {
        if (line.trim()) event.sender.send('skills:log', { id: skill.id, line });
      }
    });

    child.on('close', (code) => {
      let envelope;
      try {
        envelope = JSON.parse(stdout.trim());
      } catch {
        envelope = {
          skill: skill.id,
          ok: false,
          ran_at: new Date().toISOString(),
          summary: 'Skill failed to produce a valid envelope',
          sections: [],
          error: stderr.trim() || `process exited with code ${code}`,
        };
      }
      db.recordSkillRun(skill.id, envelope);
      resolve(envelope);
    });
  });
});

// --- classes ---

ipcMain.handle('classes:list', () => db.listClasses());

ipcMain.handle('classes:add', (event, name) => {
  if (typeof name !== 'string' || !name.trim()) throw new Error('class name must be a non-empty string');
  return db.addClass(name.trim());
});

ipcMain.handle('classes:rename', (event, id, name) => {
  if (!Number.isInteger(id)) throw new Error('class id must be an integer');
  if (typeof name !== 'string' || !name.trim()) throw new Error('class name must be a non-empty string');
  db.renameClass(id, name.trim());
});

ipcMain.handle('classes:delete', (event, id) => {
  if (!Number.isInteger(id)) throw new Error('class id must be an integer');
  db.deleteClass(id);
});

// --- assignments ---

ipcMain.handle('assignments:list', () => db.listAssignments());

ipcMain.handle('assignments:add', (event, input) => {
  const { classId, title, dueDate } = input || {};
  if (!Number.isInteger(classId)) throw new Error('classId must be an integer');
  if (typeof title !== 'string' || !title.trim()) throw new Error('title must be a non-empty string');
  if (typeof dueDate !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(dueDate)) {
    throw new Error('dueDate must be an ISO date string (YYYY-MM-DD)');
  }
  return db.addAssignment({ classId, title: title.trim(), dueDate });
});

ipcMain.handle('assignments:setDone', (event, id, done) => {
  if (!Number.isInteger(id)) throw new Error('assignment id must be an integer');
  db.setAssignmentDone(id, Boolean(done));
});

ipcMain.handle('assignments:delete', (event, id) => {
  if (!Number.isInteger(id)) throw new Error('assignment id must be an integer');
  db.deleteAssignment(id);
});

// --- settings ---

ipcMain.handle('settings:get', () => settings.getSettings(RESOURCES_ROOT));

ipcMain.handle('settings:updateHome', (event, home) => settings.updateHome(RESOURCES_ROOT, home));

ipcMain.handle('settings:updateMeals', (event, meals) => settings.updateMeals(RESOURCES_ROOT, meals));

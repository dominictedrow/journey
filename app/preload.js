// contextBridge — the ONLY surface the renderer can touch. No Node, no fs,
// no shell; every call here maps to one validated IPC handler in main.js.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cwork', {
  skills: {
    list: () => ipcRenderer.invoke('skills:list'),
    run: (id, input) => ipcRenderer.invoke('skills:run', id, input),
    lastResults: () => ipcRenderer.invoke('skills:lastResults'),
    onLog: (callback) => {
      const handler = (_event, payload) => callback(payload);
      ipcRenderer.on('skills:log', handler);
      return () => ipcRenderer.removeListener('skills:log', handler);
    },
  },
  classes: {
    list: () => ipcRenderer.invoke('classes:list'),
    add: (name) => ipcRenderer.invoke('classes:add', name),
    rename: (id, name) => ipcRenderer.invoke('classes:rename', id, name),
    delete: (id) => ipcRenderer.invoke('classes:delete', id),
  },
  assignments: {
    list: () => ipcRenderer.invoke('assignments:list'),
    add: (input) => ipcRenderer.invoke('assignments:add', input),
    setDone: (id, done) => ipcRenderer.invoke('assignments:setDone', id, done),
    delete: (id) => ipcRenderer.invoke('assignments:delete', id),
  },
  settings: {
    get: () => ipcRenderer.invoke('settings:get'),
    updateHome: (home) => ipcRenderer.invoke('settings:updateHome', home),
    updateMeals: (meals) => ipcRenderer.invoke('settings:updateMeals', meals),
  },
});

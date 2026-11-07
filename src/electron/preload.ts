import { contextBridge } from 'electron'

const apiPortArg = process.argv.find(arg => arg.startsWith('--api-port='))
const apiPort = apiPortArg ? parseInt(apiPortArg.split('=')[1]!) : 3001

contextBridge.exposeInMainWorld('electronAPI', {
  apiPort,
})

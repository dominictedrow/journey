import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import electron from 'vite-plugin-electron/simple'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'path'
import { builtinModules } from 'module'

// All node_modules that must stay external in the Electron main/preload bundles
const electronExternal = [
  'electron',
  ...builtinModules,
  ...builtinModules.map((m) => `node:${m}`),
  'better-sqlite3',
  'keytar',
  'hnswlib-node',
  'sharp',
  'fluent-ffmpeg',
  'tesseract.js',
  'ffmpeg-static',
  'ffprobe-static',
]

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    electron({
      main: {
        entry: 'src/electron/main.ts',
        vite: {
          build: {
            rollupOptions: {
              external: electronExternal,
            },
          },
        },
      },
      preload: {
        input: 'src/electron/preload.ts',
        vite: {
          build: {
            rollupOptions: {
              external: electronExternal,
            },
          },
        },
      },
      renderer: {},
    }),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
})

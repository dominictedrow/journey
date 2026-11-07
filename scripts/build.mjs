import * as esbuild from 'esbuild'
import { compile, optimize } from '@tailwindcss/node'
import { Scanner } from '@tailwindcss/oxide'
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { builtinModules } from 'module'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, '..')

rmSync(resolve(root, 'dist'), { recursive: true, force: true })
rmSync(resolve(root, 'dist-electron'), { recursive: true, force: true })
mkdirSync(resolve(root, 'dist/assets'), { recursive: true })
mkdirSync(resolve(root, 'dist-electron'), { recursive: true })

const nodeExternal = [
  'electron',
  ...builtinModules,
  ...builtinModules.map(m => `node:${m}`),
  'better-sqlite3',
  'keytar',
  'hnswlib-node',
  'sharp',
  'fluent-ffmpeg',
  'tesseract.js',
  'ffmpeg-static',
  'ffprobe-static',
]

// Intercept CSS imports during JS bundling; Tailwind is compiled separately below
const skipCSSPlugin = {
  name: 'skip-css',
  setup(build) {
    // Return a no-op JS module so no CSS file is emitted alongside index.js
    build.onLoad({ filter: /\.css$/ }, () => ({ contents: '', loader: 'js' }))
  },
}

console.log('Building main process…')
await esbuild.build({
  entryPoints: [resolve(root, 'src/electron/main.ts')],
  bundle: true,
  platform: 'node',
  target: 'node20',
  format: 'cjs',
  outfile: resolve(root, 'dist-electron/main.js'),
  external: nodeExternal,
  alias: { '@': resolve(root, 'src') },
  minify: true,
})

console.log('Building preload…')
await esbuild.build({
  entryPoints: [resolve(root, 'src/electron/preload.ts')],
  bundle: true,
  platform: 'node',
  target: 'node20',
  format: 'cjs',
  outfile: resolve(root, 'dist-electron/preload.js'),
  external: nodeExternal,
  alias: { '@': resolve(root, 'src') },
  minify: true,
})

console.log('Building renderer…')
await esbuild.build({
  entryPoints: [resolve(root, 'src/renderer/main.tsx')],
  bundle: true,
  platform: 'browser',
  target: ['chrome120'],
  format: 'iife',
  outfile: resolve(root, 'dist/assets/index.js'),
  jsx: 'automatic',
  alias: { '@': resolve(root, 'src') },
  minify: true,
  plugins: [skipCSSPlugin],
})

console.log('Building Tailwind CSS…')
const cssInput = readFileSync(resolve(root, 'tailwind.css'), 'utf8')
const compiler = await compile(cssInput, {
  base: root,
  onDependency: () => {},
})
const scanner = new Scanner({
  sources: [{ base: resolve(root, 'src'), pattern: '**/*.{ts,tsx}', negated: false }],
})
const candidates = scanner.scan()
const css = optimize(compiler.build(candidates)).code
writeFileSync(resolve(root, 'dist/assets/style.css'), css)

console.log('Writing index.html…')
writeFileSync(
  resolve(root, 'dist/index.html'),
  `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Instana</title>
    <link rel="stylesheet" href="./assets/style.css">
  </head>
  <body>
    <div id="root"></div>
    <script src="./assets/index.js"></script>
  </body>
</html>
`,
)

console.log('Done.')

import { existsSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { build } from 'electron-builder'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const tokensPath = resolve(root, 'tokens.json')

if (!existsSync(tokensPath)) {
  console.error('\nError: tokens.json not found.')
  console.error('Create tokens.json in the project root before building the .pkg:\n')
  console.error('  {')
  console.error('    "xaiApiKey": "<your xAI key>",')
  console.error('    "instagramToken": "<long-lived Instagram token>",')
  console.error('    "instagramUserId": "<numeric Instagram user ID>",')
  console.error('    "tokenExpiresAt": "<ISO timestamp ~60 days from now>"')
  console.error('  }\n')
  process.exit(1)
}

await build({
  mac: ['pkg'],
  config: {
    extraResources: [
      { from: 'eng.traineddata', to: 'traineddata/eng.traineddata' },
      { from: 'eng.traineddata.gz', to: 'traineddata/eng.traineddata.gz' },
      { from: 'tokens.json', to: 'tokens.json' },
    ],
    mac: {
      target: [{ target: 'pkg', arch: ['arm64'] }],
    },
    pkg: {
      scripts: 'build/pkg-scripts',
      installLocation: '/Applications',
    },
  },
})

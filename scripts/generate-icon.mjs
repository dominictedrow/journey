import sharp from 'sharp'
import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.join(__dirname, '..')
const ASSETS = path.join(ROOT, 'assets')
const ICONSET = path.join(ASSETS, 'icon.iconset')

fs.mkdirSync(ICONSET, { recursive: true })

const SIZE = 1024

// Render the U on black to measure its actual pixel bounding box
const measureSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}" viewBox="0 0 ${SIZE} ${SIZE}">
  <rect width="${SIZE}" height="${SIZE}" fill="black"/>
  <text
    x="512" y="512"
    text-anchor="middle"
    dominant-baseline="central"
    font-family="Snell Roundhand"
    font-size="780"
    font-weight="bold"
    fill="white"
  >U</text>
</svg>`

const measureBuf = await sharp(Buffer.from(measureSvg)).png().toBuffer()
const { info } = await sharp(measureBuf)
  .trim({ background: { r: 0, g: 0, b: 0 }, threshold: 10 })
  .toBuffer({ resolveWithObject: true })

// sharp returns negative offsets: -trimOffsetLeft = pixels from left edge to content
const left = -info.trimOffsetLeft
const top = -info.trimOffsetTop
const right = left + info.width
const bottom = top + info.height
const visualCenterX = (left + right) / 2
const visualCenterY = (top + bottom) / 2

// Shift the text position so the visual center lands exactly at (512, 512), nudged up 70px
const textX = 512 + (512 - visualCenterX)
const textY = 512 + (512 - visualCenterY)

console.log(`Bounds: (${left}, ${top}) → (${right}, ${bottom})`)
console.log(`Visual center: (${visualCenterX.toFixed(1)}, ${visualCenterY.toFixed(1)})`)
console.log(`Adjusted text position: (${textX.toFixed(1)}, ${textY.toFixed(1)})`)

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}" viewBox="0 0 ${SIZE} ${SIZE}">
  <defs>
    <linearGradient id="ig" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#FCAF45"/>
      <stop offset="25%"  stop-color="#FD1D1D"/>
      <stop offset="50%"  stop-color="#E1306C"/>
      <stop offset="75%"  stop-color="#833AB4"/>
      <stop offset="100%" stop-color="#405DE6"/>
    </linearGradient>
  </defs>
  <rect width="${SIZE}" height="${SIZE}" rx="235" ry="235" fill="url(#ig)"/>
  <text
    x="${textX}"
    y="${textY}"
    text-anchor="middle"
    dominant-baseline="central"
    font-family="Snell Roundhand"
    font-size="780"
    font-weight="bold"
    fill="white"
  >U</text>
</svg>`

const master = await sharp(Buffer.from(svg)).png().toBuffer()
fs.writeFileSync(path.join(ASSETS, 'icon.png'), master)
console.log('✓ assets/icon.png')

const sizes = [
  { size: 16,   name: 'icon_16x16.png' },
  { size: 32,   name: 'icon_16x16@2x.png' },
  { size: 32,   name: 'icon_32x32.png' },
  { size: 64,   name: 'icon_32x32@2x.png' },
  { size: 128,  name: 'icon_128x128.png' },
  { size: 256,  name: 'icon_128x128@2x.png' },
  { size: 256,  name: 'icon_256x256.png' },
  { size: 512,  name: 'icon_256x256@2x.png' },
  { size: 512,  name: 'icon_512x512.png' },
  { size: 1024, name: 'icon_512x512@2x.png' },
]

for (const { size, name } of sizes) {
  const buf = await sharp(master).resize(size, size).png().toBuffer()
  fs.writeFileSync(path.join(ICONSET, name), buf)
}

execSync(`iconutil -c icns "${ICONSET}" -o "${path.join(ASSETS, 'icon.icns')}"`)
console.log('✓ assets/icon.icns')

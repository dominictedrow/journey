import ffmpeg from 'fluent-ffmpeg'

if (process.env['FFMPEG_PATH']) ffmpeg.setFfmpegPath(process.env['FFMPEG_PATH'])
if (process.env['FFPROBE_PATH']) ffmpeg.setFfprobePath(process.env['FFPROBE_PATH'])

export function detectAudio(videoPath: string): Promise<boolean> {
  return new Promise((resolve) => {
    ffmpeg.ffprobe(videoPath, (err, meta) => {
      if (err) {
        resolve(false)
        return
      }
      const hasAudio = meta.streams.some((s) => s.codec_type === 'audio')
      resolve(hasAudio)
    })
  })
}

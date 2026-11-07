export function computeEngagementRate(
  likes: number,
  comments: number,
  reposts: number,
  sends: number,
  plays: number,
): number {
  if (plays === 0) return 0
  return ((likes + comments + reposts + sends) / plays) * 100
}

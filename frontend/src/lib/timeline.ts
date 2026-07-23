export interface TimelineMarker {
  left: number
  width: number
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

export function timelineMarker(start: number, end: number, duration: number): TimelineMarker {
  if (!Number.isFinite(duration) || duration <= 0) return { left: 0, width: 0 }
  const safeStart = clamp(start, 0, duration)
  const safeEnd = clamp(Math.max(end, safeStart), 0, duration)
  const left = (safeStart / duration) * 100
  const naturalWidth = ((safeEnd - safeStart) / duration) * 100
  return {
    left,
    width: Math.min(100 - left, naturalWidth > 0 ? Math.max(0.8, naturalWidth) : 0),
  }
}

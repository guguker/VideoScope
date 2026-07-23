import type { ClipDraft, SearchResult } from '../types'

export function addClip(queue: ClipDraft[], result: SearchResult): ClipDraft[] {
  const id = `${result.video_id}:${result.start.toFixed(3)}:${result.end.toFixed(3)}`
  if (queue.some((clip) => clip.id === id)) return queue
  return [
    ...queue,
    {
      id,
      videoId: result.video_id,
      videoName: result.video_name,
      start: Math.max(0, result.start),
      end: Math.max(result.start + 0.2, result.end),
      thumbnailUrl: result.thumbnail_url,
    },
  ]
}

export function updateClip(
  clip: ClipDraft,
  changes: Partial<Pick<ClipDraft, 'start' | 'end'>>,
  duration: number,
): ClipDraft {
  const requestedStart = changes.start ?? clip.start
  const requestedEnd = changes.end ?? clip.end
  const [orderedStart, orderedEnd] = [requestedStart, requestedEnd].sort((left, right) => left - right)
  const start = Math.max(0, Math.min(duration, orderedStart))
  const end = Math.max(start, Math.min(duration, orderedEnd))
  return { ...clip, start, end }
}


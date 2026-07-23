import { addClip, updateClip } from './clipQueue'
import type { ClipDraft, SearchResult } from '../types'

const result: SearchResult = {
  id: 'result-1',
  video_id: 'video-1',
  video_name: 'match.mp4',
  start: 10,
  end: 16,
  score: 0.9,
  modalities: ['visual'],
  evidence: [],
  thumbnail_url: null,
  intent: 'visual',
  explanation: 'Визуальный поиск',
  refined: false,
}

describe('clip queue', () => {
  it('adds a search result only once', () => {
    const first = addClip([], result)
    const second = addClip(first, result)

    expect(first).toHaveLength(1)
    expect(second).toEqual(first)
  })

  it('keeps edited boundaries ordered and inside video duration', () => {
    const clip: ClipDraft = addClip([], result)[0]

    const updated = updateClip(clip, { start: 30, end: -5 }, 20)

    expect(updated.start).toBe(0)
    expect(updated.end).toBe(20)
  })
})

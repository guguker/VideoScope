import { describe, expect, it } from 'vitest'
import { timelineMarker } from './timeline'

describe('timelineMarker', () => {
  it('converts an interval to stable percentages', () => {
    expect(timelineMarker(30, 45, 120)).toEqual({ left: 25, width: 12.5 })
  })

  it('clamps intervals to the video duration and keeps a visible marker', () => {
    expect(timelineMarker(-5, 0.1, 100)).toEqual({ left: 0, width: 0.8 })
    expect(timelineMarker(95, 110, 100)).toEqual({ left: 95, width: 5 })
  })

  it('handles an unavailable duration', () => {
    expect(timelineMarker(10, 20, 0)).toEqual({ left: 0, width: 0 })
  })
})

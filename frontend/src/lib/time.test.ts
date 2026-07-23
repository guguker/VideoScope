import { formatDuration, formatMomentRange } from './time'

describe('time formatting', () => {
  it('uses compact minute format below one hour', () => {
    expect(formatDuration(125.4)).toBe('02:05')
  })

  it('adds hours for long videos', () => {
    expect(formatDuration(3723)).toBe('01:02:03')
  })

  it('formats a readable moment range', () => {
    expect(formatMomentRange(65, 72.2)).toBe('01:05–01:12')
  })
})


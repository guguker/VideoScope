import { describe, expect, it } from 'vitest'
import { apiErrorMessage } from './api'


describe('apiErrorMessage', () => {
  it('uses a textual API detail as-is', () => {
    expect(apiErrorMessage({ detail: 'Video not found' }, 'fallback')).toBe('Video not found')
  })

  it('renders FastAPI validation issues without object stringification', () => {
    expect(apiErrorMessage({
      detail: [
        { loc: ['body', 'mode'], msg: "Input should be 'all', 'speech', 'visual' or 'ocr'" },
        { loc: ['body', 'limit'], msg: 'Input should be less than or equal to 50' },
      ],
    }, 'fallback')).toBe(
      "mode: Input should be 'all', 'speech', 'visual' or 'ocr'; "
      + 'limit: Input should be less than or equal to 50',
    )
  })

  it('falls back for an unknown or unsafe response body', () => {
    expect(apiErrorMessage({ detail: [{ unexpected: 'value' }] }, 'Ошибка API: 422'))
      .toBe('Ошибка API: 422')
  })
})

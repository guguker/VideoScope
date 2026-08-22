import { afterEach, describe, expect, it, vi } from 'vitest'
import type { JobResponse, VideoItem } from '../types'
import { api, apiErrorMessage } from './api'


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

const jobResponse: JobResponse = {
  job_id: 'job-1',
  video_id: 'video-1',
  intent: 'reindex',
  state: 'running',
  progress: 0.4,
  stage: 'speech',
  attempt: 1,
  retry_of_job_id: null,
  cancel_requested_at: null,
  error_code: null,
  created_at: '2026-08-21T09:00:00Z',
  started_at: '2026-08-21T09:00:01Z',
  finished_at: null,
  updated_at: '2026-08-21T09:00:02Z',
}

const videoResponse: VideoItem = {
  id: 'video-1',
  original_name: 'match.mp4',
  display_name: null,
  size_bytes: 1024,
  status: 'ready',
  progress: 1,
  stage: 'ready',
  duration: 42,
  width: 1920,
  height: 1080,
  fps: 25,
  error: null,
  created_at: '2026-08-21T08:00:00Z',
  updated_at: '2026-08-21T09:00:02Z',
  media_url: '/api/videos/video-1/media',
  thumbnail_url: null,
  latest_job: jobResponse,
}

describe('durable video job API', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('requests cancellation and returns the public job response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(jobResponse), { status: 202 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.cancelJob('job-1')).resolves.toEqual(jobResponse)
    expect(fetchMock).toHaveBeenCalledWith('/api/jobs/job-1/cancel', { method: 'POST' })
  })

  it('requests a retry and returns the child job response', async () => {
    const child = { ...jobResponse, job_id: 'job-2', state: 'queued', retry_of_job_id: 'job-1' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(child), { status: 202 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.retryJob('job-1')).resolves.toEqual(child)
    expect(fetchMock).toHaveBeenCalledWith('/api/jobs/job-1/retry', { method: 'POST' })
  })

  it('treats a reindex response as the updated video contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(videoResponse), { status: 202 }))
    vi.stubGlobal('fetch', fetchMock)

    const response: VideoItem = await api.reindex('video-1')

    expect(response.latest_job?.job_id).toBe('job-1')
    expect(fetchMock).toHaveBeenCalledWith('/api/videos/video-1/reindex', { method: 'POST' })
  })
})

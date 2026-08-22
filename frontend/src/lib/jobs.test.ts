import { describe, expect, it } from 'vitest'
import type { JobResponse, VideoItem } from '../types'
import { jobStatusLabel, mergeJobResponse, mergeVideoResponse } from './jobs'

const video: VideoItem = {
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
  updated_at: '2026-08-21T09:00:00Z',
  media_url: '/api/videos/video-1/media',
  thumbnail_url: null,
  latest_job: {
    job_id: 'job-2',
    intent: 'reindex',
    state: 'queued',
    progress: 0,
    stage: 'queued',
    attempt: 2,
    cancel_requested_at: null,
    error_code: null,
    created_at: '2026-08-21T09:00:00Z',
    started_at: null,
    finished_at: null,
    updated_at: '2026-08-21T09:00:02Z',
  },
}

function response(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    ...video.latest_job!,
    video_id: video.id,
    retry_of_job_id: 'job-1',
    ...overrides,
  }
}

describe('durable job state helpers', () => {
  it('does not regress a job when a stale mutation response arrives', () => {
    const result = mergeJobResponse([video], response({
      progress: 0.1,
      updated_at: '2026-08-21T09:00:01Z',
    }))

    expect(result[0]).toBe(video)
  })

  it('does not replace a retry child with an unrelated parent response', () => {
    const result = mergeJobResponse([video], response({
      job_id: 'job-1',
      retry_of_job_id: null,
      state: 'failed',
      stage: 'failed',
      finished_at: '2026-08-21T09:00:03Z',
      updated_at: '2026-08-21T09:00:03Z',
    }))

    expect(result[0]).toBe(video)
  })

  it('prepends a newly uploaded video without mutating the current collection', () => {
    const uploaded = { ...video, id: 'video-2' }
    const current = [video]

    expect(mergeVideoResponse(current, uploaded)).toEqual([uploaded, video])
    expect(current).toEqual([video])
  })

  it('maps an unknown internal stage to a safe public label', () => {
    const job = { ...video.latest_job!, state: 'running' as const, stage: 'private_worker_step' }
    expect(jobStatusLabel(job)).toBe('Индексация')
  })
})

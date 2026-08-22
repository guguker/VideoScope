import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { JobState, VideoItem } from '../types'
import { LibraryPanel } from './LibraryPanel'

function videoWithJob(state: JobState, overrides: Partial<NonNullable<VideoItem['latest_job']>> = {}): VideoItem {
  return {
    id: 'video-1',
    original_name: 'Матч.mp4',
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
    latest_job: {
      job_id: 'job-1',
      intent: 'reindex',
      state,
      progress: state === 'running' ? 0.42 : 0,
      stage: state === 'running' ? 'speech' : state,
      attempt: 1,
      cancel_requested_at: null,
      error_code: state === 'failed' ? 'internal_failure_code' : null,
      created_at: '2026-08-21T09:00:00Z',
      started_at: state === 'queued' ? null : '2026-08-21T09:00:01Z',
      finished_at: state === 'running' || state === 'queued' ? null : '2026-08-21T09:00:02Z',
      updated_at: '2026-08-21T09:00:02Z',
      ...overrides,
    },
  }
}

function renderPanel(video: VideoItem) {
  const props = {
    videos: [video],
    selectedId: video.id,
    collapsed: false,
    onToggleCollapsed: vi.fn(),
    onSelect: vi.fn(),
    onReindex: vi.fn(),
    onCancelJob: vi.fn(),
    onRetryJob: vi.fn(),
    onRename: vi.fn().mockResolvedValue(undefined),
    onUpload: vi.fn(),
  }
  render(<LibraryPanel {...props} />)
  return props
}

describe('LibraryPanel durable job controls', () => {
  it('shows persisted stage progress and offers cancellation for an active job', () => {
    const video = videoWithJob('running')
    const props = renderPanel(video)

    expect(screen.getByText('Распознавание речи')).toBeInTheDocument()
    expect(screen.getByLabelText('42%')).toBeInTheDocument()
    expect(screen.queryByLabelText(/Переиндексировать/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/Повторить индексацию/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByLabelText('Отменить индексацию Матч.mp4'))
    expect(props.onCancelJob).toHaveBeenCalledWith(video)
  })

  it('keeps cancel-requested running work active without offering duplicate cancellation', () => {
    renderPanel(videoWithJob('running', { cancel_requested_at: '2026-08-21T09:00:03Z' }))

    expect(screen.getByText('Отмена запрошена')).toBeInTheDocument()
    expect(screen.getByLabelText('Отмена индексации Матч.mp4 запрошена')).toBeDisabled()
    expect(screen.queryByLabelText(/Переиндексировать/)).not.toBeInTheDocument()
  })

  it.each(['failed', 'cancelled'] satisfies JobState[])('offers retry for a %s job without exposing internal errors', (state) => {
    const video = videoWithJob(state)
    const props = renderPanel(video)

    expect(screen.queryByText('internal_failure_code')).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/Отменить индексацию/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/Переиндексировать/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByLabelText('Повторить индексацию Матч.mp4'))
    expect(props.onRetryJob).toHaveBeenCalledWith(video)
  })

  it('offers reindex only when a ready video has no active or retryable job', () => {
    const video = videoWithJob('complete', { progress: 1, stage: 'complete' })
    const props = renderPanel(video)

    expect(screen.queryByLabelText(/Отменить индексацию/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/Повторить индексацию/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Переиндексировать Матч.mp4'))
    expect(props.onReindex).toHaveBeenCalledWith(video)
  })
})

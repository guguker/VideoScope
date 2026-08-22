import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JobResponse, JobState, VideoItem } from './types'

const mocks = vi.hoisted(() => ({
  videos: vi.fn(),
  providers: vi.fn(),
  evaluation: vi.fn(),
  runEvaluation: vi.fn(),
  glossary: vi.fn(),
  updateGlossary: vi.fn(),
  reindex: vi.fn(),
  cancelJob: vi.fn(),
  retryJob: vi.fn(),
  renameVideo: vi.fn(),
  search: vi.fn(),
  export: vi.fn(),
  uploadVideo: vi.fn(),
}))

vi.mock('./lib/api', () => ({
  api: {
    videos: mocks.videos,
    providers: mocks.providers,
    evaluation: mocks.evaluation,
    runEvaluation: mocks.runEvaluation,
    glossary: mocks.glossary,
    updateGlossary: mocks.updateGlossary,
    reindex: mocks.reindex,
    cancelJob: mocks.cancelJob,
    retryJob: mocks.retryJob,
    renameVideo: mocks.renameVideo,
    search: mocks.search,
    export: mocks.export,
  },
  uploadVideo: mocks.uploadVideo,
}))

vi.mock('./components/SearchWorkspace', () => ({ SearchWorkspace: () => null }))
vi.mock('./components/ClipQueue', () => ({ ClipQueue: () => null }))
vi.mock('./components/TopBar', () => ({
  TopBar: ({ onUpload }: { onUpload: () => void }) => (
    <button type="button" onClick={onUpload}>Открыть загрузку</button>
  ),
}))
vi.mock('./components/Dialogs', () => ({
  ProviderDialog: () => null,
  QualityDialog: () => null,
  UploadDialog: ({ onUpload }: { onUpload: (file: File) => void }) => (
    <button type="button" onClick={() => onUpload(new File(['video'], 'match.mp4', { type: 'video/mp4' }))}>
      Загрузить тестовое видео
    </button>
  ),
}))

import App from './App'

const storage = {
  clear: vi.fn(),
  getItem: vi.fn(() => null),
  key: vi.fn(() => null),
  removeItem: vi.fn(),
  setItem: vi.fn(),
  length: 0,
}
Object.defineProperty(window, 'localStorage', { configurable: true, value: storage })

function videoWithJob(state: JobState, cancelRequestedAt: string | null = null): VideoItem {
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
      progress: state === 'running' ? 0.4 : state === 'complete' ? 1 : 0,
      stage: state === 'running' ? 'speech' : state,
      attempt: 1,
      cancel_requested_at: cancelRequestedAt,
      error_code: state === 'failed' ? 'index_failed' : null,
      created_at: '2026-08-21T09:00:00Z',
      started_at: state === 'queued' ? null : '2026-08-21T09:00:01Z',
      finished_at: state === 'running' || state === 'queued' ? null : '2026-08-21T09:00:02Z',
      updated_at: '2026-08-21T09:00:02Z',
    },
  }
}

function jobResponse(video: VideoItem, overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    ...video.latest_job!,
    video_id: video.id,
    retry_of_job_id: null,
    ...overrides,
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  storage.getItem.mockReturnValue(null)
  mocks.providers.mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
})

describe('App durable job state', () => {
  it('polls the video collection while a cancel-requested job is still running', async () => {
    vi.useFakeTimers()
    const active = videoWithJob('running', '2026-08-21T09:00:03Z')
    mocks.videos.mockResolvedValue([active])

    render(<App />)
    await act(async () => Promise.resolve())
    expect(mocks.videos).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTimeAsync(1500))
    expect(mocks.videos).toHaveBeenCalledTimes(2)
  })

  it('stops collection polling after the latest job becomes terminal', async () => {
    vi.useFakeTimers()
    mocks.videos
      .mockResolvedValueOnce([videoWithJob('running')])
      .mockResolvedValue([videoWithJob('complete')])

    render(<App />)
    await act(async () => Promise.resolve())
    await act(async () => vi.advanceTimersByTimeAsync(1500))
    await act(async () => Promise.resolve())
    await act(async () => vi.advanceTimersByTimeAsync(1500))

    expect(mocks.videos).toHaveBeenCalledTimes(2)
  })

  it('merges cancellation locally without a per-video or extra collection fetch', async () => {
    const active = videoWithJob('running')
    mocks.videos.mockResolvedValue([active])
    mocks.cancelJob.mockResolvedValue(jobResponse(active, {
      cancel_requested_at: '2026-08-21T09:00:03Z',
      updated_at: '2026-08-21T09:00:03Z',
    }))
    render(<App />)

    fireEvent.click(await screen.findByLabelText('Отменить индексацию Матч.mp4'))

    await waitFor(() => expect(screen.getByText('Отмена запрошена')).toBeInTheDocument())
    expect(mocks.cancelJob).toHaveBeenCalledWith('job-1')
    expect(mocks.videos).toHaveBeenCalledTimes(1)
  })

  it('merges the retry child locally and immediately exposes active controls', async () => {
    const failed = videoWithJob('failed')
    const child = jobResponse(failed, {
      job_id: 'job-2',
      state: 'queued',
      stage: 'queued',
      attempt: 2,
      retry_of_job_id: 'job-1',
      error_code: null,
      started_at: null,
      finished_at: null,
    })
    mocks.videos.mockResolvedValue([failed])
    mocks.retryJob.mockResolvedValue(child)
    render(<App />)

    fireEvent.click(await screen.findByLabelText('Повторить индексацию Матч.mp4'))

    await waitFor(() => expect(screen.getByLabelText('Отменить индексацию Матч.mp4')).toBeInTheDocument())
    expect(mocks.retryJob).toHaveBeenCalledWith('job-1')
    expect(mocks.videos).toHaveBeenCalledTimes(1)
  })

  it('merges the reindex video response instead of issuing a second list request', async () => {
    const complete = videoWithJob('complete')
    const reindexing = videoWithJob('running')
    reindexing.latest_job = { ...reindexing.latest_job!, job_id: 'job-2' }
    mocks.videos.mockResolvedValue([complete])
    mocks.reindex.mockResolvedValue(reindexing)
    render(<App />)

    fireEvent.click(await screen.findByLabelText('Переиндексировать Матч.mp4'))

    await waitFor(() => expect(screen.getByLabelText('Отменить индексацию Матч.mp4')).toBeInTheDocument())
    expect(mocks.reindex).toHaveBeenCalledWith('video-1')
    expect(mocks.videos).toHaveBeenCalledTimes(1)
  })

  it('uses the upload video response to start durable-job polling state', async () => {
    const uploaded = videoWithJob('queued')
    uploaded.latest_job = { ...uploaded.latest_job!, intent: 'ingest' }
    mocks.videos.mockResolvedValue([])
    mocks.uploadVideo.mockResolvedValue(uploaded)
    render(<App />)

    fireEvent.click(screen.getByText('Открыть загрузку'))
    fireEvent.click(await screen.findByText('Загрузить тестовое видео'))

    await waitFor(() => expect(screen.getByLabelText('Отменить индексацию Матч.mp4')).toBeInTheDocument())
    expect(mocks.uploadVideo).toHaveBeenCalledTimes(1)
    expect(mocks.videos).toHaveBeenCalledTimes(1)
  })
})

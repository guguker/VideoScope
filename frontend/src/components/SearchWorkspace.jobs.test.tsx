import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { VideoItem } from '../types'
import { SearchWorkspace } from './SearchWorkspace'

function video(status: VideoItem['status']): VideoItem {
  return {
    id: 'video-1',
    original_name: 'Матч.mp4',
    display_name: null,
    size_bytes: 1024,
    status,
    progress: 0.1,
    stage: 'private_stage',
    duration: 42,
    width: 1920,
    height: 1080,
    fps: 25,
    error: '/private/path/worker failure',
    created_at: '2026-08-21T08:00:00Z',
    updated_at: '2026-08-21T09:00:02Z',
    media_url: '/api/videos/video-1/media',
    thumbnail_url: null,
    latest_job: {
      job_id: 'job-1',
      intent: 'reindex',
      state: status === 'failed' ? 'failed' : 'running',
      progress: 0.42,
      stage: status === 'failed' ? 'failed' : 'speech',
      attempt: 1,
      cancel_requested_at: null,
      error_code: status === 'failed' ? 'private_error_code' : null,
      created_at: '2026-08-21T09:00:00Z',
      started_at: '2026-08-21T09:00:01Z',
      finished_at: status === 'failed' ? '2026-08-21T09:00:02Z' : null,
      updated_at: '2026-08-21T09:00:02Z',
    },
  }
}

function renderWorkspace(selectedVideo: VideoItem) {
  render(<SearchWorkspace
    videos={[selectedVideo]}
    selectedVideo={selectedVideo}
    selectedResult={null}
    query=""
    scope="all"
    searchMode="all"
    useLighthouse={false}
    lighthouseAvailable={false}
    results={[]}
    searching={false}
    hasSearched={false}
    onQuery={vi.fn()}
    onScope={vi.fn()}
    onSearchMode={vi.fn()}
    onUseLighthouse={vi.fn()}
    onSearch={vi.fn()}
    onSelectResult={vi.fn()}
    onAddClip={vi.fn()}
    onUpload={vi.fn()}
  />)
}

describe('SearchWorkspace durable job presentation', () => {
  it('shows active reindex progress while the previously ready video remains playable', () => {
    renderWorkspace(video('ready'))

    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Распознавание речи')
    expect(status).toHaveTextContent('42%')
  })

  it('shows a bounded failure message without rendering backend internals', () => {
    renderWorkspace(video('failed'))

    expect(screen.getByText('Ошибка индексации')).toBeInTheDocument()
    expect(screen.queryByText('/private/path/worker failure')).not.toBeInTheDocument()
    expect(screen.queryByText('private_error_code')).not.toBeInTheDocument()
    expect(screen.queryByText('private_stage')).not.toBeInTheDocument()
  })
})

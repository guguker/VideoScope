import type {
  ClipDraft,
  EvaluationPayload,
  EvaluationReport,
  EvaluationVariantName,
  ExportResult,
  ProviderStatus,
  SearchMode,
  SearchResult,
  VideoItem,
} from '../types'

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `Ошибка API: ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch {
      // Резервное сообщение уже содержит код состояния HTTP.
    }
    throw new ApiError(message, response.status)
  }
  return (await response.json()) as T
}

export const api = {
  videos: () => request<VideoItem[]>('/api/videos'),
  providers: () => request<ProviderStatus[]>('/api/providers'),
  evaluation: () => request<EvaluationPayload>('/api/evaluation'),
  runEvaluation: (variants: EvaluationVariantName[]) =>
    request<EvaluationReport>('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variants }),
    }),
  glossary: () => request<{ entries: Record<string, string[]> }>('/api/search/glossary'),
  updateGlossary: (entries: Record<string, string[]>) =>
    request<{ entries: Record<string, string[]> }>('/api/search/glossary', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entries }),
    }),
  reindex: (videoId: string) =>
    request<{ status: string; video_id: string }>(`/api/videos/${videoId}/reindex`, {
      method: 'POST',
    }),
  renameVideo: (videoId: string, name: string) =>
    request<VideoItem>(`/api/videos/${videoId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  search: (query: string, videoIds: string[] | null, useLighthouse: boolean, mode: SearchMode) =>
    request<SearchResult[]>('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, video_ids: videoIds, use_lighthouse: useLighthouse, mode, limit: 30 }),
    }),
  export: (name: string, clips: ClipDraft[]) =>
    request<ExportResult>('/api/exports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        selections: clips.map((clip) => ({
          video_id: clip.videoId,
          start: clip.start,
          end: clip.end,
        })),
      }),
    }),
}

export function uploadVideo(file: File, onProgress: (progress: number) => void): Promise<VideoItem> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', '/api/videos')
    request.responseType = 'json'
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total)
    })
    request.addEventListener('load', () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response as VideoItem)
        return
      }
      const detail = (request.response as { detail?: string } | null)?.detail
      reject(new ApiError(detail || `Ошибка загрузки: ${request.status}`, request.status))
    })
    request.addEventListener('error', () => reject(new ApiError('Сервер недоступен', 0)))
    const form = new FormData()
    form.append('file', file)
    request.send(form)
  })
}

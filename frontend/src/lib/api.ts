import type {
  ClipDraft,
  EvaluationPayload,
  EvaluationReport,
  EvaluationVariantName,
  ExportResult,
  JobResponse,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function apiErrorMessage(body: unknown, fallback: string): string {
  if (!isRecord(body)) return fallback
  const detail = body.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (!Array.isArray(detail)) return fallback

  const messages = detail.flatMap((issue) => {
    if (!isRecord(issue) || typeof issue.msg !== 'string' || !issue.msg.trim()) return []
    const location = Array.isArray(issue.loc)
      ? issue.loc
        .filter((part) => part !== 'body' && part !== 'query' && part !== 'path')
        .filter((part): part is string | number => typeof part === 'string' || typeof part === 'number')
        .join('.')
      : ''
    return [location ? `${location}: ${issue.msg}` : issue.msg]
  })
  return messages.length > 0 ? messages.join('; ') : fallback
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `Ошибка API: ${response.status}`
    try {
      message = apiErrorMessage(await response.json(), message)
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
    request<VideoItem>(`/api/videos/${encodeURIComponent(videoId)}/reindex`, {
      method: 'POST',
    }),
  cancelJob: (jobId: string) =>
    request<JobResponse>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    }),
  retryJob: (jobId: string) =>
    request<JobResponse>(`/api/jobs/${encodeURIComponent(jobId)}/retry`, {
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
      const fallback = `Ошибка загрузки: ${request.status}`
      reject(new ApiError(apiErrorMessage(request.response, fallback), request.status))
    })
    request.addEventListener('error', () => reject(new ApiError('Сервер недоступен', 0)))
    const form = new FormData()
    form.append('file', file)
    request.send(form)
  })
}

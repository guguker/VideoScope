import type { JobResponse, JobSummary, VideoItem } from '../types'

const stageLabels: Readonly<Record<string, string>> = {
  queued: 'В очереди',
  probe: 'Проверка видео',
  scenes: 'Разбор сцен',
  speech: 'Распознавание речи',
  ocr: 'Распознавание текста',
  objects: 'Поиск объектов',
  text_vectors: 'Подготовка поиска',
  visual_dense: 'Анализ изображения',
  lighthouse: 'Видеоиндексация',
  finalizing: 'Завершение индексации',
}

export function isActiveJob(job: JobSummary | null): job is JobSummary & { state: 'queued' | 'running' } {
  return job?.state === 'queued' || job?.state === 'running'
}

export function isRetryableJob(job: JobSummary | null): job is JobSummary & { state: 'failed' | 'cancelled' } {
  return job?.state === 'failed' || job?.state === 'cancelled'
}

export function jobStatusLabel(job: JobSummary): string {
  if (job.state === 'running' && job.cancel_requested_at) return 'Отмена запрошена'
  if (job.state === 'failed') return 'Ошибка индексации'
  if (job.state === 'cancelled') return 'Индексация отменена'
  if (job.state === 'complete') return 'Готово'
  return stageLabels[job.stage] || (job.state === 'queued' ? 'В очереди' : 'Индексация')
}

function jobSummary(response: JobResponse): JobSummary {
  const { video_id: _videoId, retry_of_job_id: _retryOfJobId, ...summary } = response
  return summary
}

export function mergeJobResponse(videos: VideoItem[], response: JobResponse): VideoItem[] {
  return videos.map((video) => {
    if (video.id !== response.video_id) return video

    const current = video.latest_job
    if (current && current.job_id !== response.job_id && response.retry_of_job_id !== current.job_id) {
      return video
    }
    if (current?.job_id === response.job_id && current.updated_at > response.updated_at) {
      return video
    }
    return { ...video, latest_job: jobSummary(response) }
  })
}

export function mergeVideoResponse(videos: VideoItem[], response: VideoItem): VideoItem[] {
  const index = videos.findIndex((video) => video.id === response.id)
  if (index < 0) return [response, ...videos]
  return videos.map((video) => video.id === response.id ? response : video)
}

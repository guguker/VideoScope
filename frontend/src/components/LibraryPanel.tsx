import {
  Check,
  Film,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  RefreshCw,
  Search,
  Video,
  X,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import type { VideoItem } from '../types'
import { isActiveJob, isRetryableJob, jobStatusLabel } from '../lib/jobs'
import { formatBytes, formatDuration } from '../lib/time'

interface LibraryPanelProps {
  videos: VideoItem[]
  selectedId: string | null
  collapsed: boolean
  onToggleCollapsed: () => void
  onSelect: (video: VideoItem) => void
  onReindex: (video: VideoItem) => void
  onCancelJob: (video: VideoItem) => void
  onRetryJob: (video: VideoItem) => void
  onRename: (video: VideoItem, name: string) => Promise<void>
  onUpload: () => void
}

const statusLabels: Record<VideoItem['status'], string> = {
  queued: 'В очереди',
  processing: 'Обработка',
  ready: 'Готово',
  failed: 'Ошибка',
}

function videoName(video: VideoItem): string {
  return video.display_name || video.original_name
}

export function LibraryPanel({
  videos,
  selectedId,
  collapsed,
  onToggleCollapsed,
  onSelect,
  onReindex,
  onCancelJob,
  onRetryJob,
  onRename,
  onUpload,
}: LibraryPanelProps) {
  const [filter, setFilter] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [saving, setSaving] = useState(false)
  const visibleVideos = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase('ru')
    if (!query) return videos
    return videos.filter((video) => videoName(video).toLocaleLowerCase('ru').includes(query))
  }, [filter, videos])

  const startRename = (video: VideoItem) => {
    setEditingId(video.id)
    setDraftName(videoName(video).replace(/\.[^.]+$/, ''))
  }

  const cancelRename = () => {
    setEditingId(null)
    setDraftName('')
  }

  const saveRename = async (video: VideoItem) => {
    const name = draftName.trim()
    if (!name || saving) return
    setSaving(true)
    try {
      await onRename(video, name)
      cancelRename()
    } catch {
      // Приложение показывает ошибку API во всплывающем сообщении; оставляем редактор открытым.
    } finally {
      setSaving(false)
    }
  }

  return (
    <aside className={`panel library-panel ${collapsed ? 'is-collapsed' : ''}`} aria-label="Библиотека видео">
      <div className="panel-heading">
        <div className="library-heading-copy">
          <h2>Библиотека</h2>
          <span className="panel-count">{videos.length}</span>
        </div>
        <button
          className="icon-button library-collapse"
          type="button"
          onClick={onToggleCollapsed}
          title={collapsed ? 'Развернуть библиотеку' : 'Свернуть библиотеку'}
          aria-label={collapsed ? 'Развернуть библиотеку' : 'Свернуть библиотеку'}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>
      <label className="compact-search library-filter">
        <Search size={15} />
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Фильтр по названию"
          aria-label="Фильтр библиотеки"
        />
      </label>
      <div className="library-list">
        {visibleVideos.map((video) => {
          const job = video.latest_job
          const activeJob = isActiveJob(job) ? job : null
          const retryableJob = isRetryableJob(job) ? job : null
          const legacyWork = !job && (video.status === 'processing' || video.status === 'queued')
          const showsProgress = Boolean(activeJob) || legacyWork
          const progress = activeJob?.progress ?? video.progress
          const statusLabel = job ? jobStatusLabel(job) : statusLabels[video.status]
          const statusClass = activeJob
            ? 'processing'
            : retryableJob
              ? 'failed'
              : video.status
          return (
          <div
            className={`library-row ${selectedId === video.id ? 'is-selected' : ''} ${activeJob ? 'has-active-job' : ''} ${retryableJob ? 'has-retryable-job' : ''}`}
            key={video.id}
          >
            {editingId === video.id && !collapsed ? (
              <form
                className="library-select library-rename-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void saveRename(video)
                }}
              >
                <input
                  autoFocus
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') cancelRename()
                  }}
                  aria-label="Новое название видео"
                  maxLength={160}
                />
                <button
                  className="icon-button"
                  type="submit"
                  disabled={!draftName.trim() || saving}
                  title="Сохранить"
                  aria-label="Сохранить название"
                >
                  {saving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}
                </button>
                <button
                  className="icon-button"
                  type="button"
                  onClick={cancelRename}
                  title="Отмена"
                  aria-label="Отменить переименование"
                >
                  <X size={15} />
                </button>
              </form>
            ) : (
              <button
                className="library-select"
                type="button"
                onClick={() => onSelect(video)}
                title={collapsed ? videoName(video) : undefined}
              >
              <span className="library-thumb">
                {video.thumbnail_url ? (
                  <img src={video.thumbnail_url} alt="" />
                ) : showsProgress ? (
                  <LoaderCircle className="spin" size={19} />
                ) : (
                  <Film size={19} />
                )}
              </span>
              <span className="library-copy">
                <strong title={videoName(video)}>{videoName(video)}</strong>
                <span className="library-meta">
                  {video.duration ? formatDuration(video.duration) : formatBytes(video.size_bytes)}
                  <i aria-hidden="true" />
                  <em className={`status-text status-${statusClass}`}>{statusLabel}</em>
                </span>
                {showsProgress && (
                  <span className="mini-progress" aria-label={`${Math.round(progress * 100)}%`}>
                    <span style={{ width: `${Math.max(3, progress * 100)}%` }} />
                  </span>
                )}
              </span>
            </button>
            )}
            {editingId !== video.id && !legacyWork && (
              <span className="library-actions">
                {!collapsed && !activeJob && (
                  <button
                    className="icon-button library-rename"
                    type="button"
                    onClick={() => startRename(video)}
                    title="Переименовать"
                    aria-label={`Переименовать ${videoName(video)}`}
                  >
                    <Pencil size={14} />
                  </button>
                )}
                {activeJob && (
                  <button
                    className="icon-button library-cancel-job"
                    type="button"
                    onClick={() => onCancelJob(video)}
                    disabled={activeJob.cancel_requested_at !== null}
                    title={activeJob.cancel_requested_at ? 'Отмена запрошена' : 'Отменить индексацию'}
                    aria-label={activeJob.cancel_requested_at
                      ? `Отмена индексации ${videoName(video)} запрошена`
                      : `Отменить индексацию ${videoName(video)}`}
                  >
                    <X size={15} />
                  </button>
                )}
                {retryableJob && (
                  <button
                    className="icon-button library-retry-job"
                    type="button"
                    onClick={() => onRetryJob(video)}
                    title="Повторить индексацию"
                    aria-label={`Повторить индексацию ${videoName(video)}`}
                  >
                    <RefreshCw size={15} />
                  </button>
                )}
                {!activeJob && !retryableJob && video.status === 'ready' && (
                  <button
                    className="icon-button library-reindex"
                    type="button"
                    onClick={() => onReindex(video)}
                    title="Переиндексировать"
                    aria-label={`Переиндексировать ${videoName(video)}`}
                  >
                    <RefreshCw size={15} />
                  </button>
                )}
              </span>
            )}
          </div>
          )
        })}
        {videos.length === 0 && (
          <button className="library-empty" type="button" onClick={onUpload}>
            <Video size={22} />
            <span>Добавить первое видео</span>
          </button>
        )}
        {videos.length > 0 && visibleVideos.length === 0 && (
          <p className="empty-filter">Совпадений нет</p>
        )}
      </div>
    </aside>
  )
}

import { Download, Film, GripVertical, Scissors, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ClipDraft, ExportResult, VideoItem } from '../types'
import { formatDuration } from '../lib/time'

interface ClipQueueProps {
  clips: ClipDraft[]
  videos: VideoItem[]
  exporting: boolean
  exported: ExportResult | null
  onUpdate: (clip: ClipDraft, start: number, end: number, duration: number) => void
  onRemove: (clipId: string) => void
  onExport: (name: string) => void
}

export function ClipQueue({
  clips,
  videos,
  exporting,
  exported,
  onUpdate,
  onRemove,
  onExport,
}: ClipQueueProps) {
  const [name, setName] = useState('Лучшие моменты')
  const durations = useMemo(
    () => new Map(videos.map((video) => [video.id, video.duration ?? Number.POSITIVE_INFINITY])),
    [videos],
  )
  const totalDuration = clips.reduce((total, clip) => total + Math.max(0, clip.end - clip.start), 0)

  return (
    <aside className="panel clip-panel" aria-label="Очередь нарезки">
      <div className="panel-heading clip-heading">
        <div>
          <h2>Нарезка</h2>
          <span className="panel-count">{clips.length}</span>
        </div>
        <span className="queue-duration">{formatDuration(totalDuration)}</span>
      </div>
      <div className="clip-list">
        {clips.map((clip, index) => {
          const duration = durations.get(clip.videoId) ?? Number.POSITIVE_INFINITY
          return (
            <div className="clip-row" key={clip.id}>
              <GripVertical className="clip-grip" size={16} aria-hidden="true" />
              <span className="clip-index">{String(index + 1).padStart(2, '0')}</span>
              <div className="clip-copy">
                <div className="clip-name-line">
                  <strong title={clip.videoName}>{clip.videoName}</strong>
                  <button
                    className="icon-button danger-button"
                    type="button"
                    onClick={() => onRemove(clip.id)}
                    title="Удалить из нарезки"
                    aria-label={`Удалить ${clip.videoName} из нарезки`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
                <div className="time-inputs">
                  <label>
                    <span>Начало</span>
                    <input
                      type="number"
                      min="0"
                      max={Number.isFinite(duration) ? duration : undefined}
                      step="0.1"
                      value={clip.start.toFixed(1)}
                      onChange={(event) => onUpdate(clip, Number(event.target.value), clip.end, duration)}
                    />
                  </label>
                  <label>
                    <span>Конец</span>
                    <input
                      type="number"
                      min="0"
                      max={Number.isFinite(duration) ? duration : undefined}
                      step="0.1"
                      value={clip.end.toFixed(1)}
                      onChange={(event) => onUpdate(clip, clip.start, Number(event.target.value), duration)}
                    />
                  </label>
                </div>
              </div>
            </div>
          )
        })}
        {clips.length === 0 && (
          <div className="clip-empty">
            <Scissors size={24} />
            <span>Очередь пуста</span>
          </div>
        )}
      </div>
      <div className="export-controls">
        <label>
          <span>Имя файла</span>
          <input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} />
        </label>
        <button
          className="button button-dark export-command"
          type="button"
          disabled={clips.length === 0 || exporting || !name.trim()}
          onClick={() => onExport(name)}
        >
          {exporting ? <Scissors className="pulse" size={17} /> : <Download size={17} />}
          {exporting ? 'Собираю…' : 'Экспорт MP4'}
        </button>
        {exported && (
          <a className="export-ready" href={exported.url} download={exported.name}>
            <Download size={15} />
            <span>{exported.name}</span>
          </a>
        )}
      </div>
    </aside>
  )
}


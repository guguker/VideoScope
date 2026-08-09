import {
  Check,
  Film,
  Info,
  Layers3,
  LoaderCircle,
  Play,
  Plus,
  Search,
  Sparkles,
  Upload,
} from 'lucide-react'
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import type { SearchMode, SearchResult, VideoItem } from '../types'
import { formatDuration, formatMomentRange } from '../lib/time'
import { timelineMarker } from '../lib/timeline'
import {
  eventCardData,
  hasQwenVerification,
  resultSourceKeys,
  type EventCardData,
  type FactState,
} from '../lib/searchPresentation'

interface SearchWorkspaceProps {
  videos: VideoItem[]
  selectedVideo: VideoItem | null
  selectedResult: SearchResult | null
  query: string
  scope: 'all' | 'current'
  searchMode: SearchMode
  useLighthouse: boolean
  lighthouseAvailable: boolean
  results: SearchResult[]
  searching: boolean
  hasSearched: boolean
  onQuery: (query: string) => void
  onScope: (scope: 'all' | 'current') => void
  onSearchMode: (mode: SearchMode) => void
  onUseLighthouse: (enabled: boolean) => void
  onSearch: () => void
  onSelectResult: (result: SearchResult) => void
  onAddClip: (result: SearchResult) => void
  onUpload: () => void
}

const modalityLabels: Record<string, string> = {
  lighthouse: 'Lighthouse',
  visual: 'Видео',
  speech: 'Речь',
  ocr: 'OCR',
  objects: 'Объекты',
  scene: 'Сцена',
  qwen_video: 'Проверка видео',
  internvideo: 'InternVideo',
}

const modeLabels: { id: SearchMode; label: string }[] = [
  { id: 'all', label: 'Всё' },
  { id: 'speech', label: 'Речь' },
  { id: 'visual', label: 'Кадр' },
  { id: 'ocr', label: 'Текст' },
]

const sourceLabels: Record<string, string> = {
  lexical: 'точное совпадение',
  qdrant: 'семантика текста',
  memory: 'локальный индекс',
  siglip2: 'SigLIP 2',
  lighthouse: 'Lighthouse',
  'lighthouse-corroborated': 'Lighthouse + кадр',
  'lighthouse-unconfirmed': 'Lighthouse без подтверждения',
  'siglip2-dense': 'SigLIP 2 · точный кадр',
  'siglip2-temporal-event': 'SigLIP 2 · событие во времени',
  'lexical-exact': 'точное слово',
  'lexical-stem': 'форма слова',
  'lexical-transliteration': 'транслитерация',
  'lexical-phonetic': 'похожее звучание',
  'lexical-fuzzy': 'нечёткое слово',
  'internvideo2.5': 'InternVideo 2.5',
  'qwen-video-verifier': 'Qwen · последовательность кадров',
}

const intentLabels: Record<string, string> = {
  entity: 'Имя',
  speech: 'Речь',
  ocr: 'Текст',
  object: 'Объект',
  action: 'Действие',
  mixed: 'Комбинация',
  visual: 'Кадр',
}

function displayVideoName(video: VideoItem): string {
  return video.display_name || video.original_name
}

function relevance(score: number): { label: string; level: string } {
  if (score >= 0.78) return { label: 'Высокая', level: 'high' }
  if (score >= 0.58) return { label: 'Средняя', level: 'medium' }
  return { label: 'Низкая', level: 'low' }
}

const factStateLabels: Record<FactState, string> = {
  yes: 'да',
  no: 'нет',
  unknown: 'не доказано',
}

function scorePercent(score: number): string {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`
}

function EventCard({ data }: { data: EventCardData }) {
  return (
    <section className="event-card" aria-label="Карточка спортивного события">
      <header>
        <strong>Карточка события</strong>
        <span>{data.eventLabel}</span>
      </header>
      {data.facts.length > 0 && (
        <div className="event-facts">
          {data.facts.map((fact) => (
            <div key={fact.id}>
              <span>{fact.label}</span>
              <strong className={`fact-${fact.state}`}>{factStateLabels[fact.state]}</strong>
            </div>
          ))}
        </div>
      )}
      {data.possibleJersey && (
        <div className="event-jersey">
          <span>Возможный номер игрока</span>
          <strong>№{data.possibleJersey}</strong>
          <i>не подтверждён</i>
        </div>
      )}
      {data.stageScores.length > 0 && (
        <div className="event-stages">
          <p>Оценки стадий <span>сходство, не вероятность</span></p>
          <div>
            {data.stageScores.map((stage) => (
              <span key={stage.id} title={`${stage.label}: ${scorePercent(stage.score)}`}>
                <i>{stage.label}</i>
                <b>{scorePercent(stage.score)}</b>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function SearchTrace({ result }: { result: SearchResult }) {
  const sources = resultSourceKeys(result).map((source) => (
    sourceLabels[source] || modalityLabels[source] || source
  ))
  const steps = [
    {
      id: 'route',
      label: 'Маршрутизация',
      detail: intentLabels[result.intent] || result.intent,
    },
    ...(sources.length > 0 ? [{
      id: 'sources',
      label: 'Источники',
      detail: sources.join(' · '),
    }] : []),
    ...(result.refined ? [{
      id: 'refinement',
      label: 'Точное уточнение',
      detail: formatMomentRange(result.start, result.end),
    }] : []),
    ...(hasQwenVerification(result) ? [{
      id: 'qwen',
      label: 'Qwen',
      detail: 'проверка последовательности кадров',
    }] : []),
  ]

  return (
    <div className="search-route" role="status" aria-label="Фактически выполненные этапы поиска">
      <div className="search-route-summary">
        <Sparkles size={14} />
        <span>{result.explanation}</span>
      </div>
      <ol className="search-trace">
        {steps.map((step, index) => (
          <li key={step.id}>
            <b>{index + 1}</b>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function VideoPlayer({
  video,
  seekTo,
  onTime,
}: {
  video: VideoItem
  seekTo: number | null
  onTime: (time: number) => void
}) {
  const player = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (seekTo === null || !player.current) return
    const seek = () => {
      if (!player.current) return
      player.current.currentTime = Math.max(0, seekTo)
      void player.current.play().catch(() => undefined)
    }
    if (player.current.readyState >= 1) seek()
    else player.current.addEventListener('loadedmetadata', seek, { once: true })
  }, [seekTo, video.id])

  return (
    <div className="player-frame">
      <video
        ref={player}
        src={video.media_url}
        controls
        preload="metadata"
        key={video.id}
        onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
      />
    </div>
  )
}

const timelineLanes = [
  'speech',
  'visual',
  'qwen_video',
  'objects',
  'ocr',
  'lighthouse',
  'internvideo',
] as const

function EvidenceTimeline({
  results,
  duration,
  currentTime,
  selectedId,
  onSelect,
}: {
  results: SearchResult[]
  duration: number
  currentTime: number
  selectedId: string | null
  onSelect: (result: SearchResult) => void
}) {
  const lanes = useMemo(() => timelineLanes.map((modality) => ({
    modality,
    items: results.flatMap((result) => result.evidence
      .filter((evidence) => evidence.modality === modality)
      .map((evidence) => ({ result, evidence }))),
  })).filter((lane) => lane.items.length > 0), [results])

  if (!results.length || !duration || lanes.length === 0) return null
  const playhead = timelineMarker(currentTime, currentTime, duration).left
  return (
    <section className="evidence-timeline" aria-label="Мультимодальная шкала">
      <header>
        <span><Layers3 size={14} /> Источники совпадений</span>
        <strong>{formatDuration(currentTime)} / {formatDuration(duration)}</strong>
      </header>
      <div className="timeline-lanes">
        {lanes.map((lane) => (
          <div className={`timeline-lane timeline-${lane.modality}`} key={lane.modality}>
            <span>{modalityLabels[lane.modality]}</span>
            <div className="timeline-track">
              {lane.items.map(({ result, evidence }, index) => {
                const marker = timelineMarker(evidence.start, evidence.end, duration)
                return (
                  <button
                    type="button"
                    key={`${result.id}:${evidence.source}:${index}`}
                    className={result.id === selectedId ? 'is-selected' : ''}
                    style={{ left: `${marker.left}%`, width: `${marker.width}%` }}
                    onClick={() => onSelect(result)}
                    title={`${modalityLabels[lane.modality]} · ${formatMomentRange(evidence.start, evidence.end)} · ${sourceLabels[evidence.source] || evidence.source}`}
                    aria-label={`${modalityLabels[lane.modality]} ${formatMomentRange(evidence.start, evidence.end)}`}
                  />
                )
              })}
              <i className="timeline-playhead" style={{ left: `${playhead}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function ResultRow({
  result,
  active,
  onSelect,
  onAdd,
}: {
  result: SearchResult
  active: boolean
  onSelect: () => void
  onAdd: () => void
}) {
  const rank = relevance(result.score)
  const event = active ? eventCardData(result) : null
  const handleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect()
    }
  }
  return (
    <div
      className={`result-row ${active ? 'is-active' : ''}`}
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={handleKey}
      data-testid="search-result"
    >
      <div className="result-thumb">
        {result.thumbnail_url ? <img src={result.thumbnail_url} alt="" /> : <Film size={22} />}
        <span><Play size={12} fill="currentColor" /> {formatDuration(result.start)}</span>
      </div>
      <div className="result-content">
        <div className="result-title-line">
          <strong>{formatMomentRange(result.start, result.end)}</strong>
          <span
            className={`relevance relevance-${rank.level}`}
            title="Относительная релевантность для сортировки, не вероятность"
          >
            {rank.label}
          </span>
          {result.refined && <span className="refined-label">Точный интервал</span>}
        </div>
        <p>{result.evidence[0]?.text || result.video_name}</p>
        <div className="result-footer">
          <span className="result-video" title={result.video_name}>{result.video_name}</span>
          <span className="modality-list">
            {result.modalities.slice(0, 3).map((modality) => (
              <i key={modality}>{modalityLabels[modality] || modality}</i>
            ))}
          </span>
        </div>
        {event && <EventCard data={event} />}
        {active && result.evidence.length > 0 && (
          <div className="result-evidence" aria-label="Причины совпадения">
            {result.evidence.slice(0, 3).map((evidence, index) => (
              <div key={`${evidence.modality}:${evidence.source}:${index}`}>
                <span>{modalityLabels[evidence.modality] || evidence.modality}</span>
                <i>{sourceLabels[evidence.source] || evidence.source}</i>
                <p>
                  {evidence.modality === 'visual'
                    ? 'Кадр соответствует запросу'
                    : evidence.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
      <button
        className="icon-button result-add"
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          onAdd()
        }}
        title="Добавить в нарезку"
        aria-label="Добавить момент в нарезку"
      >
        <Plus size={17} />
      </button>
    </div>
  )
}

export function SearchWorkspace(props: SearchWorkspaceProps) {
  const readyVideos = props.videos.filter((video) => video.status === 'ready')
  const tracedResult = props.selectedResult || props.results[0] || null
  const [currentTime, setCurrentTime] = useState(0)
  useEffect(() => setCurrentTime(0), [props.selectedVideo?.id])
  const submit = (event: FormEvent) => {
    event.preventDefault()
    props.onSearch()
  }

  return (
    <main className="search-workspace">
      <form className="search-toolbar" onSubmit={submit}>
        <label className="semantic-search">
          <Search size={20} />
          <input
            value={props.query}
            onChange={(event) => props.onQuery(event.target.value)}
            placeholder="игрок забивает трёхочковый в конце четверти"
            aria-label="Поисковый запрос"
          />
          {props.searching && <LoaderCircle className="spin search-spinner" size={18} />}
        </label>
        <button
          className="button button-primary search-command"
          type="submit"
          disabled={!props.query.trim() || props.searching}
          aria-label="Найти"
        >
          <Search size={17} />
          <span>Найти</span>
        </button>
        <div className="search-options">
          <div className="search-filter-groups">
            <div className="segmented" aria-label="Область поиска">
              <button
                type="button"
                className={props.scope === 'all' ? 'is-active' : ''}
                onClick={() => props.onScope('all')}
              >
                Все видео
              </button>
              <button
                type="button"
                className={props.scope === 'current' ? 'is-active' : ''}
                onClick={() => props.onScope('current')}
                disabled={!props.selectedVideo}
              >
                Текущее
              </button>
            </div>
            <div className="segmented mode-segmented" aria-label="Канал поиска">
              {modeLabels.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={props.searchMode === mode.id ? 'is-active' : ''}
                  onClick={() => props.onSearchMode(mode.id)}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>
          <label
            className={`switch-control ${props.lighthouseAvailable ? '' : 'is-disabled'}`}
            title={props.lighthouseAvailable ? 'Добавить ранжирование Lighthouse' : 'Lighthouse не настроен'}
          >
            <input
              type="checkbox"
              checked={props.useLighthouse}
              disabled={!props.lighthouseAvailable}
              onChange={(event) => props.onUseLighthouse(event.target.checked)}
            />
            <span aria-hidden="true"><i /></span>
            <Sparkles size={14} />
            Lighthouse
          </label>
        </div>
      </form>

      {props.hasSearched && tracedResult && <SearchTrace result={tracedResult} />}

      <div className="workspace-body">
        {props.selectedVideo?.status === 'ready' && (
          <section className="player-section" aria-label="Просмотр видео">
            <div className="section-heading player-heading">
              <div>
                <h1 title={displayVideoName(props.selectedVideo)}>{displayVideoName(props.selectedVideo)}</h1>
                <span>
                  {formatDuration(props.selectedVideo.duration)}
                  {props.selectedVideo.width && props.selectedVideo.height
                    ? ` · ${props.selectedVideo.width}×${props.selectedVideo.height}`
                    : ''}
                </span>
              </div>
              {props.selectedResult && (
                <button className="button button-secondary" type="button" onClick={() => props.onAddClip(props.selectedResult!)}>
                  <Plus size={16} />
                  В нарезку
                </button>
              )}
            </div>
            <VideoPlayer
              video={props.selectedVideo}
              seekTo={props.selectedResult?.start ?? null}
              onTime={setCurrentTime}
            />
            <EvidenceTimeline
              results={props.results}
              duration={props.selectedVideo.duration || 0}
              currentTime={currentTime}
              selectedId={props.selectedResult?.id || null}
              onSelect={props.onSelectResult}
            />
          </section>
        )}

        {props.selectedVideo && props.selectedVideo.status !== 'ready' && (
          <section className="processing-state">
            {props.selectedVideo.status === 'failed' ? <Film size={28} /> : <LoaderCircle className="spin" size={28} />}
            <div>
              <h1>{displayVideoName(props.selectedVideo)}</h1>
              <p>{props.selectedVideo.error || `Этап: ${props.selectedVideo.stage}`}</p>
              {props.selectedVideo.status !== 'failed' && (
                <span className="large-progress"><i style={{ width: `${props.selectedVideo.progress * 100}%` }} /></span>
              )}
            </div>
          </section>
        )}

        {!props.selectedVideo && props.videos.length === 0 && (
          <section className="primary-empty">
            <div className="empty-visual"><Upload size={28} /></div>
            <h1>Библиотека пуста</h1>
            <button className="button button-primary" type="button" onClick={props.onUpload}>
              <Upload size={17} /> Добавить видео
            </button>
          </section>
        )}

        {!props.selectedVideo && props.videos.length > 0 && (
          <section className="primary-empty compact">
            <Film size={27} />
            <h1>Выберите видео</h1>
          </section>
        )}

        <section className="results-section" aria-label="Результаты поиска">
          <div className="section-heading results-heading">
            <div>
              <h2>Моменты</h2>
              {props.hasSearched && <span>{props.results.length}</span>}
            </div>
            {props.hasSearched && props.results.length > 0 && (
              <span
                className="search-complete"
                title="Результаты отсортированы по относительной релевантности; это не вероятность"
              >
                <Check size={14} /> Поиск завершён <Info size={13} />
              </span>
            )}
          </div>
          <div className="results-list">
            {props.results.map((result) => (
              <ResultRow
                key={result.id}
                result={result}
                active={props.selectedResult?.id === result.id}
                onSelect={() => props.onSelectResult(result)}
                onAdd={() => props.onAddClip(result)}
              />
            ))}
            {props.hasSearched && !props.searching && props.results.length === 0 && (
              <div className="no-results">
                <Search size={22} />
                <span>Совпадений не найдено</span>
              </div>
            )}
            {!props.hasSearched && readyVideos.length > 0 && (
              <div className="results-idle">
                <Sparkles size={20} />
                <span>{readyVideos.length} видео готово к поиску</span>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}

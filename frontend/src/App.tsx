import { Clapperboard, Film, Library, Search, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { ClipQueue } from './components/ClipQueue'
import { ProviderDialog, QualityDialog, UploadDialog } from './components/Dialogs'
import { LibraryPanel } from './components/LibraryPanel'
import { SearchWorkspace } from './components/SearchWorkspace'
import { TopBar } from './components/TopBar'
import { api, uploadVideo } from './lib/api'
import { addClip, updateClip } from './lib/clipQueue'
import { isActiveJob, mergeJobResponse, mergeVideoResponse } from './lib/jobs'
import type {
  ClipDraft,
  ExportResult,
  MobileView,
  ProviderStatus,
  SearchMode,
  SearchResult,
  VideoItem,
} from './types'

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Неизвестная ошибка'
}

export default function App() {
  const [videos, setVideos] = useState<VideoItem[]>([])
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState<'all' | 'current'>('all')
  const [searchMode, setSearchMode] = useState<SearchMode>('all')
  const [useLighthouse, setUseLighthouse] = useState(true)
  const [results, setResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [clips, setClips] = useState<ClipDraft[]>([])
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState<ExportResult | null>(null)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [providerOpen, setProviderOpen] = useState(false)
  const [qualityOpen, setQualityOpen] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const [mobileView, setMobileView] = useState<MobileView>('search')
  const [libraryCollapsed, setLibraryCollapsed] = useState(
    () => window.localStorage.getItem('videoscope.library-collapsed') === 'true',
  )

  const selectedVideo = useMemo(
    () => videos.find((video) => video.id === selectedVideoId) || null,
    [selectedVideoId, videos],
  )

  const loadVideos = useCallback(async (silent = false) => {
    try {
      const next = await api.videos()
      setVideos(next)
      setSelectedVideoId((current) => current ?? next[0]?.id ?? null)
    } catch (error) {
      if (!silent) setToast(errorMessage(error))
    }
  }, [])

  useEffect(() => {
    void loadVideos()
    void api.providers().then(setProviders).catch((error) => setToast(errorMessage(error)))
  }, [loadVideos])

  useEffect(() => {
    const hasWork = videos.some((video) => isActiveJob(video.latest_job))
    if (!hasWork) return
    const timer = window.setInterval(() => void loadVideos(true), 1500)
    return () => window.clearInterval(timer)
  }, [loadVideos, videos])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), 5000)
    return () => window.clearTimeout(timer)
  }, [toast])

  const handleUpload = async (file: File) => {
    setUploadProgress(0)
    try {
      const video = await uploadVideo(file, setUploadProgress)
      setVideos((current) => mergeVideoResponse(current, video))
      setSelectedVideoId(video.id)
      setSelectedResult(null)
      setUploadOpen(false)
      setMobileView('search')
      setToast('Видео добавлено в очередь')
    } catch (error) {
      setToast(errorMessage(error))
    } finally {
      setUploadProgress(null)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) return
    setSearching(true)
    setHasSearched(true)
    try {
      const videoIds = scope === 'current' && selectedVideo ? [selectedVideo.id] : null
      const next = await api.search(query, videoIds, useLighthouse, searchMode)
      setResults(next)
      setSelectedResult(next[0] || null)
      if (next[0]) setSelectedVideoId(next[0].video_id)
    } catch (error) {
      setToast(errorMessage(error))
      setResults([])
    } finally {
      setSearching(false)
    }
  }

  const handleSelectResult = (result: SearchResult) => {
    setSelectedResult(result)
    setSelectedVideoId(result.video_id)
    setMobileView('search')
  }

  const handleAddClip = (result: SearchResult) => {
    setClips((current) => addClip(current, result))
    setToast('Момент добавлен в нарезку')
  }

  const handleRenameVideo = async (video: VideoItem, name: string) => {
    try {
      const updated = await api.renameVideo(video.id, name)
      setVideos((current) => current.map((item) => item.id === updated.id ? updated : item))
      setResults((current) => current.map((item) => (
        item.video_id === updated.id
          ? { ...item, video_name: updated.display_name || updated.original_name }
          : item
      )))
      setSelectedResult((current) => (
        current?.video_id === updated.id
          ? { ...current, video_name: updated.display_name || updated.original_name }
          : current
      ))
      setClips((current) => current.map((clip) => (
        clip.videoId === updated.id
          ? { ...clip, videoName: updated.display_name || updated.original_name }
          : clip
      )))
      setToast('Название сохранено')
    } catch (error) {
      setToast(errorMessage(error))
      throw error
    }
  }

  const handleCancelJob = async (video: VideoItem) => {
    const job = video.latest_job
    if (!isActiveJob(job) || job.cancel_requested_at) return
    try {
      const updated = await api.cancelJob(job.job_id)
      setVideos((current) => mergeJobResponse(current, updated))
      setToast(updated.state === 'cancelled' ? 'Индексация отменена' : 'Запрошена отмена индексации')
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  const handleRetryJob = async (video: VideoItem) => {
    const job = video.latest_job
    if (job?.state !== 'failed' && job?.state !== 'cancelled') return
    try {
      const updated = await api.retryJob(job.job_id)
      setVideos((current) => mergeJobResponse(current, updated))
      setToast('Повторная индексация поставлена в очередь')
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  const toggleLibrary = () => {
    setLibraryCollapsed((current) => {
      const next = !current
      window.localStorage.setItem('videoscope.library-collapsed', String(next))
      return next
    })
  }

  const handleExport = async (name: string) => {
    setExporting(true)
    setExported(null)
    try {
      const output = await api.export(name, clips)
      setExported(output)
      setToast('Нарезка готова')
    } catch (error) {
      setToast(errorMessage(error))
    } finally {
      setExporting(false)
    }
  }

  const readyProviders = providers.filter((provider) => provider.state === 'ready').length
  const lighthouseAvailable = providers.some(
    (provider) => provider.id === 'lighthouse' && provider.state === 'ready',
  )

  return (
    <div className="app-shell">
      <TopBar
        readyProviders={readyProviders}
        totalProviders={providers.length}
        onUpload={() => setUploadOpen(true)}
        onProviders={() => setProviderOpen(true)}
        onQuality={() => setQualityOpen(true)}
      />
      <div className={`app-grid mobile-view-${mobileView} ${libraryCollapsed ? 'library-is-collapsed' : ''}`}>
        <LibraryPanel
          videos={videos}
          selectedId={selectedVideoId}
          collapsed={libraryCollapsed}
          onToggleCollapsed={toggleLibrary}
          onSelect={(video) => {
            setSelectedVideoId(video.id)
            setSelectedResult(null)
            setMobileView('search')
          }}
          onReindex={async (video) => {
            try {
              const updated = await api.reindex(video.id)
              setVideos((current) => mergeVideoResponse(current, updated))
              setToast('Повторная индексация запущена')
            } catch (error) {
              setToast(errorMessage(error))
            }
          }}
          onCancelJob={(video) => void handleCancelJob(video)}
          onRetryJob={(video) => void handleRetryJob(video)}
          onRename={handleRenameVideo}
          onUpload={() => setUploadOpen(true)}
        />
        <SearchWorkspace
          videos={videos}
          selectedVideo={selectedVideo}
          selectedResult={selectedResult}
          query={query}
          scope={scope}
          searchMode={searchMode}
          useLighthouse={useLighthouse}
          lighthouseAvailable={lighthouseAvailable}
          results={results}
          searching={searching}
          hasSearched={hasSearched}
          onQuery={setQuery}
          onScope={setScope}
          onSearchMode={setSearchMode}
          onUseLighthouse={setUseLighthouse}
          onSearch={handleSearch}
          onSelectResult={handleSelectResult}
          onAddClip={handleAddClip}
          onUpload={() => setUploadOpen(true)}
        />
        <ClipQueue
          clips={clips}
          videos={videos}
          exporting={exporting}
          exported={exported}
          onUpdate={(clip, start, end, duration) =>
            setClips((current) =>
              current.map((item) => item.id === clip.id ? updateClip(item, { start, end }, duration) : item),
            )
          }
          onRemove={(clipId) => setClips((current) => current.filter((clip) => clip.id !== clipId))}
          onExport={handleExport}
        />
      </div>

      <nav className="mobile-tabs" aria-label="Разделы приложения">
        <button className={mobileView === 'library' ? 'is-active' : ''} onClick={() => setMobileView('library')}>
          <Library size={19} />
          <span>Видео</span>
        </button>
        <button className={mobileView === 'search' ? 'is-active' : ''} onClick={() => setMobileView('search')}>
          <Search size={19} />
          <span>Поиск</span>
        </button>
        <button className={mobileView === 'clips' ? 'is-active' : ''} onClick={() => setMobileView('clips')}>
          <Clapperboard size={19} />
          <span>Нарезка</span>
          {clips.length > 0 && <i>{clips.length}</i>}
        </button>
      </nav>

      {uploadOpen && (
        <UploadDialog progress={uploadProgress} onUpload={handleUpload} onClose={() => setUploadOpen(false)} />
      )}
      {providerOpen && <ProviderDialog providers={providers} onClose={() => setProviderOpen(false)} />}
      {qualityOpen && <QualityDialog onClose={() => setQualityOpen(false)} />}
      {toast && (
        <div className="toast" role="status">
          <Film size={16} />
          <span>{toast}</span>
          <button className="icon-button" onClick={() => setToast(null)} aria-label="Закрыть уведомление" title="Закрыть">
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  )
}

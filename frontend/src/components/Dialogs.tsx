import {
  CheckCircle2,
  CircleAlert,
  CloudCog,
  FileVideo,
  HardDrive,
  LoaderCircle,
  Play,
  Save,
  Upload,
  X,
} from 'lucide-react'
import { DragEvent, ReactNode, useEffect, useRef, useState } from 'react'
import type { EvaluationPayload, EvaluationVariantName, ProviderStatus } from '../types'
import { formatBytes } from '../lib/time'
import { api } from '../lib/api'

interface ModalProps {
  title: string
  children: ReactNode
  onClose: () => void
}

function Modal({ title, children, onClose }: ModalProps) {
  const dialog = useRef<HTMLDivElement>(null)
  useEffect(() => {
    dialog.current?.focus()
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        ref={dialog}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <h2>{title}</h2>
          <button className="icon-button" type="button" onClick={onClose} title="Закрыть" aria-label="Закрыть">
            <X size={18} />
          </button>
        </header>
        {children}
      </div>
    </div>
  )
}

interface UploadDialogProps {
  progress: number | null
  onUpload: (file: File) => void
  onClose: () => void
}

export function UploadDialog({ progress, onUpload, onClose }: UploadDialogProps) {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const selectDropped = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files.item(0)
    if (dropped) setFile(dropped)
  }
  return (
    <Modal title="Добавить видео" onClose={onClose}>
      <div className="modal-body upload-body">
        <label
          className={`dropzone ${dragging ? 'is-dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={selectDropped}
        >
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo"
            onChange={(event) => setFile(event.target.files?.item(0) || null)}
          />
          {file ? <FileVideo size={30} /> : <Upload size={30} />}
          <strong>{file ? file.name : 'Выберите видео'}</strong>
          <span>{file ? formatBytes(file.size) : 'MP4, MOV, MKV, WebM или AVI'}</span>
        </label>
        {progress !== null && (
          <div className="upload-progress">
            <span><i style={{ width: `${progress * 100}%` }} /></span>
            <strong>{Math.round(progress * 100)}%</strong>
          </div>
        )}
      </div>
      <footer className="modal-actions">
        <button className="button button-secondary" type="button" onClick={onClose} disabled={progress !== null}>
          Отмена
        </button>
        <button
          className="button button-primary"
          type="button"
          disabled={!file || progress !== null}
          onClick={() => file && onUpload(file)}
        >
          {progress !== null ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}
          Загрузить
        </button>
      </footer>
    </Modal>
  )
}

const providerIcons = {
  ready: CheckCircle2,
  needs_configuration: CloudCog,
  unavailable: CircleAlert,
  loading: LoaderCircle,
}

export function ProviderDialog({ providers, onClose }: { providers: ProviderStatus[]; onClose: () => void }) {
  return (
    <Modal title="ML-провайдеры" onClose={onClose}>
      <div className="provider-list">
        {providers.map((provider) => {
          const Icon = providerIcons[provider.state]
          return (
            <div className={`provider-row provider-${provider.state}`} key={provider.id}>
              <Icon className={provider.state === 'loading' ? 'spin' : ''} size={19} />
              <div>
                <strong>{provider.label}</strong>
                <span>{provider.detail}</span>
              </div>
              <em>{provider.optional ? 'опционально' : 'основной'}</em>
            </div>
          )
        })}
        {providers.length === 0 && (
          <div className="provider-empty"><HardDrive size={22} /> Нет данных</div>
        )}
      </div>
    </Modal>
  )
}

const evaluationVariants: { id: EvaluationVariantName; label: string }[] = [
  { id: 'auto', label: 'Авто' },
  { id: 'speech', label: 'Речь' },
  { id: 'visual', label: 'Кадр' },
  { id: 'visual_lighthouse', label: 'Кадр + Lighthouse' },
  { id: 'ocr', label: 'OCR' },
]

function encodeGlossary(entries: Record<string, string[]>): string {
  return Object.entries(entries).map(([term, aliases]) => `${term} = ${aliases.join(', ')}`).join('\n')
}

function decodeGlossary(value: string): Record<string, string[]> {
  const output: Record<string, string[]> = {}
  for (const line of value.split('\n')) {
    const [term, aliases = ''] = line.split('=', 2)
    const normalized = term.trim()
    if (!normalized) continue
    output[normalized] = aliases.split(',').map((alias) => alias.trim()).filter(Boolean)
  }
  return output
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

export function QualityDialog({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'metrics' | 'glossary'>('metrics')
  const [payload, setPayload] = useState<EvaluationPayload | null>(null)
  const [selectedVariants, setSelectedVariants] = useState<EvaluationVariantName[]>(['auto'])
  const [glossary, setGlossary] = useState('')
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void Promise.all([api.evaluation(), api.glossary()])
      .then(([evaluation, terms]) => {
        setPayload(evaluation)
        setGlossary(encodeGlossary(terms.entries))
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : 'Ошибка загрузки'))
  }, [])

  const run = async () => {
    if (selectedVariants.length === 0) return
    setRunning(true)
    setError(null)
    try {
      const report = await api.runEvaluation(selectedVariants)
      setPayload((current) => ({ cases: current?.cases || [], report }))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Ошибка оценки')
    } finally {
      setRunning(false)
    }
  }

  const saveGlossary = async () => {
    setSaving(true)
    setError(null)
    try {
      const result = await api.updateGlossary(decodeGlossary(glossary))
      setGlossary(encodeGlossary(result.entries))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Ошибка сохранения')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal title="Качество поиска" onClose={onClose}>
      <div className="quality-tabs segmented" aria-label="Раздел качества">
        <button type="button" className={activeTab === 'metrics' ? 'is-active' : ''} onClick={() => setActiveTab('metrics')}>Метрики</button>
        <button type="button" className={activeTab === 'glossary' ? 'is-active' : ''} onClick={() => setActiveTab('glossary')}>Словарь</button>
      </div>
      {activeTab === 'metrics' ? (
        <div className="quality-body">
          <div className="evaluation-controls">
            <div className="variant-options" aria-label="Варианты оценки">
              {evaluationVariants.map((variant) => (
                <label key={variant.id}>
                  <input
                    type="checkbox"
                    checked={selectedVariants.includes(variant.id)}
                    onChange={(event) => setSelectedVariants((current) => (
                      event.target.checked
                        ? [...current, variant.id]
                        : current.filter((item) => item !== variant.id)
                    ))}
                  />
                  <span>{variant.label}</span>
                </label>
              ))}
            </div>
            <button className="button button-primary" type="button" onClick={() => void run()} disabled={running || selectedVariants.length === 0} aria-label="Запустить оценку">
              {running ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
              Запустить
            </button>
          </div>
          <div className="quality-summary">
            <span>Контрольные запросы</span>
            <strong>{payload?.cases.length ?? 0}</strong>
          </div>
          {payload?.report?.variants.length ? (
            <div className="metrics-table-wrap">
              <table className="metrics-table">
                <thead><tr><th>Вариант</th><th>Recall@1</th><th>Recall@3</th><th>IoU</th><th>Время</th></tr></thead>
                <tbody>
                  {payload.report.variants.map((variant) => (
                    <tr key={variant.name}>
                      <td>{evaluationVariants.find((item) => item.id === variant.name)?.label || variant.name}</td>
                      <td>{percent(variant.recall_at_1)}</td>
                      <td>{percent(variant.recall_at_3)}</td>
                      <td>{percent(variant.mean_temporal_iou)}</td>
                      <td>{Math.round(variant.mean_latency_ms)} мс</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="quality-empty">Нет сохранённого прогона</div>
          )}
        </div>
      ) : (
        <div className="quality-body glossary-editor">
          <textarea value={glossary} onChange={(event) => setGlossary(event.target.value)} aria-label="Словарь имён и терминов" spellCheck={false} />
          <button className="button button-primary" type="button" onClick={() => void saveGlossary()} disabled={saving}>
            {saving ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}
            Сохранить словарь
          </button>
        </div>
      )}
      {error && <div className="quality-error"><CircleAlert size={15} /> {error}</div>}
    </Modal>
  )
}

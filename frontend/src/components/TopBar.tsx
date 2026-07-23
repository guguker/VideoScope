import { Activity, ChartNoAxesCombined, ScanSearch, Upload } from 'lucide-react'

interface TopBarProps {
  readyProviders: number
  totalProviders: number
  onUpload: () => void
  onProviders: () => void
  onQuality: () => void
}

export function TopBar({ readyProviders, totalProviders, onUpload, onProviders, onQuality }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="brand" aria-label="VideoScope">
        <span className="brand-mark"><ScanSearch size={20} strokeWidth={2.2} /></span>
        <span>VideoScope</span>
      </div>
      <div className="topbar-actions">
        <button className="provider-summary quality-command" type="button" onClick={onQuality} aria-label="Качество поиска">
          <ChartNoAxesCombined size={16} />
          <span className="provider-summary-label">Качество</span>
        </button>
        <button className="provider-summary" type="button" onClick={onProviders} aria-label="Состояние моделей">
          <Activity size={16} />
          <span className="provider-summary-label">Модели</span>
          <strong>{readyProviders}/{totalProviders}</strong>
        </button>
        <button
          className="button button-primary upload-command"
          type="button"
          onClick={onUpload}
          aria-label="Добавить видео"
        >
          <Upload size={17} />
          <span>Добавить видео</span>
        </button>
      </div>
    </header>
  )
}

export type VideoStatus = 'queued' | 'processing' | 'ready' | 'failed'

export interface VideoItem {
  id: string
  original_name: string
  display_name: string | null
  stored_name: string
  size_bytes: number
  status: VideoStatus
  progress: number
  stage: string
  duration: number | null
  width: number | null
  height: number | null
  fps: number | null
  error: string | null
  created_at: string
  updated_at: string
  media_url: string
  thumbnail_url: string | null
}

export type ProviderState = 'ready' | 'needs_configuration' | 'unavailable' | 'loading'

export interface ProviderStatus {
  id: string
  label: string
  state: ProviderState
  detail: string
  optional: boolean
}

export interface SearchEvidence {
  modality: string
  score: number
  text: string
  source: string
  confidence: number | null
  start: number
  end: number
  raw_score: number
  matched_terms: string[]
}

export interface SearchResult {
  id: string
  video_id: string
  video_name: string
  start: number
  end: number
  score: number
  modalities: string[]
  evidence: SearchEvidence[]
  thumbnail_url: string | null
  intent: string
  explanation: string
  refined: boolean
}

export interface ClipDraft {
  id: string
  videoId: string
  videoName: string
  start: number
  end: number
  thumbnailUrl: string | null
}

export interface ExportResult {
  name: string
  duration: number
  created_at: string
  url: string
}

export type MobileView = 'library' | 'search' | 'clips'
export type SearchMode = 'all' | 'speech' | 'visual' | 'ocr'

export interface EvaluationCase {
  id: string
  query: string
  video_id: string
  start: number
  end: number
  mode: SearchMode
  label_source: 'gold' | 'silver'
  notes: string
}

export interface CaseEvaluation {
  case_id: string
  query: string
  relevant_rank: number | null
  temporal_iou: number
  latency_ms: number
  result_count: number
  error: string | null
}

export interface EvaluationVariant {
  name: string
  case_count: number
  recall_at_1: number
  recall_at_3: number
  recall_at_5: number
  mrr: number
  mean_temporal_iou: number
  mean_latency_ms: number
  cases: CaseEvaluation[]
}

export interface EvaluationReport {
  generated_at: string
  variants: EvaluationVariant[]
}

export interface EvaluationPayload {
  cases: EvaluationCase[]
  report: EvaluationReport | null
}

export type EvaluationVariantName = 'auto' | 'auto_lighthouse' | 'speech' | 'visual' | 'visual_lighthouse' | 'ocr'

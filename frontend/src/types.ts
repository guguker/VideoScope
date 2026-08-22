export type VideoStatus = 'queued' | 'processing' | 'ready' | 'failed'

export type JobIntent = 'ingest' | 'reindex'
export type JobState = 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'

export interface JobSummary {
  job_id: string
  intent: JobIntent
  state: JobState
  progress: number
  stage: string
  attempt: number
  cancel_requested_at: string | null
  error_code: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  updated_at: string
}

export interface JobResponse extends JobSummary {
  video_id: string
  retry_of_job_id: string | null
}

export interface VideoItem {
  id: string
  original_name: string
  display_name: string | null
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
  latest_job: JobSummary | null
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
  details: Record<string, unknown>
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
  total_case_count: number
  successful_case_count: number
  error_count: number
  status: 'complete' | 'partial' | 'failed'
  recall_at_1: number | null
  recall_at_3: number | null
  recall_at_5: number | null
  mrr: number | null
  mean_temporal_iou: number | null
  mean_latency_ms: number | null
  cases: CaseEvaluation[]
}

export interface EvaluationReport {
  generated_at: string
  schema_version: number | null
  methodology_version: number | null
  evaluation_revision: string | null
  runtime_revision: string | null
  cases_revision: string | null
  temporal_iou_threshold: number | null
  variants: EvaluationVariant[]
}

export interface EvaluationPayload {
  cases: EvaluationCase[]
  cases_revision: string
  evaluation_revision: string
  runtime_revision: string
  report: EvaluationReport | null
}

export type EvaluationVariantName = 'auto' | 'auto_lighthouse' | 'speech' | 'visual' | 'visual_lighthouse' | 'ocr'

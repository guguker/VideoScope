import type { SearchEvidence, SearchResult } from '../types'

export type FactState = 'yes' | 'no' | 'unknown'

export interface EventFact {
  id: string
  label: string
  state: FactState
}

export interface EventStageScore {
  id: string
  label: string
  score: number
}

export interface EventCardData {
  eventLabel: string
  facts: EventFact[]
  possibleJersey: string | null
  stageScores: EventStageScore[]
}

const eventLabels: Record<string, string> = {
  made_three_point: 'Трёхочковый бросок',
  made_two_point: 'Двухочковый бросок',
  made_free_throw: 'Штрафной бросок',
}

const factDefinitions = [
  ['matches_query', 'Соответствует запросу'],
  ['shot_attempt', 'Попытка броска'],
  ['ball_through_hoop', 'Мяч прошёл через кольцо'],
  ['shooter_outside_arc', 'Игрок находился за дугой'],
  ['three_point_signal', 'Сигнал судьи «три очка»'],
] as const

const stageDefinitions: Record<string, { label: string; scoreKeys: string[] }> = {
  release: { label: 'Подготовка и выпуск', scoreKeys: ['release_score', 'raw_release_score'] },
  contrast: { label: 'Признаки типа броска', scoreKeys: ['contrast_score'] },
  outcome: { label: 'Результат у кольца', scoreKeys: ['outcome_score'] },
  followup: { label: 'Реакция после броска', scoreKeys: ['followup_score'] },
  plus: { label: 'Признаки попадания', scoreKeys: ['plus_score'] },
  reset: { label: 'Возврат к штрафному', scoreKeys: ['reset_score'] },
  miss: { label: 'Признаки промаха', scoreKeys: ['miss_score'] },
  transition: { label: 'Смена владения', scoreKeys: ['transition_score'] },
}

const structuredKeys = new Set([
  'event_type',
  'stage_order',
  'matches_query',
  'shot_attempt',
  'ball_through_hoop',
  'shooter_outside_arc',
  'three_point_signal',
  'possible_shooter_jersey',
  'shooter_jersey_confirmed',
  'model_evidence',
  'core_score',
  ...Object.values(stageDefinitions).flatMap((definition) => definition.scoreKeys),
])

function hasOwn(details: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(details, key)
}

function booleanFact(details: Record<string, unknown>, key: string): FactState {
  const value = details[key]
  if (value === true) return 'yes'
  if (value === false) return 'no'
  return 'unknown'
}

function finiteNumber(details: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = details[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return null
}

function mergeDetails(evidence: SearchEvidence[]): Record<string, unknown> {
  const merged: Record<string, unknown> = {}
  for (const item of evidence) {
    if (!item.details || typeof item.details !== 'object') continue
    for (const [key, value] of Object.entries(item.details)) {
      if (!hasOwn(merged, key) || merged[key] === null || merged[key] === undefined) {
        merged[key] = value
      }
    }
  }
  return merged
}

export function eventCardData(result: SearchResult): EventCardData | null {
  const details = mergeDetails(result.evidence)
  if (!Object.keys(details).some((key) => structuredKeys.has(key))) return null

  const eventType = typeof details.event_type === 'string' ? details.event_type : ''
  const stageOrder = Array.isArray(details.stage_order)
    ? details.stage_order.filter((value): value is string => typeof value === 'string')
    : []
  const orderedStages = stageOrder.length > 0 ? stageOrder : Object.keys(stageDefinitions)
  const seenStages = new Set<string>()
  const stageScores = orderedStages.flatMap((stage) => {
    if (seenStages.has(stage)) return []
    seenStages.add(stage)
    const definition = stageDefinitions[stage]
    if (!definition) return []
    const score = finiteNumber(details, definition.scoreKeys)
    return score === null ? [] : [{ id: stage, label: definition.label, score }]
  })
  const coreScore = finiteNumber(details, ['core_score'])
  if (coreScore !== null) {
    stageScores.push({ id: 'core', label: 'Сводная оценка', score: coreScore })
  }

  const hasQwenFacts = factDefinitions.some(([key]) => hasOwn(details, key))
  const facts = hasQwenFacts
    ? factDefinitions.map(([id, label]) => ({ id, label, state: booleanFact(details, id) }))
    : []
  const possibleJersey = typeof details.possible_shooter_jersey === 'string'
    && details.possible_shooter_jersey.trim()
    ? details.possible_shooter_jersey.trim()
    : null

  return {
    eventLabel: eventLabels[eventType] || 'Проверка спортивного события',
    facts,
    possibleJersey,
    stageScores,
  }
}

export function resultSourceKeys(result: SearchResult): string[] {
  const sourceKeys = result.evidence
    .filter((evidence) => evidence.modality !== 'qwen_video')
    .map((evidence) => evidence.source || evidence.modality)
  const fallbackKeys = result.modalities.filter((modality) => modality !== 'qwen_video')
  return [...new Set(sourceKeys.length > 0 ? sourceKeys : fallbackKeys)]
}

export function hasQwenVerification(result: SearchResult): boolean {
  return result.modalities.includes('qwen_video')
    || result.evidence.some((evidence) => (
      evidence.modality === 'qwen_video' || evidence.source === 'qwen-video-verifier'
    ))
}

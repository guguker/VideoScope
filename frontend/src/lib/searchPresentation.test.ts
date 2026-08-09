import { describe, expect, it } from 'vitest'
import type { SearchEvidence, SearchResult } from '../types'
import { eventCardData, hasQwenVerification, resultSourceKeys } from './searchPresentation'

function evidence(overrides: Partial<SearchEvidence> = {}): SearchEvidence {
  return {
    modality: 'visual',
    score: 0.8,
    text: 'кандидат',
    source: 'siglip2-temporal-event',
    confidence: null,
    start: 10,
    end: 18,
    raw_score: 0.8,
    matched_terms: [],
    details: {},
    ...overrides,
  }
}

function result(items: SearchEvidence[]): SearchResult {
  return {
    id: 'result-1',
    video_id: 'video-1',
    video_name: 'Матч.mp4',
    start: 10,
    end: 18,
    score: 0.86,
    modalities: items.map((item) => item.modality),
    evidence: items,
    thumbnail_url: null,
    intent: 'action',
    explanation: 'Визуальный запрос действия',
    refined: true,
  }
}

describe('eventCardData', () => {
  it('не показывает карточку для обычного неструктурированного результата', () => {
    expect(eventCardData(result([evidence()]))).toBeNull()
  })

  it('объединяет спортивные стадии и факты Qwen без превращения неизвестного в нет', () => {
    const card = eventCardData(result([
      evidence({
        modality: 'qwen_video',
        source: 'qwen-video-verifier',
        details: {
          matches_query: true,
          shot_attempt: true,
          ball_through_hoop: null,
          shooter_outside_arc: false,
          three_point_signal: null,
          possible_shooter_jersey: '15',
          shooter_jersey_confirmed: false,
        },
      }),
      evidence({
        details: {
          event_type: 'made_three_point',
          stage_order: ['release', 'outcome', 'followup'],
          release_score: 0.72,
          outcome_score: 0.81,
          followup_score: 0.63,
          core_score: 0.7,
        },
      }),
    ]))

    expect(card?.eventLabel).toBe('Трёхочковый бросок')
    expect(card?.possibleJersey).toBe('15')
    expect(card?.facts.map((fact) => [fact.id, fact.state])).toEqual([
      ['matches_query', 'yes'],
      ['shot_attempt', 'yes'],
      ['ball_through_hoop', 'unknown'],
      ['shooter_outside_arc', 'no'],
      ['three_point_signal', 'unknown'],
    ])
    expect(card?.stageScores.map((stage) => stage.id)).toEqual([
      'release',
      'outcome',
      'followup',
      'core',
    ])
  })
})

describe('источники результата', () => {
  it('отделяет Qwen от источников кандидата и сообщает о нём только при наличии evidence', () => {
    const item = result([
      evidence(),
      evidence({ modality: 'qwen_video', source: 'qwen-video-verifier' }),
    ])

    expect(resultSourceKeys(item)).toEqual(['siglip2-temporal-event'])
    expect(hasQwenVerification(item)).toBe(true)
  })

  it('не приписывает результату проверку Qwen', () => {
    const item = result([evidence()])
    expect(hasQwenVerification(item)).toBe(false)
  })
})

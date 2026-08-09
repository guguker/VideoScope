import { describe, expect, it } from 'vitest'
import type { EvaluationPayload } from '../types'
import { evaluationReportIsStale } from './evaluation'

function payload(
  caseCount: number,
  reportCaseCount: number,
  casesRevision = 'current-revision',
  reportRevision: string | null = casesRevision,
  evaluationRevision = 'current-evaluation-revision',
  reportEvaluationRevision: string | null = evaluationRevision,
  runtimeRevision = 'current-runtime-revision',
  reportRuntimeRevision: string | null = runtimeRevision,
): EvaluationPayload {
  return {
    cases: Array.from({ length: caseCount }, (_, index) => ({
      id: `case-${index}`,
      query: 'запрос',
      video_id: 'video-1',
      start: 0,
      end: 5,
      mode: 'all',
      label_source: 'gold',
      notes: '',
    })),
    cases_revision: casesRevision,
    evaluation_revision: evaluationRevision,
    runtime_revision: runtimeRevision,
    report: {
      generated_at: '2026-08-01T00:00:00Z',
      schema_version: 3,
      methodology_version: 3,
      evaluation_revision: reportEvaluationRevision,
      runtime_revision: reportRuntimeRevision,
      cases_revision: reportRevision,
      temporal_iou_threshold: 0.3,
      variants: [{
        name: 'auto',
        total_case_count: reportCaseCount,
        successful_case_count: reportCaseCount,
        error_count: 0,
        status: 'complete',
        recall_at_1: 1,
        recall_at_3: 1,
        recall_at_5: 1,
        mrr: 1,
        mean_temporal_iou: 1,
        mean_latency_ms: 10,
        cases: [],
      }],
    },
  }
}

describe('evaluationReportIsStale', () => {
  it('обнаруживает сохранённый отчёт на шесть сценариев при восьми актуальных', () => {
    expect(evaluationReportIsStale(payload(8, 6))).toBe(true)
  })

  it('считает отчёт актуальным при полном покрытии', () => {
    expect(evaluationReportIsStale(payload(8, 8))).toBe(false)
  })

  it('обнаруживает замену кейсов при неизменном количестве', () => {
    expect(evaluationReportIsStale(payload(8, 8, 'new-revision', 'old-revision'))).toBe(true)
  })

  it('считает legacy-отчёт без revision устаревшим', () => {
    expect(evaluationReportIsStale(payload(8, 8, 'current-revision', null))).toBe(true)
  })

  it('считает отчёт старой методологии устаревшим при тех же кейсах', () => {
    expect(evaluationReportIsStale(payload(
      8,
      8,
      'same-cases',
      'same-cases',
      'new-methodology',
      'old-methodology',
    ))).toBe(true)
  })

  it('считает legacy-отчёт без methodology fingerprint устаревшим', () => {
    expect(evaluationReportIsStale(payload(
      8,
      8,
      'same-cases',
      'same-cases',
      'new-methodology',
      null,
    ))).toBe(true)
  })

  it('обнаруживает смену runtime при неизменных кейсах', () => {
    const current = payload(8, 8)
    current.runtime_revision = 'new-runtime'
    current.report!.runtime_revision = 'old-runtime'

    expect(evaluationReportIsStale(current)).toBe(true)
  })
})

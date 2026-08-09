import type { EvaluationPayload } from '../types'

export function evaluationReportIsStale(payload: EvaluationPayload | null): boolean {
  if (!payload?.report) return false
  if (
    payload.report.cases_revision === null
    || payload.report.evaluation_revision === null
    || payload.report.runtime_revision === null
  ) return true
  return payload.report.cases_revision !== payload.cases_revision
    || payload.report.evaluation_revision !== payload.evaluation_revision
    || payload.report.runtime_revision !== payload.runtime_revision
    || payload.report.variants.some((variant) => variant.total_case_count !== payload.cases.length)
}

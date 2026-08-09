import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { EvaluationPayload } from '../types'
import { QualityDialog } from './Dialogs'

const mocks = vi.hoisted(() => ({
  evaluation: vi.fn<() => Promise<EvaluationPayload>>(),
  glossary: vi.fn<() => Promise<{ entries: Record<string, string[]> }>>(),
  updateGlossary: vi.fn<(
    entries: Record<string, string[]>,
  ) => Promise<{ entries: Record<string, string[]> }>>(),
}))

vi.mock('../lib/api', () => ({
  api: {
    evaluation: mocks.evaluation,
    glossary: mocks.glossary,
    runEvaluation: vi.fn(),
    updateGlossary: mocks.updateGlossary,
  },
}))

describe('QualityDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.glossary.mockResolvedValue({ entries: {} })
    mocks.updateGlossary.mockResolvedValue({ entries: {} })
  })

  it('явно показывает partial/failed прогоны и не рисует null как ноль', async () => {
    mocks.evaluation.mockResolvedValue({
      cases: [
        { id: 'one', query: 'one', video_id: 'video-1', start: 0, end: 1, mode: 'all', label_source: 'gold', notes: '' },
        { id: 'two', query: 'two', video_id: 'video-1', start: 1, end: 2, mode: 'all', label_source: 'gold', notes: '' },
      ],
      cases_revision: 'a'.repeat(64),
      evaluation_revision: 'b'.repeat(64),
      runtime_revision: 'c'.repeat(64),
      report: {
        generated_at: '2026-08-01T00:00:00Z',
        schema_version: 3,
        methodology_version: 3,
        evaluation_revision: 'b'.repeat(64),
        runtime_revision: 'c'.repeat(64),
        cases_revision: 'a'.repeat(64),
        temporal_iou_threshold: 0.3,
        variants: [
          {
            name: 'auto',
            total_case_count: 2,
            successful_case_count: 1,
            error_count: 1,
            status: 'partial',
            recall_at_1: 1,
            recall_at_3: 1,
            recall_at_5: 1,
            mrr: 1,
            mean_temporal_iou: 0.8,
            mean_latency_ms: 10,
            cases: [],
          },
          {
            name: 'speech',
            total_case_count: 2,
            successful_case_count: 0,
            error_count: 2,
            status: 'failed',
            recall_at_1: null,
            recall_at_3: null,
            recall_at_5: null,
            mrr: null,
            mean_temporal_iou: null,
            mean_latency_ms: null,
            cases: [],
          },
        ],
      },
    })

    render(<QualityDialog onClose={() => undefined} />)

    expect(await screen.findByText('Оценка выполнена не полностью')).toBeInTheDocument()
    expect(screen.getByText(/Ошибки поиска: 3/)).toBeInTheDocument()
    expect(screen.getByText(/только по успешно выполненным сценариям/)).toBeInTheDocument()
    const failedRow = screen.getByRole('row', { name: /Речь/ })
    expect(within(failedRow).getAllByText('—')).toHaveLength(4)
    expect(within(failedRow).getByText('Сбой')).toBeInTheDocument()
    expect(screen.getByText('Авто + Lighthouse')).toBeInTheDocument()
  })

  it('не запускает benchmark без контрольных сценариев', async () => {
    mocks.evaluation.mockResolvedValue({
      cases: [],
      cases_revision: 'a'.repeat(64),
      evaluation_revision: 'b'.repeat(64),
      runtime_revision: 'c'.repeat(64),
      report: null,
    })

    render(<QualityDialog onClose={() => undefined} />)

    expect(await screen.findByText('Контрольные сценарии не настроены')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Запустить оценку' })).toBeDisabled()
  })

  it('после сохранения словаря обновляет freshness отчёта', async () => {
    const currentRuntime = 'c'.repeat(64)
    const freshPayload: EvaluationPayload = {
      cases: [
        { id: 'one', query: 'one', video_id: 'video-1', start: 0, end: 1, mode: 'all', label_source: 'gold', notes: '' },
      ],
      cases_revision: 'a'.repeat(64),
      evaluation_revision: 'b'.repeat(64),
      runtime_revision: currentRuntime,
      report: {
        generated_at: '2026-08-01T00:00:00Z',
        schema_version: 3,
        methodology_version: 3,
        evaluation_revision: 'b'.repeat(64),
        runtime_revision: currentRuntime,
        cases_revision: 'a'.repeat(64),
        temporal_iou_threshold: 0.3,
        variants: [{
          name: 'auto',
          total_case_count: 1,
          successful_case_count: 1,
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
    mocks.evaluation
      .mockResolvedValueOnce(freshPayload)
      .mockResolvedValueOnce({
        ...freshPayload,
        runtime_revision: 'd'.repeat(64),
        evaluation_revision: 'e'.repeat(64),
      })
    mocks.updateGlossary.mockResolvedValue({ entries: { Мозгов: ['Mozgov'] } })

    render(<QualityDialog onClose={() => undefined} />)
    await screen.findByText('Полный 1/1')
    fireEvent.click(screen.getByRole('button', { name: 'Словарь' }))
    fireEvent.change(screen.getByLabelText('Словарь имён и терминов'), {
      target: { value: 'Мозгов = Mozgov' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить словарь' }))

    await waitFor(() => expect(mocks.evaluation).toHaveBeenCalledTimes(2))
    fireEvent.click(screen.getByRole('button', { name: 'Метрики' }))
    expect(await screen.findByText('Сохранённый прогон не покрывает актуальный набор')).toBeInTheDocument()
  })
})

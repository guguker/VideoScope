import { fireEvent, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The workbench is served by the isolated Python server, outside Vite's root.
// These imports exercise the exact browser modules shipped by that server.
// @ts-expect-error JavaScript server asset intentionally has no TS declaration.
import { pointFromClient } from '../../../backend/src/videoscope/annotation_workbench/web/geometry.js'
// @ts-expect-error JavaScript server asset intentionally has no TS declaration.
import { mountWorkbench } from '../../../backend/src/videoscope/annotation_workbench/web/app.js'

const { readFileSync } = await vi.importActual<{
  readFileSync: (path: string, encoding: 'utf8') => string
}>('node:fs')
const html = readFileSync('../backend/src/videoscope/annotation_workbench/web/index.html', 'utf8')

const source = {
  source_id: 'uba-01', title: 'Матч 01', sha256: 'a'.repeat(64),
  duration_seconds: 7265.25, review_allowed: true,
  role: 'development_review', media_url: '/media/uba-01',
  source_group: 'uba-game-01', training_rights: 'unknown', fps: 1,
}
const workspace = {
  schema_version: 1, workspace_id: 'uba-workbench-v1',
  workspace_revision: 'b'.repeat(64), title: 'UBA · Разметка матчей',
  sources: [source, { ...source, source_id: 'uba-03', title: 'Матч 03', media_url: null,
    review_allowed: false, role: 'reserve_uninspected' }],
  progress: null,
}
const eventRecord = (id: string, notes: string, revision = 1) => ({
  record_id: id, revision, source_id: source.source_id, source_sha256: source.sha256,
  kind: 'event', status: 'draft', archived: false,
  data: {
    event_type: 'shot', start_seconds: 120, end_seconds: 125, possession_id: null,
    actor_id: null, receiver_id: null, passer_id: null, last_pass_seconds: null,
    incoming_player_id: null, outgoing_player_id: null, shot_type: null,
    outcome: null, scoring_decision: null, play_context: null,
    presentation: null, boundary_status: null, defensive_action: null, notes,
  },
  created_at: '2026-09-11T10:00:00Z', updated_at: '2026-09-11T10:00:00Z', origin: 'human',
})
const playerRecord = (id: string) => ({
  record_id: id, revision: 1, source_id: source.source_id, source_sha256: source.sha256,
  kind: 'player', status: 'draft', archived: false,
  data: { team_id: null, jersey_number: '00', number_status: 'readable', evidence_seconds: 100, notes: '' },
  created_at: '2026-09-11T10:00:00Z', updated_at: '2026-09-11T10:00:00Z', origin: 'human',
})
const teamRecord = (id: string, name: string, color: string) => ({
  record_id: id, revision: 1, source_id: source.source_id, source_sha256: source.sha256,
  kind: 'team', status: 'human_reviewed', archived: false,
  data: { name, color, evidence_seconds: 90 },
  created_at: '2026-09-11T10:00:00Z', updated_at: '2026-09-11T10:00:00Z', origin: 'human',
})

const reply = (value: unknown, status = 200, contentType = 'application/json') =>
  new Response(contentType === 'application/json' ? JSON.stringify(value) : String(value), {
    status, headers: { 'Content-Type': contentType },
  })

let fetchMock: ReturnType<typeof vi.fn>
let screen: ReturnType<typeof within>
let dispose: undefined | (() => void)
let serverRecords: Array<Record<string, any>>

class MemoryStorage implements Storage {
  private values = new Map<string, string>()
  get length() { return this.values.size }
  clear() { this.values.clear() }
  getItem(key: string) { return this.values.get(key) ?? null }
  key(index: number) { return [...this.values.keys()][index] ?? null }
  removeItem(key: string) { this.values.delete(key) }
  setItem(key: string, value: string) { this.values.set(key, String(value)) }
}

function completeNetwork(url: string, options: RequestInit = {}) {
  if (url === '/api/workspace') return reply(workspace)
  if (url.startsWith('/api/sources/')) {
    const sourceId = url.split('/').at(-1)!
    const selectedSource = workspace.sources.find(item => item.source_id === sourceId)!
    return reply({ source: selectedSource, records: sourceId === 'uba-01' ? serverRecords : [], progress: {
      source_id: sourceId, position_seconds: 120, selected_record_id: sourceId === 'uba-01' ? serverRecords.at(-1)?.record_id ?? null : null,
      playback_rate: 1,
    } })
  }
  if (url === '/api/progress') return reply({ ok: true })
  if (/\/api\/records\/[^/]+\/history$/.test(url)) return reply({ records: serverRecords })
  if (url === '/api/records' && options.method === 'POST') {
    const payload = JSON.parse(String(options.body))
    const record = {
      ...payload, revision: payload.expected_revision + 1,
      source_sha256: source.sha256, created_at: '2026-09-11T10:00:00Z',
      updated_at: '2026-09-11T10:01:00Z', origin: 'human',
    }
    serverRecords = [...serverRecords.filter(item => item.record_id !== record.record_id), record]
    return reply(record)
  }
  throw new Error(`Unexpected request: ${options.method ?? 'GET'} ${url}`)
}

async function mount(records: Array<Record<string, any>> = []) {
  serverRecords = records
  document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
  screen = within(document.body)
  fetchMock.mockImplementation(completeNetwork)
  dispose = await mountWorkbench()
  await screen.findByRole('heading', { name: 'Матч 01' })
}

describe('full-match annotation workbench geometry', () => {
  it('normalizes points against the intrinsic 16:9 image and rejects its letterbox', () => {
    expect(pointFromClient({
      clientX: 400, clientY: 75, rect: { left: 0, top: 0, width: 800, height: 600 },
      videoWidth: 1920, videoHeight: 1080,
    })).toEqual({ x: 0.5, y: 0 })
    expect(pointFromClient({
      clientX: 400, clientY: 50, rect: { left: 0, top: 0, width: 800, height: 600 },
      videoWidth: 1920, videoHeight: 1080,
    })).toBeNull()
  })
})

describe('full-match annotation workbench editing', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', new MemoryStorage())
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  })

  afterEach(() => {
    dispose?.()
    dispose = undefined
    document.documentElement.innerHTML = '<head></head><body></body>'
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    delete (HTMLVideoElement.prototype as unknown as { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback
    Object.assign(workspace.sources[1], { media_url: null, review_allowed: false, role: 'reserve_uninspected' })
  })

  it('retains independently saved answers when several events share a match', async () => {
    await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Первый бросок' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByText('Черновик сохранён')

    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.change(screen.getByLabelText('Тип события'), { target: { value: 'pass' } })
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Передача под кольцо' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Событие:/ })).toHaveLength(2))

    fireEvent.click(screen.getByRole('button', { name: 'Событие: Первый бросок' }))
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Первый бросок')
    fireEvent.click(screen.getByRole('button', { name: 'Событие: Передача под кольцо' }))
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Передача под кольцо')
  })

  it('preserves a player jersey number as the string 00 in its save payload and editor', async () => {
    await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить игрока' }))
    fireEvent.input(screen.getByLabelText('Номер формы'), { target: { value: '00' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByText('Черновик сохранён')
    const payload = JSON.parse(fetchMock.mock.calls.find(([, options]) => options?.method === 'POST' &&
      JSON.parse(String(options.body)).kind === 'player')![1].body)
    expect(payload.data.jersey_number).toBe('00')
    expect(screen.getByLabelText('Номер формы')).toHaveValue('00')
  })

  it('distinguishes the same jersey number by linked team in lists and participant selectors', async () => {
    const blueId = `team-${'5'.repeat(32)}`
    const redId = `team-${'6'.repeat(32)}`
    const bluePlayer = { ...playerRecord(`player-${'7'.repeat(32)}`), data: {
      ...playerRecord(`player-${'7'.repeat(32)}`).data, team_id: blueId,
    } }
    const redPlayer = { ...playerRecord(`player-${'8'.repeat(32)}`), data: {
      ...playerRecord(`player-${'8'.repeat(32)}`).data, team_id: redId,
    } }
    await mount([teamRecord(blueId, 'Синие', 'синий'), teamRecord(redId, 'Красные', 'красный'), bluePlayer, redPlayer])
    expect(screen.getByRole('button', { name: 'Игрок: Синие · №00' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Игрок: Красные · №00' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    const actor = within(screen.getByLabelText('Бросающий'))
    expect(actor.getByRole('option', { name: 'Синие · №00' })).toBeInTheDocument()
    expect(actor.getByRole('option', { name: 'Красные · №00' })).toBeInTheDocument()
  })

  it('restores a locally persisted draft after the page is mounted again', async () => {
    await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Возвратить после перезапуска' } })
    dispose?.()
    document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
    screen = within(document.body)
    dispose = await mountWorkbench()
    await screen.findByRole('heading', { name: 'Матч 01' })
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Возвратить после перезапуска')
    expect(screen.getByText('Восстановлен локальный черновик')).toBeInTheDocument()
  })

  it('shows the original clip context and missing-review state for an imported v1 event', async () => {
    const legacy = {
      ...eventRecord(`event-${'3'.repeat(32)}`, 'Импортированный бросок'),
      status: 'human_reviewed', needs_review: true, origin: 'legacy',
      legacy: {
        batch_revision: 'c'.repeat(64), example_id: 'example-01', event_id: 'primary',
        source_id: 'uba-01', source_sha256: source.sha256,
        source_start_seconds: 3500, source_end_seconds: 3520,
        schema_version: 1, label_status: 'human_reviewed',
      },
    }
    await mount([legacy])
    expect(screen.getByText('Нужно дополнить два решения из новой схемы')).toBeInTheDocument()
    expect(screen.getByText('1 объектов · 0 подтверждено')).toBeInTheDocument()
    expect(screen.queryByText('Подтверждено')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Состояние'), { target: { value: 'human_reviewed' } })
    expect(screen.queryByRole('button', { name: 'Событие: Импортированный бросок' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Состояние'), { target: { value: 'draft' } })
    expect(screen.getByRole('button', { name: 'Событие: Импортированный бросок' })).toBeInTheDocument()
    const context = screen.getByRole('button', { name: 'Перейти к исходному контексту' })
    expect(context).toHaveTextContent('00:58:20–00:58:40')
    fireEvent.click(context)
    expect((screen.getByLabelText('Видео матча') as HTMLVideoElement).currentTime).toBe(3500)
  })

  it.each([409, 500, 'network'])('keeps edited fields after an unacknowledged %s save', async failure => {
    await mount([eventRecord(`event-${'1'.repeat(32)}`, 'Старое значение')])
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Новая несохранённая правка' } })
    fetchMock.mockImplementationOnce((_url, options) => failure === 'network'
      ? Promise.reject(new TypeError('offline'))
      : Promise.resolve(reply({ detail: 'conflict' }, Number(failure))))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByRole('alert')
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Новая несохранённая правка')
    expect(screen.getByText('Сохранено локально, ожидает отправки')).toBeInTheDocument()
    expect(screen.queryByText('Черновик сохранён')).not.toBeInTheDocument()
  })

  it('shows both stale versions and explicitly rebases the local draft onto the latest revision', async () => {
    const recordId = `event-${'9'.repeat(32)}`
    await mount([eventRecord(recordId, 'Серверная версия 1', 1)])
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Моя локальная правка' } })
    serverRecords = [eventRecord(recordId, 'Серверная версия 2', 2)]
    fetchMock.mockImplementationOnce(() => Promise.resolve(reply({ detail: 'stale revision' }, 409)))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    const conflict = (await screen.findByText('Конфликт версий')).closest('section')!
    expect(within(conflict).getByText('Моя локальная правка')).toBeInTheDocument()
    expect(within(conflict).getByText('Серверная версия 2')).toBeInTheDocument()
    expect(within(conflict).getByText(/Версия сервера: 2\./)).toBeInTheDocument()

    dispose?.()
    document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
    screen = within(document.body)
    dispose = await mountWorkbench()
    await screen.findByRole('heading', { name: 'Матч 01' })
    expect(await screen.findByText('Конфликт версий')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Загрузить серверную версию' }))
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Серверная версия 2')
    fireEvent.click(screen.getByRole('button', { name: 'Вернуть мои локальные изменения' }))
    expect(screen.getByLabelText('Комментарий к событию')).toHaveValue('Моя локальная правка')
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить мои изменения черновиком' }))
    await screen.findByText('Черновик сохранён')
    const finalSave = fetchMock.mock.calls.filter(([url, options]) => url === '/api/records' && options?.method === 'POST').at(-1)!
    const payload = JSON.parse(String(finalSave[1].body))
    expect(payload.expected_revision).toBe(2)
    expect(payload.status).toBe('draft')
    expect(payload.data.notes).toBe('Моя локальная правка')
  })

  it('normalizes a cleared optional enum to null in the saved record', async () => {
    const saved: Record<string, any> = eventRecord(`event-${'a'.repeat(32)}`, 'Очистить защиту')
    saved.data.defensive_action = 'help'
    await mount([saved])
    fireEvent.change(screen.getByLabelText('Защитное действие'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByText('Черновик сохранён')
    const payload = JSON.parse(String(fetchMock.mock.calls.filter(([url, options]) =>
      url === '/api/records' && options?.method === 'POST').at(-1)![1].body))
    expect(payload.data.defensive_action).toBeNull()
    const card = screen.getByRole('button', { name: 'Событие: Очистить защиту' })
    expect(card).toHaveTextContent('черновик')
    expect(card).not.toHaveTextContent('нужно дополнить')
  })

  it('does not replace the active field or cursor when an older save is acknowledged', async () => {
    await mount([eventRecord(`event-${'b'.repeat(32)}`, 'Начало')])
    const notes = screen.getByLabelText('Комментарий к событию') as HTMLTextAreaElement
    fireEvent.input(notes, { target: { value: 'Первая правка' } })
    let finishSave!: (response: Response) => void
    fetchMock.mockImplementationOnce((_url: string, options: RequestInit) => {
      const payload = JSON.parse(String(options.body))
      return new Promise<Response>(resolve => { finishSave = response => resolve(response) }).then(() =>
        reply({ ...payload, revision: 2, source_sha256: source.sha256, needs_review: true,
          created_at: '2026-09-11T10:00:00Z', updated_at: '2026-09-11T10:01:00Z', origin: 'human' }))
    })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await waitFor(() => expect(finishSave).toBeTypeOf('function'))
    fireEvent.input(notes, { target: { value: 'Вторая правка продолжается' } })
    notes.focus()
    notes.setSelectionRange(8, 8)
    finishSave(reply({ ok: true }))
    await waitFor(() => expect(screen.getByText('Сохраняем…')).toBeInTheDocument())
    expect(screen.getByLabelText('Комментарий к событию')).toBe(notes)
    expect(notes).toHaveValue('Вторая правка продолжается')
    expect(document.activeElement).toBe(notes)
    expect(notes.selectionStart).toBe(8)
  })

  it('autosaves edits only as drafts and requires a separate review action', async () => {
    await mount()
    vi.useFakeTimers()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Только черновик' } })
    await vi.advanceTimersByTimeAsync(900)
    const payloads = fetchMock.mock.calls
      .filter(([, options]) => options?.method === 'POST' && String(options.body).includes('"kind":"event"'))
      .map(([, options]) => JSON.parse(String(options.body)))
    expect(payloads).toHaveLength(1)
    expect(payloads[0].status).toBe('draft')
    expect(screen.getByRole('button', { name: 'Подтвердить разметку' })).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('chains two saves for one event onto the acknowledged revision', async () => {
    await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Первая правка' } })
    let finishFirst!: (response: Response) => void
    let firstPayload: Record<string, any> | null = null
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url === '/api/records' && options.method === 'POST' && firstPayload === null) {
        firstPayload = JSON.parse(String(options.body))
        return new Promise<Response>(resolve => { finishFirst = resolve })
      }
      return completeNetwork(url, options)
    })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await waitFor(() => expect(finishFirst).toBeTypeOf('function'))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Вторая правка' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    finishFirst(reply({ ...(firstPayload || {}), revision: 1, source_sha256: source.sha256,
      created_at: '2026-09-11T10:00:00Z', updated_at: '2026-09-11T10:01:00Z', origin: 'human' }))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url, options]) =>
      url === '/api/records' && options?.method === 'POST')).toHaveLength(2))
    const saves = fetchMock.mock.calls.filter(([url, options]) => url === '/api/records' && options?.method === 'POST')
    const second = JSON.parse(String(saves[1][1].body))
    expect(second.expected_revision).toBe(1)
    expect(second.data.notes).toBe('Вторая правка')
  })

  it('cancels a pending source draft autosave when switching matches', async () => {
    Object.assign(workspace.sources[1], { media_url: '/media/uba-03', review_allowed: true, role: 'development_review' })
    await mount()
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    fireEvent.input(screen.getByLabelText('Комментарий к событию'), { target: { value: 'Только первый матч' } })
    fireEvent.click(screen.getByRole('button', { name: 'Матч 03' }))
    await screen.findByRole('heading', { name: 'Матч 03' })
    await new Promise(resolve => setTimeout(resolve, 700))
    const wrongSourceSave = fetchMock.mock.calls.some(([url, options]) => url === '/api/records' &&
      options?.method === 'POST' && JSON.parse(String(options.body)).data.notes === 'Только первый матч')
    expect(wrongSourceSave).toBe(false)
  })

  it('hides frame points as soon as playback moves away from their timestamp', async () => {
    await mount()
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    Object.defineProperties(video, {
      videoWidth: { value: 1920, configurable: true }, videoHeight: { value: 1080, configurable: true },
      currentTime: { value: 120, writable: true, configurable: true },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить кадр с точками' }))
    fireEvent.click(screen.getByLabelText('Мяч'))
    const surface = screen.getByLabelText('Кадр видео для отметки точек')
    const rectSpy = vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600, x: 0, y: 0, toJSON: () => ({}),
    })
    fireEvent.click(surface, { clientX: 400, clientY: 300 })
    expect(screen.getByLabelText('Точка: мяч')).toBeVisible()
    expect(screen.getByLabelText('Точка: мяч')).toHaveStyle({ left: '400px', top: '300px' })
    rectSpy.mockReturnValue({
      left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300, x: 0, y: 0, toJSON: () => ({}),
    })
    fireEvent.resize(window)
    expect(screen.getByLabelText('Точка: мяч')).toHaveStyle({ left: '200px', top: '150px' })
    video.currentTime = 121
    fireEvent.timeUpdate(video)
    expect(screen.queryByLabelText('Точка: мяч')).not.toBeInTheDocument()
  })

  it('uses presented-frame media time for a new spatial frame when the browser exposes it', async () => {
    const callbacks: Array<(now: number, metadata: { mediaTime: number }) => void> = []
    Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', {
      configurable: true,
      value: vi.fn((callback: (now: number, metadata: { mediaTime: number }) => void) => {
        callbacks.push(callback)
        return callbacks.length
      }),
    })
    await mount()
    callbacks[0](0, { mediaTime: 119.875 })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить кадр с точками' }))
    expect(screen.getByLabelText('Время кадра, секунды')).toHaveValue(119.875)
  })

  it('adds another spatial point without replacing the first point or its identity', async () => {
    const playerId = `player-${'2'.repeat(32)}`
    await mount([playerRecord(playerId)])
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    Object.defineProperties(video, {
      videoWidth: { value: 1920, configurable: true }, videoHeight: { value: 1080, configurable: true },
      currentTime: { value: 120, writable: true, configurable: true },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить кадр с точками' }))
    fireEvent.change(screen.getByLabelText('Игрок для точки'), { target: { value: playerId } })
    const surface = screen.getByLabelText('Кадр видео для отметки точек')
    vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600, x: 0, y: 0, toJSON: () => ({}),
    })
    fireEvent.click(surface, { clientX: 300, clientY: 300 })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить следующую точку' }))
    fireEvent.click(screen.getByLabelText('Мяч'))
    fireEvent.click(surface, { clientX: 500, clientY: 300 })
    expect(screen.getByLabelText('Точка: игрок 00')).toBeInTheDocument()
    expect(screen.getByLabelText('Точка: мяч')).toBeInTheDocument()
  })

  it('does not attach a point after playback has moved away from the frame', async () => {
    await mount()
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    Object.defineProperties(video, {
      videoWidth: { value: 1920, configurable: true }, videoHeight: { value: 1080, configurable: true },
      currentTime: { value: 120, writable: true, configurable: true },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить кадр с точками' }))
    expect(video.pause).toHaveBeenCalled()
    fireEvent.click(screen.getByLabelText('Мяч'))
    const surface = screen.getByLabelText('Кадр видео для отметки точек')
    vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600, x: 0, y: 0, toJSON: () => ({}),
    })
    video.currentTime = 121
    fireEvent.timeUpdate(video)
    fireEvent.click(surface, { clientX: 400, clientY: 300 })
    expect(screen.getByRole('alert')).toHaveTextContent('Вернитесь к времени кадра')
    expect(screen.queryByLabelText('Точка: мяч')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Вернуться к кадру' }))
    expect(video.currentTime).toBe(120)
  })

  it('steps one frame using the measured source FPS', async () => {
    await mount()
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    Object.defineProperty(video, 'currentTime', { value: 120, writable: true, configurable: true })
    fireEvent.click(screen.getByRole('button', { name: 'Следующий кадр' }))
    expect(video.currentTime).toBe(121)
  })

  it('does not run playback shortcuts while focus is in an editable control', async () => {
    await mount()
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    Object.defineProperty(video, 'currentTime', { value: 120, writable: true, configurable: true })
    const exactSeek = screen.getByLabelText('Точное время, чч:мм:сс')
    exactSeek.focus()
    fireEvent.keyDown(exactSeek, { key: 'ArrowRight' })
    expect(video.currentTime).toBe(120)
    fireEvent.keyDown(document.body, { key: 'ArrowRight' })
    expect(video.currentTime).toBe(120.25)
  })

  it('does not overwrite resume progress from passive initial media events', async () => {
    await mount([eventRecord(`event-${'4'.repeat(32)}`, 'Сохранённый момент')])
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    fireEvent.seeked(video)
    fireEvent.pause(video)
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/progress')).toBe(false)
    fireEvent.input(screen.getByLabelText('Точное время, чч:мм:сс'), { target: { value: '01:00:01' } })
    fireEvent.click(screen.getByRole('button', { name: 'Перейти' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === '/api/progress')).toBe(true))
    const payload = JSON.parse(String(fetchMock.mock.calls.find(([url]) => url === '/api/progress')![1].body))
    expect(payload.position_seconds).toBe(3601)
  })

  it('exposes native long-match playback controls except while placing frame points', async () => {
    await mount()
    const video = screen.getByLabelText('Видео матча') as HTMLVideoElement
    expect(video.controls).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Добавить кадр с точками' }))
    expect(video.controls).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Добавить событие' }))
    expect(video.controls).toBe(true)
  })
})

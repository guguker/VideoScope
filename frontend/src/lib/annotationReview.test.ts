import { fireEvent, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// These assets belong to the isolated review server, outside Vite's serving root.
// Read them in the test runner without adding Node types to the production app.
const { readFileSync } = await vi.importActual<{
  readFileSync: (path: string, encoding: 'utf8') => string
}>('node:fs')
const webRoot = '../backend/src/videoscope/annotation_review/web'
const html = readFileSync(`${webRoot}/index.html`, 'utf8')
const script = readFileSync(`${webRoot}/app.js`, 'utf8')

const example = (id: string) => ({
  example_id: id,
  source_alias: 'UBA 01',
  source_start_seconds: 125.5,
  source_end_seconds: 145.5,
  clip_duration_seconds: 20,
  video_url: `/media/${id}`,
  poster_url: `/posters/${id}`,
})
const annotation = {
  schema_version: 2, scoring_decision: 'counted', play_context: 'in_play',
  revision: 1, shot_type: 'three', outcome: 'made', presentation: 'live',
  boundary_status: 'complete', start_seconds: 1, end_seconds: 18, notes: 'Мяч виден',
}
const review = (annotations = {}) => ({
  batch_id: 'pilot', batch_revision: 'revision-1', title: 'Проверка эпизодов UBA',
  examples: [example('e1'), example('e2')], annotations,
})
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
})

let fetchMock: ReturnType<typeof vi.fn>
let screen: ReturnType<typeof within>

async function mount(data: ReturnType<typeof review> & { events?: Record<string, unknown[]> } = review()) {
  fetchMock.mockResolvedValueOnce(reply(data))
  document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
  screen = within(document.body)
  window.eval(script)
  await screen.findByRole('heading', { name: /^Эпизод 0[12]$/ })
}

function chooseLabels() {
  fireEvent.click(screen.getByLabelText('Трёхочковый'))
  fireEvent.click(screen.getByLabelText('Попадание'))
  fireEvent.click(screen.getByLabelText('Засчитаны'))
  fireEvent.click(screen.getByLabelText('В игре'))
  fireEvent.click(screen.getByLabelText('Основной эпизод'))
  fireEvent.click(screen.getByLabelText('Контекста хватает'))
}

describe('standalone local annotation review', () => {
  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    document.body.innerHTML = ''
  })

  it('keeps the first draft and creates a separate blank second shot without saving implicitly', async () => {
    await mount()
    chooseLabels()
    fireEvent.input(screen.getByLabelText('Конец фрагмента, секунды'), { target: { value: '5.844' } })
    fireEvent.input(screen.getByLabelText('Комментарий'), { target: { value: 'Первый бросок с фолом' } })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить ещё один бросок в этом клипе' }))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(screen.getAllByRole('radio').every((input: HTMLElement) => !(input as HTMLInputElement).checked)).toBe(true)
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(5.844)
    fireEvent.click(screen.getByRole('button', { name: /Бросок 1/ }))
    expect(screen.getByLabelText('Комментарий')).toHaveValue('Первый бросок с фолом')
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(5.844)
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
  })

  it('saves the second event independently and stays in its clip', async () => {
    const first = { ...annotation, event_id: 'primary', end_seconds: 5.844 }
    await mount({ ...review({ e1: first }), events: { e1: [first], e2: [] } })
    fireEvent.click(screen.getByRole('button', { name: 'Предыдущий' }))
    fireEvent.click(screen.getByRole('button', { name: 'Добавить ещё один бросок в этом клипе' }))
    chooseLabels()
    fireEvent.click(screen.getByLabelText('Не засчитаны'))
    fireEvent.click(screen.getByLabelText('Новый бросок после свистка'))
    fetchMock.mockImplementationOnce((_url: string, options: RequestInit) => {
      const payload = JSON.parse(String(options.body))
      return reply({ annotation: { ...payload, revision: 1 }, reviewed_count: 1, total_count: 2 })
    })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить бросок', exact: true }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const payload = JSON.parse(fetchMock.mock.calls[1][1].body)
    expect(payload.schema_version).toBe(3)
    expect(payload.event_id).toMatch(/^event-[a-f0-9]{32}$/)
    expect(payload.expected_revision).toBe(0)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Сохранить бросок', exact: true })).toBeEnabled())
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Бросок 1/ }))
    expect(screen.getByLabelText('Засчитаны')).toBeChecked()
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(5.844)
  })

  it('starts every human label blank and requires an explicit decision', async () => {
    await mount()
    expect(screen.getAllByRole('radio').every((input: HTMLElement) => !(input as HTMLInputElement).checked)).toBe(true)
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(0)
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(20)
    fireEvent.submit(screen.getByRole('form', { name: 'Разметка эпизода' }))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('alert')).toHaveTextContent('Выберите ответ в каждой из шести групп')
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
  })

  it('advances and marks reviewed only after a successful server acknowledgement', async () => {
    await mount()
    chooseLabels()
    let resolve!: (response: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Следующий' })).toBeDisabled()
    const [url, options] = fetchMock.mock.calls[1]
    expect(url).toBe('/api/annotations')
    expect(JSON.parse(options.body)).toEqual({
      schema_version: 3, event_id: 'primary', scoring_decision: 'counted', play_context: 'in_play',
      batch_revision: 'revision-1', example_id: 'e1', expected_revision: 0,
      shot_type: 'three', outcome: 'made', presentation: 'live',
      boundary_status: 'complete', start_seconds: 0, end_seconds: 20, notes: '',
    })
    resolve(reply({ annotation, reviewed_count: 1, total_count: 2 }))
    await screen.findByRole('heading', { name: 'Эпизод 02' })
    expect(screen.getByText('1 из 2 сохранено')).toBeInTheDocument()
    expect(screen.getAllByRole('radio').every((input: HTMLElement) => !(input as HTMLInputElement).checked)).toBe(true)
  })

  it.each([409, 422, 500])('keeps input and position after HTTP %s without false progress', async status => {
    await mount()
    chooseLabels()
    fireEvent.input(screen.getByLabelText('Комментарий'), { target: { value: 'Проверяю границы' } })
    fetchMock.mockResolvedValueOnce(reply({ detail: 'server detail' }, status))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    await waitFor(() => expect(screen.getByRole('alert')).not.toHaveTextContent(/^$/))
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
    expect(screen.getByLabelText('Комментарий')).toHaveValue('Проверяю границы')
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
    if (status === 409) expect(screen.getByRole('alert')).toHaveTextContent('обновите страницу')
  })

  it.each(['network', 'malformed'])('retains answers after an unconfirmed %s save', async failure => {
    await mount()
    chooseLabels()
    if (failure === 'network') fetchMock.mockRejectedValueOnce(new TypeError('connection lost'))
    else fetchMock.mockResolvedValueOnce(reply({ reviewed_count: 1, total_count: 2 }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    await screen.findByRole('alert')
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
  })

  it('blocks resubmission after a stale revision without discarding the draft', async () => {
    await mount()
    chooseLabels()
    fetchMock.mockResolvedValueOnce(reply({}, 409))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    await screen.findByRole('alert')
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
  })

  it('preserves unsaved drafts during navigation without treating them as reviewed', async () => {
    await mount()
    fireEvent.click(screen.getByLabelText('Штрафной'))
    fireEvent.click(screen.getByRole('button', { name: 'Следующий' }))
    expect(screen.getByLabelText('Штрафной')).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: 'Предыдущий' }))
    expect(screen.getByLabelText('Штрафной')).toBeChecked()
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('restores saved answers and submits their expected revision when editing', async () => {
    await mount(review({ e1: annotation }))
    expect(screen.getByRole('heading', { name: 'Эпизод 02' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Предыдущий' }))
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(1)
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(18)
    fetchMock.mockResolvedValueOnce(reply({ annotation: { ...annotation, revision: 2 }, reviewed_count: 1, total_count: 2 }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).expected_revision).toBe(1)
    await screen.findByRole('heading', { name: 'Эпизод 02' })
  })

  it('opens nullable saved records as unfinished, without inventing labels or losing revision', async () => {
    await mount(review({ e1: { ...annotation, shot_type: null, start_seconds: null, end_seconds: null } }))
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    expect(screen.getByText('Нужно завершить')).toBeInTheDocument()
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
    expect(screen.getByLabelText('Трёхочковый')).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
    chooseLabels()
    fetchMock.mockResolvedValueOnce(reply({ annotation: { ...annotation, revision: 2 }, reviewed_count: 1, total_count: 2 }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).expected_revision).toBe(1)
    await screen.findByRole('heading', { name: 'Эпизод 02' })
  })

  it.each(['missing', 'null'])('restores v1 history with %s new fields without guessing or counting it complete', async representation => {
    const legacy: Record<string, unknown> = { ...annotation, schema_version: 1, notes: 'Фол, затем чужое добивание' }
    for (const field of ['scoring_decision', 'play_context']) {
      if (representation === 'missing') delete legacy[field]
      else legacy[field] = null
    }
    await mount(review({ e1: legacy }))
    expect(screen.getByRole('heading', { name: 'Эпизод 01' })).toBeInTheDocument()
    expect(screen.getByText('0 из 2 сохранено')).toBeInTheDocument()
    expect(screen.getByLabelText('Трёхочковый')).toBeChecked()
    expect(screen.getByLabelText('Попадание')).toBeChecked()
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(1)
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(18)
    expect(screen.getByLabelText('Комментарий')).toHaveValue('Фол, затем чужое добивание')
    expect(screen.getByLabelText('Засчитаны')).not.toBeChecked()
    expect(screen.getByLabelText('В игре')).not.toBeChecked()
    expect(document.getElementById('draft-hint')).toHaveTextContent('два новых вопроса')
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
    fireEvent.click(screen.getByLabelText('Не засчитаны'))
    fireEvent.click(screen.getByLabelText('Новый бросок после свистка'))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    fetchMock.mockResolvedValueOnce(reply({ annotation: { ...annotation, revision: 2, scoring_decision: 'not_counted', play_context: 'after_whistle' }, reviewed_count: 1, total_count: 2 }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toMatchObject({
      schema_version: 3, event_id: 'primary', expected_revision: 1, shot_type: 'three', outcome: 'made',
      scoring_decision: 'not_counted', play_context: 'after_whistle',
      start_seconds: 1, end_seconds: 18, notes: 'Фол, затем чужое добивание',
    })
    await screen.findByRole('heading', { name: 'Эпизод 02' })
  })

  it('accepts an explicitly counted foul shot and keeps physical result independent', async () => {
    await mount()
    chooseLabels()
    fireEvent.click(screen.getByLabelText('Фол на броске'))
    expect(screen.getByLabelText('Засчитаны')).toBeChecked()
    expect(screen.getByLabelText('Попадание')).toBeChecked()
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeEnabled()
    fireEvent.click(screen.getByLabelText('Промах'))
    expect(screen.getByLabelText('Засчитаны')).toBeChecked()
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeEnabled()
    fetchMock.mockResolvedValueOnce(reply({ annotation: { ...annotation, play_context: 'foul_on_shot', outcome: 'miss' }, reviewed_count: 1, total_count: 2 }))
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить и дальше' }))
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toMatchObject({ scoring_decision: 'counted', play_context: 'foul_on_shot', outcome: 'miss' })
    await screen.findByRole('heading', { name: 'Эпизод 02' })
  })

  it.each(['Новый бросок после свистка', 'Новый бросок при другой остановке'])('explains contradictory counted + %s without resetting labels', async context => {
    await mount()
    chooseLabels()
    fireEvent.click(screen.getByLabelText(context))
    expect(screen.getByLabelText('Засчитаны')).toBeChecked()
    expect(screen.getByLabelText('Попадание')).toBeChecked()
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
    expect(document.getElementById('decision-hint')).toHaveTextContent('Проверьте решение об очках')
    fireEvent.submit(screen.getByRole('form', { name: 'Разметка эпизода' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Проверьте решение об очках')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByLabelText('Не засчитаны'))
    expect(screen.getByLabelText('Попадание')).toBeChecked()
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeEnabled()
  })

  it('accepts explicit uncertainty in all six groups and never infers answers for non-shots', async () => {
    await mount()
    fireEvent.click(screen.getByLabelText('Броска нет'))
    expect(screen.getAllByRole('radio').filter((input: HTMLElement) => (input as HTMLInputElement).checked)).toHaveLength(1)
    for (const input of screen.getAllByRole('radio')) {
      if ((input as HTMLInputElement).value === 'unclear') fireEvent.click(input)
    }
    expect(screen.getAllByRole('radio').filter((input: HTMLElement) => (input as HTMLInputElement).checked)).toHaveLength(6)
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeEnabled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(document.body).toHaveTextContent('Остальные броски добавляй отдельно')
  })

  it('rejects invalid boundaries and notes before contacting the server', async () => {
    await mount()
    chooseLabels()
    fireEvent.input(screen.getByLabelText('Начало фрагмента, секунды'), { target: { value: '21' } })
    fireEvent.submit(screen.getByRole('form', { name: 'Разметка эпизода' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Границы должны быть внутри клипа')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    fireEvent.input(screen.getByLabelText('Начало фрагмента, секунды'), { target: { value: '0' } })
    fireEvent.input(screen.getByLabelText('Комментарий'), { target: { value: 'я'.repeat(2001) } })
    fireEvent.submit(screen.getByRole('form', { name: 'Разметка эпизода' }))
    expect(screen.getByRole('alert')).toHaveTextContent('2000')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('uses clip-relative boundary controls and displays the source clock separately', async () => {
    await mount()
    const player = document.querySelector('video')!
    player.currentTime = 3.25
    fireEvent.click(screen.getByRole('button', { name: 'Начало здесь' }))
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(3.25)
    fireEvent.click(screen.getByRole('button', { name: 'Вперёд на 0,25 секунды' }))
    expect(player.currentTime).toBe(3.5)
    fireEvent.click(screen.getByRole('button', { name: 'Конец здесь' }))
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(3.5)
    expect(screen.getByTestId('source-time')).toHaveTextContent('00:02:09')
  })

  it('shows a recoverable loading failure and never inserts source text as HTML', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('network unavailable'))
    document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
    screen = within(document.body)
    window.eval(script)
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Не удалось открыть подборку'))
    const data = review()
    data.examples[0].source_alias = '<img src=x onerror=alert(1)>'
    fetchMock.mockResolvedValueOnce(reply(data))
    fireEvent.click(screen.getByRole('button', { name: 'Повторить загрузку' }))
    await screen.findByRole('heading', { name: 'Эпизод 01' })
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Скачать разметку JSON' })).toHaveAttribute('href', '/api/export')
  })

  it('refuses remote media before assigning any source to the player', async () => {
    const data = review()
    data.examples[0].video_url = 'https://example.com/match.mp4'
    fetchMock.mockResolvedValueOnce(reply(data))
    document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
    screen = within(document.body)
    window.eval(script)
    await screen.findByRole('alert')
    expect(document.querySelector('video')).not.toHaveAttribute('src')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

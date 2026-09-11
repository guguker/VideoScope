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

async function mount(data = review()) {
  fetchMock.mockResolvedValueOnce(reply(data))
  document.documentElement.innerHTML = html.replace(/<!doctype html>/i, '')
  screen = within(document.body)
  window.eval(script)
  await screen.findByRole('heading', { name: /^Эпизод 0[12]$/ })
}

function chooseLabels() {
  fireEvent.click(screen.getByLabelText('Трёхочковый'))
  fireEvent.click(screen.getByLabelText('Попадание'))
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

  it('starts every human label blank and requires an explicit decision', async () => {
    await mount()
    expect(screen.getAllByRole('radio').every((input: HTMLElement) => !(input as HTMLInputElement).checked)).toBe(true)
    expect(screen.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
    expect(screen.getByLabelText('Начало фрагмента, секунды')).toHaveValue(0)
    expect(screen.getByLabelText('Конец фрагмента, секунды')).toHaveValue(20)
    fireEvent.submit(screen.getByRole('form', { name: 'Разметка эпизода' }))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('alert')).toHaveTextContent('Выберите ответ в каждой из четырёх групп')
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

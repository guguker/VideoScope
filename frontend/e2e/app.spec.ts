import { expect, test } from '@playwright/test'

const video = {
  id: 'video-1',
  original_name: 'basketball-final.mp4',
  display_name: null,
  stored_name: 'video-1.mp4',
  size_bytes: 104857600,
  status: 'ready',
  progress: 1,
  stage: 'ready',
  duration: 120,
  width: 1920,
  height: 1080,
  fps: 30,
  error: null,
  created_at: '2026-07-20T10:00:00Z',
  updated_at: '2026-07-20T10:10:00Z',
  media_url: '/api/videos/video-1/media',
  thumbnail_url: null,
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/videos', (route) => route.fulfill({ json: [video] }))
  await page.route('**/api/videos/video-1', async (route) => {
    if (route.request().method() !== 'PATCH') {
      await route.fallback()
      return
    }
    const body = route.request().postDataJSON() as { name: string }
    await route.fulfill({ json: { ...video, display_name: body.name } })
  })
  await page.route('**/api/providers', (route) => route.fulfill({ json: [
    { id: 'ffmpeg', label: 'FFmpeg', state: 'ready', detail: 'local', optional: false },
    { id: 'whisper', label: 'Whisper MLX', state: 'ready', detail: 'small', optional: false },
  ] }))
  await page.route('**/api/evaluation', (route) => route.fulfill({ json: {
    cases: [{ id: 'case-1', query: 'трёхочковый бросок', video_id: 'video-1', start: 44, end: 51, mode: 'visual', label_source: 'gold', notes: '' }],
    report: null,
  } }))
  await page.route('**/api/evaluation/run', (route) => route.fulfill({ json: {
    generated_at: '2026-07-21T10:00:00Z',
    variants: [{ name: 'auto', case_count: 1, recall_at_1: 1, recall_at_3: 1, recall_at_5: 1, mrr: 1, mean_temporal_iou: 0.8, mean_latency_ms: 120, cases: [] }],
  } }))
  await page.route('**/api/search/glossary', async (route) => {
    if (route.request().method() === 'PUT') {
      await route.fulfill({ json: route.request().postDataJSON() })
      return
    }
    await route.fulfill({ json: { entries: { 'трёхочковый': ['треха'] } } })
  })
  await page.route('**/api/search', (route) => route.fulfill({ json: [
    {
      id: 'result-1',
      video_id: 'video-1',
      video_name: 'basketball-final.mp4',
      start: 44,
      end: 51,
      score: 0.94,
      modalities: ['speech', 'objects', 'lighthouse'],
      evidence: [{
        modality: 'speech',
        score: 0.9,
        text: 'трёхочковый бросок',
        source: 'lexical',
        confidence: 0.92,
        start: 44,
        end: 51,
        raw_score: 0.9,
        matched_terms: ['трёхочковый'],
      }],
      thumbnail_url: null,
      intent: 'mixed',
      explanation: 'Объединяются речь и кадр',
      refined: true,
    },
  ] }))
})

test('searches a moment and adds it to the clip queue', async ({ page }, testInfo) => {
  await page.goto('/')
  await page.getByLabel('Поисковый запрос').fill('трёхочковый бросок')
  await page.getByRole('button', { name: 'Найти' }).click()
  await expect(page.getByTestId('search-result')).toContainText('00:44–00:51')
  await expect(page.getByLabel('Мультимодальная шкала')).toBeVisible()
  await page.getByTitle('Добавить в нарезку').click()

  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: /Нарезка/ }).click()
  }
  await expect(page.getByLabel('Очередь нарезки')).toContainText('basketball-final.mp4')
})

test('opens quality metrics and runs the benchmark', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Качество поиска' }).click()
  await page.getByRole('button', { name: 'Запустить оценку' }).click()

  await expect(page.getByRole('dialog', { name: 'Качество поиска' })).toContainText('Recall@1')
  await expect(page.getByRole('dialog', { name: 'Качество поиска' })).toContainText('100%')
})

test('collapses the library and renames a video inline', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile')
  await page.goto('/')

  await page.getByRole('button', { name: 'Свернуть библиотеку' }).click()
  await expect(page.getByLabel('Библиотека видео')).toHaveClass(/is-collapsed/)
  await page.getByRole('button', { name: 'Развернуть библиотеку' }).click()

  await page.getByLabel('Библиотека видео').getByText('basketball-final.mp4').hover()
  await page.getByRole('button', { name: 'Переименовать basketball-final.mp4' }).click()
  await page.getByLabel('Новое название видео').fill('Финал турнира')
  await page.getByRole('button', { name: 'Сохранить название' }).click()
  await expect(page.getByLabel('Библиотека видео').getByText('Финал турнира')).toBeVisible()
})

test('keeps the player visible while opening a deep search result', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'mobile')
  await page.unroute('**/api/search')
  await page.route('**/api/search', (route) => route.fulfill({ json: Array.from({ length: 30 }, (_, index) => ({
    id: `result-${index}`,
    video_id: 'video-1',
    video_name: 'basketball-final.mp4',
    start: index * 3,
    end: index * 3 + 2,
    score: 0.9 - index * 0.005,
    modalities: ['speech'],
    evidence: [{
      modality: 'speech',
      score: 0.9,
      text: `момент ${index + 1}`,
      source: 'lexical',
      confidence: 0.9,
      start: index * 3,
      end: index * 3 + 2,
      raw_score: 0.9,
      matched_terms: ['момент'],
    }],
    thumbnail_url: null,
    intent: 'speech',
    explanation: 'Поиск по речи',
    refined: false,
  })) }))
  await page.goto('/')
  await page.getByLabel('Поисковый запрос').fill('момент')
  await page.getByRole('button', { name: 'Найти' }).click()

  const lastResult = page.getByTestId('search-result').last()
  await lastResult.scrollIntoViewIfNeeded()
  await lastResult.click()
  await expect(page.locator('video')).toBeVisible()
  const player = await page.locator('video').boundingBox()
  expect(player?.y).toBeGreaterThanOrEqual(0)
  expect((player?.y || 0) + (player?.height || 0)).toBeLessThanOrEqual(900)
})

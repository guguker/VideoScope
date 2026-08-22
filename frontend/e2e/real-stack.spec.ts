import { expect, test } from '@playwright/test'

test('uploads through the real API and completes one durable index job', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'The real stack smoke test runs once on desktop Chromium')

  await page.goto('/')
  await page.getByRole('button', { name: 'Добавить видео' }).first().click()
  await page.locator('input[type="file"]').setInputFiles({
    name: 'deterministic.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('videoscope deterministic e2e fixture'),
  })

  const uploadResponsePromise = page.waitForResponse((response) => (
    response.request().method() === 'POST'
      && new URL(response.url()).pathname === '/api/videos'
  ))
  await page.getByRole('button', { name: 'Загрузить' }).click()
  const uploadResponse = await uploadResponsePromise

  expect(uploadResponse.status()).toBe(202)
  const uploaded = await uploadResponse.json() as {
    id: string
    latest_job: { job_id: string; state: string }
  }
  const location = uploadResponse.headers().location
  expect(location).toBe(`/api/jobs/${uploaded.latest_job.job_id}`)
  expect(uploaded.latest_job.state).toBe('queued')

  const row = page.locator('.library-row').filter({ hasText: 'deterministic.mp4' })
  await expect(row).toContainText('В очереди')
  await expect(row).toContainText('Разбор сцен')
  await expect(row).toContainText('Готово')

  const persisted = await page.request.get(location)
  expect(persisted.status()).toBe(200)
  await expect(persisted.json()).resolves.toMatchObject({
    job_id: uploaded.latest_job.job_id,
    video_id: uploaded.id,
    state: 'complete',
    progress: 1,
  })
})

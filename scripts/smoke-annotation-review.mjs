#!/usr/bin/env node
/** Real browser / real review server smoke. Only disposable synthetic media. */
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { spawn, execFile } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, mkdir, readFile, writeFile, stat } from 'node:fs/promises'
import { createRequire } from 'node:module'
import net from 'node:net'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const run = promisify(execFile)
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'frontend/package.json'))
const { chromium, expect } = require('@playwright/test')
const temporary = await mkdtemp(path.join(tmpdir(), 'videoscope-review-smoke-'))
const batchDir = path.join(temporary, 'batch')
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex')
let browser
let server
let serverLog = ''
let serverStopped = false
const pageErrors = []
const externalRequests = []
const receipt = { status: 'failed', purpose: 'synthetic_annotation_review_smoke', temporary }

async function freePort() {
  const socket = net.createServer()
  socket.listen(0, '127.0.0.1')
  await once(socket, 'listening')
  const { port } = socket.address()
  await new Promise((resolve, reject) => socket.close((error) => error ? reject(error) : resolve()))
  return port
}

async function ffmpeg(args) {
  await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin', ...args], {
    timeout: 30000, maxBuffer: 1024 * 1024,
  })
}

async function startServer(port, origin) {
  server = spawn(path.join(root, '.venv/bin/python'), [
    '-m', 'videoscope.annotation_review.server', '--batch', batchDir, '--port', String(port),
  ], { cwd: root, env: { ...process.env, PYTHONPATH: path.join(root, 'backend/src') }, stdio: ['ignore', 'pipe', 'pipe'] })
  server.stdout.on('data', (data) => { serverLog = (serverLog + data).slice(-16000) })
  server.stderr.on('data', (data) => { serverLog = (serverLog + data).slice(-16000) })
  const deadline = Date.now() + 20000
  while (Date.now() < deadline) {
    if (server.exitCode !== null || server.signalCode !== null) throw new Error(`Review server exited: ${serverLog}`)
    try {
      const response = await fetch(`${origin}/api/health`, { signal: AbortSignal.timeout(1000) })
      const health = await response.json()
      if (response.ok && health.batch_id === 'synthetic-review-smoke') return
    } catch { /* Wait for this owned server only. */ }
    await sleep(100)
  }
  throw new Error(`Review server readiness timeout: ${serverLog}`)
}

async function stopServer() {
  if (!server || server.exitCode !== null || server.signalCode !== null) return
  const exited = once(server, 'exit')
  server.kill('SIGTERM')
  const stopped = await Promise.race([exited.then(() => true), sleep(5000).then(() => false)])
  if (!stopped) {
    server.kill('SIGKILL')
    await exited
  }
}

try {
  await mkdir(path.join(batchDir, 'clips'), { recursive: true })
  await mkdir(path.join(batchDir, 'posters'))
  const sourcePath = path.join(temporary, 'synthetic-source.mp4')
  await ffmpeg([
    '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=24',
    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
    '-t', '6', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
    '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', sourcePath,
  ])
  const sourceBytes = await readFile(sourcePath)
  const examples = []
  for (const [index, start] of [0, 3].entries()) {
    const id = `synthetic-${index + 1}`
    const clip = path.join(batchDir, 'clips', `${id}.mp4`)
    const poster = path.join(batchDir, 'posters', `${id}.jpg`)
    await ffmpeg(['-ss', String(start), '-i', sourcePath, '-t', '3', '-c:v', 'libx264',
      '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-threads', '2', '-c:a', 'aac', '-movflags', '+faststart', clip])
    await ffmpeg(['-i', clip, '-frames:v', '1', '-q:v', '3', '-threads', '1', poster])
    const bytes = await readFile(clip)
    examples.push({
      example_id: id, source_id: 'synthetic-source', source_start_seconds: start,
      source_end_seconds: start + 3, clip_duration_seconds: 3,
      prepared_input_sha256: sha256(bytes), prepared_input_byte_size: bytes.length,
      clip_path: `clips/${id}.mp4`, poster_path: `posters/${id}.jpg`,
      selection_method: 'synthetic_ui_smoke', selection_notes: 'Generated test pattern, no user video or model labels.',
    })
  }
  const { stdout: codeSha } = await run('git', ['rev-parse', 'HEAD'], { cwd: root })
  const manifest = {
    schema_version: 1, batch_id: 'synthetic-review-smoke', title: 'Проверка эпизодов · синтетический тест',
    created_at: new Date().toISOString(), code_sha: codeSha.trim(), purpose: 'annotation_pilot',
    training_allowed: false, promotion_allowed: false,
    sources: [{
      source_id: 'synthetic-source', sha256: sha256(sourceBytes), byte_size: sourceBytes.length,
      duration_seconds: 6, source_group: 'synthetic-game', usage: 'development_review',
      review_allowed: true, training_rights: 'unknown',
    }], examples,
  }
  const batchBytes = JSON.stringify(manifest, null, 2) + '\n'
  await writeFile(path.join(batchDir, 'batch.json'), batchBytes)
  const port = await freePort()
  const origin = `http://127.0.0.1:${port}`
  await startServer(port, origin)
  browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } })
  const page = await context.newPage()
  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('request', (request) => {
    if (new URL(request.url()).origin !== origin) externalRequests.push(request.url())
  })
  await page.goto(origin)
  await expect(page.getByRole('heading', { name: 'Эпизод 01' })).toBeVisible()
  await expect(page.locator('input[type=radio]:checked')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Сохранить и дальше' })).toBeDisabled()
  await expect(page.getByText('0 из 2 сохранено')).toBeVisible()
  const video = page.locator('video')
  await expect.poll(() => video.evaluate((element) => element.duration)).toBeGreaterThan(2.9)
  await video.evaluate((element) => element.play())
  await expect.poll(() => video.evaluate((element) => element.currentTime)).toBeGreaterThan(0.2)
  await video.evaluate((element) => element.pause())
  const media = await context.request.get(`${origin}/media/synthetic-1`, { headers: { Range: 'bytes=0-31' } })
  assert.equal(media.status(), 206)
  assert.equal((await media.body()).length, 32)
  await page.screenshot({ path: path.join(temporary, 'desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true)
  await page.screenshot({ path: path.join(temporary, 'mobile.png'), fullPage: true })
  await page.setViewportSize({ width: 1440, height: 1100 })

  await page.getByLabel('Броска нет', { exact: true }).check()
  await page.getByLabel('Не применимо', { exact: true }).check()
  await page.getByLabel('Основной эпизод', { exact: true }).check()
  await page.getByLabel('Контекста хватает', { exact: true }).check()
  await page.getByLabel('Начало фрагмента, секунды').fill('0.25')
  await page.getByLabel('Конец фрагмента, секунды').fill('2.75')
  await page.getByLabel('Комментарий', { exact: true }).fill('Синтетический тест интерфейса; реальных событий нет.')
  await expect(page.getByRole('button', { name: 'Сохранить и дальше' })).toBeEnabled()
  // Delay the genuine server response: UI must not claim success while its
  // acknowledgement is still in transit, even though the server has persisted it.
  let acknowledge
  const acknowledgementGate = new Promise((resolve) => { acknowledge = resolve })
  let serverPersisted = false
  await page.route('**/api/annotations', async (route) => {
    const response = await route.fetch()
    assert.equal(response.status(), 200)
    serverPersisted = true
    await acknowledgementGate
    await route.fulfill({ response })
  }, { times: 1 })
  await page.getByRole('button', { name: 'Сохранить и дальше' }).click()
  await expect.poll(() => serverPersisted).toBe(true)
  await expect(page.getByRole('heading', { name: 'Эпизод 01' })).toBeVisible()
  await expect(page.getByText('0 из 2 сохранено')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Следующий', exact: true })).toBeDisabled()
  acknowledge()
  await expect(page.getByRole('heading', { name: 'Эпизод 02' })).toBeVisible()
  await expect(page.getByText('1 из 2 сохранено')).toBeVisible()
  await expect(page.locator('input[type=radio]:checked')).toHaveCount(0)

  await page.getByRole('button', { name: 'Предыдущий', exact: true }).click()
  await expect(page.getByLabel('Броска нет', { exact: true })).toBeChecked()
  await page.getByLabel('Начало фрагмента, секунды').fill('0.5')
  await page.getByLabel('Комментарий', { exact: true }).fill('Синтетический тест: исправленная версия.')
  await page.getByRole('button', { name: 'Сохранить и дальше' }).click()
  await expect(page.getByRole('heading', { name: 'Эпизод 02' })).toBeVisible()
  await stopServer()
  await startServer(port, origin)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Эпизод 02' })).toBeVisible()
  await page.getByRole('button', { name: 'Предыдущий', exact: true }).click()
  await expect(page.getByLabel('Начало фрагмента, секунды')).toHaveValue('0.5')
  await expect(page.getByLabel('Комментарий', { exact: true })).toHaveValue('Синтетический тест: исправленная версия.')
  await expect(page.getByText('Сохранено локально · версия 2. Ответы можно исправить.')).toBeVisible()

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Скачать разметку JSON' }).click()
  const download = await downloadPromise
  const exportPath = path.join(temporary, 'exported-reviews.json')
  await download.saveAs(exportPath)
  const exported = JSON.parse(await readFile(exportPath, 'utf8'))
  assert.equal(exported.records.length, 2)
  assert.deepEqual(exported.records.map((record) => record.revision), [1, 2])
  for (const record of exported.records) {
    assert.equal(record.gold, false)
    assert.equal(record.training_allowed, false)
    assert.equal(record.promotion_allowed, false)
    assert.equal(record.destination, 'annotation_inbox')
    assert.equal(record.source_sha256, sha256(sourceBytes))
    assert.equal(record.prepared_input_sha256, examples[0].prepared_input_sha256)
  }
  assert.equal(await readFile(path.join(batchDir, 'batch.json'), 'utf8'), batchBytes)
  assert.equal(sha256(await readFile(sourcePath)), sha256(sourceBytes))
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(externalRequests, [])
  const webAssets = {}
  for (const filename of ['index.html', 'app.js', 'style.css']) {
    webAssets[filename] = sha256(await readFile(path.join(root, 'backend/src/videoscope/annotation_review/web', filename)))
  }
  const { stdout: gitStatus } = await run('git', ['status', '--porcelain'], { cwd: root })
  Object.assign(receipt, {
    status: 'passed', code_sha: codeSha.trim(), examples: 2, annotation_revisions: 2,
    working_tree_dirty: Boolean(gitStatus.trim()),
    runner_sha256: sha256(await readFile(fileURLToPath(import.meta.url))), web_assets_sha256: webAssets,
    playable_video: true, http_range_status: 206, blank_labels: true,
    save_acknowledged_before_advance: true, restored_after_server_restart: true,
    export_provenance_verified: true, source_and_batch_unchanged: true,
    external_requests: externalRequests, javascript_errors: pageErrors,
    desktop_screenshot: path.join(temporary, 'desktop.png'),
    mobile_screenshot: path.join(temporary, 'mobile.png'),
    exported_reviews_bytes: (await stat(exportPath)).size,
  })
} catch (error) {
  receipt.error = error.stack || String(error)
  receipt.server_log = serverLog
  process.exitCode = 1
} finally {
  if (browser) await browser.close()
  await stopServer()
  serverStopped = !server || server.exitCode !== null || server.signalCode !== null
  receipt.server_stopped = serverStopped
  await writeFile(path.join(temporary, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n')
  process.stdout.write(JSON.stringify(receipt, null, 2) + '\n')
}

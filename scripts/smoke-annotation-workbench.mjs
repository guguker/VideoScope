#!/usr/bin/env node
/** Real local workbench smoke, using generated video and disposable data only. */
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { spawn, execFile } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, mkdir, readFile, writeFile, copyFile, readdir, realpath } from 'node:fs/promises'
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
const temporary = await realpath(await mkdtemp(path.join(tmpdir(), 'videoscope-workbench-smoke-')))
const python = path.join(root, '.venv/bin/python')
const env = { ...process.env, PYTHONPATH: path.join(root, 'backend/src') }
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex')
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const id = (kind, character) => `${kind}-${character.repeat(32)}`
const receipt = { status: 'failed', purpose: 'synthetic_full_match_annotation_workbench', temporary }
const externalRequests = []
const pageErrors = []
const servers = new Set()
let browser
let activePage

async function jsonFile(file, data) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, JSON.stringify(data, null, 2) + '\n')
}

async function command(executable, args, options = {}) {
  return run(executable, args, { cwd: root, env, timeout: 60000, maxBuffer: 1024 * 1024, ...options })
}

async function ffmpeg(args) {
  return command('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin', '-n', ...args])
}

async function freePort() {
  const socket = net.createServer()
  socket.listen(0, '127.0.0.1')
  await once(socket, 'listening')
  const { port } = socket.address()
  await new Promise((resolve, reject) => socket.close((error) => error ? reject(error) : resolve()))
  return port
}

async function start(workspace, selectedPort = null) {
  const port = selectedPort ?? await freePort()
  const origin = `http://127.0.0.1:${port}`
  const child = spawn(python, ['-m', 'videoscope.annotation_workbench.server',
    '--workspace', workspace, '--port', String(port)], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] })
  servers.add(child)
  let log = ''
  for (const stream of [child.stdout, child.stderr]) stream.on('data', (chunk) => { log = (log + chunk).slice(-16000) })
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (child.exitCode !== null) throw new Error(`Workbench server exited: ${log}`)
    try {
      const response = await fetch(`${origin}/api/health`, { signal: AbortSignal.timeout(500) })
      if (response.ok && (await response.json()).status === 'ready') return { child, port, origin }
    } catch { /* Wait only for the child started by this smoke. */ }
    await sleep(100)
  }
  throw new Error(`Workbench readiness timeout: ${log}`)
}

async function stop(child) {
  if (child.exitCode !== null || child.signalCode !== null) return
  const exited = once(child, 'exit')
  child.kill('SIGTERM')
  if (!await Promise.race([exited.then(() => true), sleep(5000).then(() => false)])) {
    child.kill('SIGKILL')
    await exited
  }
  servers.delete(child)
}

async function api(origin, route, body) {
  const response = await fetch(origin + route, body === undefined ? {} : {
    method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const result = await response.json()
  assert.ok(response.ok, `${route}: ${response.status} ${JSON.stringify(result)}`)
  return result
}

async function fixture(project, mediaPaths, legacy = true) {
  await mkdir(project, { recursive: true })
  const sources = []
  for (let i = 0; i < 8; i += 1) {
    const filename = `source-${i + 1}.mp4`
    const available = i < 2
    let bytes
    if (available) {
      await copyFile(mediaPaths[i], path.join(project, filename))
      bytes = await readFile(mediaPaths[i])
    }
    sources.push({ source_id: `source-${i + 1}`, source_relpath: filename,
      source_sha256: available ? sha(bytes) : sha(`unopened-synthetic-reserve-${i}`),
      bytes: available ? bytes.length : 100, duration_seconds: i === 0 ? 3612 : 8,
      video_streams: [{ avg_frame_rate: i === 0 ? '1/1' : '2/1', width: 320, height: 180 }],
      grouping: { source_group_id: `synthetic-game-${i + 1}` } })
  }
  const audit = path.join(project, 'audit')
  await jsonFile(path.join(audit, 'inventory.json'), { sources })
  await jsonFile(path.join(audit, 'source-roles.json'), { sources: sources.map((source, i) => ({
    source_id: source.source_id, role: i < 2 ? 'development_review' : i === 7 ? 'regression_suspected' : 'reserve_uninspected',
    content_decode_allowed_current_slice: i < 2, training_allowed: false, promotion_eligible: false,
  })) })
  await jsonFile(path.join(audit, 'source-rights-ledger.json'), { sources: sources.map((source) => ({
    source_id: source.source_id, source_sha256: source.source_sha256, rights_status: 'unknown',
    training_allowed: false, external_upload_allowed: false, publication_allowed: false,
  })) })
  const batch = path.join(project, 'legacy')
  const manifest = {
    schema_version: 1, batch_id: 'synthetic-workbench-legacy', title: 'Синтетические старые ответы',
    created_at: '2026-09-11T00:00:00+00:00', code_sha: 'a'.repeat(40), purpose: 'annotation_pilot',
    training_allowed: false, promotion_allowed: false,
    sources: [{ source_id: 'old-alias', sha256: sources[0].source_sha256, byte_size: sources[0].bytes,
      duration_seconds: 3612, source_group: 'synthetic-game-1', usage: 'development_review',
      review_allowed: true, training_rights: 'unknown' }],
    examples: [{ example_id: 'old-clip', source_id: 'old-alias', source_start_seconds: 3600,
      source_end_seconds: 3608, clip_duration_seconds: 8, prepared_input_sha256: sha('synthetic-clip'),
      prepared_input_byte_size: 14, clip_path: 'clips/old-clip.mp4', poster_path: 'posters/old-clip.jpg',
      selection_method: 'synthetic', selection_notes: 'Only generated media and labels.' }],
  }
  await jsonFile(path.join(batch, 'batch.json'), manifest)
  const { stdout } = await command(python, ['-c',
    'import json,sys; from videoscope.annotation_review.schema import batch_revision; print(batch_revision(json.load(open(sys.argv[1]))))',
    path.join(batch, 'batch.json')])
  if (legacy) {
    const record = {
      schema_version: 1, batch_id: manifest.batch_id, batch_revision: stdout.trim(), example_id: 'old-clip',
      revision: 1, created_at: '2026-09-11T00:01:00+00:00', reviewer: 'local_owner', label_status: 'human_reviewed',
      destination: 'annotation_inbox', gold: false, training_allowed: false, promotion_allowed: false,
      source_id: 'old-alias', source_sha256: sources[0].source_sha256, source_start_seconds: 3600,
      source_end_seconds: 3608, prepared_input_sha256: manifest.examples[0].prepared_input_sha256,
      shot_type: 'two', outcome: 'miss', presentation: 'live', boundary_status: 'complete',
      start_seconds: 0.5, end_seconds: 2.5, notes: 'Сохранённый синтетический ответ v1.',
    }
    const parent = path.join(batch, 'annotations/old-clip')
    await jsonFile(path.join(parent, '000001.json'), record)
    await jsonFile(path.join(parent, '000002.json'), { ...record, schema_version: 2, revision: 2,
      created_at: '2026-09-11T00:02:00+00:00', scoring_decision: 'not_applicable', play_context: 'foul_on_shot' })
    await jsonFile(path.join(parent, `events/${id('event', 'b')}/000001.json`), {
      ...record, schema_version: 3, event_id: id('event', 'b'), revision: 1,
      start_seconds: 3, end_seconds: 5, outcome: 'made', scoring_decision: 'not_counted', play_context: 'after_whistle',
      created_at: '2026-09-11T00:03:00+00:00', notes: 'Второй синтетический бросок после свистка.',
    })
  }
  const workspace = path.join(project, 'workspace')
  await command(python, ['-m', 'videoscope.annotation_workbench.prepare', '--audit', audit,
    '--project-root', project, '--legacy-batch', batch, '--output', workspace])
  return { workspace, batch, sources }
}

async function treeHashes(directory) {
  const result = {}
  async function walk(folder) {
    for (const item of await readdir(folder, { withFileTypes: true })) {
      const file = path.join(folder, item.name)
      if (item.isDirectory()) await walk(file)
      else result[path.relative(directory, file)] = sha(await readFile(file))
    }
  }
  await walk(directory)
  return result
}

try {
  const { stdout: codeSha } = await command('git', ['rev-parse', 'HEAD'])
  receipt.code_sha = codeSha.trim()
  const { stdout: gitStatus } = await command('git', ['status', '--porcelain', '--untracked-files=normal'])
  receipt.working_tree_dirty = gitStatus.split('\n').filter((line) => line && line !== '?? .venv').length > 0
  if (process.argv.includes('--require-clean')) assert.equal(receipt.working_tree_dirty, false, 'Commit implementation before attested acceptance')
  receipt.executed_files_sha256 = {}
  for (const folder of ['backend/src/videoscope/annotation_workbench', 'backend/src/videoscope/annotation_workbench/web']) {
    for (const item of await readdir(path.join(root, folder), { withFileTypes: true })) {
      if (item.isFile() && /\.(py|js|css|html)$/.test(item.name)) {
        const file = `${folder}/${item.name}`
        receipt.executed_files_sha256[file] = sha(await readFile(path.join(root, file)))
      }
    }
  }
  receipt.runner_sha256 = sha(await readFile(fileURLToPath(import.meta.url)))
  receipt.started_at = new Date().toISOString()
  await command(python, ['-m', 'videoscope.annotation_workbench.prepare', '--help'])
  const media = [path.join(temporary, 'long.mp4'), path.join(temporary, 'short.mp4')]
  await ffmpeg(['-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=1', '-f', 'lavfi', '-i',
    'sine=frequency=440:sample_rate=8000', '-t', '3612', '-c:v', 'libx264', '-preset', 'ultrafast',
    '-threads', '2', '-g', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '16k', '-movflags', '+faststart', media[0]])
  await ffmpeg(['-f', 'lavfi', '-i', 'color=c=blue:size=320x180:rate=2', '-t', '8',
    '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', media[1]])
  const first = await fixture(path.join(temporary, 'project-a'), media)
  const legacyBefore = await treeHashes(first.batch)
  let current = await start(first.workspace)
  const workspace = await api(current.origin, '/api/workspace')
  assert.equal(workspace.sources.length, 8)
  assert.equal(workspace.sources.filter((source) => source.review_allowed).length, 2)
  assert.ok(workspace.sources.filter((source) => !source.review_allowed).every((source) => source.media_url === null))
  const sourceId = workspace.sources[0].source_id
  const initial = await api(current.origin, `/api/sources/${sourceId}`)
  assert.equal(initial.records.filter((record) => record.kind === 'event').length, 2)
  assert.ok(initial.records.some((record) => record.data.start_seconds === 3600.5))
  const range = await fetch(current.origin + workspace.sources[0].media_url, { headers: { Range: 'bytes=0-15' } })
  assert.equal(range.status, 206)
  assert.equal((await range.arrayBuffer()).byteLength, 16)
  const blocked = await fetch(current.origin + `/media/${workspace.sources[2].source_id}`)
  assert.ok([403, 404, 409].includes(blocked.status))

  browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true })
  const page = await context.newPage()
  activePage = page
  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('request', (request) => {
    if (new URL(request.url()).origin !== current.origin) externalRequests.push(request.url())
  })
  await page.goto(current.origin)
  const video = page.locator('video')
  await expect(video).toBeVisible()
  await expect.poll(() => video.evaluate((element) => element.duration)).toBeGreaterThan(3600)
  await video.evaluate((element) => { element.currentTime = 3605; return element.play() })
  await expect.poll(() => video.evaluate((element) => element.currentTime)).toBeGreaterThan(3605.1)
  await video.evaluate((element) => element.pause())

  const records = async () => (await api(current.origin, `/api/sources/${sourceId}`)).records
  const field = (name) => page.getByLabel(name, { exact: true })
  const save = async (status = 'human_reviewed') => {
    await page.locator(status === 'draft' ? '#save-draft' : '#confirm-record').click()
    await expect(page.locator('#save-state')).toHaveText(status === 'draft' ? 'Черновик сохранён' : 'Подтверждено')
    await expect(page.locator('#editor-error')).toBeHidden()
  }
  const seek = async (seconds) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor(seconds % 3600 / 60)
    await field('Точное время, чч:мм:сс').fill(`${h}:${m}:${seconds % 60}`)
    await page.getByRole('button', { name: 'Перейти', exact: true }).click()
    await expect.poll(() => video.evaluate((element) => element.currentTime)).toBeCloseTo(seconds, 2)
    await expect.poll(() => video.evaluate((element) => element.seeking)).toBe(false)
  }
  const uniqueRecord = async (predicate) => {
    const matches = (await records()).filter(predicate)
    assert.equal(matches.length, 1)
    return matches[0]
  }
  const createPlayer = async (number, teamId) => {
    await page.getByRole('button', { name: 'Добавить игрока', exact: true }).click()
    await field('Команда').selectOption(teamId)
    await field('Номер формы').fill(number)
    await save()
    return uniqueRecord((record) => record.kind === 'player' && record.data.jersey_number === number)
  }
  await seek(3601)
  await page.getByRole('button', { name: 'Добавить команду', exact: true }).click()
  await field('Название команды').fill('Синтетическая команда')
  await field('Цвет формы').fill('Синий')
  await save()
  const team = await uniqueRecord((record) => record.kind === 'team')
  const shooter = await createPlayer('00', team.record_id)
  const passer = await createPlayer('7', team.record_id)
  assert.equal(shooter.data.jersey_number, '00')

  await page.getByRole('button', { name: 'Добавить владение', exact: true }).click()
  await field('Начало владения, секунды').fill('3599')
  await field('Конец владения, секунды').fill('3610')
  await field('Команда с мячом').selectOption(team.record_id)
  await field('Направление атаки на изображении').selectOption('right')
  await save()
  const possession = await uniqueRecord((record) => record.kind === 'possession')

  await page.getByRole('button', { name: 'Добавить событие', exact: true }).click()
  await field('Начало события, секунды').fill('3601')
  await field('Конец события, секунды').fill('3605')
  await field('Владение').selectOption(possession.record_id)
  await field('Бросающий').selectOption(shooter.record_id)
  await field('Последний пасующий перед броском').selectOption(passer.record_id)
  await field('Время последней передачи, секунды').fill('3601')
  for (const [label, value] of [['Тип броска', 'three'], ['Исход броска', 'made'],
    ['Решение по очкам', 'counted'], ['Контекст игры', 'in_play'],
    ['Показ момента', 'live'], ['Полнота границ', 'complete']]) await field(label).selectOption(value)
  await field('Комментарий к событию').fill('Синтетический подтверждённый бросок')
  await save()
  const shot = await uniqueRecord((record) => record.data.notes === 'Синтетический подтверждённый бросок')
  assert.equal(shot.data.actor_id, shooter.record_id)

  for (const type of ['pass', 'substitution']) {
    await page.getByRole('button', { name: 'Добавить событие', exact: true }).click()
    await field('Тип события').selectOption(type)
    await field('Начало события, секунды').fill('3600')
    await field('Конец события, секунды').fill('3601')
    await field('Показ момента').selectOption('live')
    await field('Полнота границ').selectOption('complete')
    if (type === 'pass') {
      await field('Исполнитель').selectOption(passer.record_id)
      await field('Получатель передачи').selectOption(shooter.record_id)
    } else {
      await field('Вышел на площадку').selectOption(shooter.record_id)
      await field('Покинул площадку').selectOption(passer.record_id)
    }
    await save()
    assert.equal((await uniqueRecord((record) => record.data.event_type === type)).status, 'human_reviewed')
  }

  await seek(3603)
  await page.getByRole('button', { name: 'Добавить кадр с точками', exact: true }).click()
  await field('Событие').selectOption(shot.record_id)
  await field('Владение').selectOption(possession.record_id)
  await field('Игрок для точки').selectOption(shooter.record_id)
  const surface = page.getByRole('button', { name: 'Кадр видео для отметки точек', exact: true })
  const rect = await surface.boundingBox()
  assert.ok(rect)
  await surface.click({ position: { x: rect.width * 0.4, y: rect.height * 0.6 } })
  const shooterMarker = page.getByRole('button', { name: /^Точка: игрок Синтетическая команда · 00$/ })
  await expect(shooterMarker).toBeVisible()
  await page.getByRole('button', { name: 'Добавить следующую точку', exact: true }).click()
  await field('Мяч').check()
  await surface.click({ position: { x: rect.width * 0.6, y: rect.height * 0.4 } })
  await save()
  const frame = await uniqueRecord((record) => record.kind === 'frame')
  assert.equal(frame.data.timestamp_seconds, 3603)
  assert.equal(frame.data.points.length, 2)
  assert.equal(frame.data.points.filter((point) => point.entity === 'player').length, 1)
  assert.equal(frame.data.points.filter((point) => point.entity === 'ball').length, 1)
  assert.ok(frame.data.points.every((point) => point.x > 0 && point.x < 1 && point.y > 0 && point.y < 1))
  await page.screenshot({ path: path.join(temporary, 'desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 430, height: 932 })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false)
  await expect.poll(async () => {
    const picture = await video.boundingBox()
    const marker = await shooterMarker.boundingBox()
    return Boolean(picture && marker && marker.x >= picture.x && marker.x + marker.width <= picture.x + picture.width)
  }).toBe(true)
  await page.screenshot({ path: path.join(temporary, 'narrow.png'), fullPage: true })
  await page.setViewportSize({ width: 1440, height: 1100 })

  await page.getByRole('button', { name: 'Добавить событие', exact: true }).click()
  await field('Комментарий к событию').fill('Незаконченный второй эпизод')
  await save('draft')
  const draft = await uniqueRecord((record) => record.data.notes === 'Незаконченный второй эпизод')
  assert.equal(draft.data.end_seconds, null)
  await seek(3606)
  await expect.poll(async () => (await api(current.origin, `/api/sources/${sourceId}`)).progress.position_seconds).toBe(3606)
  await page.reload()
  await expect(field('Комментарий к событию')).toHaveValue('Незаконченный второй эпизод')
  await expect.poll(() => video.evaluate((element) => element.currentTime)).toBeCloseTo(3606, 2)
  await stop(current.child)
  current = await start(first.workspace, current.port)
  await page.reload()
  await expect(field('Комментарий к событию')).toHaveValue('Незаконченный второй эпизод')
  assert.equal((await uniqueRecord((record) => record.record_id === draft.record_id)).status, 'draft')

  await page.locator(`[data-record-id="${shot.record_id}"]`).click()
  await field('Комментарий к событию').fill('Исправленный синтетический бросок')
  await save()
  const corrected = await uniqueRecord((record) => record.record_id === shot.record_id)
  assert.ok(corrected.revision > shot.revision)
  assert.equal((await uniqueRecord((record) => record.record_id === draft.record_id)).data.notes, draft.data.notes)
  await page.getByRole('button', { name: 'Показать историю', exact: true }).click()
  await expect.poll(() => page.locator('#history-list li').count()).toBeGreaterThan(1)

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Скачать резервную копию', exact: true }).click()
  const download = await downloadPromise
  const backupPath = path.join(temporary, 'backup.ndjson')
  await download.saveAs(backupPath)
  const backupBytes = await readFile(backupPath)
  assert.ok(backupBytes.length > 100)
  assert.equal(backupBytes.includes(Buffer.from(first.workspace)), false)
  const beforeRestore = await records()
  const progressBeforeRestore = (await api(current.origin, `/api/sources/${sourceId}`)).progress

  await stop(current.child)
  const second = await fixture(path.join(temporary, 'different-root/project-b'), media, false)
  current = await start(second.workspace)
  const secondWorkspace = await api(current.origin, '/api/workspace')
  assert.equal(secondWorkspace.workspace_revision, workspace.workspace_revision)
  await page.goto(current.origin)
  await page.locator('summary').filter({ hasText: 'Восстановление из копии' }).click()
  await field('Файл резервной копии').setInputFiles(backupPath)
  await page.getByRole('button', { name: 'Восстановить из копии', exact: true }).click()
  await expect(page.locator('#restore-summary')).toContainText('Добавлено:')
  assert.deepEqual(await records(), beforeRestore)
  const restoredProgress = (await api(current.origin, `/api/sources/${sourceId}`)).progress
  assert.equal(restoredProgress.selected_record_id, progressBeforeRestore.selected_record_id)
  assert.equal(restoredProgress.position_seconds, progressBeforeRestore.position_seconds)
  await page.getByRole('button', { name: 'Восстановить из копии', exact: true }).click()
  await expect(page.locator('#restore-summary')).toContainText('Добавлено: 0')
  assert.deepEqual(await records(), beforeRestore)
  assert.deepEqual(await treeHashes(first.batch), legacyBefore)
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(externalRequests, [])
  receipt.status = 'ready'
  receipt.checks = ['long_video_range_and_hour_seek', 'eight_sources_two_allowed_six_unopened',
    'legacy_v1_v2_v3_history_and_absolute_times', 'team_players_00_possession_shot_pass_substitution',
    'frame_player_and_ball_points', 'desktop_and_narrow_layout', 'independent_draft_reload_and_server_restart',
    'correction_retains_other_events_and_history', 'download_restore_different_root_and_idempotence',
    'legacy_bytes_unchanged', 'no_external_requests_or_browser_errors']
  receipt.record_count = beforeRestore.length
  receipt.backup_sha256 = sha(backupBytes)
  receipt.workspace_revision = workspace.workspace_revision
} catch (error) {
  receipt.error = error instanceof Error ? error.stack : String(error)
  if (activePage && !activePage.isClosed()) {
    await activePage.screenshot({ path: path.join(temporary, 'failure.png'), fullPage: true }).catch(() => {})
    await writeFile(path.join(temporary, 'failure-dom.txt'), await activePage.locator('body').ariaSnapshot()).catch(() => {})
  }
  console.error(receipt.error)
  process.exitCode = 1
} finally {
  if (browser) await browser.close()
  for (const server of servers) await stop(server)
  receipt.owned_servers_stopped = [...servers].every((server) => server.exitCode !== null || server.signalCode !== null)
  receipt.page_errors = pageErrors
  receipt.external_requests = externalRequests
  await jsonFile(path.join(temporary, 'receipt.json'), receipt)
  console.log(`Workbench smoke receipt: ${path.join(temporary, 'receipt.json')}`)
}

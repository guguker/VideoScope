import { pointFromClient, pointToStyle } from "./geometry.js";
import {
  SCHEMA_VERSION, clone, createEntitySaveQueue, formatTime, loadLocalDrafts,
  newRecordId, parseTime, writeLocalDrafts,
} from "./state.js";

const $ = id => document.getElementById(id);
const KINDS = { team: "Команда", player: "Игрок", possession: "Владение", event: "Событие", frame: "Кадр" };
const EVENT_NAMES = {
  shot: "Бросок", pass: "Передача", rebound: "Подбор", turnover: "Потеря", foul: "Фол",
  substitution: "Замена", screen: "Заслон", defense: "Защита", other: "Другое", unclear: "Неясно",
};
const emptyData = {
  team: () => ({ name: "", color: "", evidence_seconds: null }),
  player: () => ({ team_id: null, jersey_number: null, number_status: "not_reviewed", evidence_seconds: null, notes: "" }),
  possession: time => ({ start_seconds: time, end_seconds: null, team_id: null, attack_direction: "unclear", notes: "" }),
  event: time => ({
    event_type: "shot", start_seconds: time, end_seconds: null, possession_id: null,
    actor_id: null, receiver_id: null, passer_id: null, last_pass_seconds: null,
    incoming_player_id: null, outgoing_player_id: null, shot_type: null, outcome: null,
    scoring_decision: null, play_context: null, presentation: null, boundary_status: null,
    defensive_action: null, notes: "",
  }),
  frame: time => ({ timestamp_seconds: time, event_id: null, possession_id: null, notes: "", points: [] }),
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function numberValue(value) { return value == null ? "" : escapeHtml(value); }
function nullableNumber(value) {
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
function option(value, label, selected) {
  return `<option value="${escapeHtml(value)}"${String(selected ?? "") === String(value) ? " selected" : ""}>${escapeHtml(label)}</option>`;
}
function selectField(label, name, current, options, extra = "") {
  return `<label class="field"><span>${escapeHtml(label)}</span><select data-field="${name}" aria-label="${escapeHtml(label)}" ${extra}>${options.map(([value, text]) => option(value, text, current)).join("")}</select></label>`;
}
function textField(label, name, value, { type = "text", step = "any", max = 2000, placeholder = "" } = {}) {
  return `<label class="field"><span>${escapeHtml(label)}</span><input data-field="${name}" type="${type}" ${type === "number" ? `step="${step}" min="0"` : `maxlength="${max}"`} value="${numberValue(value)}" placeholder="${escapeHtml(placeholder)}"></label>`;
}
function textArea(label, name, value, placeholder = "") {
  return `<label class="field wide"><span>${escapeHtml(label)}</span><textarea data-field="${name}" aria-label="${escapeHtml(label)}" rows="3" maxlength="2000" placeholder="${escapeHtml(placeholder)}">${escapeHtml(value)}</textarea></label>`;
}
function responseDetail(payload, fallback) {
  return typeof payload?.detail === "string" ? payload.detail : fallback;
}

export async function mountWorkbench() {
  const tabStorageKey = "videoscope.annotation-workbench.tab";
  const recoveryTabId = sessionStorage.getItem(tabStorageKey);
  // Duplicate Tab can copy sessionStorage; each document must own a fresh writer.
  const draftTabId = newRecordId("tab");
  sessionStorage.setItem(tabStorageKey, draftTabId);
  const controller = new AbortController();
  const { signal } = controller;
  const queue = createEntitySaveQueue();
  const saveTimers = new Map();
  let progressTimer = null;
  let workspace = null;
  let source = null;
  let records = new Map();
  let drafts = new Map();
  let selectedId = null;
  let recoveredIds = new Set();
  let selectedPointId = null;
  let pointTool = { entity: "player", playerId: null, visibility: "visible", anchor: "floor_contact" };
  let sourceSession = 0;
  let progressInteractionStarted = false;
  let disposed = false;
  let presentedMediaTime = null;
  let videoFrameCallbackId = null;
  let resizeObserver = null;

  const video = $("match-video");
  const editor = $("record-editor");

  function listen(target, type, handler, options = {}) {
    target?.addEventListener(type, handler, { ...options, signal });
  }
  function showMessage(id, message = "") {
    const node = $(id);
    if (!node) return;
    node.textContent = message;
    node.hidden = !message;
  }
  async function request(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const error = new Error(responseDetail(payload, `Ошибка HTTP ${response.status}`));
      error.status = response.status;
      throw error;
    }
    return payload;
  }
  function currentTime() {
    return Number.isFinite(video.currentTime) ? Number(video.currentTime.toFixed(3)) : 0;
  }
  function frameTime() {
    return Number.isFinite(presentedMediaTime) ? Number(presentedMediaTime.toFixed(6)) : currentTime();
  }
  function sourceFps() {
    const raw = source?.fps ?? source?.video_fps ?? source?.frame_rate ?? source?.video_frame_rate;
    if (typeof raw === "string" && raw.includes("/")) {
      const [numerator, denominator] = raw.split("/").map(Number);
      if (numerator > 0 && denominator > 0) return numerator / denominator;
    }
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 60;
  }
  function trackPresentedFrame(_now, metadata) {
    if (Number.isFinite(metadata?.mediaTime)) presentedMediaTime = metadata.mediaTime;
    if (!disposed && typeof video.requestVideoFrameCallback === "function") {
      videoFrameCallbackId = video.requestVideoFrameCallback(trackPresentedFrame);
    }
  }
  function startFrameTracking() {
    if (typeof video.requestVideoFrameCallback === "function" && videoFrameCallbackId == null) {
      videoFrameCallbackId = video.requestVideoFrameCallback(trackPresentedFrame);
    }
  }
  function selectedDraft() { return selectedId ? drafts.get(selectedId) : null; }
  function persistPending(context = { source, drafts }) {
    if (!workspace || !context.source) return;
    writeLocalDrafts(localStorage, workspace.workspace_id, workspace.workspace_revision, context.source.source_id, context.drafts, draftTabId);
  }
  function setDirty(draft) {
    draft.dirty = true;
    draft.saveState = "local";
    draft.error = "";
    draft.generation = (draft.generation || 0) + 1;
    if (draft.conflict) {
      draft.conflict.local_data = clone(draft.data);
      draft.recoverable_data = clone(draft.data);
    }
    persistPending();
    updateSaveState();
    renderRecordList();
    scheduleAutosave(draft);
  }
  function clearSaveTimer(recordId) {
    const timer = saveTimers.get(recordId);
    if (timer) clearTimeout(timer);
    saveTimers.delete(recordId);
  }
  function scheduleAutosave(draft) {
    clearSaveTimer(draft.record_id);
    saveTimers.set(draft.record_id, setTimeout(() => {
      saveTimers.delete(draft.record_id);
      saveRecord(draft, "draft");
    }, 650));
  }
  function updateSaveState() {
    const draft = selectedDraft();
    const node = $("save-state");
    if (!draft) { node.textContent = "Нет изменений"; node.className = "state-tag"; return; }
    const incompleteReviewed = draft.status === "human_reviewed" && draft.needs_review;
    const states = {
      saving: ["Сохраняем…", "saving"], saved: [incompleteReviewed ? "Нужно дополнить" : draft.status === "human_reviewed" ? "Подтверждено" : "Черновик сохранён", incompleteReviewed ? "warning" : "saved"],
      error: ["Сохранено локально, ожидает отправки", "error"], local: ["Сохранено локально", "local"],
    };
    const [text, className] = states[draft.saveState] || states.local;
    node.textContent = text;
    node.className = `state-tag ${className}`;
  }
  function payloadFor(draft, status, contextSource, data, expectedRevision) {
    return {
      schema_version: SCHEMA_VERSION, workspace_revision: workspace.workspace_revision,
      source_id: contextSource.source_id, record_id: draft.record_id,
      expected_revision: expectedRevision, kind: draft.kind,
      status, archived: false, data: clone(data),
    };
  }
  function validLatestRevision(history, draft, contextSource) {
    return (Array.isArray(history) ? history : [])
      .filter(item => item?.record_id === draft.record_id && item.source_id === contextSource.source_id &&
        item.source_sha256 === contextSource.sha256 && item.kind === draft.kind &&
        Number.isInteger(item.revision) && item.revision > draft.expected_revision && !item.archived)
      .sort((left, right) => right.revision - left.revision)[0] || null;
  }
  async function recoverConflict(draft, context) {
    const result = await request(`/api/records/${encodeURIComponent(draft.record_id)}/history`);
    const history = result?.records || result?.revisions || result?.history || (Array.isArray(result) ? result : []);
    const latest = validLatestRevision(history, draft, context.source);
    if (!latest) throw new Error("Не удалось получить более новую версию этой записи");
    // Include edits made while the save or history request was in flight.
    const localData = clone(draft.conflict?.local_data || draft.data);
    draft.conflict = {
      source_id: context.source.source_id, source_sha256: context.source.sha256,
      record_id: draft.record_id, kind: draft.kind, latest_revision: latest.revision,
      server_data: clone(latest.data), server_status: latest.status,
      server_needs_review: Boolean(latest.needs_review), local_data: clone(localData),
    };
    draft.recoverable_data = clone(localData);
    draft.error = "Сервер сохранил более новую версию. Сравните обе версии и явно выберите дальнейшее действие.";
  }
  async function saveRecord(draft, status = "draft", { refreshEditor = false } = {}) {
    if (!draft || disposed) return;
    clearSaveTimer(draft.record_id);
    const generation = draft.generation || 0;
    const data = clone(draft.data);
    const context = { session: sourceSession, source, records, drafts };
    draft.saveState = "saving";
    updateSaveState();
    return queue.enqueue(draft.record_id, async () => {
      const snapshot = payloadFor(draft, status, context.source, data, draft.expected_revision);
      try {
        const saved = await request("/api/records", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(snapshot),
        });
        if (!saved || saved.record_id !== draft.record_id || saved.revision <= snapshot.expected_revision) throw new Error("Сервер вернул некорректное подтверждение сохранения");
        context.records.set(saved.record_id, saved);
        draft.expected_revision = saved.revision;
        draft.status = saved.status;
        draft.needs_review = saved.needs_review ?? saved.status !== "human_reviewed";
        draft.error = "";
        draft.conflict = null;
        draft.recoverable_data = null;
        if ((draft.generation || 0) === generation) {
          draft.dirty = false;
          draft.saveState = "saved";
          recoveredIds.delete(draft.record_id);
        } else {
          draft.dirty = true;
          draft.saveState = queue.pending(draft.record_id) ? "saving" : "local";
        }
        persistPending(context);
        if (context.session === sourceSession) {
          showMessage("editor-error");
          renderRecordList();
          if (refreshEditor && selectedId === draft.record_id && (draft.generation || 0) === generation) renderEditor();
          else updateEditorChrome();
          updateProgressCopy();
          if (selectedId === draft.record_id) saveProgress();
        }
        return saved;
      } catch (error) {
        draft.dirty = true;
        draft.saveState = "error";
        if (error.status === 409) {
          try { await recoverConflict(draft, context); }
          catch (historyError) { draft.error = `Конфликт версий: ${historyError.message}. Ваша правка сохранена локально.`; }
        } else draft.error = `Не удалось отправить черновик: ${error.message}. Правка сохранена локально.`;
        persistPending(context);
        if (context.session === sourceSession) {
          if (selectedId === draft.record_id) renderEditor();
          else updateSaveState();
          renderRecordList();
        }
        throw error;
      }
    }).catch(() => undefined);
  }

  function draftFromRecord(record) {
    return {
      record_id: record.record_id, kind: record.kind, status: record.status,
      archived: false, data: clone(record.data), expected_revision: record.revision,
      generation: 0, dirty: false, saveState: "saved", error: "",
      origin: record.origin, legacy: clone(record.legacy), needs_review: Boolean(record.needs_review),
    };
  }
  function createDraft(kind) {
    if (kind === "frame") video.pause();
    progressInteractionStarted = true;
    const time = kind === "frame" ? frameTime() : currentTime();
    const draft = {
      record_id: newRecordId(kind), kind, status: "draft", archived: false,
      data: emptyData[kind](time), expected_revision: 0, generation: 0,
      dirty: true, saveState: "local", error: "",
    };
    drafts.set(draft.record_id, draft);
    selectedId = draft.record_id;
    selectedPointId = null;
    pointTool = { entity: "player", playerId: null, visibility: "visible", anchor: "floor_contact" };
    persistPending();
    renderRecordList();
    renderEditor();
    updateProgressCopy();
    return draft;
  }

  function sourceButtonLabel(item) {
    return `${item.title}${item.review_allowed ? "" : ", просмотр недоступен"}`;
  }
  function renderSources() {
    const node = $("source-list");
    node.replaceChildren();
    for (const item of workspace.sources) {
      const wrap = document.createElement("div");
      wrap.className = `source-card${source?.source_id === item.source_id ? " active" : ""}`;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.sourceId = item.source_id;
      button.disabled = !item.review_allowed || !item.media_url;
      button.setAttribute("aria-label", sourceButtonLabel(item));
      const title = document.createElement("strong"); title.textContent = item.title;
      const duration = document.createElement("span"); duration.textContent = formatTime(item.duration_seconds);
      button.append(title, duration);
      const meta = document.createElement("p");
      meta.textContent = item.review_allowed ? "Можно размечать" : "Резерв · содержимое не открывается";
      wrap.append(button, meta); node.append(wrap);
    }
    $("source-count").textContent = `${workspace.sources.length} матчей`;
  }
  function recordTime(draft) {
    const data = draft.data;
    if (draft.kind === "frame") return data.timestamp_seconds;
    if (draft.kind === "event" || draft.kind === "possession") return data.start_seconds;
    return Number.POSITIVE_INFINITY;
  }
  function playerDisplay(draft) {
    if (!draft) return "Игрок не найден";
    const team = draft.data.team_id ? drafts.get(draft.data.team_id) : null;
    const teamLabel = team?.data.name?.trim() || team?.data.color?.trim() || "";
    const players = [...drafts.values()].filter(item => item.kind === "player");
    const ordinal = Math.max(1, players.findIndex(item => item.record_id === draft.record_id) + 1);
    const identity = draft.data.jersey_number ? `№${draft.data.jersey_number}`
      : draft.data.notes?.trim() ? draft.data.notes.trim().slice(0, 32)
      : draft.data.evidence_seconds != null ? `без номера · ${formatTime(draft.data.evidence_seconds)}`
      : `Игрок ${ordinal}`;
    return teamLabel ? `${teamLabel} · ${identity}` : identity;
  }
  function recordLabel(draft) {
    const data = draft.data;
    if (draft.kind === "team") return `Команда: ${data.name || "без названия"}`;
    if (draft.kind === "player") return `Игрок: ${playerDisplay(draft)}`;
    if (draft.kind === "possession") return `Владение: ${data.notes || formatTime(data.start_seconds)}`;
    if (draft.kind === "frame") return `Кадр: ${formatTime(data.timestamp_seconds)}`;
    return `Событие: ${data.notes || EVENT_NAMES[data.event_type] || "без описания"}`;
  }
  function renderRecordList() {
    if (!source) return;
    const kind = $("kind-filter").value;
    const status = $("status-filter").value;
    const list = [...drafts.values()].filter(draft => {
      if (kind !== "all" && draft.kind !== kind) return false;
      if (status === "local") return draft.dirty;
      if (status === "human_reviewed") return draft.status === "human_reviewed" && !draft.needs_review;
      if (status === "draft") return draft.status === "draft" || draft.needs_review;
      return status === "all" || draft.status === status;
    }).sort((a, b) => recordTime(a) - recordTime(b) || a.record_id.localeCompare(b.record_id));
    const node = $("record-list");
    node.replaceChildren();
    if (!list.length) {
      const empty = document.createElement("p"); empty.className = "empty-list"; empty.textContent = "По этим фильтрам пока ничего нет."; node.append(empty); return;
    }
    for (const draft of list) {
      const button = document.createElement("button");
      button.type = "button"; button.dataset.recordId = draft.record_id;
      button.className = `record-card${draft.record_id === selectedId ? " active" : ""}`;
      button.setAttribute("aria-label", recordLabel(draft));
      const top = document.createElement("span"); top.className = "record-card-top";
      const type = document.createElement("strong"); type.textContent = KINDS[draft.kind];
      const time = document.createElement("span"); const seconds = recordTime(draft);
      time.textContent = Number.isFinite(seconds) ? formatTime(seconds) : "справочник";
      top.append(type, time);
      const title = document.createElement("span"); title.className = "record-card-title";
      title.textContent = recordLabel(draft).replace(/^.+?: /, "");
      const state = document.createElement("small");
      state.textContent = draft.dirty ? "• локальная правка" : draft.status === "human_reviewed" && draft.needs_review ? "! нужно дополнить" : draft.status === "human_reviewed" ? "✓ подтверждено" : "черновик";
      button.append(top, title, state); node.append(button);
    }
  }
  function updateProgressCopy() {
    const all = [...drafts.values()];
    const reviewed = all.filter(item => item.status === "human_reviewed" && !item.needs_review && !item.dirty).length;
    $("match-progress").textContent = `${all.length} объектов · ${reviewed} подтверждено`;
  }

  function teamOptions(value, blank = "Не выбрана") {
    return [["", blank], ...[...drafts.values()].filter(item => item.kind === "team")
      .map(item => [item.record_id, item.data.name || "Команда без названия"])]
      .map(([id, label]) => option(id, label, value)).join("");
  }
  function playerOptions(value, blank = "Не выбран") {
    return [["", blank], ...[...drafts.values()].filter(item => item.kind === "player")
      .map(item => [item.record_id, playerDisplay(item)])]
      .map(([id, label]) => option(id, label, value)).join("");
  }
  function possessionOptions(value) {
    return [["", "Не связано"], ...[...drafts.values()].filter(item => item.kind === "possession")
      .map(item => [item.record_id, `${formatTime(item.data.start_seconds)} · ${item.data.notes || "владение"}`])]
      .map(([id, label]) => option(id, label, value)).join("");
  }
  function eventOptions(value) {
    return [["", "Не связано"], ...[...drafts.values()].filter(item => item.kind === "event")
      .map(item => [item.record_id, `${formatTime(item.data.start_seconds)} · ${EVENT_NAMES[item.data.event_type]}`])]
      .map(([id, label]) => option(id, label, value)).join("");
  }
  function editorForTeam(data) {
    return `<div class="field-grid">${textField("Название команды", "name", data.name, { max: 80, placeholder: "Например, хозяева" })}${textField("Цвет формы", "color", data.color, { max: 40, placeholder: "Наблюдаемый цвет" })}${textField("Время подтверждения, секунды", "evidence_seconds", data.evidence_seconds, { type: "number", step: "0.01" })}</div>`;
  }
  function editorForPlayer(data) {
    return `<div class="field-grid"><label class="field"><span>Команда</span><select data-field="team_id" aria-label="Команда">${teamOptions(data.team_id)}</select></label>${textField("Номер формы", "jersey_number", data.jersey_number, { max: 3, placeholder: "00" })}${selectField("Видимость номера", "number_status", data.number_status, [["not_reviewed", "Ещё не проверен"], ["readable", "Читается"], ["unreadable", "Не читается"], ["offscreen", "Вне кадра"], ["unclear", "Неясно"]])}${textField("Время наблюдения, секунды", "evidence_seconds", data.evidence_seconds, { type: "number", step: "0.01" })}${textArea("Комментарий об игроке", "notes", data.notes)}</div>`;
  }
  function editorForPossession(data) {
    return `<div class="field-grid">${textField("Начало владения, секунды", "start_seconds", data.start_seconds, { type: "number", step: "0.01" })}${textField("Конец владения, секунды", "end_seconds", data.end_seconds, { type: "number", step: "0.01" })}<label class="field"><span>Команда с мячом</span><select data-field="team_id" aria-label="Команда с мячом">${teamOptions(data.team_id)}</select></label>${selectField("Направление атаки на изображении", "attack_direction", data.attack_direction, [["unclear", "Неясно"], ["left", "Влево"], ["right", "Вправо"]])}${textArea("Комментарий к владению", "notes", data.notes, "Опишите неопределённость, если команда не видна")}</div><div class="mark-actions"><button type="button" data-mark="start_seconds">Начало владения здесь</button><button type="button" data-mark="end_seconds">Конец владения здесь</button></div>`;
  }
  function editorForEvent(data) {
    const actorLabel = data.event_type === "shot" ? "Бросающий" : "Исполнитель";
    const people = `<label class="field"><span>${actorLabel}</span><select data-field="actor_id" aria-label="${actorLabel}">${playerOptions(data.actor_id)}</select></label><label class="field"><span>Получатель передачи</span><select data-field="receiver_id" aria-label="Получатель передачи">${playerOptions(data.receiver_id)}</select></label><label class="field"><span>Последний пасующий перед броском</span><select data-field="passer_id" aria-label="Последний пасующий перед броском">${playerOptions(data.passer_id)}</select></label>${textField("Время последней передачи, секунды", "last_pass_seconds", data.last_pass_seconds, { type: "number", step: "0.01" })}`;
    const substitution = data.event_type === "substitution" ? `<label class="field"><span>Вышел на площадку</span><select data-field="incoming_player_id" aria-label="Вышел на площадку">${playerOptions(data.incoming_player_id)}</select></label><label class="field"><span>Покинул площадку</span><select data-field="outgoing_player_id" aria-label="Покинул площадку">${playerOptions(data.outgoing_player_id)}</select></label>` : "";
    const shot = data.event_type === "shot" ? `${selectField("Тип броска", "shot_type", data.shot_type, [["", "Не выбрано"], ["two", "Двухочковый"], ["three", "Трёхочковый"], ["free_throw", "Штрафной"], ["non_shot", "Броска нет"], ["unclear", "Неясно"]])}${selectField("Исход броска", "outcome", data.outcome, [["", "Не выбрано"], ["made", "Попадание"], ["miss", "Промах"], ["not_applicable", "Не применимо"], ["unclear", "Неясно"]])}${selectField("Решение по очкам", "scoring_decision", data.scoring_decision, [["", "Не выбрано"], ["counted", "Засчитаны"], ["not_counted", "Не засчитаны"], ["not_applicable", "Не применимо"], ["unclear", "Неясно"]])}${selectField("Контекст игры", "play_context", data.play_context, [["", "Не выбрано"], ["in_play", "В игре"], ["foul_on_shot", "Фол на броске"], ["after_whistle", "После свистка"], ["other_dead_ball", "Другая остановка"], ["not_applicable", "Не применимо"], ["unclear", "Неясно"]])}` : "";
    return `<div class="field-grid">${selectField("Тип события", "event_type", data.event_type, Object.entries(EVENT_NAMES))}${textField("Начало события, секунды", "start_seconds", data.start_seconds, { type: "number", step: "0.01" })}${textField("Конец события, секунды", "end_seconds", data.end_seconds, { type: "number", step: "0.01" })}<label class="field"><span>Владение</span><select data-field="possession_id" aria-label="Владение">${possessionOptions(data.possession_id)}</select></label>${people}${substitution}${shot}${selectField("Показ момента", "presentation", data.presentation, [["", "Не выбрано"], ["live", "Основной эфир"], ["replay", "Повтор"], ["unclear", "Неясно"]])}${selectField("Полнота границ", "boundary_status", data.boundary_status, [["", "Не выбрано"], ["complete", "Контекста хватает"], ["too_short", "Не хватает начала или конца"], ["unclear", "Неясно"]])}${selectField("Защитное действие", "defensive_action", data.defensive_action, [["", "Не отмечено"], ["on_ball", "Защита по мячу"], ["help", "Подстраховка"], ["switch", "Смена"], ["trap", "Дабл-тим"], ["zone", "Зона"], ["man_to_man", "Персональная"], ["unclear", "Неясно"]])}${textArea("Комментарий к событию", "notes", data.notes)}</div><div class="mark-actions"><button type="button" data-mark="start_seconds">Начало события здесь</button><button type="button" data-mark="end_seconds">Конец события здесь</button>${data.event_type === "shot" ? '<button type="button" data-mark="last_pass_seconds">Последний пас здесь</button>' : ""}</div><details class="help"><summary>Подсказка по участникам и заменам</summary><p>Для броска исполнитель — бросающий, а последний пасующий указывается отдельно. Замена означает наблюдаемый вход или выход игрока, а не исчезновение из кадра.</p></details>`;
  }
  function editorForFrame(data) {
    const visibilityNames = { visible: "виден", occluded: "перекрыт", offscreen: "вне кадра", unclear: "неясно" };
    const pointRows = (data.points || []).map(point => {
      const entity = point.entity === "ball" ? "Мяч" : playerDisplay(drafts.get(point.player_id));
      return `<li><button type="button" data-point-id="${escapeHtml(point.point_id)}" class="${selectedPointId === point.point_id ? "selected" : ""}" aria-label="Выбрать точку: ${escapeHtml(entity.toLowerCase())}">${escapeHtml(entity)} · ${escapeHtml(visibilityNames[point.visibility] || point.visibility)}</button><button type="button" data-remove-point="${escapeHtml(point.point_id)}" aria-label="Удалить точку: ${escapeHtml(entity.toLowerCase())}">Удалить</button></li>`;
    }).join("");
    return `<div class="field-grid">${textField("Время кадра, секунды", "timestamp_seconds", data.timestamp_seconds, { type: "number", step: "0.001" })}<label class="field"><span>Событие</span><select data-field="event_id" aria-label="Событие">${eventOptions(data.event_id)}</select></label><label class="field"><span>Владение</span><select data-field="possession_id" aria-label="Владение">${possessionOptions(data.possession_id)}</select></label>${textArea("Комментарий к кадру", "notes", data.notes)}</div><fieldset class="point-tools"><legend>Точки на кадре</legend><div class="radio-row"><label><input type="radio" name="point-entity" value="player"${pointTool.entity === "player" ? " checked" : ""}> Игрок</label><label><input type="radio" name="point-entity" value="ball" aria-label="Мяч"${pointTool.entity === "ball" ? " checked" : ""}> Мяч</label></div><label class="field"><span>Игрок для точки</span><select id="point-player" aria-label="Игрок для точки">${playerOptions(pointTool.playerId, "Выберите игрока")}</select></label><div class="point-options">${selectField("Видимость", "point_visibility", pointTool.visibility, [["visible", "Виден"], ["occluded", "Перекрыт"], ["offscreen", "Вне кадра"], ["unclear", "Неясно"]])}${selectField("Опорная точка", "point_anchor", pointTool.anchor, [["floor_contact", "Контакт с площадкой"], ["image_center", "Центр изображения"]])}</div><p>Поставьте видео на паузу и нажмите на стопы игрока. Для мяча используется центр. Нажмите существующую метку, чтобы переставить её.</p><div class="mark-actions"><button id="return-to-frame" type="button">Вернуться к кадру</button><button id="new-point" type="button">Добавить следующую точку</button><button id="add-point-without-position" type="button">Добавить без координат</button></div><ul class="point-list">${pointRows || "<li class=\"muted\">Точек пока нет</li>"}</ul></fieldset>`;
  }
  function conflictValue(value) {
    if (value == null || value === "") return "не указано";
    if (Array.isArray(value)) return value.length ? `${value.length} отметок` : "нет отметок";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
  const conflictFieldNames = {
    name: "Название", color: "Цвет формы", team_id: "Команда", jersey_number: "Номер формы",
    number_status: "Видимость номера", evidence_seconds: "Время подтверждения", start_seconds: "Начало",
    end_seconds: "Конец", attack_direction: "Направление атаки", event_type: "Тип события",
    possession_id: "Владение", actor_id: "Исполнитель", receiver_id: "Получатель передачи",
    passer_id: "Последний пасующий", last_pass_seconds: "Время последней передачи",
    incoming_player_id: "Вышел на площадку", outgoing_player_id: "Покинул площадку",
    shot_type: "Тип броска", outcome: "Исход броска", scoring_decision: "Решение по очкам",
    play_context: "Контекст игры", presentation: "Показ момента", boundary_status: "Полнота границ",
    defensive_action: "Защитное действие", notes: "Комментарий", timestamp_seconds: "Время кадра",
    event_id: "Событие", points: "Точки на кадре",
  };
  function conflictHtml(draft) {
    const conflict = draft.conflict;
    if (!conflict) return "";
    const local = conflict.local_data || draft.recoverable_data || draft.data;
    const server = conflict.server_data || {};
    const fields = [...new Set([...Object.keys(local), ...Object.keys(server)])]
      .filter(field => JSON.stringify(local[field]) !== JSON.stringify(server[field]));
    const comparisons = fields.map(field => `<div class="conflict-field"><strong>${escapeHtml(conflictFieldNames[field] || field)}</strong><dl><div><dt>Моя версия</dt><dd>${escapeHtml(conflictValue(local[field]))}</dd></div><div><dt>Серверная версия</dt><dd>${escapeHtml(conflictValue(server[field]))}</dd></div></dl></div>`).join("");
    return `<section class="conflict-panel" aria-labelledby="conflict-title"><div><p class="eyebrow">ТРЕБУЕТ РЕШЕНИЯ</p><h3 id="conflict-title">Конфликт версий</h3><p>Версия сервера: ${conflict.latest_revision}. Обе версии сохранены локально до вашего выбора.</p></div><div class="conflict-comparison">${comparisons || "<p>Поля совпадают, но серверная ревизия новее.</p>"}</div><div class="conflict-actions"><button type="button" data-conflict-action="server">Загрузить серверную версию</button><button type="button" data-conflict-action="local">Вернуть мои локальные изменения</button><button type="button" class="primary-button" data-conflict-action="save-local">Сохранить мои изменения черновиком</button></div></section>`;
  }
  function updateEditorChrome() {
    const draft = selectedDraft();
    $("history-panel").hidden = !draft || draft.expected_revision === 0;
    if (!draft) { updateSaveState(); return; }
    $("confirm-record").textContent = draft.status === "human_reviewed" && !draft.needs_review
      ? "Подтвердить исправление" : "Подтвердить разметку";
    showMessage("editor-error", draft.error);
    updateSaveState();
  }
  function activeConflict(draft) {
    const conflict = draft?.conflict;
    if (!conflict || conflict.source_id !== source?.source_id || conflict.source_sha256 !== source?.sha256 ||
      conflict.record_id !== draft.record_id || conflict.kind !== draft.kind || !Number.isInteger(conflict.latest_revision)) return null;
    return conflict;
  }
  function chooseConflictVersion(draft, choice) {
    const conflict = activeConflict(draft);
    if (!conflict) return;
    draft.expected_revision = conflict.latest_revision;
    if (choice === "server") {
      draft.data = clone(conflict.server_data);
      draft.status = conflict.server_status;
      draft.needs_review = conflict.server_needs_review;
    } else {
      draft.data = clone(conflict.local_data || draft.recoverable_data);
      draft.status = "draft";
      draft.needs_review = true;
    }
    draft.dirty = true;
    draft.saveState = "local";
    draft.error = "";
    draft.generation = (draft.generation || 0) + 1;
    persistPending();
    renderEditor();
    renderRecordList();
    updateProgressCopy();
  }
  function saveConflictAsDraft(draft) {
    const conflict = activeConflict(draft);
    if (!conflict) return;
    draft.data = clone(conflict.local_data || draft.recoverable_data);
    draft.expected_revision = conflict.latest_revision;
    draft.status = "draft";
    draft.needs_review = true;
    draft.dirty = true;
    draft.saveState = "local";
    draft.error = "";
    draft.generation = (draft.generation || 0) + 1;
    persistPending();
    renderEditor();
    saveRecord(draft, "draft", { refreshEditor: true });
  }
  function renderEditor() {
    const draft = selectedDraft();
    $("empty-editor").hidden = Boolean(draft);
    editor.hidden = !draft;
    if (!draft) { $("point-surface").hidden = true; $("point-layer").replaceChildren(); video.controls = true; updateEditorChrome(); return; }
    $("editor-kind").textContent = KINDS[draft.kind].toUpperCase();
    $("editor-title").textContent = recordLabel(draft);
    const legacyContext = draft.legacy ? `<section class="legacy-context" aria-label="Контекст импортированной разметки"><strong>${draft.needs_review ? "Нужно дополнить два решения из новой схемы" : "Импортированная разметка"}</strong><p>Исходный клип ${escapeHtml(draft.legacy.example_id)} · схема ${escapeHtml(draft.legacy.schema_version)}</p><button type="button" data-seek-legacy="${numberValue(draft.legacy.source_start_seconds)}" aria-label="Перейти к исходному контексту">${formatTime(draft.legacy.source_start_seconds)}–${formatTime(draft.legacy.source_end_seconds)}</button></section>` : "";
    $("editor-fields").innerHTML = conflictHtml(draft) + legacyContext + ({ team: editorForTeam, player: editorForPlayer, possession: editorForPossession, event: editorForEvent, frame: editorForFrame })[draft.kind](draft.data);
    updateEditorChrome();
    $("point-surface").hidden = draft.kind !== "frame";
    video.controls = draft.kind !== "frame";
    renderPointLayer();
  }

  function setField(field, target) {
    const draft = selectedDraft();
    if (!draft) return;
    const numeric = new Set(["evidence_seconds", "start_seconds", "end_seconds", "last_pass_seconds", "timestamp_seconds"]);
    const nullable = new Set(["team_id", "possession_id", "actor_id", "receiver_id", "passer_id", "incoming_player_id", "outgoing_player_id", "event_id"]);
    const nullableEnums = new Set(["shot_type", "outcome", "scoring_decision", "play_context", "presentation", "boundary_status", "defensive_action"]);
    let value = numeric.has(field) ? nullableNumber(target.value) : nullable.has(field) || nullableEnums.has(field) ? (target.value || null) : target.value;
    if (field === "jersey_number") {
      value = target.value.replace(/\D/g, "").slice(0, 3) || null;
      target.value = value || "";
      draft.data.number_status = value ? "readable" : "not_reviewed";
    }
    draft.data[field] = value;
    setDirty(draft);
    if (field === "event_type") renderEditor();
  }

  function seek(seconds, save = true) {
    const max = source?.duration_seconds ?? Number.POSITIVE_INFINITY;
    presentedMediaTime = null;
    video.currentTime = Math.max(0, Math.min(max, Number(seconds) || 0));
    updateClock();
    renderPointLayer();
    if (save) { progressInteractionStarted = true; saveProgress(); }
  }
  function updateClock() {
    const formatted = formatTime(currentTime());
    $("current-clock").textContent = formatted;
    if (document.activeElement !== $("exact-seek")) $("exact-seek").value = formatted;
  }
  async function saveProgress() {
    if (!workspace || !source || !progressInteractionStarted) return;
    try {
      await request("/api/progress", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        schema_version: SCHEMA_VERSION, workspace_revision: workspace.workspace_revision,
        source_id: source.source_id, position_seconds: currentTime(), selected_record_id: selectedId,
        playback_rate: Number(video.playbackRate) || 1,
      }) });
    } catch { /* Progress is convenient state; record drafts remain independently durable. */ }
  }
  function openRecord(recordId) {
    progressInteractionStarted = true;
    selectedId = recordId; selectedPointId = null;
    pointTool = { entity: "player", playerId: null, visibility: "visible", anchor: "floor_contact" };
    const draft = selectedDraft();
    if (draft) {
      if (draft.kind === "frame") video.pause();
      const seconds = recordTime(draft);
      if (Number.isFinite(seconds)) seek(seconds, false);
    }
    renderRecordList(); renderEditor(); saveProgress();
  }
  function renderPointLayer() {
    const layer = $("point-layer"); layer.replaceChildren();
    const draft = selectedDraft();
    if (!draft || draft.kind !== "frame" || Math.abs(currentTime() - draft.data.timestamp_seconds) > 0.04) return;
    const rect = $("point-surface").getBoundingClientRect();
    for (const point of draft.data.points || []) {
      if (point.x == null || point.y == null || point.visibility !== "visible") continue;
      const style = pointToStyle(point, rect, video.videoWidth || 1920, video.videoHeight || 1080);
      if (!style) continue;
      const marker = document.createElement("button");
      marker.type = "button"; marker.className = `point-marker ${point.entity}${selectedPointId === point.point_id ? " selected" : ""}`;
      marker.dataset.pointId = point.point_id; marker.style.left = style.left; marker.style.top = style.top;
      marker.setAttribute("aria-label", point.entity === "ball" ? "Точка: мяч" : `Точка: игрок ${playerDisplay(drafts.get(point.player_id)).replace("№", "")}`);
      marker.textContent = point.entity === "ball" ? "●" : drafts.get(point.player_id)?.data.jersey_number || "•";
      layer.append(marker);
    }
  }
  function pointConfiguration() {
    return { entity: pointTool.entity, player_id: pointTool.entity === "player" ? pointTool.playerId : null,
      visibility: pointTool.visibility, anchor: pointTool.entity === "ball" ? "image_center" : pointTool.anchor };
  }
  function selectPoint(pointId) {
    const point = selectedDraft()?.data.points?.find(item => item.point_id === pointId);
    selectedPointId = pointId;
    if (point) pointTool = { entity: point.entity, playerId: point.player_id,
      visibility: point.visibility, anchor: point.anchor };
    renderEditor();
  }
  function addOrMovePoint(position) {
    const draft = selectedDraft(); if (!draft || draft.kind !== "frame") return;
    if (Math.abs(currentTime() - draft.data.timestamp_seconds) > 0.04) {
      showMessage("editor-error", "Вернитесь к времени кадра перед добавлением точки."); return;
    }
    const configuration = pointConfiguration();
    if (configuration.entity === "player" && !configuration.player_id) {
      showMessage("editor-error", "Сначала выберите игрока для точки."); return;
    }
    if (selectedPointId) {
      const existing = draft.data.points.find(point => point.point_id === selectedPointId);
      if (existing) Object.assign(existing, configuration, position);
    } else {
      if (configuration.entity === "player" && draft.data.points.some(point => point.player_id === configuration.player_id)) {
        showMessage("editor-error", "Этот игрок уже отмечен на кадре. Выберите его точку, чтобы исправить."); return;
      }
      const point = { point_id: newRecordId("point").slice(6), ...configuration, ...position, notes: "" };
      draft.data.points.push(point); selectedPointId = point.point_id;
    }
    setDirty(draft); renderEditor();
  }
  async function openSource(sourceId) {
    const selectedSource = workspace.sources.find(item => item.source_id === sourceId);
    if (!selectedSource?.review_allowed || !selectedSource.media_url) return;
    const session = ++sourceSession;
    progressInteractionStarted = false;
    for (const timer of saveTimers.values()) clearTimeout(timer);
    saveTimers.clear();
    showMessage("global-error");
    const result = await request(`/api/sources/${encodeURIComponent(sourceId)}`);
    if (session !== sourceSession) return;
    source = result.source;
    records = new Map((result.records || []).map(record => [record.record_id, record]));
    drafts = new Map([...records.values()].map(record => [record.record_id, draftFromRecord(record)]));
    recoveredIds = new Set();
    for (const local of loadLocalDrafts(localStorage, workspace.workspace_id, workspace.workspace_revision, source.source_id, draftTabId, recoveryTabId)) {
      const base = records.get(local.record_id);
      const recovered = { ...clone(local), dirty: true, saveState: "local", error: "", generation: local.generation || 1,
        origin: base?.origin, legacy: clone(base?.legacy), needs_review: Boolean(base?.needs_review) };
      const storedConflict = recovered.conflict;
      if (!storedConflict || storedConflict.source_id !== source.source_id || storedConflict.source_sha256 !== source.sha256 ||
        storedConflict.record_id !== recovered.record_id || storedConflict.kind !== recovered.kind || !Number.isInteger(storedConflict.latest_revision)) {
        recovered.conflict = null;
      }
      if (base && base.revision > recovered.expected_revision && (!recovered.conflict || base.revision > recovered.conflict.latest_revision)) {
        // Previewing the server version must not replace the recoverable local copy.
        const localData = clone(recovered.conflict?.local_data || recovered.data);
        recovered.conflict = {
          source_id: source.source_id, source_sha256: source.sha256, record_id: recovered.record_id,
          kind: recovered.kind, latest_revision: base.revision, server_data: clone(base.data),
          server_status: base.status, server_needs_review: Boolean(base.needs_review), local_data: localData,
        };
        recovered.recoverable_data = clone(localData);
        recovered.error = "Сервер сохранил более новую версию. Сравните обе версии и явно выберите дальнейшее действие.";
        recovered.saveState = "error";
      }
      drafts.set(local.record_id, recovered);
      recoveredIds.add(local.record_id);
    }
    persistPending();
    const progress = result.progress;
    selectedId = progress?.selected_record_id && drafts.has(progress.selected_record_id) ? progress.selected_record_id : null;
    selectedPointId = null;
    pointTool = { entity: "player", playerId: null, visibility: "visible", anchor: "floor_contact" };
    if (!selectedId && recoveredIds.size) selectedId = [...recoveredIds].at(-1);
    if (!selectedId) {
      selectedId = [...drafts.values()].filter(item => item.kind === "event")
        .sort((a, b) => String(records.get(a.record_id)?.updated_at || "").localeCompare(String(records.get(b.record_id)?.updated_at || ""))).at(-1)?.record_id || null;
    }
    video.src = selectedSource.media_url;
    video.load();
    startFrameTracking();
    video.playbackRate = progress?.playback_rate || 1;
    $("playback-rate").value = String(video.playbackRate);
    seek(progress?.position_seconds || (selectedDraft() ? recordTime(selectedDraft()) : 0), false);
    $("match-title").textContent = source.title;
    $("match-role").textContent = `${source.role === "development_review" ? "РАЗРЕШЁН ДЛЯ ПРОСМОТРА" : source.role} · ${formatTime(source.duration_seconds)}`;
    renderSources(); renderRecordList(); renderEditor(); updateProgressCopy();
    $("match-title").focus();
    if (recoveredIds.size) showMessage("global-status", "Восстановлен локальный черновик");
    else showMessage("global-status");
  }

  listen($("source-list"), "click", event => {
    const button = event.target.closest("button[data-source-id]");
    if (button) openSource(button.dataset.sourceId).catch(error => showMessage("global-error", error.message));
  });
  listen(document, "click", event => {
    const create = event.target.closest("button[data-create]");
    if (create) createDraft(create.dataset.create);
  });
  listen($("record-list"), "click", event => {
    const button = event.target.closest("button[data-record-id]"); if (button) openRecord(button.dataset.recordId);
  });
  listen($("kind-filter"), "change", renderRecordList);
  listen($("status-filter"), "change", renderRecordList);
  listen(editor, "input", event => {
    const field = event.target.dataset.field;
    if (field && !field.startsWith("point_")) setField(field, event.target);
  });
  listen(editor, "change", event => {
    const field = event.target.dataset.field;
    if (event.target.name === "point-entity") {
      pointTool.entity = event.target.value;
      if (pointTool.entity === "ball") pointTool.anchor = "image_center";
      return;
    }
    if (event.target.id === "point-player") { pointTool.playerId = event.target.value || null; return; }
    if (field?.startsWith("point_")) {
      pointTool[field === "point_visibility" ? "visibility" : "anchor"] = event.target.value;
      return;
    }
    if (field) setField(field, event.target);
  });
  listen(editor, "click", event => {
    const conflictAction = event.target.closest("button[data-conflict-action]");
    if (conflictAction) {
      const draft = selectedDraft();
      if (conflictAction.dataset.conflictAction === "save-local") saveConflictAsDraft(draft);
      else chooseConflictVersion(draft, conflictAction.dataset.conflictAction);
      return;
    }
    const mark = event.target.closest("button[data-mark]");
    if (mark) {
      const draft = selectedDraft(); draft.data[mark.dataset.mark] = currentTime(); setDirty(draft); renderEditor(); return;
    }
    const legacySeek = event.target.closest("button[data-seek-legacy]");
    if (legacySeek) { seek(Number(legacySeek.dataset.seekLegacy)); return; }
    const pointButton = event.target.closest("button[data-point-id]");
    if (pointButton) { selectPoint(pointButton.dataset.pointId); return; }
    const removePoint = event.target.closest("button[data-remove-point]");
    if (removePoint) {
      const draft = selectedDraft(); draft.data.points = draft.data.points.filter(point => point.point_id !== removePoint.dataset.removePoint);
      if (selectedPointId === removePoint.dataset.removePoint) selectedPointId = null;
      setDirty(draft); renderEditor(); return;
    }
    if (event.target.id === "new-point") { selectedPointId = null; renderEditor(); return; }
    if (event.target.id === "return-to-frame") { const draft = selectedDraft(); video.pause(); seek(draft.data.timestamp_seconds); return; }
    if (event.target.id === "add-point-without-position") addOrMovePoint({ x: null, y: null, visibility: pointConfiguration().visibility === "visible" ? "unclear" : pointConfiguration().visibility });
  });
  listen($("point-surface"), "click", event => {
    const draft = selectedDraft();
    if (!draft || Math.abs(currentTime() - draft.data.timestamp_seconds) > 0.04) {
      showMessage("editor-error", "Вернитесь к времени кадра перед добавлением точки."); return;
    }
    const point = pointFromClient({ clientX: event.clientX, clientY: event.clientY,
      rect: event.currentTarget.getBoundingClientRect(), videoWidth: video.videoWidth || 1920, videoHeight: video.videoHeight || 1080 });
    if (!point) { showMessage("editor-error", "Нажмите внутри изображения, а не на чёрной полосе."); return; }
    addOrMovePoint({ ...point, visibility: "visible" });
  });
  listen($("point-layer"), "click", event => {
    const marker = event.target.closest("button[data-point-id]"); if (marker) selectPoint(marker.dataset.pointId);
  });
  listen($("save-draft"), "click", () => saveRecord(selectedDraft(), "draft"));
  listen($("confirm-record"), "click", () => saveRecord(selectedDraft(), "human_reviewed"));
  listen($("load-history"), "click", async () => {
    const draft = selectedDraft(); if (!draft) return;
    try {
      const result = await request(`/api/records/${encodeURIComponent(draft.record_id)}/history`);
      const history = result.records || result.revisions || result.history || (Array.isArray(result) ? result : []);
      $("history-list").replaceChildren(...history.map(item => {
        const li = document.createElement("li"); li.textContent = `Версия ${item.revision} · ${item.status === "human_reviewed" ? "подтверждено" : "черновик"} · ${item.updated_at || item.created_at}`; return li;
      }));
    } catch (error) { showMessage("editor-error", `Не удалось открыть историю: ${error.message}`); }
  });
  listen($("restore-button"), "click", async () => {
    const file = $("restore-file").files?.[0];
    if (!file) { $("restore-summary").textContent = "Сначала выберите файл NDJSON."; return; }
    try {
      const response = await fetch("/api/restore", { method: "POST", headers: { "Content-Type": "application/x-ndjson" }, body: await file.text() });
      const result = await response.json(); if (!response.ok) throw new Error(responseDetail(result, `Ошибка HTTP ${response.status}`));
      $("restore-summary").textContent = `Добавлено: ${result.added ?? result.added_count ?? 0}; уже было: ${result.unchanged ?? result.unchanged_count ?? 0}.`;
      await openSource(source.source_id);
    } catch (error) { $("restore-summary").textContent = `Копия не восстановлена: ${error.message}`; }
  });
  listen($("play-pause"), "click", () => video.paused ? video.play() : video.pause());
  listen($("frame-back"), "click", () => seek(frameTime() - 1 / sourceFps()));
  listen($("frame-forward"), "click", () => seek(frameTime() + 1 / sourceFps()));
  for (const [id, delta] of [["step-back", -0.25], ["short-back", -2], ["short-forward", 2], ["step-forward", 0.25]]) listen($(id), "click", () => seek(currentTime() + delta));
  function seekFromInput() {
    const parsed = parseTime($("exact-seek").value);
    if (parsed == null) { showMessage("global-error", "Введите время в формате чч:мм:сс."); return; }
    showMessage("global-error"); seek(parsed);
  }
  listen($("seek-button"), "click", seekFromInput);
  listen($("exact-seek"), "keydown", event => { if (event.key === "Enter") { event.preventDefault(); seekFromInput(); } });
  listen($("playback-rate"), "change", event => { progressInteractionStarted = true; video.playbackRate = Number(event.target.value); saveProgress(); });
  listen(video, "timeupdate", () => { updateClock(); renderPointLayer(); });
  listen(video, "resize", renderPointLayer);
  listen(window, "resize", renderPointLayer);
  listen(video, "pause", saveProgress);
  listen(video, "seeked", () => { updateClock(); renderPointLayer(); saveProgress(); });
  listen(video, "pointerdown", () => { progressInteractionStarted = true; });
  listen(video, "play", () => { progressInteractionStarted = true; if (!progressTimer) progressTimer = setInterval(saveProgress, 15000); });
  listen(video, "ended", () => { if (progressTimer) clearInterval(progressTimer); progressTimer = null; saveProgress(); });
  listen(document, "keydown", event => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.matches("input, textarea, select, button") || target.isContentEditable)) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); seek(currentTime() - 0.25); }
    if (event.key === "ArrowRight") { event.preventDefault(); seek(currentTime() + 0.25); }
    if (event.key === " ") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
  });
  listen(window, "beforeunload", event => {
    if ([...drafts.values()].some(draft => draft.dirty)) { event.preventDefault(); event.returnValue = ""; }
  });
  if (typeof globalThis.ResizeObserver === "function") {
    resizeObserver = new ResizeObserver(renderPointLayer);
    resizeObserver.observe($("video-stage"));
  }

  try {
    workspace = await request("/api/workspace");
    if (!workspace || workspace.schema_version !== SCHEMA_VERSION || !Array.isArray(workspace.sources)) throw new Error("Рабочее пространство имеет неподдерживаемый формат");
    renderSources();
    const initial = workspace.progress?.source_id && workspace.sources.find(item => item.source_id === workspace.progress.source_id && item.review_allowed)
      || workspace.sources.find(item => item.review_allowed && item.media_url);
    if (!initial) throw new Error("Нет матчей, разрешённых для разметки");
    await openSource(initial.source_id);
    $("loading").hidden = true; $("main-workbench").hidden = false;
  } catch (error) {
    $("loading").hidden = true; showMessage("global-error", `Не удалось открыть рабочее пространство: ${error.message}`);
  }

  return () => {
    disposed = true; controller.abort();
    for (const timer of saveTimers.values()) clearTimeout(timer);
    saveTimers.clear();
    if (progressTimer) clearInterval(progressTimer);
    resizeObserver?.disconnect();
    if (videoFrameCallbackId != null && typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(videoFrameCallbackId);
  };
}

if (document.getElementById("workbench-app")) mountWorkbench();

/* This review surface deliberately shows no machine-generated labels. */
(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const schemaVersion = 2;
  const newFields = ["scoring_decision", "play_context"];
  const fields = ["shot_type", "outcome", ...newFields, "presentation", "boundary_status"];
  const choices = {
    shot_type: ["two", "three", "free_throw", "non_shot", "unclear"],
    outcome: ["made", "miss", "not_applicable", "unclear"],
    scoring_decision: ["counted", "not_counted", "not_applicable", "unclear"],
    play_context: ["in_play", "foul_on_shot", "after_whistle", "other_dead_ball", "not_applicable", "unclear"],
    presentation: ["live", "replay", "unclear"],
    boundary_status: ["complete", "too_short", "unclear"],
  };
  const player = byId("player");
  const form = byId("review-form");
  const drafts = new Map();
  let batch = null;
  let position = 0;
  let saving = false;
  let saveError = "";
  let conflict = false;

  const round = (value) => Math.round(value * 1000) / 1000;
  const numberLabel = (index) => String(index + 1).padStart(2, "0");
  const current = () => batch.examples[position];
  const saved = (example) => batch.annotations[example.example_id];
  const completed = (example) => completeAnnotation(saved(example), example);
  const reviewedCount = () => batch.examples.filter(completed).length;

  function clock(value) {
    const seconds = Math.max(0, Math.floor(value));
    return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
      .map((part) => String(part).padStart(2, "0")).join(":");
  }

  function localMediaURL(value) {
    const url = new URL(value, window.location.href);
    if (url.origin !== window.location.origin || !/^\/(media|posters)\/[A-Za-z0-9_-]+$/.test(url.pathname)) {
      throw new Error("Non-local media URL");
    }
    return url.href;
  }

  function validateBatch(data) {
    if (!data || typeof data.batch_id !== "string" || typeof data.batch_revision !== "string"
      || !Array.isArray(data.examples) || !data.examples.length || !data.annotations
      || typeof data.annotations !== "object" || Array.isArray(data.annotations)) {
      throw new Error("Invalid review batch");
    }
    const ids = new Set();
    for (const example of data.examples) {
      if (typeof example.example_id !== "string" || !example.example_id || ids.has(example.example_id)
        || typeof example.source_alias !== "string"
        || !Number.isFinite(example.clip_duration_seconds) || example.clip_duration_seconds <= 0
        || !Number.isFinite(example.source_start_seconds) || example.source_start_seconds < 0
        || !Number.isFinite(example.source_end_seconds)
        || example.source_end_seconds <= example.source_start_seconds) {
        throw new Error("Invalid review example");
      }
      localMediaURL(example.video_url);
      if (example.poster_url) localMediaURL(example.poster_url);
      if (data.annotations[example.example_id]
        && !validAnnotation(data.annotations[example.example_id], example)) {
        throw new Error("Invalid saved annotation");
      }
      ids.add(example.example_id);
    }
    return data;
  }

  function validAnnotation(annotation, example) {
    return annotation && Number.isInteger(annotation.revision) && annotation.revision > 0
      && [1, schemaVersion].includes(annotation.schema_version)
      && fields.every((field) => annotation[field] === null || choices[field].includes(annotation[field])
        || (annotation.schema_version === 1 && newFields.includes(field) && annotation[field] === undefined))
      && ((annotation.start_seconds === null && annotation.end_seconds === null)
        || validBounds(annotation.start_seconds, annotation.end_seconds, example.clip_duration_seconds))
      && typeof annotation.notes === "string" && annotation.notes.length <= 2000;
  }

  function completeAnnotation(annotation, example) {
    return annotation?.schema_version === schemaVersion
      && fields.every((field) => choices[field].includes(annotation[field]))
      && !decisionConflict(annotation)
      && validBounds(annotation.start_seconds, annotation.end_seconds, example.clip_duration_seconds);
  }

  function decisionConflict(annotation) {
    return annotation.scoring_decision === "counted"
      && ["after_whistle", "other_dead_ball"].includes(annotation.play_context);
  }

  const decisionConflictMessage = "Проверьте решение об очках и обстоятельства: новый бросок после остановки не может одновременно быть засчитан. Если это продолжение броска с фолом, выберите «Фол на броске». При сомнении отметьте «Неясно».";

  function validBounds(start, end, duration) {
    return Number.isFinite(start) && Number.isFinite(end)
      && start >= 0 && start < end && end <= duration;
  }

  function freshDraft(example) {
    const record = saved(example);
    return {
      ...Object.fromEntries(fields.map((field) => [field, record?.[field] ?? null])),
      start_seconds: record?.start_seconds ?? 0,
      end_seconds: record?.end_seconds ?? example.clip_duration_seconds,
      notes: record ? record.notes : "",
      dirty: false,
    };
  }

  function draftFor(example) {
    if (!drafts.has(example.example_id)) drafts.set(example.example_id, freshDraft(example));
    return drafts.get(example.example_id);
  }

  function showError(message) {
    byId("error").textContent = message;
    byId("error").hidden = !message;
    if (message) byId("notice").hidden = true;
  }

  function notice(message) {
    byId("notice").textContent = message;
    byId("notice").hidden = !message;
  }

  async function request(url, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      return await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
    } finally {
      clearTimeout(timeout);
    }
  }

  function updateProgress() {
    const count = reviewedCount();
    byId("progress-label").textContent = `${count} из ${batch.examples.length} сохранено`;
    byId("progress").max = batch.examples.length;
    byId("progress").value = count;
    const list = byId("episode-list");
    list.replaceChildren();
    batch.examples.forEach((example, index) => {
      const button = document.createElement("button");
      const isSaved = Boolean(completed(example));
      const dirty = drafts.get(example.example_id)?.dirty;
      const state = dirty ? "черновик" : isSaved ? "сохранён" : saved(example) ? "нужно завершить" : "не сохранён";
      button.type = "button";
      button.textContent = numberLabel(index);
      button.setAttribute("aria-label", `Эпизод ${numberLabel(index)} — ${state}`);
      if (index === position) button.setAttribute("aria-current", "step");
      if (isSaved) button.classList.add("reviewed");
      if (dirty || isSaved) {
        const mark = document.createElement("span");
        mark.className = "mark";
        mark.setAttribute("aria-hidden", "true");
        mark.textContent = dirty ? "•" : "✓";
        button.append(mark);
      }
      button.disabled = saving;
      button.addEventListener("click", () => navigate(index));
      list.append(button);
    });
  }

  function updateState() {
    const example = current();
    const draft = draftFor(example);
    const state = byId("episode-state");
    state.textContent = draft.dirty ? "Черновик" : completed(example) ? "Сохранён" : saved(example) ? "Нужно завершить" : "Не сохранён";
    state.className = `state-tag${draft.dirty ? " dirty" : completed(example) ? " saved" : ""}`;
    byId("draft-hint").textContent = draft.dirty
      ? "Есть несохранённые ответы. При переходе они остаются черновиком в этой вкладке."
      : completed(example)
        ? `Сохранено локально · версия ${saved(example).revision}. Ответы можно исправить.`
        : saved(example)?.schema_version === 1
          ? "Прежние ответы сохранены. Подтвердите два новых вопроса: зачёт очков и обстоятельства броска. Затем сохраните новую версию."
        : saved(example)
          ? "Сохранён незавершённый ответ. Дополните пустые группы и проверьте границы."
        : "Ответы сохраняются кнопкой ниже. Без догадок — только то, что видно.";
    updateSaveAvailability();
  }

  function updateSaveAvailability() {
    const draft = draftFor(current());
    const invalidDecision = decisionConflict(draft);
    byId("decision-hint").textContent = invalidDecision ? decisionConflictMessage : "";
    byId("decision-hint").hidden = !invalidDecision;
    byId("save").disabled = saving || conflict
      || invalidDecision
      || fields.some((field) => !choices[field].includes(draft[field]))
      || !validBounds(draft.start_seconds, draft.end_seconds, current().clip_duration_seconds)
      || draft.notes.length > 2000;
  }

  function updateClock() {
    if (!batch) return;
    byId("source-time").textContent = clock(current().source_start_seconds + player.currentTime);
  }

  function setBusy(value) {
    saving = value;
    byId("labels").disabled = value;
    for (const id of ["start-seconds", "end-seconds", "mark-start", "mark-end", "step-back", "step-forward"]) {
      byId(id).disabled = value;
    }
    byId("previous").disabled = value || position === 0;
    byId("next").disabled = value || position === batch.examples.length - 1;
    byId("save").textContent = value ? "Сохраняем…" : "Сохранить и дальше";
    updateSaveAvailability();
    form.setAttribute("aria-busy", String(value));
    updateProgress();
  }

  function render() {
    const example = current();
    const draft = draftFor(example);
    player.pause();
    player.src = localMediaURL(example.video_url);
    if (example.poster_url) player.poster = localMediaURL(example.poster_url);
    else player.removeAttribute("poster");
    player.load();
    player.playbackRate = Number(byId("speed").value);
    byId("episode-title").textContent = `Эпизод ${numberLabel(position)}`;
    byId("source-alias").textContent = example.source_alias;
    byId("source-range").textContent = `В матче ${clock(example.source_start_seconds)}–${clock(example.source_end_seconds)}`;
    byId("source-time").textContent = clock(example.source_start_seconds);
    for (const field of fields) {
      for (const input of form.querySelectorAll(`input[name="${field}"]`)) {
        input.checked = input.value === draft[field];
      }
    }
    for (const [id, field] of [["start-seconds", "start_seconds"], ["end-seconds", "end_seconds"]]) {
      byId(id).max = String(example.clip_duration_seconds);
      byId(id).value = String(draft[field]);
    }
    byId("notes").value = draft.notes;
    updateState();
    setBusy(false);
  }

  function navigate(index) {
    if (saving || index < 0 || index >= batch.examples.length || index === position) return;
    position = index;
    showError(conflict ? saveError : "");
    render();
    byId("episode-title").focus({ preventScroll: true });
  }

  function edited() {
    if (!batch || saving) return;
    const draft = draftFor(current());
    for (const field of fields) {
      draft[field] = form.querySelector(`input[name="${field}"]:checked`)?.value || null;
    }
    draft.start_seconds = byId("start-seconds").value === "" ? "" : Number(byId("start-seconds").value);
    draft.end_seconds = byId("end-seconds").value === "" ? "" : Number(byId("end-seconds").value);
    draft.notes = byId("notes").value;
    draft.dirty = true;
    if (!conflict) showError("");
    notice("");
    updateState();
    updateProgress();
  }

  function markBoundary(id) {
    if (!batch || saving) return;
    byId(id).value = String(round(Math.max(0, Math.min(current().clip_duration_seconds, player.currentTime))));
    edited();
  }

  function step(delta) {
    if (!batch || saving) return;
    player.pause();
    player.currentTime = round(Math.max(0, Math.min(current().clip_duration_seconds, player.currentTime + delta)));
    updateClock();
  }

  async function save(event) {
    event.preventDefault();
    if (!batch || saving) return;
    if (conflict) {
      showError(saveError);
      byId("error").focus();
      return;
    }
    const example = current();
    const draft = draftFor(example);
    if (fields.some((field) => !choices[field].includes(draft[field]))) {
      showError("Выберите ответ в каждой из шести групп. Если не уверены, можно выбрать «Неясно».");
      byId("error").focus();
      return;
    }
    if (decisionConflict(draft)) {
      showError(decisionConflictMessage);
      byId("error").focus();
      return;
    }
    if (!validBounds(draft.start_seconds, draft.end_seconds, example.clip_duration_seconds)) {
      showError("Границы должны быть внутри клипа: начало ≥ 0, конец позже начала и не дальше конца клипа.");
      byId("error").focus();
      return;
    }
    if (draft.notes.length > 2000) {
      showError("Сократите комментарий до 2000 символов.");
      byId("error").focus();
      return;
    }
    const expectedRevision = saved(example)?.revision || 0;
    const payload = {
      schema_version: schemaVersion,
      batch_revision: batch.batch_revision,
      example_id: example.example_id,
      expected_revision: expectedRevision,
      ...Object.fromEntries(fields.map((field) => [field, draft[field]])),
      start_seconds: draft.start_seconds,
      end_seconds: draft.end_seconds,
      notes: draft.notes,
    };
    showError("");
    notice("");
    setBusy(true);
    try {
      const response = await request("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (response.status === 409) {
        conflict = true;
        saveError = "Подборка или сохранённые ответы изменились в другой вкладке. Ваш ввод остался здесь. Скопируйте нужные изменения и обновите страницу перед повторным сохранением; устаревшая версия не будет перезаписана.";
        throw new Error(saveError);
      }
      if (!response.ok) {
        throw new Error(response.status === 422
          ? "Ответы не прошли проверку сервера. Проверьте выбранные метки и границы. Ввод сохранён в этой вкладке."
          : `Не удалось сохранить ответы (HTTP ${response.status}). Ввод остался здесь. Попробуйте ещё раз.`);
      }
      const result = await response.json();
      if (!validAnnotation(result.annotation, example) || !completeAnnotation(result.annotation, example)
        || result.annotation.revision <= expectedRevision
        || result.total_count !== batch.examples.length || !Number.isInteger(result.reviewed_count)
        || result.reviewed_count < 1 || result.reviewed_count > result.total_count) {
        throw new Error("Сервер вернул неполное подтверждение. Ввод остался здесь. Обновите страницу после проверки соединения, чтобы узнать статус сохранения.");
      }
      batch.annotations[example.example_id] = result.annotation;
      drafts.delete(example.example_id);
      saving = false;
      const nextPosition = position + 1 < batch.examples.length ? position + 1
        : batch.examples.findIndex((item) => !completed(item));
      if (nextPosition >= 0) position = nextPosition;
      render();
      notice(reviewedCount() === batch.examples.length
        ? "Все эпизоды проверены, ответы сохранены на этом Mac. Спасибо! Можно исправить любой ответ или скачать разметку. Спорные случаи разберём отдельно."
        : `Эпизод ${numberLabel(batch.examples.indexOf(example))} сохранён на этом Mac.`);
      byId("episode-title").focus({ preventScroll: true });
    } catch (error) {
      showError(error instanceof TypeError || error?.name === "AbortError"
        ? "Нет подтверждения сохранения от сервера. Ввод остался здесь. Проверьте, что локальный сервер работает, и попробуйте ещё раз."
        : error.message || "Не удалось сохранить ответы. Ввод остался здесь.");
      byId("error").focus();
    } finally {
      setBusy(false);
    }
  }

  async function load() {
    byId("retry").hidden = true;
    byId("loading").hidden = false;
    showError("");
    try {
      const response = await request("/api/review");
      if (!response.ok) throw new Error("Review load failed");
      batch = validateBatch(await response.json());
      if (typeof batch.title === "string" && batch.title) byId("page-title").textContent = batch.title;
      position = Math.max(0, batch.examples.findIndex((example) => !completed(example)));
      byId("loading").hidden = true;
      byId("review").hidden = false;
      render();
      if (reviewedCount() === batch.examples.length) notice("Все эпизоды уже сохранены. Можно просмотреть и исправить ответы.");
    } catch {
      batch = null;
      byId("review").hidden = true;
      byId("retry").hidden = false;
      byId("progress-label").textContent = "Подборка недоступна";
      showError("Не удалось открыть подборку. Проверьте, что локальный сервер запущен, и повторите загрузку.");
    }
  }

  form.addEventListener("submit", save);
  form.addEventListener("input", edited);
  // Boundary inputs are outside the form visually, but belong to it semantically.
  byId("start-seconds").addEventListener("input", edited);
  byId("end-seconds").addEventListener("input", edited);
  byId("mark-start").addEventListener("click", () => markBoundary("start-seconds"));
  byId("mark-end").addEventListener("click", () => markBoundary("end-seconds"));
  byId("step-back").addEventListener("click", () => step(-0.25));
  byId("step-forward").addEventListener("click", () => step(0.25));
  byId("previous").addEventListener("click", () => navigate(position - 1));
  byId("next").addEventListener("click", () => navigate(position + 1));
  byId("retry").addEventListener("click", load);
  byId("speed").addEventListener("change", () => { player.playbackRate = Number(byId("speed").value); });
  player.addEventListener("timeupdate", updateClock);
  player.addEventListener("error", () => {
    if (batch) showError("Не удалось воспроизвести этот клип. Не угадывайте метки: попробуйте открыть эпизод ещё раз или отложите его кнопкой «Следующий».");
  });
  load();
})();

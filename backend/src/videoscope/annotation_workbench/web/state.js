export const SCHEMA_VERSION = 1;
const STORAGE_PREFIX = "videoscope.annotation-workbench.drafts";

export function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export function newRecordId(kind) {
  const value = globalThis.crypto?.randomUUID?.().replaceAll("-", "") ||
    Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  return `${kind}-${value}`;
}

export function storageKey(workspaceId, workspaceRevision, sourceId, tabId = null) {
  const base = `${STORAGE_PREFIX}:${workspaceId}:${workspaceRevision}:${sourceId}`;
  return tabId ? `${base}:tab:${tabId}` : base;
}

export function loadLocalDrafts(storage, workspaceId, workspaceRevision, sourceId, tabId = null, recoveryTabId = null) {
  try {
    const key = storageKey(workspaceId, workspaceRevision, sourceId, tabId);
    const own = storage.getItem(key);
    let keys = [key];
    if (tabId && own === null) {
      const previous = recoveryTabId && storageKey(workspaceId, workspaceRevision, sourceId, recoveryTabId);
      // Reloads prefer their previous writer, including its empty snapshot.
      if (previous && storage.getItem(previous) !== null) keys = [previous];
      else {
        // A fresh tab can recover a closed tab or the legacy shared format.
        const base = storageKey(workspaceId, workspaceRevision, sourceId);
        keys.push(base);
        for (let index = 0; index < storage.length; index += 1) {
          const candidate = storage.key(index);
          if (candidate?.startsWith(`${base}:tab:`) && candidate !== key) keys.push(candidate);
        }
      }
    }
    return keys.flatMap(candidate => {
      try {
        const parsed = JSON.parse(storage.getItem(candidate) || "[]");
        if (!Array.isArray(parsed)) return [];
        return parsed.filter(item => item && item.record_id && item.kind && item.data && Number.isInteger(item.expected_revision));
      } catch { return []; }
    });
  } catch {
    return [];
  }
}

export function writeLocalDrafts(storage, workspaceId, workspaceRevision, sourceId, drafts, tabId = null) {
  const pending = [...drafts.values()]
    .filter(draft => draft.dirty)
    .map(({ record_id, kind, status, archived, data, expected_revision, generation, conflict, recoverable_data }) => ({
      record_id, kind, status, archived, data: clone(data), expected_revision, generation,
      conflict: clone(conflict), recoverable_data: clone(recoverable_data),
    }));
  const key = storageKey(workspaceId, workspaceRevision, sourceId, tabId);
  // Keep an empty owned snapshot so a reload does not import another tab's drafts.
  if (pending.length || tabId) storage.setItem(key, JSON.stringify(pending));
  else storage.removeItem(key);
}

/** Serialize writes per entity while allowing different records to save independently. */
export function createEntitySaveQueue() {
  const tails = new Map();
  return {
    enqueue(recordId, operation) {
      const previous = tails.get(recordId) || Promise.resolve();
      const current = previous.catch(() => undefined).then(operation);
      tails.set(recordId, current);
      current.finally(() => {
        if (tails.get(recordId) === current) tails.delete(recordId);
      }).catch(() => undefined);
      return current;
    },
    pending(recordId) { return tails.has(recordId); },
  };
}

export function formatTime(seconds) {
  const finite = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const hours = Math.floor(finite / 3600);
  const minutes = Math.floor((finite % 3600) / 60);
  const whole = Math.floor(finite % 60);
  const fraction = Math.round((finite - Math.floor(finite)) * 100);
  const base = [hours, minutes, whole].map(value => String(value).padStart(2, "0")).join(":");
  return fraction ? `${base}.${String(fraction).padStart(2, "0")}` : base;
}

export function parseTime(value) {
  const parts = String(value).trim().split(":");
  if (parts.length !== 3 || parts.some(part => part === "" || !Number.isFinite(Number(part)))) return null;
  const [hours, minutes, seconds] = parts.map(Number);
  if (hours < 0 || minutes < 0 || minutes >= 60 || seconds < 0 || seconds >= 60) return null;
  return hours * 3600 + minutes * 60 + seconds;
}

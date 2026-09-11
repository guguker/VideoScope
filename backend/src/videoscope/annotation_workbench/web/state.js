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

export function storageKey(workspaceId, workspaceRevision, sourceId) {
  return `${STORAGE_PREFIX}:${workspaceId}:${workspaceRevision}:${sourceId}`;
}

export function loadLocalDrafts(storage, workspaceId, workspaceRevision, sourceId) {
  try {
    const parsed = JSON.parse(storage.getItem(storageKey(workspaceId, workspaceRevision, sourceId)) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(item => item && item.record_id && item.kind && item.data && Number.isInteger(item.expected_revision));
  } catch {
    return [];
  }
}

export function writeLocalDrafts(storage, workspaceId, workspaceRevision, sourceId, drafts) {
  const pending = [...drafts.values()]
    .filter(draft => draft.dirty)
    .map(({ record_id, kind, status, archived, data, expected_revision, generation }) => ({
      record_id, kind, status, archived, data: clone(data), expected_revision, generation,
    }));
  const key = storageKey(workspaceId, workspaceRevision, sourceId);
  if (pending.length) storage.setItem(key, JSON.stringify(pending));
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

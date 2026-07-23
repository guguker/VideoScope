export function formatDuration(value: number | null | undefined): string {
  const total = Math.max(0, Math.floor(value ?? 0))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const seconds = total % 60
  if (hours > 0) {
    return [hours, minutes, seconds].map((part) => String(part).padStart(2, '0')).join(':')
  }
  return [minutes, seconds].map((part) => String(part).padStart(2, '0')).join(':')
}

export function formatMomentRange(start: number, end: number): string {
  return `${formatDuration(start)}–${formatDuration(end)}`
}

export function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} КБ`
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} МБ`
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} ГБ`
}


// Formatting helpers shared by the console widgets. Kept in one place so a GiB is a GiB
// everywhere and the sidebar gauge cannot disagree with the cache panel.

export function gib(bytes: number | null | undefined, digits = 1): string {
  if (!bytes || bytes <= 0) return '0'
  return (bytes / 2 ** 30).toFixed(digits)
}

export function bytesLabel(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let value = bytes
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(value >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

/** 512000 -> "512K", 1048576 -> "1.0M". Used for token counts on the cache sliders. */
export function compactTokens(n: number | null | undefined): string {
  if (!n || n <= 0) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`
  return String(n)
}

/** 950 -> "950", 12_345 -> "12.3K", 1_234_567 -> "1.2M", 2.5e9 -> "2.5B". For running
 *  totals, where compactTokens' whole-K rounding would hide a few thousand tokens. */
export function humanCount(n: number | null | undefined): string {
  if (!n || n <= 0) return '0'
  const units: [number, string][] = [
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ]
  for (let i = 0; i < units.length; i++) {
    const [div, suffix] = units[i]
    if (n < div) continue
    const v = n / div
    const text = v >= 100 ? String(Math.round(v)) : v.toFixed(1).replace(/\.0$/, '')
    // 999_960 would read "1000K"; say "1M" instead.
    if (Number(text) >= 1000 && i > 0) return `1${units[i - 1][1]}`
    return `${text}${suffix}`
  }
  return String(Math.round(n))
}

export function duration(seconds: number | null | undefined): string {
  const s = Math.max(0, Math.floor(seconds ?? 0))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  const rem = m % 60
  if (h < 24) return rem ? `${h}h ${rem}m` : `${h}h`
  return `${Math.floor(h / 24)}d ${h % 24}h`
}

export function clockTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false })
}

/** Round `value` to the nearest multiple of `step`, clamped to [min, max]. */
export function snap(value: number, step: number, min: number, max: number): number {
  const snapped = Math.round(value / step) * step
  return Math.min(max, Math.max(min, snapped))
}

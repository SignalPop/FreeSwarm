'use client'

import { useEffect, useState } from 'react'

/**
 * Published quality scores for the LLMs this console runs, so the model lists can be sorted
 * by them.
 *
 * - SWE: SWE-bench Verified, % of real GitHub issues resolved (coding ability).
 * - AA: Artificial Analysis Intelligence Index (artificialanalysis.ai), a composite of
 *   reasoning, knowledge, math and coding evals.
 *
 * Scores belong to the base model; quantized releases (NVFP4, FP8) are matched to it by name.
 * A null score means no published number was found. Nothing here is estimated. Each value
 * carries its source; update the table when newer scores are published.
 */

export type Rating = {
  key: string
  label: string
  /** Regex source, matched against the model's id / repo / path, lower-cased. First match wins. */
  match: string
  swe: number | null
  swe_source: string | null
  aa: number | null
  aa_source: string | null
}

// The table lives in the backend (app/ratings.py) so the swarm and every page read the same
// numbers. Fetched once per page load and shared by every component that asks.
let table: { rx: RegExp; r: Rating }[] = []
let aaVersion = ''
let loading: Promise<void> | null = null
const listeners = new Set<() => void>()

function load(): Promise<void> {
  loading ??= fetch('/api/ratings')
    .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
    .then((d: { ratings: Rating[]; aa_version: string }) => {
      table = d.ratings.map((r) => ({ rx: new RegExp(r.match), r }))
      aaVersion = d.aa_version
      listeners.forEach((f) => f())
    })
    .catch(() => {
      loading = null // retry on the next mount
    })
  return loading
}

export function ratingFor(...names: (string | null | undefined)[]): Rating | null {
  const hay = names.filter(Boolean).join(' ').toLowerCase()
  return table.find(({ rx }) => rx.test(hay))?.r ?? null
}

/** Re-renders the caller once the ratings table has arrived. */
export function useRatings(): typeof ratingFor {
  const [, bump] = useState(0)
  useEffect(() => {
    const f = () => bump((n) => n + 1)
    listeners.add(f)
    void load()
    return () => { listeners.delete(f) }
  }, [])
  return ratingFor
}

export type RatingSort = 'default' | 'swe' | 'aa'

/** Highest score first; models without that score keep their original order, after the rest. */
export function sortByRating<T>(items: T[], by: RatingSort, rating: (t: T) => Rating | null): T[] {
  if (by === 'default') return items
  const score = (t: T) => rating(t)?.[by] ?? null
  return items
    .map((t, i) => ({ t, i, s: score(t) }))
    .sort((a, b) => (a.s == null ? 1 : 0) - (b.s == null ? 1 : 0) || (b.s ?? 0) - (a.s ?? 0) || a.i - b.i)
    .map((x) => x.t)
}

export function SortByRating({ value, onChange }: { value: RatingSort; onChange: (v: RatingSort) => void }) {
  const opts: [RatingSort, string, string][] = [
    ['default', 'Default', 'Original order'],
    ['swe', 'SWE-bench', 'Coding: SWE-bench Verified, % of real GitHub issues resolved. Highest first.'],
    ['aa', 'AA Intelligence', 'Artificial Analysis Intelligence Index: overall reasoning, knowledge, math and coding. Highest first.'],
  ]
  return (
    <span className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-faint">
      Sort by
      <span className="inline-flex overflow-hidden rounded-lg border border-seam">
        {opts.map(([v, label, tip]) => (
          <button key={v} type="button" title={tip} onClick={(e) => { e.stopPropagation(); onChange(v) }}
            className={`px-2 py-0.5 font-mono text-[11px] ${value === v ? 'bg-accent/15 text-accent' : 'text-ink-dim hover:bg-panel-hi'}`}>
            {label}
          </button>
        ))}
      </span>
    </span>
  )
}

/** "SWE 64.2%  AA 45" chips; hover shows where each number comes from. */
export function RatingChips({ rating }: { rating: Rating | null }) {
  if (!rating || (rating.swe == null && rating.aa == null)) return null
  const chip = 'rounded-md border border-seam px-1.5 py-px font-mono text-[10.5px] text-ink-dim'
  return (
    <span className="inline-flex items-center gap-1">
      {rating.swe != null && (
        <span className={chip} title={`SWE-bench Verified (${rating.label}): ${rating.swe}% resolved\n${rating.swe_source ?? ''}`}>
          SWE {rating.swe}%
        </span>
      )}
      {rating.aa != null && (
        <span className={chip} title={`Artificial Analysis Intelligence Index${aaVersion ? ` ${aaVersion}` : ''} (${rating.label}): ${rating.aa}\n${rating.aa_source ?? ''}`}>
          AA {rating.aa}
        </span>
      )}
    </span>
  )
}

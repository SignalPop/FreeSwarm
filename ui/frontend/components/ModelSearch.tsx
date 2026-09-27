'use client'

import { useEffect, useState } from 'react'
import { api, type ModelKind, type ModelSearchResult, type ModelSearchSort } from '@/lib/api'
import { duration, humanCount } from '@/lib/format'
import { Button, Pill } from '@/components/ui'
import { RatingChips } from '@/lib/ratings'

const SORTS: [ModelSearchSort, string][] = [
  ['trending', 'trending'],
  ['new', 'newest'],
  ['recent', 'recently updated'],
  ['downloads', 'most downloaded'],
  ['likes', 'most liked'],
]

function params(n: number | null): string | null {
  if (!n) return null
  return n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B` : `${Math.round(n / 1e6)}M`
}

function ago(iso: string | null): string | null {
  return iso ? `${duration((Date.now() - Date.parse(iso)) / 1000)} ago` : null
}

/**
 * Search Hugging Face for LLMs or time-series forecasters, each marked with whether this install
 * can run it -- the engine's architectures and weight formats for an LLM, a forecaster adapter
 * for a time-series model -- and whether it is already downloaded or on the list. With the box
 * empty it shows what is trending: new models worth considering. "Add to list" puts one with the
 * known models (one-click download there); "Check" plans a download straight away.
 */
export default function ModelSearch({
  onListed,
  onCheck,
}: {
  /** The download list changed: re-read it. */
  onListed: () => void
  /** Plan a download of this repo in the "Another model" box. */
  onCheck: (repo: string) => void
}) {
  const [kind, setKind] = useState<ModelKind>('llm')
  const [q, setQ] = useState('')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<ModelSearchSort>('trending')
  const [runnable, setRunnable] = useState(true)
  const [hideHad, setHideHad] = useState(true)
  const [rows, setRows] = useState<ModelSearchResult[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [adding, setAdding] = useState<string | null>(null)
  const [added, setAdded] = useState<Set<string>>(new Set())

  // Typing settles for a moment before it searches; Enter searches at once.
  useEffect(() => {
    const t = setTimeout(() => setQuery(q.trim()), 450)
    return () => clearTimeout(t)
  }, [q])

  useEffect(() => {
    let live = true
    setLoading(true)
    setErr(null)
    api
      .searchModels({ q: query, kind, sort, runnable })
      .then((r) => live && setRows(r.results))
      .catch((e: Error) => live && setErr(e.message))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [query, kind, sort, runnable])

  async function add(r: ModelSearchResult) {
    setAdding(r.repo)
    setErr(null)
    try {
      await api.addToDownloadList(r.repo, kind, r.why)
      setAdded((s) => new Set([...s, r.repo]))
      onListed()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setAdding(null)
    }
  }

  const had = (r: ModelSearchResult) => r.installed || r.listed || added.has(r.repo)
  const shown = (rows ?? []).filter((r) => !hideHad || !had(r))
  const fresh = (rows ?? []).filter((r) => r.runs !== 'no' && !had(r)).length
  const seg = (on: boolean) =>
    `rounded px-2.5 py-1 ${on ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:text-ink-dim'}`

  return (
    <div className="rounded-xl border border-seam p-3">
      <div className="mb-2 flex flex-wrap items-baseline gap-2">
        <span className="text-[12.5px] font-medium text-ink">Find new models</span>
        <span className="text-[12px] text-ink-faint">
          — search Hugging Face; each result says whether it runs on this install
        </span>
        {rows && !loading && fresh > 0 && (
          <span className="ml-auto">
            <Pill tone="accent">
              {fresh} new to consider
            </Pill>
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex rounded-lg border border-seam p-0.5 font-mono text-[11px]">
          <button type="button" className={seg(kind === 'llm')} onClick={() => setKind('llm')}>
            LLMs
          </button>
          <button type="button" className={seg(kind === 'timeseries')} onClick={() => setKind('timeseries')}>
            time series
          </button>
        </span>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setQuery(q.trim())}
          placeholder={kind === 'llm' ? 'search, e.g. qwen, muse, nvfp4, gemma — or leave empty for trending' : 'search, e.g. chronos, kronos, timesfm — or leave empty for trending'}
          className="min-w-[240px] flex-1 rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
        />
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as ModelSearchSort)}
          className="rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[11.5px] text-ink"
        >
          {SORTS.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-4 text-[11.5px] text-ink-dim">
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={runnable} onChange={(e) => setRunnable(e.target.checked)} />
          only ones that run here
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={hideHad} onChange={(e) => setHideHad(e.target.checked)} />
          hide ones I have or listed
        </label>
        {loading && <span className="font-mono text-[11px] text-ink-faint">searching…</span>}
      </div>
      {err && <div className="mt-2 text-[12px] text-bad">✗ {err}</div>}

      {rows && (
        <div className="mt-2 space-y-1.5">
          {shown.length === 0 && !loading && (
            <div className="py-3 text-center text-[12px] text-ink-faint">
              Nothing {hideHad ? 'new ' : ''}matches{runnable ? ' that runs here' : ''}.{' '}
              {runnable && 'Untick "only ones that run here" to see everything.'}
            </div>
          )}
          {shown.map((r) => {
            const meta = [
              params(r.params),
              r.downloads != null && `${humanCount(r.downloads)} downloads`,
              r.likes != null && `${humanCount(r.likes)} likes`,
              ago(r.last_modified) && `updated ${ago(r.last_modified)}`,
            ].filter(Boolean)
            const isHad = had(r)
            return (
              <div key={r.repo} className="rounded-lg border border-seam/70 bg-panel-hi/30 px-3 py-2">
                <div className="flex flex-wrap items-center gap-2">
                  <a
                    href={`https://huggingface.co/${r.repo}`}
                    target="_blank"
                    rel="noreferrer"
                    className="font-mono text-[12.5px] text-ink hover:text-accent"
                  >
                    {r.repo}
                  </a>
                  <Pill tone={r.runs === 'yes' ? 'good' : r.runs === 'maybe' ? 'warn' : 'bad'}>
                    {r.runs === 'yes' ? 'runs here' : r.runs === 'maybe' ? 'may run' : "won't run"}
                  </Pill>
                  {kind === 'llm' &&
                    (r.swe != null || r.aa != null ? (
                      <RatingChips
                        rating={{ key: r.repo, label: r.rating_label ?? r.repo, match: '', swe: r.swe, swe_source: r.swe_source, aa: r.aa, aa_source: r.aa_source }}
                      />
                    ) : (
                      <span className="font-mono text-[10.5px] text-ink-faint" title="No published SWE-bench Verified or AA Intelligence Index for this model">
                        SWE — · AA —
                      </span>
                    ))}
                  {r.gated && <Pill tone="neutral">gated — needs a token</Pill>}
                  {r.installed && <Pill tone="good">downloaded</Pill>}
                  {(r.listed || added.has(r.repo)) && !r.installed && <Pill tone="accent">on your list</Pill>}
                  <span className="ml-auto flex items-center gap-2">
                    {!isHad && r.runs !== 'no' && (
                      <Button tone="primary" disabled={adding !== null} onClick={() => add(r)}>
                        {adding === r.repo ? 'Adding…' : 'Add to list'}
                      </Button>
                    )}
                    {!r.installed && (
                      <Button tone="ghost" onClick={() => onCheck(r.repo)}>
                        Check size
                      </Button>
                    )}
                  </span>
                </div>
                <div className="mt-0.5 font-mono text-[10.5px] text-ink-faint">
                  {meta.join(' · ')}
                </div>
                <div className={`mt-0.5 text-[11.5px] ${r.runs === 'no' ? 'text-ink-faint' : 'text-ink-dim'}`}>{r.why}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

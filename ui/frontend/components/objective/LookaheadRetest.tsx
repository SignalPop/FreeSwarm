'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button } from '@/components/ui'

type Run = { label: string; kind: string; state: string; started?: number; seconds?: number }
type Progress = { phase: string; cuts_done: number; cuts_total: number; runs: Run[] }

type Status = {
  total: number
  queue?: number[]
  progress?: Record<string, Progress>
  results: { seq: number; verdict: 'pass' | 'fail' | 'error'; detail: string }[]
  current?: number | null
  current_rank?: number | null
  started_at?: number
  done_at: number | null
  recrowned?: number | null
  cancel?: boolean
  cancelled?: boolean
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json' } })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body?.detail ?? `${res.status}`)
  return body as T
}

const mmss = (s: number) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`)

/**
 * Re-run the look-ahead test on the leaderboard with the dense cuts (placed just after each
 * candidate's own trades). Results scored under the older two-cut test may hide leaks; any
 * found are disqualified exactly as the harness would have at submission. Runs in the
 * background, one candidate at a time, and can be cancelled.
 */
export default function LookaheadRetest({ objectiveId, onChange }: { objectiveId: string; onChange: () => void }) {
  const [st, setSt] = useState<Status | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now() / 1000)
  const url = `/api/objectives/${encodeURIComponent(objectiveId)}/lookahead/retest`

  const load = useCallback(async () => {
    try {
      const s = await call<Status>(url)
      setSt((prev) => {
        if (prev && prev.results.length !== s.results.length) onChange()
        return s
      })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [url, onChange])

  const running = !!st && st.total > 0 && !st.done_at
  useEffect(() => {
    void load()
    const t = setInterval(load, running ? 3000 : 30_000)
    return () => clearInterval(t)
  }, [load, running])
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [running])

  async function start() {
    setErr(null)
    try {
      setSt(await call<Status>(url, { method: 'POST', body: JSON.stringify({ top: 25 }) }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }
  async function cancel() {
    setErr(null)
    try {
      setSt(await call<Status>(`${url}/cancel`, { method: 'POST' }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const done = st?.results.length ?? 0
  const total = st?.total ?? 0
  const candPct = (seq: number): number => {
    const r = st?.results.find((x) => x.seq === seq)
    if (r) return 100
    const p = st?.progress?.[String(seq)]
    if (!p || !p.runs?.length) return 0
    const finished = p.runs.filter((x) => !['queued', 'running'].includes(x.state)).length
    // The cut list is only known after the full run: count that as 1 of ~11 until then.
    const planned = p.runs.length > 1 ? p.runs.length : 11
    return Math.min(99, Math.round((finished / planned) * 100))
  }
  const inFlight = st?.current != null ? candPct(st.current) / 100 : 0
  const pct = total ? Math.round(((done + (running ? inFlight : 0)) / total) * 100) : 0
  const passed = st?.results.filter((r) => r.verdict === 'pass').length ?? 0
  const fails = st?.results.filter((r) => r.verdict === 'fail') ?? []
  const errors = st?.results.filter((r) => r.verdict === 'error').length ?? 0
  const elapsed = st?.started_at ? now - st.started_at : 0
  const eta = running && done + inFlight > 0.05 ? (elapsed / (done + inFlight)) * (total - done - inFlight) : null

  let status = ''
  if (running && st) {
    status = st.cancel
      ? 'Cancelling…'
      : st.current
        ? `Testing #${st.current} (rank ${st.current_rank} of ${total}) · ${done} of ${total} done · ${mmss(elapsed)} elapsed` +
          (eta != null ? ` · about ${mmss(eta)} left` : '')
        : `Starting · ${done} of ${total} done`
  } else if (st?.done_at) {
    status = `${st.cancelled ? 'Cancelled' : 'Finished'} after ${done} of ${total} candidates`
  }

  return (
    <div className="mb-2 rounded-lg border border-seam px-3 py-2 text-[12px] text-ink-dim">
      <div className="flex flex-wrap items-center gap-2">
        <span>
          Look-ahead test: every candidate is re-run with the data cut just after its own trades; any past decision
          that changes is a leak and disqualifies it.
        </span>
        <span className="ml-auto flex items-center gap-2">
          {running ? (
            <Button tone="danger" disabled={!!st?.cancel} onClick={cancel}>
              {st?.cancel ? 'Cancelling…' : 'Cancel'}
            </Button>
          ) : (
            <Button tone="ghost" onClick={start}>Re-test top 25</Button>
          )}
        </span>
      </div>

      {st && total > 0 && (
        <>
          <div className="mt-2 flex items-center gap-2">
            <div className="h-2 flex-1 overflow-hidden rounded bg-seam">
              <div className={`h-full transition-all ${fails.length ? 'bg-bad' : 'bg-accent'}`} style={{ width: `${pct}%` }} />
            </div>
            <span className="w-[42px] text-right font-mono text-[11.5px] text-ink">{pct}%</span>
          </div>
          <div className="mt-1 text-[11.5px]">{status}</div>
          <div className="mt-0.5 font-mono text-[11px]">
            {passed} passed · <span className={fails.length ? 'text-bad' : ''}>{fails.length} disqualified</span> · {errors} could not run
            {st.recrowned != null ? ` · new champion #${st.recrowned}` : ''}
          </div>

          {/* One chip per candidate: queued, its % while testing, then its verdict. */}
          <div className="mt-2 flex flex-wrap gap-1">
            {(st.queue ?? []).map((seq, i) => {
              const r = st.results.find((x) => x.seq === seq)
              const now_ = st.current === seq && running
              const p = candPct(seq)
              const tone = r
                ? r.verdict === 'pass' ? 'border-good/50 text-good' : r.verdict === 'fail' ? 'border-bad/60 bg-bad/10 text-bad' : 'border-warn/50 text-warn'
                : now_ ? 'border-accent/60 text-accent' : 'border-seam text-ink-faint'
              return (
                <span key={seq} title={r ? `${r.verdict}: ${r.detail}` : now_ ? 'testing now' : 'queued'}
                  className={`relative overflow-hidden rounded-md border px-1.5 py-0.5 font-mono text-[10.5px] ${tone}`}>
                  {now_ && <span className="absolute inset-y-0 left-0 bg-accent/15" style={{ width: `${p}%` }} />}
                  <span className="relative">
                    {i + 1}. #{seq} {r ? (r.verdict === 'pass' ? '✓' : r.verdict === 'fail' ? '✗ leak' : '? error') : `${p}%`}
                  </span>
                </span>
              )
            })}
          </div>

          {/* The runs of the candidate being tested: the full run, then every cut. */}
          {running && st.current != null && st.progress?.[String(st.current)] && (
            <div className="mt-2 rounded-md border border-seam/70 p-2">
              <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">runs for #{st.current}</div>
              {st.progress[String(st.current)].runs.map((r, i) => (
                <div key={i} className="flex items-center gap-2 font-mono text-[11px]">
                  <span className={r.state === 'running' ? 'text-accent' : r.state === 'pass' ? 'text-good' : r.state === 'fail' ? 'text-bad' : r.state === 'error' ? 'text-warn' : 'text-ink-faint'}>
                    {r.state === 'running' ? '◐' : r.state === 'pass' ? '✓' : r.state === 'fail' ? '✗' : r.state === 'error' ? '!' : r.state === 'skipped' ? '–' : '○'}
                  </span>
                  <span className="text-ink-dim">{r.label}</span>
                  <span className="text-ink-faint">{r.kind}</span>
                  <span className="ml-auto text-ink-faint">
                    {r.state === 'running' && r.started ? `${Math.round(now - r.started)}s…` : r.seconds != null ? `${r.seconds}s` : r.state}
                  </span>
                </div>
              ))}
              {st.progress[String(st.current)].runs.length === 1 && (
                <div className="font-mono text-[10.5px] text-ink-faint">the cuts are placed once the full run shows where this candidate trades</div>
              )}
            </div>
          )}
        </>
      )}
      {fails.map((r) => (
        <div key={r.seq} className="mt-0.5 text-[11px] text-bad">#{r.seq}: {r.detail}</div>
      ))}
      {err && <div className="mt-1 text-[11px] text-bad">✗ {err}</div>}
    </div>
  )
}

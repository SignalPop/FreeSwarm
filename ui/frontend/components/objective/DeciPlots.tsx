'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  insight,
  num,
  signed,
  VERDICT_TONE,
  type DeciBatch,
  type DeciCell,
  type DeciStudy,
  type DeciStudyRow,
} from '@/lib/insight'

/**
 * Decile studies ("deci-plots"): for one signal, the mean forward return in each of its ten
 * deciles, per timeframe and horizon, and whether the shape holds in every sub-period. All
 * in-sample, with ROLLING decile edges from past sessions only -- a full-sample qcut would put
 * next year's distribution into this morning's bucket. Studies are stored and never re-run,
 * and agents read the summary in their brief.
 */
export default function DeciPlots({ objectiveId }: { objectiveId: string }) {
  const [rows, setRows] = useState<DeciStudyRow[] | null>(null)
  const [batch, setBatch] = useState<DeciBatch | null>(null)
  const [split, setSplit] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [openId, setOpenId] = useState<number | null>(null)
  const [filter, setFilter] = useState('')
  const [sort, setSort] = useState<'recent' | 't'>('t')

  const load = useCallback(() => {
    insight
      .deciList(objectiveId)
      .then((d) => (setRows(d.studies), setBatch(d.batch), setSplit(d.split_date), setErr(null)))
      .catch((x) => setErr(x instanceof Error ? x.message : String(x)))
  }, [objectiveId])

  useEffect(() => {
    load()
    const t = setInterval(load, batch?.phase === 'running' ? 5000 : 15000)
    return () => clearInterval(t)
  }, [load, batch?.phase])

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase()
    const out = (rows ?? []).filter((r) => !f || r.signal.toLowerCase().includes(f) || r.summary.verdict.includes(f))
    if (sort === 't') out.sort((a, b) => Math.abs(b.summary.best?.t_spread ?? 0) - Math.abs(a.summary.best?.t_spread ?? 0))
    return out
  }, [rows, filter, sort])

  return (
    <div className="space-y-4">
      <RunForm objectiveId={objectiveId} batch={batch} onDone={(id) => (load(), id != null && setOpenId(id))} />
      {err && <div className="text-[12px] text-bad">✗ {err}</div>}
      {openId !== null && <StudyDetail id={openId} onClose={() => setOpenId(null)} />}
      <section>
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <div className="font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
            Studies ({rows?.length ?? '…'}) · in-sample{split ? ` before ${split}` : ''}
          </div>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter signal or verdict"
            className="ml-auto rounded border border-seam bg-panel-hi px-2 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
          />
          <button onClick={() => setSort(sort === 't' ? 'recent' : 't')} className="font-mono text-[10.5px] text-ink-faint hover:text-accent">
            sort: {sort === 't' ? 'strongest |t|' : 'newest'}
          </button>
        </div>
        {rows && rows.length === 0 ? (
          <div className="text-[12px] text-ink-faint">No studies yet — run one above, or let the agents (deci_plot) build them up.</div>
        ) : (
          <table className="w-full font-mono text-[11px]">
            <thead>
              <tr className="text-ink-faint">
                <th className="py-1 text-left font-normal">signal</th>
                <th className="py-1 text-left font-normal">verdict</th>
                <th className="py-1 text-left font-normal">best cell</th>
                <th className="py-1 text-right font-normal" title="mean fwd return, top decile minus bottom decile (bps)">spread</th>
                <th className="py-1 text-right font-normal" title="t of the spread, overlap-adjusted">t</th>
                <th className="py-1 text-right font-normal" title="Spearman of decile vs mean return">monotone</th>
                <th className="py-1 text-right font-normal" title="share of sub-periods with the same sign">stable</th>
                <th className="py-1 pl-3 text-left font-normal">by · when</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const b = r.summary.best
                return (
                  <tr
                    key={r.id}
                    onClick={() => setOpenId(r.id)}
                    className={`cursor-pointer border-t border-seam/60 hover:bg-panel-hi/60 ${openId === r.id ? 'bg-panel-hi/60' : ''}`}
                  >
                    <td className="py-1 pr-2 text-ink" title={r.signal}>
                      {r.signal.length > 48 ? `${r.signal.slice(0, 46)}…` : r.signal}
                    </td>
                    <td className={`py-1 ${VERDICT_TONE[r.summary.verdict] ?? ''}`}>{r.summary.verdict}</td>
                    <td className="py-1 text-ink-dim">{b ? `${b.timeframe} h${b.horizon}` : '—'}</td>
                    <td className={`py-1 text-right ${b?.spread_bps ? (b.spread_bps > 0 ? 'text-good' : 'text-bad') : 'text-ink-faint'}`}>
                      {signed(b?.spread_bps ?? null, 2)}
                    </td>
                    <td className="py-1 text-right text-ink">{num(b?.t_spread ?? null, 1)}</td>
                    <td className="py-1 text-right text-ink-dim">{num(b?.spearman ?? null, 2)}</td>
                    <td className="py-1 text-right text-ink-dim">
                      {b?.consistency != null ? `${Math.round(b.consistency * 100)}%` : '—'}
                    </td>
                    <td className="py-1 pl-3 text-ink-faint">
                      {r.author || '—'} · {stamp(r.ts)}
                      {r.runs && r.runs > 1 ? ` · ${r.runs} runs` : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}

const TIMEFRAMES = ['10s', '20s', '30s', '1min', '5min']

function RunForm({ objectiveId, batch, onDone }: { objectiveId: string; batch: DeciBatch | null; onDone: (id: number | null) => void }) {
  const [signal, setSignal] = useState('')
  const [tfs, setTfs] = useState<string[]>(TIMEFRAMES)
  const [horizons, setHorizons] = useState('1, 3, 6, 12')
  const [windowDays, setWindowDays] = useState(20)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [options, setOptions] = useState<string[]>([])

  useEffect(() => {
    insight
      .deciSignals(objectiveId)
      .then((d) => setOptions([...d.columns, ...d.features.flatMap((f) => f.columns.map((c) => `${f.view}:${c}`))]))
      .catch(() => undefined)
  }, [objectiveId])

  const hs = horizons
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0)

  async function run(force = false) {
    setBusy(true)
    setMsg(null)
    try {
      const s = await insight.deciRun(objectiveId, {
        signal: signal.trim(),
        timeframes: tfs,
        horizons: hs,
        window_days: windowDays,
        author: 'operator',
        force,
      })
      setMsg(s.cached ? 'Already studied — showing the stored study (no recomputation).' : 'Done.')
      onDone(s.id)
    } catch (x) {
      setMsg(`✗ ${x instanceof Error ? x.message : String(x)}`)
    } finally {
      setBusy(false)
    }
  }

  async function runAll() {
    setMsg(null)
    try {
      const b = await insight.deciBatch(objectiveId, { author: 'operator' })
      setMsg(
        b.total
          ? `Queued ${b.total} columns (${b.already_studied} already studied) — one sandbox run per ${8} columns, in the background.`
          : `Every column is already studied (${b.already_studied}).`,
      )
      onDone(null)
    } catch (x) {
      setMsg(`✗ ${x instanceof Error ? x.message : String(x)}`)
    }
  }

  const field = 'rounded border border-seam bg-panel-hi px-2 py-1 font-mono text-[11px] text-ink outline-none focus:border-accent'
  return (
    <section className="rounded-lg border border-seam p-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="min-w-[260px] flex-1 text-[10.5px] text-ink-faint">
          signal — a column, an expression (GEX / Pinning_TotalAbsGex), or fc_&lt;feature&gt;:&lt;column&gt;
          <input
            value={signal}
            list="deci-signal-options"
            onChange={(e) => setSignal(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && signal.trim() && !busy && run()}
            className={`${field} mt-0.5 block w-full`}
          />
          <datalist id="deci-signal-options">
            {options.map((o) => (
              <option key={o} value={o} />
            ))}
          </datalist>
        </label>
        <div className="flex gap-1 text-[10.5px]">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => setTfs(tfs.includes(tf) ? tfs.filter((x) => x !== tf) : TIMEFRAMES.filter((x) => x === tf || tfs.includes(x)))}
              className={`rounded border px-1.5 py-1 font-mono ${tfs.includes(tf) ? 'border-accent/50 text-accent' : 'border-seam text-ink-faint'}`}
            >
              {tf}
            </button>
          ))}
        </div>
        <label className="text-[10.5px] text-ink-faint">
          horizons (bars)
          <input value={horizons} onChange={(e) => setHorizons(e.target.value)} className={`${field} ml-1 w-24`} />
        </label>
        <label className="text-[10.5px] text-ink-faint" title="decile edges come from this many PAST sessions (never the current one)">
          window days
          <input type="number" min={2} max={250} value={windowDays} onChange={(e) => setWindowDays(+e.target.value)} className={`${field} ml-1 w-14`} />
        </label>
        <button
          disabled={busy || !signal.trim() || !tfs.length || !hs.length}
          onClick={() => run()}
          className="rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent hover:bg-accent/10 disabled:opacity-40"
        >
          {busy ? 'running…' : 'run study'}
        </button>
        <button
          disabled={busy || !signal.trim()}
          onClick={() => run(true)}
          title="Run again even though it is stored (history is kept)"
          className="font-mono text-[10.5px] text-ink-faint hover:text-accent disabled:opacity-40"
        >
          re-run
        </button>
        <button
          disabled={batch?.phase === 'running'}
          onClick={runAll}
          title="Study every numeric column not studied yet, with the default settings, one sandbox run at a time"
          className="rounded-md border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:border-accent/40 hover:text-accent disabled:opacity-40"
        >
          run all columns
        </button>
      </div>
      {msg && <div className="mt-1.5 text-[11.5px] text-ink-dim">{msg}</div>}
      {batch && (batch.phase === 'running' || batch.failed.length > 0 || batch.error) && (
        <div className="mt-1.5 font-mono text-[11px] text-ink-dim">
          {batch.phase === 'running' && <span className="mr-1 inline-block size-1.5 animate-dot rounded-full bg-accent" />}
          batch {batch.phase}: {batch.done}/{batch.total} studied
          {batch.current ? ` · now ${batch.current.slice(0, 120)}` : ''}
          {batch.failed.length > 0 && <span className="text-warn"> · {batch.failed.length} failed</span>}
          {batch.error && <span className="text-bad"> · {batch.error}</span>}
          {batch.phase === 'running' && (
            <button onClick={() => insight.deciBatchCancel(objectiveId).then(() => onDone(null))} className="ml-2 text-ink-faint hover:text-bad">
              cancel
            </button>
          )}
        </div>
      )}
    </section>
  )
}

function StudyDetail({ id, onClose }: { id: number; onClose: () => void }) {
  const [s, setS] = useState<DeciStudy | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tf, setTf] = useState<string | null>(null)
  const [h, setH] = useState<string | null>(null)

  useEffect(() => {
    setS(null)
    insight
      .deciStudy(id)
      .then((d) => {
        setS(d)
        const b = d.summary.best
        setTf(b?.timeframe ?? Object.keys(d.result.timeframes)[0] ?? null)
        setH(b ? String(b.horizon) : null)
      })
      .catch((x) => setErr(x instanceof Error ? x.message : String(x)))
  }, [id])

  if (err) return <div className="text-[12px] text-bad">✗ {err}</div>
  if (!s) return <div className="text-[12px] text-ink-faint">loading study…</div>
  const tfs = Object.keys(s.result.timeframes)
  const t = tf ? s.result.timeframes[tf] : null
  const hs = t ? Object.keys(t.horizons) : []
  const hh = h && hs.includes(h) ? h : hs[0]
  const cell = t && hh ? t.horizons[hh] : null

  return (
    <section className="rounded-lg border border-accent/30 bg-panel-hi/30 p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-[12.5px] text-ink">{s.signal}</span>
        <span className={`font-mono text-[11px] ${VERDICT_TONE[s.summary.verdict] ?? ''}`}>{s.summary.verdict}</span>
        {s.summary.direction && <span className="font-mono text-[11px] text-ink-dim">{s.summary.direction}</span>}
        <span className="font-mono text-[10.5px] text-ink-faint">
          {s.result.rows.toLocaleString()} rows before {s.result.cut ?? '—'} · deciles from the previous{' '}
          {s.window.bars ? `${s.window.bars} bars` : `${s.window.days} sessions`} · {s.author || '—'} · {stamp(s.ts)}
        </span>
        <button onClick={onClose} className="ml-auto font-mono text-[11px] text-ink-faint hover:text-accent">
          close
        </button>
      </div>

      <div className="mt-2 font-mono text-[10.5px] text-ink-faint">t of the top-minus-bottom spread · timeframe × horizon (click a cell)</div>
      <TGrid s={s} tf={tf} h={hh} onPick={(a, b) => (setTf(a), setH(b))} />

      {t && cell && (
        <>
          <div className="mt-3 flex flex-wrap items-baseline gap-3 font-mono text-[11px]">
            <span className="text-ink">
              {tf} bars · {hh} bar{hh === '1' ? '' : 's'} ahead
            </span>
            <span className={VERDICT_TONE[cell.verdict] ?? ''}>{cell.verdict}</span>
            <span className="text-ink-dim">
              spread {signed(cell.spread_bps, 3)} bps (t {num(cell.t_spread, 1)}) · Spearman {num(cell.spearman, 2)} · same sign in{' '}
              {cell.consistency != null ? `${Math.round(cell.consistency * 100)}%` : '—'} of periods · n {cell.n.toLocaleString()}
            </span>
            <span className="text-ink-faint">
              {t.bucketed.toLocaleString()} of {t.bars.toLocaleString()} bars bucketed (warm-up until {t.warmup_until ?? '—'})
            </span>
          </div>
          <DecileBars cell={cell} />
          <div className="mt-2 font-mono text-[10.5px] text-ink-faint">the same, in each sub-period — a real relationship keeps its shape</div>
          <div className="grid gap-2 sm:grid-cols-3">
            {cell.periods.map((p) => (
              <div key={p.from}>
                <div className="font-mono text-[10px] text-ink-faint">
                  {p.from} → {p.to} · {signed(p.spread_bps, 2)} bps (t {num(p.t_spread, 1)})
                </div>
                <MiniBars values={p.mean_by_decile} />
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

const W = 640

function stamp(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

/** Mean forward return per decile, centred on zero; hover a bar for n, median, hit rate, t. */
function DecileBars({ cell }: { cell: DeciCell }) {
  const H = 190
  const pad = { l: 48, r: 10, t: 12, b: 30 }
  const vals = cell.deciles.map((d) => d.mean_bps ?? 0)
  const ext = Math.max(1e-9, ...vals.map(Math.abs))
  const y = (v: number) => pad.t + (1 - (v + ext) / (2 * ext)) * (H - pad.t - pad.b)
  const bw = (W - pad.l - pad.r) / 10
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" role="img" aria-label="mean forward return by decile">
      <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} className="stroke-ink-faint" strokeWidth={1} />
      <text x={pad.l - 6} y={y(ext) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {signed(ext, 3)}
      </text>
      <text x={pad.l - 6} y={y(0) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        0 bps
      </text>
      <text x={pad.l - 6} y={y(-ext) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {signed(-ext, 3)}
      </text>
      {cell.deciles.map((d, i) => {
        const v = d.mean_bps ?? 0
        const x = pad.l + i * bw + 2
        const top = Math.min(y(v), y(0))
        const weak = d.t === null || Math.abs(d.t) < 2
        return (
          <g key={d.decile}>
            <rect
              x={x}
              y={top}
              width={bw - 4}
              height={Math.max(1, Math.abs(y(v) - y(0)))}
              rx={3}
              className={v >= 0 ? 'fill-good' : 'fill-bad'}
              opacity={weak ? 0.45 : 0.9}
            >
              <title>
                {`decile ${d.decile}: mean ${signed(d.mean_bps, 4)} bps, median ${signed(d.median_bps, 4)}, hit ${num(d.hit, 3)}, t ${num(d.t, 2)}, n ${d.n}`}
              </title>
            </rect>
            <text x={x + (bw - 4) / 2} y={H - pad.b + 13} textAnchor="middle" className="fill-ink-dim font-mono text-[9px]">
              D{d.decile}
            </text>
            <text x={x + (bw - 4) / 2} y={H - pad.b + 24} textAnchor="middle" className="fill-ink-faint font-mono text-[8px]">
              {d.hit != null ? `${Math.round(d.hit * 100)}%` : ''}
            </text>
          </g>
        )
      })}
      <text x={W - pad.r} y={10} textAnchor="end" className="fill-ink-faint font-mono text-[8.5px]">
        low signal → high · faded = |t| &lt; 2 · % = hit rate
      </text>
    </svg>
  )
}

function MiniBars({ values }: { values: (number | null)[] }) {
  const H = 60
  const w = 200
  const vals = values.map((v) => v ?? 0)
  const ext = Math.max(1e-9, ...vals.map(Math.abs))
  const y = (v: number) => 4 + (1 - (v + ext) / (2 * ext)) * (H - 8)
  const bw = w / 10
  return (
    <svg viewBox={`0 0 ${w} ${H}`} className="w-full" role="img" aria-label="decile means in a sub-period">
      <line x1={0} x2={w} y1={y(0)} y2={y(0)} className="stroke-seam" />
      {vals.map((v, i) => (
        <rect
          key={i}
          x={i * bw + 1}
          y={Math.min(y(v), y(0))}
          width={bw - 2}
          height={Math.max(0.5, Math.abs(y(v) - y(0)))}
          rx={1.5}
          className={v >= 0 ? 'fill-good' : 'fill-bad'}
          opacity={0.8}
        >
          <title>{`decile ${i + 1}: ${signed(values[i], 4)} bps`}</title>
        </rect>
      ))}
    </svg>
  )
}

/** Timeframe × horizon: the t of the spread, coloured by its sign, stronger = more opaque. */
function TGrid({ s, tf, h, onPick }: { s: DeciStudy; tf: string | null; h: string | undefined; onPick: (tf: string, h: string) => void }) {
  const tfs = Object.keys(s.result.timeframes)
  const hs = Array.from(new Set(tfs.flatMap((k) => Object.keys(s.result.timeframes[k].horizons)))).sort((a, b) => +a - +b)
  return (
    <table className="mt-1 font-mono text-[10.5px]">
      <thead>
        <tr className="text-ink-faint">
          <th className="pr-2 text-left font-normal">bars \ ahead</th>
          {hs.map((x) => (
            <th key={x} className="px-1 font-normal">
              h{x}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {tfs.map((k) => (
          <tr key={k}>
            <td className="pr-2 text-ink-dim">{k}</td>
            {hs.map((x) => {
              const c = s.result.timeframes[k].horizons[x]
              const tv = c?.t_spread ?? null
              const on = k === tf && x === h
              const strength = tv === null ? 0 : Math.min(1, Math.abs(tv) / 6)
              return (
                <td key={x} className="p-0.5">
                  <button
                    onClick={() => onPick(k, x)}
                    title={c ? `${c.verdict}: spread ${signed(c.spread_bps, 3)} bps, t ${num(tv, 2)}, Spearman ${num(c.spearman, 2)}` : 'no data'}
                    className={`w-14 rounded px-1 py-0.5 text-right ${on ? 'ring-1 ring-accent' : ''} ${
                      tv === null ? 'text-ink-faint' : Math.abs(tv) >= 2 ? 'text-ink' : 'text-ink-dim'
                    }`}
                    style={{
                      background:
                        tv === null
                          ? undefined
                          : `color-mix(in srgb, var(--color-${tv >= 0 ? 'good' : 'bad'}) ${Math.round(8 + strength * 40)}%, transparent)`,
                    }}
                  >
                    {num(tv, 1)}
                  </button>
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

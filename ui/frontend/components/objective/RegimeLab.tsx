'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/ui'
import {
  preview,
  regimeLab,
  sameRoutes,
  shortLabel,
  type Eligible,
  type LabResult,
  type Overview,
  type Routes,
  type Run,
  type Seg,
  type Split,
} from '@/lib/regimes'

/**
 * The Regime Lab: which verified candidate works in which market regime, and a router that
 * trades each regime with the one that works there.
 *
 * Colour carries one meaning per job: a MEMBER (a candidate) always wears its series colour, in
 * the run's member order; a SHARPE is blue above zero and red below, through a grey "nothing".
 * Every number an agent can see is in-sample; the holdout columns are the operator's check on
 * whether a regime edge persisted -- routes are never chosen from them.
 */

const memberColor = (i: number) => (i >= 0 && i < 8 ? `var(--color-series-${i + 1})` : 'var(--color-ink-faint)')
const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(d))
const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)
const ago = (ts: number) => {
  const s = Date.now() / 1000 - ts
  return s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)}m ago` : s < 129600 ? `${Math.round(s / 3600)}h ago` : `${Math.round(s / 86400)}d ago`
}

/** Blue above zero, red below, grey at zero; saturates at |sharpe| = 4, never past 70%. */
function divColor(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'var(--div-mid)'
  const p = Math.round(Math.min(1, Math.abs(v) / 4) * 70)
  return `color-mix(in oklab, ${v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)'} ${p}%, var(--div-mid))`
}

type SegKey = 'is' | 'a' | 'b' | 'ho'
/** Labels that are not a regime: the warm-up before a field has history, a detector's unknown. */
const NOT_REGIME = new Set(['warmup', 'unknown', 'nan', 'None', ''])
const SEG_LABEL: Record<SegKey, string> = { is: 'in-sample', a: '1st half', b: '2nd half', ho: 'holdout' }

export default function RegimeLab({ objectiveId }: { objectiveId: string }) {
  const [ov, setOv] = useState<Overview | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [runId, setRunId] = useState<number | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const lastJobRun = useRef<number | null>(null)

  const load = useCallback(() => {
    regimeLab
      .overview(objectiveId)
      .then((o) => {
        setOv(o)
        setErr(null)
        // A run that just finished is the one the operator wants to see.
        const fresh = o.job?.phase === 'done' ? o.job.run_id : null
        if (fresh && fresh !== lastJobRun.current) {
          lastJobRun.current = fresh
          setRunId(fresh)
        } else setRunId((cur) => cur ?? o.runs[0]?.id ?? null)
      })
      .catch((x) => setErr(x instanceof Error ? x.message : String(x)))
  }, [objectiveId])

  const running = ov?.job?.phase === 'running'
  useEffect(() => {
    load()
    const t = setInterval(load, running ? 2000 : 30000)
    return () => clearInterval(t)
  }, [load, running])

  useEffect(() => {
    if (runId === null) return
    let alive = true
    regimeLab
      .run(runId)
      .then((r) => alive && setRun(r))
      .catch((x) => alive && setErr(x instanceof Error ? x.message : String(x)))
    return () => {
      alive = false
    }
  }, [runId])

  async function start(body: { split: Split; members?: number[] | null; routes?: Routes | null }) {
    try {
      setErr(null)
      await regimeLab.start(objectiveId, body)
      load()
    } catch (x) {
      setErr(x instanceof Error ? x.message : String(x))
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-[12px] leading-relaxed text-ink-dim">
        Which verified candidate works in which market regime. Regimes are fields ranked causally against their own
        recent history (or a library detector); every member&apos;s net P&amp;L is split by the regime in force. The router
        starts from the best single member and switches a regime to another member only when it wins in{' '}
        <span className="text-ink">both</span> in-sample halves. Agents see the in-sample numbers and the router script
        (tool <span className="font-mono">regime_lab</span>); holdout columns are for you.
      </p>
      {ov && <LabForm ov={ov} running={running} onRun={start} />}
      {ov?.job && ov.job.phase !== 'done' && <JobBar job={ov.job} />}
      {err && <div className="text-[12px] text-bad">✗ {err}</div>}
      {ov && ov.runs.length > 0 && <RunList runs={ov.runs} selected={runId} onPick={setRunId} />}
      {run?.result && run.id === runId && (
        <RunView key={run.id} run={run} result={run.result} running={running} onMeasure={(routes) =>
          start({ split: run.spec, members: run.members, routes })} />
      )}
    </div>
  )
}

// =========================================================================================
// Starting a run
// =========================================================================================
function LabForm({ ov, running, onRun }: { ov: Overview; running: boolean; onRun: (b: { split: Split; members: number[] | null }) => void }) {
  const [kind, setKind] = useState<'fields' | 'module'>('fields')
  const [fa, setFa] = useState(ov.default_fields[0] ?? ov.fields[0] ?? '')
  const [na, setNa] = useState(3)
  const [fb, setFb] = useState(ov.default_fields[1] ?? '')
  const [nb, setNb] = useState(3)
  const [smooth, setSmooth] = useState(360)
  const [windowDays, setWindowDays] = useState(20)
  const [mod, setMod] = useState(ov.modules[0] ?? '')
  const [members, setMembers] = useState<number[] | null>(null)
  const [picking, setPicking] = useState(false)
  const chosen = members ?? ov.default_members
  const bySeq = useMemo(() => new Map(ov.eligible.map((e) => [e.seq, e])), [ov.eligible])
  const secs = chosen.reduce((s, q) => s + (bySeq.get(q)?.eval_seconds ?? 0), 0)
  const slow = ov.eval_timeout_s ? secs > ov.eval_timeout_s * 0.6 : false

  const select = 'rounded border border-seam bg-panel-hi px-1.5 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent'
  const split: Split =
    kind === 'module'
      ? { kind: 'module', module: mod }
      : { kind: 'fields', fields: [{ field: fa, n: na }, ...(fb && fb !== fa ? [{ field: fb, n: nb }] : [])], smooth, window_days: windowDays }

  return (
    <div className="rounded-xl border border-seam bg-panel-hi/30 p-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[11px] text-ink-dim">
        <div className="flex overflow-hidden rounded border border-seam font-mono text-[10.5px]">
          {(['fields', 'module'] as const).map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              disabled={k === 'module' && !ov.modules.length}
              className={`px-2 py-0.5 ${kind === k ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:text-ink-dim'} disabled:opacity-40`}
            >
              {k === 'fields' ? 'split by fields' : 'library detector'}
            </button>
          ))}
        </div>
        {kind === 'fields' ? (
          <>
            <span className="flex flex-wrap items-center gap-1.5">
              <FieldPick label="x" value={fa} n={na} fields={ov.fields} onField={setFa} onN={setNa} select={select} />
              <span className="text-ink-faint">×</span>
              <FieldPick label="y" value={fb} n={nb} fields={ov.fields} onField={setFb} onN={setNb} select={select} optional />
            </span>
            <label className="flex items-center gap-1" title="trailing mean over this many bars before ranking (360 = 1 hour of 10 s bars); 0 = raw">
              smooth
              <input type="number" min={0} value={smooth} onChange={(e) => setSmooth(Math.max(0, +e.target.value || 0))} className={`${select} w-16`} />
              bars
            </label>
            <label className="flex items-center gap-1" title="each field is ranked against its own trailing window of this many sessions">
              rank vs last
              <input type="number" min={2} max={250} value={windowDays} onChange={(e) => setWindowDays(Math.min(250, Math.max(2, +e.target.value || 20)))} className={`${select} w-14`} />
              days
            </label>
          </>
        ) : (
          <select value={mod} onChange={(e) => setMod(e.target.value)} className={select}>
            {ov.modules.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-dim">
        <span className="text-ink-faint">members{members ? '' : ' (auto: best ranked, no near-copies)'}:</span>
        {chosen.map((s) => (
          <span key={s} className="rounded border border-seam px-1.5 font-mono text-[10.5px] text-ink">
            #{s}
          </span>
        ))}
        <button onClick={() => setPicking(!picking)} className="font-mono text-[10.5px] text-accent hover:underline">
          {picking ? 'done' : 'choose…'}
        </button>
        {members && (
          <button onClick={() => setMembers(null)} className="font-mono text-[10.5px] text-ink-faint hover:text-ink-dim">
            auto
          </button>
        )}
        <span className="ml-auto flex items-center gap-2">
          {slow && (
            <span className="font-mono text-[10px] text-warn" title="a router runs every member inside its own script">
              ⚠ members take ~{Math.round(secs)} s — the router may time out when scored
            </span>
          )}
          <Button tone="primary" disabled={running || chosen.length < 1 || (kind === 'fields' && !fa) || (kind === 'module' && !mod)} onClick={() => onRun({ split, members })}>
            {running ? 'Running…' : 'Run'}
          </Button>
        </span>
      </div>
      {picking && <MemberPicker eligible={ov.eligible} chosen={chosen} onChange={setMembers} />}
      <div className="mt-1.5 font-mono text-[10px] text-ink-faint">
        The first run replays each member once (5–90 s each); after that a new split takes seconds.
      </div>
    </div>
  )
}

function FieldPick({ label, value, n, fields, onField, onN, select, optional = false }: {
  label: string; value: string; n: number; fields: string[]; onField: (f: string) => void; onN: (n: number) => void; select: string; optional?: boolean
}) {
  return (
    <span className="flex items-center gap-1">
      <span className="font-mono text-[10px] text-ink-faint">{label}</span>
      <select value={value} onChange={(e) => onField(e.target.value)} className={`${select} max-w-[160px]`}>
        {optional && <option value="">— none —</option>}
        {fields.map((f) => (
          <option key={f}>{f}</option>
        ))}
      </select>
      <select value={n} onChange={(e) => onN(+e.target.value)} disabled={optional && !value} className={select} title="buckets">
        {[2, 3, 4, 5].map((k) => (
          <option key={k} value={k}>
            {k === 2 ? 'lo/hi' : k === 3 ? 'lo/mid/hi' : `${k} bins`}
          </option>
        ))}
      </select>
    </span>
  )
}

function MemberPicker({ eligible, chosen, onChange }: { eligible: Eligible[]; chosen: number[]; onChange: (m: number[]) => void }) {
  const set = new Set(chosen)
  return (
    <div className="mt-2 max-h-[220px] overflow-y-auto rounded border border-seam">
      <table className="w-full font-mono text-[10.5px]">
        <thead className="sticky top-0 bg-panel text-ink-faint">
          <tr>
            <th className="w-6" />
            <th className="py-1 text-left font-normal">#</th>
            <th className="py-1 text-right font-normal">score</th>
            <th className="py-1 text-right font-normal">in-sample</th>
            <th className="py-1 text-right font-normal">run s</th>
            <th className="py-1 pl-3 text-left font-normal">hypothesis</th>
          </tr>
        </thead>
        <tbody>
          {eligible.map((e) => {
            const on = set.has(e.seq)
            return (
              <tr key={e.seq} className="border-t border-seam/50 text-ink-dim hover:bg-panel-hi/60">
                <td className="text-center">
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={!on && chosen.length >= 8}
                    onChange={() => onChange(on ? chosen.filter((s) => s !== e.seq) : [...chosen, e.seq])}
                  />
                </td>
                <td className="py-0.5 text-ink">{e.seq}</td>
                <td className="py-0.5 text-right">{fmt(e.score)}</td>
                <td className="py-0.5 text-right">{fmt(e.is_score)}</td>
                <td className={`py-0.5 text-right ${(e.eval_seconds ?? 0) > 120 ? 'text-warn' : ''}`}>{fmt(e.eval_seconds, 0)}</td>
                <td className="max-w-[360px] truncate py-0.5 pl-3 font-sans text-[11px]" title={e.rationale}>
                  <span className="text-ink-faint">{e.model} · </span>
                  {e.rationale}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function JobBar({ job }: { job: NonNullable<Overview['job']> }) {
  const p = job.total ? Math.round((job.done / job.total) * 100) : 0
  return (
    <div className="rounded-lg border border-seam p-2 font-mono text-[11px]">
      {job.phase === 'running' ? (
        <>
          <div className="flex justify-between text-ink-dim">
            <span>{job.step ?? 'starting'}</span>
            <span className="text-ink-faint">{job.describe}</span>
          </div>
          <div className="mt-1 h-1 overflow-hidden rounded bg-seam">
            <div className="h-full rounded bg-accent transition-all" style={{ width: `${Math.max(4, p)}%` }} />
          </div>
        </>
      ) : (
        <div className="text-bad">✗ {job.error}</div>
      )}
      {Object.entries(job.errors).map(([s, e]) => (
        <div key={s} className="mt-1 text-warn">
          #{s}: {e}
        </div>
      ))}
    </div>
  )
}

function RunList({ runs, selected, onPick }: { runs: Run[]; selected: number | null; onPick: (id: number) => void }) {
  const [all, setAll] = useState(false)
  return (
    <div>
      <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">Runs ({runs.length})</div>
      <table className="w-full font-mono text-[10.5px]">
        <thead className="text-ink-faint">
          <tr>
            <th className="py-0.5 text-left font-normal">regimes</th>
            <th className="py-0.5 text-right font-normal" title="router Sharpe in-sample / holdout">router IS · HO</th>
            <th className="py-0.5 text-right font-normal" title="the best single member alone, in-sample / holdout">best single IS · HO</th>
            <th className="py-0.5 text-right font-normal">routed</th>
            <th className="py-0.5 text-right font-normal" />
          </tr>
        </thead>
        <tbody>
          {(all ? runs : runs.slice(0, 5)).map((r) => {
            const s = r.summary
            const beats = (s.router.is ?? -9) > (s.best_single.is ?? -9)
            return (
              <tr
                key={r.id}
                onClick={() => onPick(r.id)}
                className={`cursor-pointer border-t border-seam/50 ${r.id === selected ? 'bg-accent/10 text-ink' : 'text-ink-dim hover:bg-panel-hi/60'}`}
              >
                <td className="max-w-[300px] truncate py-1" title={s.describe}>
                  <span className="text-ink-faint">{r.id} · </span>
                  {r.spec.kind === 'fields' ? r.spec.fields.map((f) => `${f.field}×${f.n}`).join(' · ') : r.spec.module}
                  {s.routes_source === 'custom' && <span className="text-ink-faint"> · custom routes</span>}
                </td>
                <td className="py-1 text-right">
                  <span className={beats ? 'text-ink' : ''}>{fmt(s.router.is)}</span> · {fmt(s.router.ho)}
                </td>
                <td className="py-1 text-right">
                  #{s.best_single.seq} {fmt(s.best_single.is)} · {fmt(s.best_single.ho)}
                </td>
                <td className="py-1 text-right">
                  {s.routed}/{s.regimes}
                </td>
                <td className="py-1 text-right text-ink-faint">
                  {r.candidate_seq ? <span className="text-accent">→ #{r.candidate_seq} · </span> : null}
                  {r.author === 'operator' ? '' : `${r.author.split('-')[0]} · `}
                  {ago(r.ts)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {runs.length > 5 && (
        <button onClick={() => setAll(!all)} className="mt-1 font-mono text-[10.5px] text-ink-faint hover:text-accent">
          {all ? 'show 5' : `show all ${runs.length}`}
        </button>
      )}
    </div>
  )
}

// =========================================================================================
// One run
// =========================================================================================
function RunView({ run, result: r, running, onMeasure }: { run: Run; result: LabResult; running: boolean; onMeasure: (routes: Routes) => void }) {
  const [routes, setRoutes] = useState<Routes>(r.routes)
  const [focus, setFocus] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState<number | null>(run.candidate_seq)
  const [submitErr, setSubmitErr] = useState<string | null>(null)
  const [showCode, setShowCode] = useState(false)
  const dirty = !sameRoutes(routes, r.routes)
  const idx = useMemo(() => new Map(r.members.map((m, i) => [m.seq, i])), [r.members])
  const pv = useMemo(() => (dirty ? preview(r, routes) : null), [dirty, r, routes])
  const stats = pv ? pv.stats : r.router
  const contrib = pv ? pv.contrib : r.daily.contrib
  const routerDaily = pv ? pv.router : r.daily.router
  const best = r.best_single

  function route(label: string, seq: number) {
    setRoutes((cur) => ({ ...cur, [label]: cur[label] === seq ? null : seq }))
  }

  async function submit() {
    try {
      setSubmitErr(null)
      const out = await regimeLab.submit(run.id, dirty ? routes : null)
      setSubmitted(out.seq)
    } catch (x) {
      setSubmitErr(x instanceof Error ? x.message : String(x))
    }
  }

  return (
    <div className="space-y-4 border-t border-seam pt-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <div className="text-[13px] text-ink">{r.describe}</div>
        <div className="font-mono text-[10.5px] text-ink-faint">
          {r.switches_per_day !== null && `${fmt(r.switches_per_day, 1)} regime changes / day · `}
          costs {r.cost_bps} bps · in-sample to {r.split_date ?? 'end'} (halves split {r.mid_date})
        </div>
      </div>

      {/* The headline: does routing beat simply trading the best member everywhere? */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile label={`router · in-sample${pv ? ' (preview)' : ''}`} value={fmt(stats.is.sharpe)} sub={`halves ${fmt(stats.a.sharpe)} / ${fmt(stats.b.sharpe)}`}
          tone={(stats.is.sharpe ?? -9) > (best.is?.sharpe ?? -9) ? 'good' : 'dim'} />
        <Tile label={`router · holdout${pv ? ' (preview)' : ''}`} value={fmt(stats.ho.sharpe)} sub="operator only"
          tone={(stats.ho.sharpe ?? -9) > (best.ho?.sharpe ?? -9) ? 'good' : 'dim'} />
        <Tile label={`best single · #${best.seq ?? '—'}`} value={`${fmt(best.is?.sharpe)} · ${fmt(best.ho?.sharpe)}`} sub="in-sample · holdout" tone="dim" />
        <Tile label="regimes routed" value={`${Object.values(routes).filter((v) => v != null).length} / ${r.regimes.filter((g) => !NOT_REGIME.has(g.label)).length}`}
          sub={`${r.members.length} members${Object.keys(r.errors).length ? ` · ${Object.keys(r.errors).length} failed` : ''}`} tone="dim" />
      </div>
      {pv && (
        <div className="rounded border border-warn/30 bg-warn/5 px-2 py-1 text-[11px] text-warn">
          Preview of your edited routes: it leaves out the trade made at each regime change from one member&apos;s position to
          another&apos;s — at {fmt(r.switches_per_day, 1)} changes a day that cost is real. <b>Measure exactly</b> before trusting it.
        </div>
      )}

      <Legend r={r} />

      <section>
        <SectionTitle>Regime map</SectionTitle>
        <RegimeMap r={r} routes={routes} idx={idx} focus={focus} onFocus={setFocus} />
      </section>

      <section>
        <SectionTitle>Every member in every regime · click a cell to route that regime to it (again: flat)</SectionTitle>
        <Matrix r={r} routes={routes} idx={idx} focus={focus} onFocus={setFocus} onRoute={route} />
      </section>

      <section>
        <SectionTitle>Where the router&apos;s P&amp;L came from</SectionTitle>
        <ContributionChart r={r} routes={routes} contrib={contrib} router={routerDaily} idx={idx} />
      </section>

      <div className="flex flex-wrap items-center gap-2">
        {dirty && (
          <>
            <Button tone="ghost" onClick={() => setRoutes(r.routes)}>Reset routes</Button>
            <Button tone="ghost" disabled={running} onClick={() => onMeasure(routes)}>Measure exactly</Button>
          </>
        )}
        {!dirty && r.routes_source === 'custom' && (
          <Button tone="ghost" disabled={running} onClick={() => onMeasure(r.suggested)}>Back to suggested routes</Button>
        )}
        <Button tone="primary" disabled={!Object.values(routes).some((v) => v != null)} onClick={submit}>
          Submit router as a candidate
        </Button>
        {submitted && <span className="font-mono text-[11px] text-accent">submitted as #{submitted} — scored like any candidate (see Recent)</span>}
        {submitErr && <span className="text-[11px] text-bad">✗ {submitErr}</span>}
        {r.code && (
          <button onClick={() => setShowCode(!showCode)} className="ml-auto font-mono text-[10.5px] text-ink-faint hover:text-accent">
            {showCode ? 'hide' : 'show'} router script{dirty ? ' (as measured)' : ''}
          </button>
        )}
      </div>
      {showCode && r.code && (
        <pre className="max-h-[320px] overflow-auto rounded border border-seam bg-canvas p-2 font-mono text-[10.5px] leading-relaxed text-ink-dim">{r.code}</pre>
      )}
      {Object.keys(r.errors).length > 0 && (
        <div className="font-mono text-[10.5px] text-warn">
          {Object.entries(r.errors).map(([s, e]) => (
            <div key={s}>
              #{s} left out: {e}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="mb-1.5 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{children}</div>
}

function Tile({ label, value, sub, tone }: { label: string; value: string; sub: string; tone: 'good' | 'dim' }) {
  return (
    <div className="rounded-xl border border-seam bg-panel-hi/40 p-2.5">
      <div className="truncate text-[10px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className={`mt-0.5 font-mono text-[17px] ${tone === 'good' ? 'text-good' : 'text-ink'}`}>{value}</div>
      <div className="truncate font-mono text-[10px] text-ink-faint">{sub}</div>
    </div>
  )
}

function Legend({ r }: { r: LabResult }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
      {r.members.map((m, i) => (
        <span key={m.seq} className="flex items-center gap-1.5" title={`${m.model}: ${m.rationale}`}>
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: memberColor(i) }} />
          <span className="font-mono text-ink">#{m.seq}</span>
          <span className="font-mono text-[10px] text-ink-faint">
            {fmt(r.member_stats[String(m.seq)]?.is.sharpe)} · {fmt(r.member_stats[String(m.seq)]?.ho.sharpe)}
          </span>
        </span>
      ))}
      <span className="flex items-center gap-1.5 text-ink-faint">
        <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'repeating-linear-gradient(45deg, var(--color-seam) 0 2px, transparent 2px 4px)' }} />
        flat
      </span>
      <span className="ml-auto flex items-center gap-1 font-mono text-[10px] text-ink-faint">
        Sharpe
        <span className="inline-block h-2 w-16 rounded-sm" style={{ background: 'linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))' }} />
        −4 … +4
      </span>
    </div>
  )
}

// -----------------------------------------------------------------------------------------
// Regime map: the regimes laid out as the grid they are (x field low -> high, y field high at
// the top). Each tile wears the colour of the member it is routed to, deeper the better that
// member did there in its WORSE in-sample half, and shows how that member (coloured) and the
// others (grey) earned inside the regime over time -- in-sample left of the dashed line.
// -----------------------------------------------------------------------------------------
function RegimeMap({ r, routes, idx, focus, onFocus }: {
  r: LabResult; routes: Routes; idx: Map<number, number>; focus: string | null; onFocus: (l: string | null) => void
}) {
  const byLabel = useMemo(() => new Map(r.regimes.map((g) => [g.label, g])), [r.regimes])
  const splitAt = useMemo(() => r.dates.findIndex((d) => r.split_date !== null && d >= r.split_date), [r.dates, r.split_date])
  const tile = (label: string) =>
    byLabel.has(label) ? (
      <RegimeTile key={label} r={r} label={label} seq={routes[label] ?? null} idx={idx} splitAt={splitAt}
        focused={focus === label} onFocus={() => onFocus(focus === label ? null : label)} />
    ) : (
      <div key={label} className="rounded-lg border border-dashed border-seam p-2 text-[10px] text-ink-faint">never occurred</div>
    )

  if (r.axes.length === 2) {
    const [ax, ay] = r.axes
    return (
      <div className="flex gap-2">
        <div className="flex w-5 flex-col items-center justify-around">
          <span className="rotate-180 font-mono text-[10px] text-ink-faint [writing-mode:vertical-rl]">{ay.field} →</span>
        </div>
        <div className="flex-1">
          <div className="grid gap-1.5" style={{ gridTemplateColumns: `auto repeat(${ax.buckets.length}, minmax(0, 1fr))` }}>
            {[...ay.buckets].reverse().map((by) => (
              <div key={by} className="contents">
                <div className="flex items-center pr-1 font-mono text-[10px] text-ink-faint">{by}</div>
                {ax.buckets.map((bx) => tile(`${ax.field}:${bx}|${ay.field}:${by}`))}
              </div>
            ))}
            <div />
            {ax.buckets.map((bx) => (
              <div key={bx} className="text-center font-mono text-[10px] text-ink-faint">{bx}</div>
            ))}
          </div>
          <div className="mt-0.5 text-center font-mono text-[10px] text-ink-faint">{ax.field} →</div>
        </div>
      </div>
    )
  }
  const main = r.regimes.filter((g) => !NOT_REGIME.has(g.label))
  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${Math.min(4, Math.max(1, main.length))}, minmax(0, 1fr))` }}>
      {main.map((g) => tile(g.label))}
    </div>
  )
}

function RegimeTile({ r, label, seq, idx, splitAt, focused, onFocus }: {
  r: LabResult; label: string; seq: number | null; idx: Map<number, number>; splitAt: number; focused: boolean; onFocus: () => void
}) {
  const g = r.regimes.find((x) => x.label === label)!
  const cell = seq !== null ? r.cells[label]?.[String(seq)] : null
  const worse = cell ? Math.min(cell.a.sharpe ?? 0, cell.b.sharpe ?? 0) : 0
  const color = seq !== null ? memberColor(idx.get(seq) ?? -1) : null
  const depth = Math.round(10 + Math.min(1, Math.max(0, worse) / 4) * 40)
  const background = color
    ? `color-mix(in oklab, ${color} ${depth}%, var(--color-panel))`
    : 'repeating-linear-gradient(45deg, var(--color-panel-hi) 0 4px, var(--color-panel) 4px 8px)'
  return (
    <button
      onClick={onFocus}
      className={`min-w-0 rounded-lg border p-1.5 text-left transition ${focused ? 'border-ink' : 'border-seam hover:border-ink-faint'}`}
      style={{ background }}
      title={`${label}\n${pct(g.share.is)} of in-sample bars on ${g.days.is} days · stays ~${fmt(g.dwell_min, 0)} min at a time`}
    >
      <div className="flex items-baseline justify-between gap-1 font-mono text-[10px]">
        <span className="text-ink-dim">{pct(g.share.is)}</span>
        <span className="truncate text-ink">{seq !== null ? `#${seq}` : 'flat'}</span>
      </div>
      <Spark r={r} label={label} seq={seq} idx={idx} splitAt={splitAt} />
      <div className="flex justify-between font-mono text-[10px] text-ink-dim">
        <span title="in-sample daily Sharpe of the routed member here (1st / 2nd half)">
          {cell ? `${fmt(cell.is.sharpe, 1)} (${fmt(cell.a.sharpe, 1)}/${fmt(cell.b.sharpe, 1)})` : '—'}
        </span>
        <span title="holdout Sharpe of the routed member here" className="text-ink-faint">
          HO {cell ? fmt(cell.ho.sharpe, 1) : '—'}
        </span>
      </div>
    </button>
  )
}

/** Cumulative P&L inside one regime: the routed member in its colour, the others faint grey. */
function Spark({ r, label, seq, idx, splitAt }: { r: LabResult; label: string; seq: number | null; idx: Map<number, number>; splitAt: number }) {
  const W = 120
  const H = 34
  const series = useMemo(() => {
    const out: { seq: number; ys: number[] }[] = []
    for (const m of r.members) {
      const d = r.daily.cells[label]?.[String(m.seq)]
      if (!d) continue
      let c = 0
      out.push({ seq: m.seq, ys: d.map((v) => (c += v)) })
    }
    return out
  }, [r, label])
  const all = series.flatMap((s) => s.ys)
  const lo = Math.min(0, ...all)
  const hi = Math.max(0, ...all)
  const n = r.dates.length
  const x = (i: number) => (i / Math.max(1, n - 1)) * W
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4)
  const path = (ys: number[]) => ys.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="my-1 block h-[34px] w-full" aria-hidden>
      <line x1={0} x2={W} y1={y(0)} y2={y(0)} className="stroke-ink-faint/40" strokeWidth={0.5} />
      {splitAt > 0 && <line x1={x(splitAt)} x2={x(splitAt)} y1={0} y2={H} className="stroke-ink-faint" strokeDasharray="2 2" strokeWidth={0.6} />}
      {series.filter((s) => s.seq !== seq).map((s) => (
        <path key={s.seq} d={path(s.ys)} fill="none" className="stroke-ink-faint/45" strokeWidth={0.8} vectorEffect="non-scaling-stroke" />
      ))}
      {series.filter((s) => s.seq === seq).map((s) => (
        <path key={s.seq} d={path(s.ys)} fill="none" stroke={memberColor(idx.get(s.seq) ?? -1)} strokeWidth={2} vectorEffect="non-scaling-stroke"
          className="[filter:brightness(1.25)]" />
      ))}
    </svg>
  )
}

// -----------------------------------------------------------------------------------------
// Matrix: members x regimes, each cell the member's daily Sharpe inside that regime.
// -----------------------------------------------------------------------------------------
function Matrix({ r, routes, idx, focus, onFocus, onRoute }: {
  r: LabResult; routes: Routes; idx: Map<number, number>; focus: string | null; onFocus: (l: string | null) => void; onRoute: (label: string, seq: number) => void
}) {
  const [seg, setSeg] = useState<SegKey>('is')
  const [tip, setTip] = useState<{ x: number; y: number; label: string; seq: number } | null>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const regimes = r.regimes.filter((g) => !NOT_REGIME.has(g.label))
  return (
    <div ref={wrap} className="relative">
      <div className="mb-1 flex items-center gap-1 font-mono text-[10.5px]">
        {(Object.keys(SEG_LABEL) as SegKey[]).map((k) => (
          <button key={k} onClick={() => setSeg(k)}
            className={`rounded px-1.5 py-0.5 ${seg === k ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:text-ink-dim'}`}>
            {SEG_LABEL[k]}
          </button>
        ))}
        <span className="ml-auto text-ink-faint">● positive in both in-sample halves · ○ in one</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-separate [border-spacing:2px] font-mono text-[10.5px]">
          <thead>
            <tr>
              <th className="text-left font-normal text-ink-faint">member</th>
              {regimes.map((g) => (
                <th key={g.label} onClick={() => onFocus(focus === g.label ? null : g.label)}
                  className={`cursor-pointer px-0.5 text-center font-normal leading-tight ${focus === g.label ? 'text-ink' : 'text-ink-faint hover:text-ink-dim'}`}
                  title={`${g.label} — ${pct(g.share.is)} of in-sample bars`}>
                  {shortLabel(g.label).split(' · ').map((p) => (
                    <div key={p}>{p}</div>
                  ))}
                </th>
              ))}
              <th className="px-1 text-center font-normal text-ink-faint">all</th>
            </tr>
          </thead>
          <tbody>
            {r.members.map((m, i) => {
              const all = r.member_stats[String(m.seq)]
              return (
                <tr key={m.seq}>
                  <td className="whitespace-nowrap pr-1">
                    <span className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: memberColor(i) }} />
                    <span className="text-ink">#{m.seq}</span>
                  </td>
                  {regimes.map((g) => {
                    const c = r.cells[g.label]?.[String(m.seq)]
                    const v = c?.[seg].sharpe
                    const on = routes[g.label] === m.seq
                    const stable = c && (c.a.sharpe ?? 0) > 0 && (c.b.sharpe ?? 0) > 0
                    const half = c && !stable && ((c.a.sharpe ?? 0) > 0 || (c.b.sharpe ?? 0) > 0)
                    return (
                      <td key={g.label}
                        onClick={() => onRoute(g.label, m.seq)}
                        onMouseEnter={(e) => {
                          const box = wrap.current?.getBoundingClientRect()
                          const cell = e.currentTarget.getBoundingClientRect()
                          if (box) setTip({ x: cell.left - box.left + cell.width / 2, y: cell.top - box.top, label: g.label, seq: m.seq })
                        }}
                        onMouseLeave={() => setTip(null)}
                        className={`relative h-7 min-w-[44px] cursor-pointer rounded text-center text-ink ${focus && focus !== g.label ? 'opacity-50' : ''}`}
                        style={{ background: divColor(v), boxShadow: on ? `inset 0 0 0 2px ${memberColor(i)}, inset 0 0 0 3px var(--color-panel)` : undefined }}>
                        {fmt(v, 1)}
                        <span className="absolute right-0.5 top-0 text-[8px] leading-none text-ink-dim">{stable ? '●' : half ? '○' : ''}</span>
                      </td>
                    )
                  })}
                  <td className="h-7 min-w-[44px] rounded text-center text-ink" style={{ background: divColor(all?.[seg].sharpe) }}>
                    {fmt(all?.[seg].sharpe, 1)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {tip && <CellTip r={r} {...tip} />}
    </div>
  )
}

function CellTip({ r, x, y, label, seq }: { r: LabResult; x: number; y: number; label: string; seq: number }) {
  const c = r.cells[label]?.[String(seq)]
  if (!c) return null
  const row = (k: SegKey, s: Seg) => (
    <tr key={k}>
      <td className="pr-2 text-ink-faint">{SEG_LABEL[k]}</td>
      <td className="pr-2 text-right text-ink">{fmt(s.sharpe)}</td>
      <td className="pr-2 text-right">{s.days} d</td>
      <td className="pr-2 text-right">{s.pnl === null ? '—' : `${(s.pnl * 1e4).toFixed(0)} bps`}</td>
      <td className="text-right">{s.hit === null ? '—' : pct(s.hit)}</td>
    </tr>
  )
  return (
    <div className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-lg border border-seam bg-panel p-2 font-mono text-[10.5px] text-ink-dim shadow-lg"
      style={{ left: x, top: y - 4 }}>
      <div className="mb-1 text-ink">#{seq} in {label}</div>
      <table>
        <tbody>{(Object.keys(SEG_LABEL) as SegKey[]).map((k) => row(k, c[k]))}</tbody>
      </table>
      <div className="mt-1 text-[9.5px] text-ink-faint">daily Sharpe · days in force · P&amp;L · days up</div>
    </div>
  )
}

// -----------------------------------------------------------------------------------------
// Contribution: the router's cumulative P&L split by the member that earned it -- gains stacked
// above zero, losses below -- with the router's total and the best member traded alone on top,
// and a ribbon underneath showing which member was trading each day.
// -----------------------------------------------------------------------------------------
function ContributionChart({ r, routes, contrib, router, idx }: {
  r: LabResult; routes: Routes; contrib: Record<string, number[]>; router: number[]; idx: Map<number, number>
}) {
  const W = 720
  const H = 230
  const RIB = 12
  const PAD = { l: 44, r: 10, t: 10, b: 18 }
  const [hover, setHover] = useState<number | null>(null)
  const n = r.dates.length
  const keys = useMemo(() => [...r.members.map((m) => String(m.seq)), 'flat'].filter((k) => contrib[k]), [r.members, contrib])
  const model = useMemo(() => {
    const cum: Record<string, number[]> = {}
    for (const k of keys) {
      let c = 0
      cum[k] = contrib[k].map((v) => (c += v))
    }
    let t = 0
    const total = router.map((v) => (t += v))
    const bestKey = r.best_single.seq !== null ? String(r.best_single.seq) : null
    let b = 0
    const bestAlone = bestKey && r.daily.members[bestKey] ? r.daily.members[bestKey].map((v) => (b += v)) : null
    // Diverging stack per day: each member's cumulative contribution sits on the positive or
    // the negative pile depending on its sign that day.
    const bands: Record<string, { y0: number[]; y1: number[] }> = {}
    const pos = new Array(n).fill(0)
    const neg = new Array(n).fill(0)
    for (const k of keys) {
      const y0: number[] = []
      const y1: number[] = []
      for (let i = 0; i < n; i++) {
        const v = cum[k][i]
        if (v >= 0) {
          y0.push(pos[i])
          pos[i] += v
          y1.push(pos[i])
        } else {
          y0.push(neg[i])
          neg[i] += v
          y1.push(neg[i])
        }
      }
      bands[k] = { y0, y1 }
    }
    const vals = [...pos, ...neg, ...total, ...(bestAlone ?? [])]
    const lo = Math.min(0, ...vals)
    const hi = Math.max(0, ...vals)
    const pad = (hi - lo) * 0.06 || 0.01
    return { cum, total, bestAlone, bands, lo: lo - pad, hi: hi + pad }
  }, [keys, contrib, router, r, n])

  const plotB = H - PAD.b - RIB - 6
  const x = (i: number) => PAD.l + (i / Math.max(1, n - 1)) * (W - PAD.l - PAD.r)
  const y = (v: number) => PAD.t + (1 - (v - model.lo) / (model.hi - model.lo)) * (plotB - PAD.t)
  const line = (ys: number[]) => ys.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const area = (b: { y0: number[]; y1: number[] }) =>
    b.y1.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('') +
    b.y0.map((_, j) => {
      const i = n - 1 - j
      return `L${x(i).toFixed(1)},${y(b.y0[i]).toFixed(1)}`
    }).join('') + 'Z'
  const splitAt = r.split_date ? r.dates.findIndex((d) => d >= (r.split_date as string)) : -1
  // Zero always; each end only when it is not crowding zero.
  const ticks = [0, model.hi, model.lo].filter((v, i, a) => a.findIndex((w) => Math.abs(w - v) < (model.hi - model.lo) * 0.12) === i)
  const months = r.dates.map((d, i) => [d, i] as const).filter(([d], i, a) => i === 0 || d.slice(0, 7) !== a[i - 1][0].slice(0, 7))
  const ribbonColor = (i: number) => {
    const g = r.daily.regime[i]
    const label = g >= 0 ? r.regimes[g]?.label : undefined
    const seq = label ? routes[label] : null
    return seq != null ? memberColor(idx.get(seq) ?? -1) : null
  }
  const colorOf = (k: string) => (k === 'flat' ? 'var(--color-ink-faint)' : memberColor(idx.get(Number(k)) ?? -1))

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const i = Math.round(((vx - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1))
    setHover(i >= 0 && i < n ? i : null)
  }
  const h = hover
  const dayLabel = h !== null && r.daily.regime[h] >= 0 ? r.regimes[r.daily.regime[h]]?.label : null

  return (
    <div>
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {h !== null ? (
          <>
            <span className="text-ink">{r.dates[h]}</span>
            <span>{dayLabel ? `${shortLabel(dayLabel)} → ${routes[dayLabel] != null ? `#${routes[dayLabel]}` : 'flat'}` : '—'}</span>
            <span>router {(model.total[h] * 100).toFixed(2)}%</span>
            {model.bestAlone && <span>#{r.best_single.seq} alone {(model.bestAlone[h] * 100).toFixed(2)}%</span>}
            {keys.filter((k) => Math.abs(model.cum[k][h]) > 1e-9).map((k) => (
              <span key={k} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: colorOf(k) }} />
                {k === 'flat' ? 'closing' : `#${k}`} {(model.cum[k][h] * 100).toFixed(2)}%
              </span>
            ))}
          </>
        ) : (
          <span className="text-ink-faint">hover for the day: its dominant regime, who traded it, and each member&apos;s share of the router so far</span>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label="router cumulative P&L by contributing member"
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {splitAt > 0 && (
          <>
            <rect x={x(splitAt)} y={PAD.t} width={W - PAD.r - x(splitAt)} height={plotB - PAD.t} className="fill-accent/[0.06]" />
            <line x1={x(splitAt)} x2={x(splitAt)} y1={PAD.t} y2={plotB + RIB + 6} className="stroke-accent" strokeDasharray="3 3" />
            <text x={x(splitAt) + 4} y={PAD.t + 10} className="fill-accent font-mono text-[9px]">holdout (hidden from agents)</text>
          </>
        )}
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className={t === 0 ? 'stroke-ink-faint/60' : 'stroke-seam'} />
            <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">{(t * 100).toFixed(1)}%</text>
          </g>
        ))}
        {keys.map((k) => (
          <path key={k} d={area(model.bands[k])} style={{ fill: colorOf(k) }} opacity={k === 'flat' ? 0.35 : 0.55}
            className="stroke-panel" strokeWidth={0.6} />
        ))}
        {model.bestAlone && <path d={line(model.bestAlone)} fill="none" className="stroke-ink-dim" strokeWidth={1.4} strokeDasharray="4 3" />}
        <path d={line(model.total)} fill="none" className="stroke-ink" strokeWidth={2} />
        {h !== null && (
          <>
            <line x1={x(h)} x2={x(h)} y1={PAD.t} y2={plotB + RIB + 6} className="stroke-ink-dim" />
            <circle cx={x(h)} cy={y(model.total[h])} r={4} className="fill-ink stroke-panel" strokeWidth={2} />
          </>
        )}
        {/* Who traded each day: the member routed to that day's dominant regime. */}
        {r.dates.map((_, i) => {
          const c = ribbonColor(i)
          const w = (W - PAD.l - PAD.r) / Math.max(1, n)
          return (
            <rect key={i} x={x(i) - w / 2} y={plotB + 6} width={Math.max(0.5, w - 0.3)} height={RIB}
              style={{ fill: c ?? 'var(--color-seam)' }} opacity={c ? 0.9 : 0.6} />
          )
        })}
        {months.filter((_, j) => j % Math.max(1, Math.ceil(months.length / 8)) === 0).map(([d, i]) => (
          <text key={d} x={x(i)} y={H - 4} className="fill-ink-faint font-mono text-[9px]">{d.slice(0, 7)}</text>
        ))}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-3 font-mono text-[10px] text-ink-faint">
        <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 bg-ink" /> router</span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 border-t border-dashed border-ink-dim" /> #{r.best_single.seq} traded everywhere
        </span>
        <span>bands: each member&apos;s cumulative contribution (gains above zero, losses below) · strip: who traded that day</span>
      </div>
    </div>
  )
}

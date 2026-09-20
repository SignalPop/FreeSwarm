'use client'

import { useEffect, useState } from 'react'
import { duration } from '@/lib/format'
import { fmtMetric, type MetricKind } from '@/lib/objectives'
import { library, type Feature, type LibModule, type LibModuleFull, type Verdict } from '@/lib/library'
import Markdown from '@/components/Markdown'
import CopyButton from '@/components/CopyButton'
import { Button, Pill } from '@/components/ui'
import { moduleReview, type ModuleReviewResult, type ReviewProgress } from '@/lib/review'

const KIND_TONE = { regime: 'accent', signal: 'good', risk: 'warn', util: 'neutral' } as const
const VERDICT_TONE = { works: 'good', broken: 'bad', note: 'neutral' } as const

function ago(ts: number) {
  return `${duration(Date.now() / 1000 - ts)} ago`
}

/**
 * The project's code library as a tab of the objective panel: every module with the
 * evidence behind it (how many candidates used it and how they did), and the objective's
 * forecast features with their measured skill. Click a module to review its code.
 */
export default function LibraryTab({
  projectId,
  objectiveId,
  kind,
}: {
  projectId: string
  objectiveId: string
  kind: MetricKind
}) {
  const [mods, setMods] = useState<LibModule[] | null>(null)
  const [feats, setFeats] = useState<Feature[]>([])
  const [open, setOpen] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const [m, f] = await Promise.all([library.list(projectId), library.features(objectiveId)])
        if (!alive) return
        setMods(m.modules)
        setFeats(f.features)
        setErr(null)
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      }
    }
    void load()
    const t = setInterval(load, 8000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [projectId, objectiveId, tick])

  if (err) return <div className="text-[12px] text-bad">{err}</div>
  if (!mods) return <div className="text-[12px] text-ink-faint">Loading…</div>

  return (
    <div className="space-y-3">
      {mods.length === 0 ? (
        <div className="text-[12px] text-ink-faint">
          Empty. Agents save reusable regime detectors, signals and risk rules here with
          library_save; every candidate that imports one becomes evidence for it.
        </div>
      ) : (
        <table className="w-full table-fixed font-mono text-[11px]">
          <thead>
            <tr className="text-left text-ink-faint">
              <th className="py-1 font-normal">module</th>
              <th className="w-[62px] py-1 font-normal">kind</th>
              <th className="w-[64px] py-1 text-right font-normal">used</th>
              <th className="w-[70px] py-1 text-right font-normal">best IS</th>
              <th className="w-[80px] py-1 pl-3 font-normal">comments</th>
            </tr>
          </thead>
          <tbody>
            {mods.map((m) => (
              <tr
                key={m.name}
                onClick={() => setOpen(m.name)}
                className={`cursor-pointer border-t border-seam/60 hover:bg-panel-hi ${m.status === 'retired' ? 'opacity-50' : ''}`}
              >
                <td className="truncate py-1 text-ink" title={m.status === 'quarantined' ? m.warning : m.description}>
                  {m.name} <span className="text-ink-faint">v{m.version}</span>
                  {m.evidence.champions > 0 && <span className="text-good"> ★{m.evidence.champions}</span>}
                  {m.status === 'quarantined' && <span className="text-bad"> · do not use</span>}
                  <div className="truncate font-sans text-[11px] text-ink-faint">
                    {m.status === 'quarantined' ? <span className="text-bad">{m.warning.split('\n')[0]}</span> : m.description}
                  </div>
                </td>
                <td className="py-1">
                  <Pill tone={KIND_TONE[m.kind]}>{m.kind}</Pill>
                </td>
                <td className="py-1 text-right text-ink-dim">
                  {m.evidence.uses}
                  {m.evidence.lookahead_fails > 0 && <span className="text-bad"> ·{m.evidence.lookahead_fails}✗</span>}
                </td>
                <td className="py-1 text-right text-ink-dim">{fmtMetric(kind, m.evidence.best_in_sample)}</td>
                <td className="py-1 pl-3">
                  <span className="text-good">{m.comments.works ?? 0}✓</span>{' '}
                  <span className="text-bad">{m.comments.broken ?? 0}✗</span>{' '}
                  <span className="text-ink-faint">{m.comments.note ?? 0}✎</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <BuildForecast objectiveId={objectiveId} onBuilt={() => setTick((n) => n + 1)} />

      {feats.length > 0 && (
        <div>
          <div className="mb-1 text-[10.5px] uppercase tracking-wide text-ink-faint">
            Forecast features (in-sample skill)
          </div>
          <ul className="space-y-1">
            {feats.map((f) => (
              <li key={f.view} className="font-mono text-[11px] text-ink-dim">
                <span className="text-ink">{f.view}</span> · {f.params.model.split('/').pop()} · {f.params.horizon} bars ahead ·
                every {f.params.every} · {f.rows} rows
                {Object.entries(f.skill ?? {}).map(([s, k]) => (
                  <div key={s} className="ml-3 text-ink-faint">
                    {s}: skill{' '}
                    <span className={(k.skill_vs_no_change ?? 0) > 0 ? 'text-good' : 'text-ink-dim'}>
                      {k.skill_vs_no_change ?? '—'}
                    </span>{' '}
                    · direction{' '}
                    <span className={(k.direction_accuracy ?? 0) > 0.55 ? 'text-good' : 'text-ink-dim'}>
                      {k.direction_accuracy ?? '—'}
                    </span>{' '}
                    · 10–90% band {k.band_coverage_10_90 ?? '—'}
                  </div>
                ))}
              </li>
            ))}
          </ul>
        </div>
      )}

      {open && (
        <ModuleView
          projectId={projectId}
          name={open}
          kind={kind}
          onClose={() => setOpen(null)}
          onChanged={() => setTick((n) => n + 1)}
        />
      )}
    </div>
  )
}

function ModuleView({
  projectId,
  name,
  kind,
  onClose,
  onChanged,
}: {
  projectId: string
  name: string
  kind: MetricKind
  onClose: () => void
  onChanged: () => void
}) {
  const [m, setM] = useState<LibModuleFull | null>(null)
  const [version, setVersion] = useState<number | undefined>(undefined)
  const [tab, setTab] = useState<'code' | 'evidence' | 'regimes' | 'test'>('code')
  const [verdict, setVerdict] = useState<Verdict>('note')
  const [text, setText] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  // Reviewing the module itself rather than a candidate that imported it. A defect here has
  // already contaminated every result built on it, so the verdict can retire it and sweep them.
  const [reviewing, setReviewing] = useState(false)
  const [progress, setProgress] = useState<ReviewProgress | null>(null)
  const [reviewed, setReviewed] = useState<ModuleReviewResult | null>(null)
  const [retiring, setRetiring] = useState(false)

  async function runReview() {
    setReviewing(true)
    setErr(null)
    setReviewed(null)
    setProgress(null)
    try {
      setReviewed(await moduleReview.run(projectId, name, (p) => setProgress((prev) => ({ ...prev, ...p }))))
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setReviewing(false)
      setProgress(null)
    }
  }

  async function retireFromVerdict() {
    if (!reviewed) return
    setRetiring(true)
    setErr(null)
    try {
      await moduleReview.retire(projectId, name, reviewed.board_text, reviewed.model)
      setReviewed(null)
      setTick((n) => n + 1)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setRetiring(false)
    }
  }

  useEffect(() => {
    let alive = true
    library
      .get(projectId, name, version)
      .then((d) => alive && setM(d))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [projectId, name, version, tick])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function act(fn: () => Promise<unknown>) {
    setErr(null)
    try {
      await fn()
      setTick((n) => n + 1)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const ev = m?.evidence
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex max-h-[88vh] w-full max-w-[980px] flex-col overflow-hidden rounded-2xl border border-seam bg-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-seam px-5 py-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[15px] text-ink">lib.{name}</span>
              {m && <Pill tone={KIND_TONE[m.kind]}>{m.kind}</Pill>}
              {m?.status === 'retired' && <Pill tone="warn">retired</Pill>}
              {m?.status === 'quarantined' && <Pill tone="bad">quarantined</Pill>}
              {m && (
                <select
                  className="rounded-lg border border-seam bg-panel-hi px-2 py-0.5 font-mono text-[11px] text-ink"
                  value={m.shown_version}
                  onChange={(e) => setVersion(Number(e.target.value))}
                >
                  {m.versions.map((v) => (
                    <option key={v.version} value={v.version}>
                      v{v.version}
                      {v.version === m.version ? ' (current)' : ''} · {v.author ?? '?'}
                    </option>
                  ))}
                </select>
              )}
            </div>
            {m && <div className="mt-1 text-[12.5px] text-ink-dim">{m.description}</div>}
            {reviewing && progress?.note && (
              <div className="mt-2 rounded-lg border border-remote/30 bg-remote/5 px-3 py-2">
                <div className="flex items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-wide text-remote">
                  <span className="inline-block size-1.5 animate-dot rounded-full bg-remote" />
                  {progress.phase === 'writing' ? 'writing the verdict' : 'reasoning'}
                  {progress.output_tokens ? ` · ${progress.output_tokens.toLocaleString()} tok` : ''}
                  {progress.modules?.length ? ` · +${progress.modules.length} imported` : ''}
                </div>
                <div className="mt-1 line-clamp-3 text-[12px] leading-relaxed text-ink-dim">…{progress.note}</div>
              </div>
            )}

            {reviewed && (
              <div
                className={`mt-2 rounded-lg border px-3 py-2 ${
                  reviewed.verdict.disqualify ? 'border-bad/40 bg-bad/5' : 'border-good/40 bg-good/5'
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[12.5px] font-medium text-ink">Review by {reviewed.model}</span>
                  <Pill tone={reviewed.verdict.disqualify ? 'bad' : 'good'}>
                    {reviewed.verdict.disqualify ? 'retire this module' : 'no blocking problem'}
                  </Pill>
                  <span className="ml-auto font-mono text-[10.5px] text-ink-faint">
                    {reviewed.seconds}s · {reviewed.usage.input_tokens} in / {reviewed.usage.output_tokens} out
                    {reviewed.modules_reviewed.length > 1 ? ` · read ${reviewed.modules_reviewed.length} modules` : ''}
                  </span>
                </div>
                <div className="mt-1.5 whitespace-pre-wrap text-[12px] leading-relaxed text-ink-dim">
                  {reviewed.verdict.summary}
                </div>
                {reviewed.verdict.look_ahead_detail && (
                  <div className="mt-1.5 whitespace-pre-wrap text-[12px] leading-relaxed text-ink-dim">
                    <span className="text-[10.5px] uppercase tracking-wide text-ink-faint">Look-ahead</span>
                    <br />
                    {reviewed.verdict.look_ahead_detail}
                  </div>
                )}
                {reviewed.verdict.findings.length > 0 && (
                  <ul className="mt-1.5 space-y-1 text-[12px] text-ink-dim">
                    {reviewed.verdict.findings.map((f, i) => (
                      <li key={i}>
                        <span className={f.severity === 'critical' ? 'text-bad' : 'text-warn'}>
                          [{f.severity}]
                        </span>{' '}
                        <span className="text-ink">{f.title}</span> — {f.detail}
                      </li>
                    ))}
                  </ul>
                )}
                {reviewed.awaiting_confirmation && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-bad/25 pt-2">
                    <Button tone="primary" onClick={retireFromVerdict} disabled={retiring}>
                      {retiring ? 'retiring…' : 'Retire it and demote everything using it'}
                    </Button>
                    <Button tone="ghost" onClick={() => setReviewed(null)}>
                      Keep it
                    </Button>
                    <span className="text-[11.5px] text-ink-faint">
                      Retiring marks the module do-not-use and disqualifies every result that imports it,
                      across the project. Nothing changes until you confirm.
                    </span>
                  </div>
                )}
              </div>
            )}

            {m?.status === 'quarantined' && m.warning && (
              <div className="mt-2 rounded-lg border border-bad/40 bg-bad/5 px-3 py-2">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-bad">
                  Do not build on this module
                </div>
                <div className="mt-1 whitespace-pre-wrap text-[12px] text-ink-dim">{m.warning}</div>
              </div>
            )}
            {m?.version_note && <div className="mt-0.5 font-mono text-[10.5px] text-ink-faint">this version: {m.version_note}</div>}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {m && <CopyButton text={m.code} label="copy" />}
            {m && m.shown_version !== m.version && (
              <Button tone="ghost" onClick={() => act(() => library.patch(projectId, name, { restore_version: m.shown_version }))}>
                Restore v{m.shown_version}
              </Button>
            )}
            {m && m.status !== 'quarantined' && (
              <button
                onClick={runReview}
                disabled={reviewing}
                title="Send this module — and everything it imports — to Claude to check for look-ahead bias"
                className="rounded-md border border-remote/45 px-2.5 py-1 font-mono text-[11px] text-remote hover:bg-remote/10 disabled:opacity-40"
              >
                {reviewing ? 'reviewing…' : 'review with Claude'}
              </button>
            )}
            {m && (
              <Button
                tone={m.status === 'active' ? 'ghost' : 'primary'}
                onClick={() => act(() => library.patch(projectId, name, { status: m.status === 'active' ? 'retired' : 'active' }))}
              >
                {m.status === 'active' ? 'Retire' : m.status === 'quarantined' ? 'Clear quarantine' : 'Reactivate'}
              </Button>
            )}
            <button onClick={onClose} className="font-mono text-[11px] text-ink-faint hover:text-accent">
              close
            </button>
          </div>
        </div>

        <div className="flex gap-1 border-b border-seam px-5">
          {(['code', 'evidence', ...(m?.kind === 'regime' ? (['regimes'] as const) : []), 'test'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`-mb-px border-b-2 px-3 py-2 font-mono text-[11.5px] ${
                tab === t ? 'border-accent text-accent' : 'border-transparent text-ink-faint hover:text-ink-dim'
              }`}
            >
              {t === 'evidence' && ev ? `evidence (${ev.uses})` : t}
            </button>
          ))}
        </div>

        <div className="grid min-h-0 flex-1 gap-0 overflow-hidden md:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-h-0 overflow-y-auto p-5">
            {err && <div className="mb-2 text-[12px] text-bad">{err}</div>}
            {!m && !err && <div className="text-[12px] text-ink-faint">Loading…</div>}
            {m && tab === 'code' && <Markdown source={'```python\n' + m.code + '\n```'} />}
            {m && tab === 'test' && (
              <pre className="whitespace-pre-wrap rounded-lg bg-panel-hi p-3 font-mono text-[11.5px] text-ink-dim">
                {m.test_output || '(no output)'}
              </pre>
            )}
            {m && ev && tab === 'evidence' && (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-2 font-mono text-[11.5px]">
                  <Stat label="candidates" value={ev.uses} />
                  <Stat label="ran ok" value={ev.ok} />
                  <Stat label="errors" value={ev.errors} />
                  <Stat label="look-ahead fails" value={ev.lookahead_fails} bad={ev.lookahead_fails > 0} />
                  <Stat label="champions" value={ev.champions} good={ev.champions > 0} />
                  <Stat label="best holdout" value={fmtMetric(kind, ev.best_holdout)} />
                </div>
                <table className="w-full font-mono text-[11px]">
                  <thead>
                    <tr className="text-left text-ink-faint">
                      <th className="py-1 font-normal">candidate</th>
                      <th className="py-1 font-normal">module v</th>
                      <th className="py-1 text-right font-normal">holdout</th>
                      <th className="py-1 text-right font-normal">in-sample</th>
                      <th className="py-1 pl-3 font-normal">look-ahead</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ev.recent ?? []).map((c) => (
                      <tr key={c.id} className="border-t border-seam/60 text-ink-dim">
                        <td className="py-1">
                          #{c.seq}
                          {c.champion_at ? <span className="text-good"> ★</span> : ''}
                        </td>
                        <td className="py-1">v{c.version}</td>
                        <td className="py-1 text-right text-ink">{c.status === 'error' ? <span className="text-bad">error</span> : fmtMetric(kind, c.score)}</td>
                        <td className="py-1 text-right">{fmtMetric(kind, c.is_score)}</td>
                        <td className={`py-1 pl-3 ${c.lookahead === 'fail' ? 'text-bad' : c.lookahead === 'pass' ? 'text-good' : ''}`}>{c.lookahead}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {m && tab === 'regimes' && <RegimeTable map={m.regime_map} />}
          </div>

          <div className="flex min-h-0 flex-col border-t border-seam md:border-l md:border-t-0">
            <div className="px-4 pt-3 text-[10.5px] uppercase tracking-wide text-ink-faint">Comments</div>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-4 py-2">
              {(m?.comments ?? []).length === 0 && <div className="text-[12px] text-ink-faint">No comments yet.</div>}
              {(m?.comments ?? []).map((c) => (
                <div key={c.id} className="rounded-lg border border-seam/60 bg-panel-hi/40 p-2">
                  <div className="flex items-center gap-1.5 font-mono text-[10px] text-ink-faint">
                    <Pill tone={VERDICT_TONE[c.verdict]}>{c.verdict}</Pill>
                    <span className="truncate">{c.author}</span>
                    <span className="ml-auto shrink-0">v{c.version} · {ago(c.ts)}</span>
                  </div>
                  <div className="mt-1 whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{c.text}</div>
                </div>
              ))}
            </div>
            <div className="border-t border-seam p-3">
              <div className="mb-1.5 flex gap-1">
                {(['works', 'broken', 'note'] as const).map((v) => (
                  <button
                    key={v}
                    onClick={() => setVerdict(v)}
                    className={`rounded-md border px-2 py-0.5 font-mono text-[10.5px] ${
                      verdict === v ? 'border-accent text-accent' : 'border-seam text-ink-faint'
                    }`}
                  >
                    {v}
                  </button>
                ))}
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Your review — agents read it before using this module."
                className="h-[70px] w-full resize-none rounded-lg border border-seam bg-panel-hi px-2 py-1.5 text-[12px] text-ink outline-none focus:border-accent"
              />
              <Button
                tone="primary"
                className="mt-1.5 w-full"
                disabled={!text.trim()}
                onClick={() =>
                  act(async () => {
                    await library.comment(projectId, name, verdict, text.trim())
                    setText('')
                  })
                }
              >
                Add comment
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Run the forecaster yourself: pick series (columns or expressions), horizon and spacing.
 *  The result becomes a feature every agent can load, with its measured skill. */
function BuildForecast({ objectiveId, onBuilt }: { objectiveId: string; onBuilt: () => void }) {
  const [series, setSeries] = useState('IntrVol, GEX')
  const [horizon, setHorizon] = useState(30)
  const [every, setEvery] = useState(30)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)

  async function run() {
    setBusy(true)
    setMsg(null)
    const t0 = Date.now()
    try {
      const f = await library.buildFeature(objectiveId, {
        columns: series.split(',').map((x) => x.trim()).filter(Boolean),
        horizon,
        every,
      })
      const skill = Object.entries(f.skill ?? {})
        .map(([k, v]) => `${k}: skill ${v.skill_vs_no_change ?? '—'}, direction ${v.direction_accuracy ?? '—'}`)
        .join(' · ')
      setMsg({
        ok: true,
        text: `${f.view}: ${f.rows} forecasts${f.cached ? ' (cached)' : ` in ${((Date.now() - t0) / 1000).toFixed(0)}s`}. ${skill}${f.adjusted ? ` — ${f.adjusted}` : ''}`,
      })
      onBuilt()
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) })
    } finally {
      setBusy(false)
    }
  }

  const input = 'rounded-md border border-seam bg-panel-hi px-2 py-1 font-mono text-[11px] text-ink outline-none focus:border-accent'
  return (
    <div className="rounded-lg border border-seam p-2.5">
      <div className="mb-1.5 text-[10.5px] uppercase tracking-wide text-ink-faint">Build a forecast feature</div>
      <div className="flex flex-wrap items-center gap-2">
        <input className={`${input} min-w-[180px] flex-1`} value={series} onChange={(e) => setSeries(e.target.value)}
          title="Columns or expressions, comma separated, e.g. IntrVol, GEX / Pinning_TotalAbsGex" />
        <label className="font-mono text-[10.5px] text-ink-faint">
          horizon <input type="number" min={1} max={256} className={`${input} w-[60px]`} value={horizon}
            onChange={(e) => setHorizon(Math.max(1, Number(e.target.value) || 1))} />
        </label>
        <label className="font-mono text-[10.5px] text-ink-faint">
          every <input type="number" min={1} className={`${input} w-[64px]`} value={every}
            onChange={(e) => setEvery(Math.max(1, Number(e.target.value) || 1))} />
        </label>
        <Button tone="primary" onClick={run} disabled={busy || !series.trim()}>
          {busy ? 'Forecasting…' : 'Run forecast'}
        </Button>
      </div>
      <div className="mt-1 text-[10.5px] text-ink-faint">
        Bars are 10 s: horizon 30 = 5 min ahead, every 30 = one forecast per 5 min. Takes up to a minute or two.
      </div>
      {msg && <div className={`mt-1.5 font-mono text-[11px] ${msg.ok ? 'text-good' : 'text-bad'}`}>{msg.text}</div>}
    </div>
  )
}

function Stat({ label, value, good, bad }: { label: string; value: React.ReactNode; good?: boolean; bad?: boolean }) {
  return (
    <div className="rounded-lg border border-seam bg-panel-hi/40 px-2 py-1.5">
      <div className="text-[9.5px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className={`text-[14px] ${good ? 'text-good' : bad ? 'text-bad' : 'text-ink'}`}>{value}</div>
    </div>
  )
}

/** Regime x signal: net Sharpe per cell (gross and trades/day on hover), in-sample only. */
function RegimeTable({ map }: { map: LibModuleFull['regime_map'] }) {
  if (!map) {
    return (
      <div className="text-[12px] text-ink-faint">
        No regime map yet. Agents run regime_map to measure every signal module inside each regime this
        detector finds.
      </div>
    )
  }
  const regimes = Object.entries(map.result.regimes).sort((a, b) => b[1].share - a[1].share)
  const signals = Array.from(new Set(regimes.flatMap(([, r]) => Object.keys(r.signals))))
  const tone = (v: number | null | undefined) =>
    v == null ? 'text-ink-faint' : v > 1 ? 'text-good' : v < -1 ? 'text-bad' : 'text-ink-dim'
  return (
    <div className="space-y-2">
      <div className="font-mono text-[10.5px] text-ink-faint">
        v{map.regime_version} · {map.result.from} → {map.result.to} · {map.result.in_sample_only ? 'in-sample only' : 'full data'} ·
        net of costs · {ago(map.ts)} by {map.author}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[11px]">
          <thead>
            <tr className="text-left text-ink-faint">
              <th className="py-1 font-normal">regime</th>
              <th className="py-1 text-right font-normal">share</th>
              {signals.map((s) => (
                <th key={s} className="py-1 pl-3 text-right font-normal">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regimes.map(([label, r]) => (
              <tr key={label} className="border-t border-seam/60">
                <td className="py-1 text-ink">{label}</td>
                <td className="py-1 text-right text-ink-dim">{(r.share * 100).toFixed(0)}%</td>
                {signals.map((s) => {
                  const c = r.signals[s]
                  return (
                    <td
                      key={s}
                      className={`py-1 pl-3 text-right ${tone(c?.sharpe)}`}
                      title={c ? `gross ${c.sharpe_gross?.toFixed(2) ?? '—'} · ${c.trades_per_day?.toFixed(1) ?? '—'} trades/day · hit ${c.hit_rate?.toFixed(3) ?? '—'}` : ''}
                    >
                      {c?.sharpe != null ? c.sharpe.toFixed(2) : '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {Object.keys(map.result.errors ?? {}).length > 0 && (
        <div className="text-[11.5px] text-bad">
          {Object.entries(map.result.errors).map(([k, v]) => (
            <div key={k}>
              {k}: {v}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

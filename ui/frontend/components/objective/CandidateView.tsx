'use client'

import { useEffect, useState } from 'react'
import { clockTime, duration } from '@/lib/format'
import {
  fmtMetric,
  objectives,
  type CandidateFull,
  type DemoteResult,
  type ObjectiveDetail,
  type SegmentStats,
} from '@/lib/objectives'
import { chatStore } from '@/lib/chatStore'
import { review, type ReviewProgress, type ReviewResult } from '@/lib/review'
import Markdown from '@/components/Markdown'
import CopyButton from '@/components/CopyButton'
import { Pill } from '@/components/ui'
import { EquityCurve } from './Charts'

const ROWS: { key: keyof SegmentStats; label: string; kind: string }[] = [
  { key: 'sharpe', label: 'Sharpe', kind: 'sharpe' },
  { key: 'sortino', label: 'Sortino', kind: 'sortino' },
  { key: 'total_return', label: 'Total return', kind: 'total_return' },
  { key: 'cagr', label: 'CAGR', kind: 'cagr' },
  { key: 'max_drawdown', label: 'Max drawdown', kind: 'max_drawdown' },
  { key: 'calmar', label: 'Calmar', kind: 'calmar' },
  { key: 'volatility', label: 'Volatility', kind: 'cagr' },
  { key: 'win_rate', label: 'Win rate', kind: 'cagr' },
  { key: 'active_days', label: 'Active days', kind: 'count' },
  { key: 'days', label: 'Days', kind: 'count' },
]

function cell(kind: string, v: unknown) {
  if (typeof v !== 'number') return '—'
  return kind === 'count' ? String(v) : fmtMetric(kind, v)
}

export function verdictTone(v: string): 'good' | 'bad' | 'warn' | 'neutral' {
  return v === 'pass' ? 'good' : v === 'fail' ? 'bad' : v === 'error' || v === 'pending' ? 'warn' : 'neutral'
}

/** One candidate, in full: what it tried, how it scored in and out of sample, whether it
 *  passed the look-ahead test and the audit, its equity curve, and its code. */
export default function CandidateView({
  objective,
  candidateId,
  onClose,
  onOpenChat,
  onDemoted,
}: {
  objective: ObjectiveDetail
  candidateId: string
  onClose: () => void
  onOpenChat: () => void
  /** The leaderboard changed underneath: a demotion re-ranks and may re-crown. */
  onDemoted?: () => void
}) {
  const [c, setC] = useState<CandidateFull | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tab, setTab] = useState<'overview' | 'code' | 'output'>('overview')
  // Demoting: the finding is pasted in, so the panel is deliberately large and the action
  // stays disabled until there is something for the team to learn from.
  const [demoting, setDemoting] = useState(false)
  const [finding, setFinding] = useState('')
  const [lesson, setLesson] = useState('')
  const [reviewer, setReviewer] = useState('external review')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState<DemoteResult | null>(null)
  const [reviewing, setReviewing] = useState(false)
  const [reviewed, setReviewed] = useState<ReviewResult | null>(null)
  // What the reviewer is doing right now, so a minute of thinking does not look like a hang.
  const [progress, setProgress] = useState<ReviewProgress | null>(null)
  const [elapsed, setElapsed] = useState(0)

  async function runReview() {
    setReviewing(true)
    setErr(null)
    setReviewed(null)
    setProgress(null)
    const startedAt = Date.now()
    const tick = setInterval(() => setElapsed(Math.round((Date.now() - startedAt) / 1000)), 1000)
    setElapsed(0)
    try {
      const r = await review.runStreamed(objective.id, candidateId, (p) =>
        setProgress((prev) => ({ ...prev, ...p })),
      )
      setReviewed(r)
      if (r.demoted) {
        // 'auto': it already landed. Reflect the new state rather than offering the form.
        setC(await objectives.candidate(objective.id, candidateId))
        onDemoted?.()
      } else if (r.awaiting_confirmation) {
        // 'ask': hand the reviewer's own words to the demote form so the operator approves
        // rather than retypes.
        setFinding(r.board_text)
        setLesson(r.verdict.lesson)
        setReviewer(r.model)
        setDemoting(true)
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      clearInterval(tick)
      setReviewing(false)
      setProgress(null)
    }
  }

  async function submitDemotion() {
    setBusy(true)
    setErr(null)
    try {
      const r = await objectives.demote(objective.id, candidateId, {
        finding,
        lesson: lesson.trim() || undefined,
        reviewer: reviewer.trim() || 'operator',
      })
      setDone(r)
      setDemoting(false)
      setC(await objectives.candidate(objective.id, candidateId))
      onDemoted?.()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    let alive = true
    objectives
      .candidate(objective.id, candidateId)
      .then((d) => alive && setC(d))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [objective.id, candidateId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const kind = objective.metric.kind
  const m = c?.metrics ?? {}
  const isChamp = !!c?.champion_at
  const isBest = objective.best_id === candidateId

  function toChat() {
    if (!c) return
    chatStore.appendToInput(
      `Here is candidate #${c.seq} for the objective "${objective.title}" ` +
        `(${objective.metric_label} ${fmtMetric(kind, c.score)} on the holdout).\n\n` +
        `Hypothesis: ${c.rationale}\n\n\`\`\`python\n${c.code || c.answer}\n\`\`\`\n\n`,
    )
    onOpenChat()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex max-h-[88vh] w-full max-w-[980px] flex-col overflow-hidden rounded-2xl border border-seam bg-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-seam px-5 py-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[15px] font-medium text-ink">Candidate #{c?.seq ?? '…'}</span>
              {isBest && <Pill tone="good">current best</Pill>}
              {isChamp && !isBest && <Pill tone="accent">former best</Pill>}
              {c && <Pill tone={c.status === 'ok' ? 'neutral' : c.status === 'error' ? 'bad' : 'warn'}>{c.status}</Pill>}
            </div>
            {c && (
              <div className="mt-1.5 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-ink-faint">
                <span>{c.model}</span>
                <span>· {c.mode === 'improve' ? `improved ${c.parent_id ? 'a parent' : ''}` : 'explored'}</span>
                <span>· {clockTime(c.created_at)}</span>
                {c.eval_seconds != null && <span>· scored in {duration(c.eval_seconds)}</span>}
              </div>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {c?.code && <CopyButton text={c.code} label="copy code" />}
            {c && (
              <button
                onClick={toChat}
                className="rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent hover:bg-accent/10"
              >
                open in chat
              </button>
            )}
            {c && (
              <button
                onClick={runReview}
                disabled={reviewing}
                title="Send this candidate — and the library modules it imports — to Claude for a look-ahead and overfitting review"
                className="rounded-md border border-remote/45 px-2.5 py-1 font-mono text-[11px] text-remote hover:bg-remote/10 disabled:opacity-40"
              >
                {reviewing ? 'reviewing…' : 'review with Claude'}
              </button>
            )}
            {reviewing && (
              <span
                className="flex items-center gap-1.5 font-mono text-[11px] text-remote"
                title={
                  progress?.modules?.length
                    ? `Reviewing the candidate and ${progress.modules.length} library module(s): ${progress.modules.join(', ')}`
                    : 'Reviewing the candidate'
                }
              >
                <span className="inline-block size-1.5 animate-dot rounded-full bg-remote" />
                {progress?.phase === 'writing'
                  ? 'writing the verdict'
                  : progress?.phase === 'thinking'
                    ? 'reasoning about alignment'
                    : 'sending'}
                <span className="text-ink-faint">
                  · {elapsed}s
                  {progress?.output_tokens ? ` · ${progress.output_tokens.toLocaleString()} tok` : ''}
                  {progress?.modules?.length ? ` · +${progress.modules.length} module` : ''}
                </span>
              </span>
            )}
            {c && c.audit !== 'fail' && (
              <button
                onClick={() => setDemoting((v) => !v)}
                title="Disqualify this result and record why, so the team stops reproducing it"
                className="rounded-md border border-bad/40 px-2.5 py-1 font-mono text-[11px] text-bad hover:bg-bad/10"
              >
                demote
              </button>
            )}
            <button onClick={onClose} className="font-mono text-[11px] text-ink-faint hover:text-accent">
              close
            </button>
          </div>
        </div>

        <div className="flex gap-1 border-b border-seam px-5">
          {(['overview', 'code', 'output'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`-mb-px border-b-2 px-3 py-2 font-mono text-[11.5px] ${
                tab === t ? 'border-accent text-accent' : 'border-transparent text-ink-faint hover:text-ink-dim'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {/* While the reviewer works: what it is reading and what it is currently reasoning
              about. The note is the model's own summary of its reasoning, not a guess. */}
          {reviewing && progress?.note && (
            <div className="mb-3 rounded-lg border border-remote/30 bg-remote/5 px-3 py-2">
              <div className="flex items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-wide text-remote">
                <span className="inline-block size-1.5 animate-dot rounded-full bg-remote" />
                {progress.phase === 'writing' ? 'writing the verdict' : 'reasoning'}
                {progress.modules?.length ? ` · read ${progress.modules.join(', ')}` : ''}
              </div>
              <div className="mt-1 line-clamp-3 text-[12px] leading-relaxed text-ink-dim">…{progress.note}</div>
            </div>
          )}
          {err && <div className="mb-3 text-[12px] text-bad">{err}</div>}
          {!c && !err && <div className="text-[12px] text-ink-faint">Loading…</div>}

          {reviewed && (
            <div
              className={`mb-5 rounded-xl border p-4 ${
                reviewed.verdict.disqualify ? 'border-bad/40 bg-bad/5' : 'border-good/35 bg-good/5'
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[13px] font-medium text-ink">Review by {reviewed.model}</span>
                <Pill tone={reviewed.verdict.disqualify ? 'bad' : reviewed.verdict.trustworthy ? 'good' : 'warn'}>
                  {reviewed.verdict.disqualify ? 'disqualify' : reviewed.verdict.trustworthy ? 'trustworthy' : 'doubtful'}
                </Pill>
                {reviewed.verdict.look_ahead_found && <Pill tone="bad">look-ahead found</Pill>}
                <span className="ml-auto font-mono text-[10.5px] text-ink-faint">
                  {reviewed.seconds}s · {reviewed.usage.input_tokens ?? '—'} in / {reviewed.usage.output_tokens ?? '—'} out
                </span>
              </div>

              <div className="mt-2 text-[12.5px] leading-relaxed text-ink">{reviewed.verdict.summary}</div>

              {reviewed.verdict.look_ahead_detail && (
                <div className="mt-2">
                  <div className="text-[11px] uppercase tracking-wide text-ink-faint">Look-ahead</div>
                  <div className="whitespace-pre-wrap text-[12px] leading-relaxed text-ink-dim">
                    {reviewed.verdict.look_ahead_detail}
                  </div>
                </div>
              )}

              {reviewed.verdict.findings.length > 0 && (
                <div className="mt-2">
                  <div className="text-[11px] uppercase tracking-wide text-ink-faint">Findings</div>
                  {reviewed.verdict.findings.map((f, i) => (
                    <div key={i} className="mt-1 text-[12px] leading-relaxed text-ink-dim">
                      <span
                        className={`font-mono text-[10.5px] uppercase ${
                          f.severity === 'critical' ? 'text-bad' : f.severity === 'major' ? 'text-warn' : 'text-ink-faint'
                        }`}
                      >
                        [{f.severity}]
                      </span>{' '}
                      <span className="text-ink">{f.title}</span> — {f.detail}
                    </div>
                  ))}
                </div>
              )}

              {reviewed.verdict.improvements.length > 0 && (
                <div className="mt-2">
                  <div className="text-[11px] uppercase tracking-wide text-ink-faint">Suggested improvements</div>
                  <ul className="mt-1 list-disc pl-5 text-[12px] leading-relaxed text-ink-dim">
                    {reviewed.verdict.improvements.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-2.5 text-[11.5px] text-ink-faint">
                {reviewed.demoted
                  ? `Demoted automatically (#${reviewed.demoted.demoted})${
                      reviewed.demoted.recrowned ? ` — #${reviewed.demoted.recrowned.seq} is champion now` : ''
                    }. Posted to the board and recorded as a lesson and a pitfall.`
                  : reviewed.awaiting_confirmation
                    ? 'It recommends disqualifying this result — the demotion below is pre-filled with its own words. Nothing changes until you confirm.'
                    : 'Posted to the message board. Nothing was demoted.'}
              </div>
            </div>
          )}

          {demoting && (
            <div className="mb-5 rounded-xl border border-bad/40 bg-bad/5 p-4">
              <div className="text-[13px] font-medium text-bad">Demote #{c?.seq}</div>
              <div className="mt-1 text-[12px] leading-relaxed text-ink-dim">
                For a result the harness passed but review disqualified — look-ahead the mechanical test cannot see,
                a leak in an imported module, a score that is not real. It drops off the leaderboard, is never crowned
                again, and the reason becomes a team lesson, a project pitfall every future agent reads, and a post on
                the message board.
              </div>

              <div className="mt-3 text-[11px] uppercase tracking-wide text-ink-faint">The finding (pasted in full)</div>
              <textarea
                value={finding}
                onChange={(e) => setFinding(e.target.value)}
                rows={10}
                placeholder="Paste the review here — the whole analysis, including the corrected code if you have it."
                className="mt-1 w-full resize-y rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[11.5px] leading-relaxed text-ink outline-none focus:border-bad"
              />

              <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-ink-faint">
                    One-line lesson <span className="normal-case text-ink-faint/70">— this is what agents actually read</span>
                  </div>
                  <input
                    value={lesson}
                    onChange={(e) => setLesson(e.target.value)}
                    placeholder="Never align a resampled position onto bars inside its own window — lag by one full bar."
                    className="mt-1 w-full rounded-lg border border-seam bg-panel-hi px-3 py-1.5 text-[12px] text-ink outline-none focus:border-bad"
                  />
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-ink-faint">Reviewer</div>
                  <input
                    value={reviewer}
                    onChange={(e) => setReviewer(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-bad"
                  />
                </div>
              </div>

              <div className="mt-3 flex items-center gap-2">
                <button
                  disabled={busy || !finding.trim()}
                  onClick={submitDemotion}
                  className="rounded-lg border border-bad/50 bg-bad/10 px-3 py-1.5 font-mono text-[12px] text-bad disabled:opacity-40"
                >
                  {busy ? 'Demoting…' : 'Demote and teach the team'}
                </button>
                <button
                  onClick={() => setDemoting(false)}
                  className="font-mono text-[11.5px] text-ink-faint hover:text-ink-dim"
                >
                  cancel
                </button>
                {!finding.trim() && (
                  <span className="text-[11.5px] text-ink-faint">The finding is required — a demotion with no reason teaches nothing.</span>
                )}
              </div>
            </div>
          )}

          {done && (
            <div className="mb-5 rounded-xl border border-warn/40 bg-warn/5 p-4 text-[12.5px] leading-relaxed text-ink-dim">
              <div className="font-medium text-warn">Demoted #{done.demoted}. Recorded as a lesson and a project pitfall.</div>
              {done.quarantined?.length > 0 && (
                <div className="mt-1 text-bad">
                  Quarantined {done.quarantined.length === 1 ? 'the module' : 'the modules'} it was built on —{' '}
                  <span className="font-mono">{done.quarantined.join(', ')}</span>. Agents now see a do-not-use warning
                  with the reason, in the library and in the module source.
                </div>
              )}
              {done.cascaded?.length > 0 && (
                <div className="mt-1 text-bad">
                  Disqualified {done.cascaded.length} other result{done.cascaded.length === 1 ? '' : 's'} built on the
                  same module{done.cascaded.length === 1 ? '' : 's'} —{' '}
                  <span className="font-mono">#{done.cascaded.join(', #')}</span>. They inherited the defect, so their
                  scores were not real either; the leaderboard has been re-crowned from what is left.
                </div>
              )}
              {done.was_champion && (
                <div className="mt-1">
                  It held the title.{' '}
                  {done.recrowned ? `#${done.recrowned.seq} is the champion now.` : 'No eligible candidate is left to crown.'}
                </div>
              )}
              {done.descendants.length > 0 ? (
                <div className="mt-2">
                  <div className="text-ink">
                    {done.descendants.length} candidate{done.descendants.length === 1 ? '' : 's'} built on it — likely to
                    share the flaw. Nothing was changed automatically; open each and demote it if it does.
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5 font-mono text-[11px]">
                    {done.descendants.map((d) => (
                      <span
                        key={d.id}
                        title={d.rationale}
                        className={`rounded-md border px-2 py-0.5 ${
                          d.audit === 'fail' ? 'border-seam text-ink-faint line-through' : 'border-warn/40 text-warn'
                        }`}
                      >
                        #{d.seq}
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mt-1">Nothing was built on it.</div>
              )}
            </div>
          )}

          {c && tab === 'overview' && (
            <div className="space-y-5">
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">Hypothesis</div>
                <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink">{c.rationale || '—'}</div>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <Stat label={`${objective.metric_label} · holdout`} value={fmtMetric(kind, c.score)} strong />
                <Stat label={`${objective.metric_label} · in-sample`} value={fmtMetric(kind, c.is_score)} />
                <Stat
                  label="Look-ahead test"
                  value={<Pill tone={verdictTone(c.lookahead)}>{c.lookahead}</Pill>}
                />
              </div>
              {c.score_note && <Note tone="warn">{c.score_note}</Note>}
              {m.warning && <Note tone="warn">{m.warning}</Note>}
              {c.lookahead_detail && (
                <Note tone={c.lookahead === 'fail' ? 'bad' : 'neutral'}>{c.lookahead_detail}</Note>
              )}
              {c.audit !== 'none' && (
                <Note tone={c.audit === 'pass' ? 'good' : c.audit === 'fail' ? 'bad' : 'warn'}>
                  Audit {c.audit}
                  {c.audit_notes ? ` — ${c.audit_notes}` : c.audit === 'pending' ? ' — a model is reviewing the code before it can take the title' : ''}
                </Note>
              )}

              {c.returns?.length > 1 && (
                <div>
                  <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
                    Equity (growth of 1, daily, net of costs)
                  </div>
                  <EquityCurve returns={c.returns} split={objective.split_date} />
                </div>
              )}

              {(m.in_sample || m.holdout) && (
                <table className="w-full font-mono text-[11.5px]">
                  <thead>
                    <tr className="text-ink-faint">
                      <th className="py-1 text-left font-normal">metric</th>
                      <th className="py-1 text-right font-normal">in-sample</th>
                      {m.holdout && <th className="py-1 text-right font-normal">holdout</th>}
                      <th className="py-1 text-right font-normal">full</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ROWS.map((r) => (
                      <tr key={r.key} className="border-t border-seam/60">
                        <td className="py-1 text-ink-dim">{r.label}</td>
                        <td className="py-1 text-right text-ink">{cell(r.kind, m.in_sample?.[r.key])}</td>
                        {m.holdout && <td className="py-1 text-right text-ink">{cell(r.kind, m.holdout?.[r.key])}</td>}
                        <td className="py-1 text-right text-ink-dim">{cell(r.kind, m.full?.[r.key])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {(m.execution || m.extra || m.source) && (
                <div className="font-mono text-[11px] text-ink-faint">
                  {m.source && <div>returns: {m.source}</div>}
                  {m.execution && (
                    <div>
                      {m.execution.positions} positions · {m.execution.position_changes} changes · {m.execution.bars} bars
                      · {m.execution.cost_bps} bps cost · max |position| {m.execution.max_leverage} · priced on{' '}
                      {m.execution.price_column}
                    </div>
                  )}
                  {m.extra && (
                    <div>
                      {Object.entries(m.extra)
                        .map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toFixed(4) : v}`)
                        .join(' · ')}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {c && tab === 'code' && (
            <Markdown source={c.code ? '```python\n' + c.code + '\n```' : c.answer || '(no code)'} />
          )}

          {c && tab === 'output' && (
            <div className="space-y-4">
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">stdout</div>
                <pre className="max-h-[300px] overflow-auto whitespace-pre-wrap rounded-lg bg-panel-hi p-3 font-mono text-[11.5px] text-ink-dim">
                  {c.stdout || '(empty)'}
                </pre>
              </div>
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">stderr</div>
                <pre className="max-h-[300px] overflow-auto whitespace-pre-wrap rounded-lg bg-panel-hi p-3 font-mono text-[11.5px] text-bad/90">
                  {c.stderr || '(empty)'}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, strong }: { label: string; value: React.ReactNode; strong?: boolean }) {
  return (
    <div className="rounded-xl border border-seam bg-panel-hi/40 p-3">
      <div className="text-[10.5px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className={`mt-1 font-mono ${strong ? 'text-[20px] text-ink' : 'text-[15px] text-ink-dim'}`}>{value}</div>
    </div>
  )
}

function Note({ tone, children }: { tone: 'good' | 'bad' | 'warn' | 'neutral'; children: React.ReactNode }) {
  const cls =
    tone === 'good'
      ? 'border-good/30 bg-good/5 text-good'
      : tone === 'bad'
        ? 'border-bad/30 bg-bad/5 text-bad'
        : tone === 'warn'
          ? 'border-warn/30 bg-warn/5 text-warn'
          : 'border-seam bg-panel-hi/40 text-ink-dim'
  return <div className={`rounded-lg border px-3 py-2 text-[12px] leading-relaxed ${cls}`}>{children}</div>
}

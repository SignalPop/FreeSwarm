'use client'

import { useEffect, useRef, useState } from 'react'
import { Button, Panel, Pill } from '@/components/ui'
import CopyButton from '@/components/CopyButton'
import type { TimelineEntry } from '@/lib/agents'
import { clockTime, duration } from '@/lib/format'
import {
  purposeLabel,
  refusalLabel,
  refusalsTitle,
  workApi,
  type CandidateDoc,
  type ChatMarks,
  type IterationDoc,
  type ToolMarks,
  type WorkOutcome,
} from '@/lib/work'

// The two expanded views of the Work page: one iteration with its whole tool timeline, and one
// candidate with its scores, output and code. Each is fetched when its row opens.

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Math.abs(v) >= 1000 ? v.toFixed(0) : String(Number(v.toFixed(digits)))
}

export function stamp(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { hour12: false }) : '—'
}

export const OUTCOME_LABEL: Record<WorkOutcome, string> = {
  ok: 'submitted ok',
  error: 'submitted · errored',
  no_submission: 'no submission',
  interrupted: 'interrupted',
  running: 'running',
  done: 'done',
}

export const OUTCOME_TONE: Record<WorkOutcome, 'good' | 'bad' | 'warn' | 'accent' | 'neutral'> = {
  ok: 'good',
  error: 'bad',
  no_submission: 'warn',
  interrupted: 'warn',
  running: 'accent',
  done: 'neutral',
}

export const TONE_TEXT = {
  good: 'text-good',
  bad: 'text-bad',
  warn: 'text-warn',
  accent: 'text-accent',
  neutral: 'text-ink-faint',
} as const

export function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="border-t border-seam/60 pt-3">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-faint">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  )
}

export function CodeBlock({ text, empty, tone }: { text: string; empty: string; tone?: 'bad' }) {
  if (!text) return <div className="text-[12px] text-ink-faint">{empty}</div>
  return (
    <div className="relative">
      <CopyButton text={text} className="absolute right-2 top-2" />
      <pre
        className={`max-h-[360px] overflow-auto whitespace-pre-wrap break-all rounded-xl border border-seam bg-canvas p-3 pr-12 font-mono text-[11.5px] leading-relaxed ${
          tone === 'bad' ? 'text-bad/90' : 'text-ink-dim'
        }`}
      >
        {text}
      </pre>
    </div>
  )
}

function Kv({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 text-[12px]">
      <span className="w-[92px] shrink-0 font-mono text-[11px] text-ink-faint">{label}</span>
      <div className="min-w-0 flex-1 text-ink-dim">{children}</div>
    </div>
  )
}

/** A long text, shown as one line until opened. */
function Fold({ label, text, tone, open = false }: { label: string; text: string; tone?: 'bad' | 'warn'; open?: boolean }) {
  const [shown, setShown] = useState(open)
  if (!text || text === '{}') return null
  return (
    <div className="mt-1">
      <div className="flex items-center gap-1">
        <button onClick={() => setShown((v) => !v)} aria-expanded={shown} className="min-w-0 truncate text-left text-ink-faint hover:text-ink-dim">
          {shown ? '▾' : '▸'} {label}
          {!shown && <span className="ml-1.5 text-ink-faint/80">{text.replace(/\s+/g, ' ').slice(0, 140)}</span>}
        </button>
        {shown && <CopyButton text={text} label={`copy ${label}`} />}
      </div>
      {shown && (
        <pre
          className={`mt-1 max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded bg-panel p-2 font-mono text-[11px] ${
            tone === 'bad' ? 'text-bad/90' : tone === 'warn' ? 'text-warn' : 'text-ink-dim'
          }`}
        >
          {text}
        </pre>
      )}
    </div>
  )
}

/** One line saying what a tool call was asked to do. */
function argsSummary(name: string, args: unknown): string {
  if (!args || typeof args !== 'object') return args == null ? '' : String(args)
  const a = args as Record<string, unknown>
  if (typeof a.code === 'string') {
    const code = a.code as string
    const lines = code.split('\n')
    const first = lines.find((l) => {
      const t = l.trim()
      return t && !t.startsWith('import ') && !t.startsWith('from ') && !t.startsWith('#')
    })
    const extra = Object.entries(a)
      .filter(([k, v]) => k !== 'code' && k !== 'rationale' && v !== null && v !== '' && typeof v !== 'object')
      .map(([k, v]) => `${k}=${String(v).slice(0, 40)}`)
    return [`${lines.length} lines`, ...extra, first ? `· ${first.trim().slice(0, 100)}` : ''].filter(Boolean).join(' ')
  }
  const text = JSON.stringify(a)
  return name === 'team_board' && text === '{}' ? '' : text.slice(0, 180)
}

type Filter = 'all' | 'failures' | 'tools' | 'skipped'
type Entry = TimelineEntry & ToolMarks & ChatMarks

/** Not a failure: a policy refusal, a soft runner step, or a side request skipped (model busy). */
const refusedOrSkipped = (e: Entry) => (e.kind === 'tool' && (!!e.refused || !!e.soft)) || (e.kind === 'chat' && !!e.soft)

/** One step as plain text: its line plus everything its folds hold, for pasting into a bug or a chat. */
function stepText(e: Entry, t0: number, self: string): string {
  const at = `+${duration(e.at - t0)}`
  const part = (label: string, text: string | null | undefined) => (text && text !== '{}' ? `${label}:\n${text}` : '')
  const join = (...parts: string[]) => parts.filter(Boolean).join('\n\n')
  if (e.kind === 'asked') return `${at} request ${e.index + 1} sent${e.model && e.model !== self ? ` to ${e.model}` : ''}`
  if (e.kind === 'message') return join(`${at} runner -> model (follow-up)`, e.text)
  if (e.kind === 'chat') {
    const who = e.model && e.model !== self ? e.model : 'model'
    const toks = e.prompt_tokens != null ? ` · ${e.prompt_tokens} -> ${e.completion_tokens ?? 0} tok` : ''
    if (e.soft) {
      return join(`${at} ${who} ${e.soft_note || 'skipped (model busy)'} after ${e.seconds}s — ${purposeLabel(e.purpose)}${toks}`,
        part('why', e.soft_error))
    }
    const head = `${at} ${who} ${e.error ? `failed after ${e.seconds}s: ${e.error}` : `replied in ${e.seconds}s`}${toks}` +
      `${e.finish === 'length' ? ' · truncated (finish: length)' : ''}${e.tool_calls.length ? ` · calls ${e.tool_calls.join(', ')}` : ''}`
    return join(head, part('said', e.said), part('reasoning', e.reasoning))
  }
  const status = e.refused
    ? `refused · ${refusalLabel(e.refusal)}`
    : e.soft
      ? 'skipped'
      : e.failed
        ? e.recovered ? 'error (recovered)' : 'error'
        : 'ok'
  return join(
    `${at} ${e.name} ${status}${e.auto_repaired ? ' · auto-repaired' : ''} · ${e.seconds}s`,
    (e.failed || e.refused) && e.error_line ? e.error_line : '',
    part(e.refused ? "runner's message" : 'traceback / error', (e.failed || e.refused) ? e.error_tail : ''),
    e.soft ? e.soft_error ?? '' : '',
    part('repaired errors', e.auto_repaired ? (e.repair_errors ?? []).join('\n') : ''),
    part('inputs', JSON.stringify(e.args ?? {}, null, 2)),
    part('result', typeof e.result === 'string' ? e.result : JSON.stringify(e.result, null, 2)),
  )
}

/**
 * One iteration in full: assignment, what it concluded, what it submitted, and every step --
 * tool calls with their inputs and results (failures carry their error line and traceback),
 * the model's replies (errors and truncations marked), and the runner's follow-ups.
 */
export function IterationDetail({
  id,
  onClose,
  onLoaded,
  onOpenCandidate,
}: {
  id: string
  onClose: () => void
  onLoaded?: () => void
  onOpenCandidate: (objectiveId: string, candidateId: string) => void
}) {
  const [doc, setDoc] = useState<IterationDoc | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const onLoadedRef = useRef(onLoaded)
  onLoadedRef.current = onLoaded

  useEffect(() => {
    let cancelled = false
    setDoc(null)
    workApi
      .iteration(id)
      .then((d) => {
        if (cancelled) return
        setDoc(d)
        setErr(null)
        onLoadedRef.current?.()
      })
      .catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      cancelled = true
    }
  }, [id])

  if (!doc) {
    return <Panel className="p-5 text-[13px] text-ink-faint">{err ? <span className="text-bad">{err}</span> : 'Loading…'}</Panel>
  }

  const rec = doc.record
  const s = doc.summary
  const t0 = rec.started_at
  const timeline = (rec.timeline ?? []) as (Entry)[]
  const shown = timeline.filter((e) =>
    filter === 'all'
      ? true
      : filter === 'tools'
        ? e.kind === 'tool'
        : filter === 'skipped'
          ? refusedOrSkipped(e)
          : (e.kind === 'tool' && (e.failed || e.auto_repaired)) ||
            (e.kind === 'chat' && !e.soft && (!!e.error || e.finish === 'length')) ||
            e.kind === 'message',
  )
  const failed = timeline.filter((e) => e.kind === 'tool' && e.failed && !e.recovered).length
  const recovered = timeline.filter((e) => e.kind === 'tool' && ((e.failed && e.recovered) || e.auto_repaired)).length
  const refusedSkipped = timeline.filter(refusedOrSkipped).length
  const refusals = s.refusals ?? 0
  const softChats = s.chat_errors_soft ?? 0
  const refusedNote = refusals ? ` · ${refusals} refused` : ''
  const fixedNote = (n: number | undefined, repaired?: number) =>
    n ? ` · ${n} recovered${repaired ? ` (${repaired} auto-repaired)` : ''}` : ''
  const objectiveId = rec.objective?.id

  return (
    <Panel className="space-y-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[11px] text-ink-faint">
            <span>iteration {doc.id}</span>
            <span>·</span>
            <span>{doc.model}</span>
            {doc.role && doc.role !== 'search' && (
              <>
                <span>·</span>
                <span>{doc.role}</span>
              </>
            )}
          </div>
          <div className="text-[16px] font-medium leading-snug text-ink">
            {doc.agent} <span className="text-ink-dim">· {rec.mode}</span>
          </div>
        </div>
        <button onClick={onClose} className="rounded-lg border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:text-ink">
          close
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Pill tone={OUTCOME_TONE[s.outcome]} pulse={s.outcome === 'running'}>
          {s.outcome_text ? `${s.outcome_text}` : OUTCOME_LABEL[s.outcome]}
        </Pill>
        <span className="font-mono text-[11px] text-ink-faint">
          {stamp(rec.started_at)} → {rec.ended_at ? clockTime(rec.ended_at) : 'still running'} ·{' '}
          {duration((rec.ended_at ?? Date.now() / 1000) - rec.started_at)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11.5px] sm:grid-cols-4">
        {[
          ['tool calls', String(s.tool_calls)],
          [
            'run_python',
            s.experiments || s.experiments_refused
              ? `${s.experiments} · ${s.experiments_failed} failed${fixedNote(s.experiments_recovered)}${
                  s.experiments_refused ? ` · ${s.experiments_refused} refused` : ''
                }`
              : '0',
          ],
          ['tool errors', `${s.tool_errors}${fixedNote(s.tool_errors_recovered, s.tool_auto_repaired)}${refusedNote}`],
          [
            'chats',
            `${s.chats}${s.chat_errors ? ` · ${s.chat_errors} failed` : ''}${softChats ? ` · ${softChats} skipped (model busy)` : ''}`,
          ],
          ['truncated', String(s.truncations)],
          ['follow-ups', String(s.followups)],
          ['tokens', s.tokens ? `${s.tokens.prompt.toLocaleString()} → ${s.tokens.completion.toLocaleString()}` : '—'],
          ['steps', `${timeline.length}${s.timeline_dropped ? ` (+${s.timeline_dropped} not kept)` : ''}`],
        ].map(([k, v]) => (
          <div
            key={k}
            title={
              k === 'tool errors' && refusals
                ? refusalsTitle(refusals, s.refusals_by_kind, s.refusals_recovered ?? 0)
                : k === 'chats' && softChats
                  ? 'Side requests the runner made (auto-repair, answer to feedback) that were skipped because the model was busy: not errors'
                  : undefined
            }
          >
            <span className="text-ink-faint">{k} </span>
            <span className="text-ink">{v}</span>
          </div>
        ))}
      </div>

      <Section title="Assignment">
        <div className="space-y-1">
          <Kv label="objective">{rec.objective?.title ?? '—'}</Kv>
          {rec.parent && (
            <Kv label="parent">
              #{rec.parent.seq}
              {rec.parent.rank != null && ` (rank ${rec.parent.rank})`}
              {rec.parent.in_sample_score != null && ` · in-sample ${fmtNum(rec.parent.in_sample_score)}`}
              {rec.parent.diagnosis && <span className="text-ink-faint"> · {rec.parent.diagnosis}</span>}
            </Kv>
          )}
          {rec.audit_of && (
            <Kv label="auditing">
              #{rec.audit_of.seq}
              {rec.audit_of.model && ` by ${rec.audit_of.model}`}
            </Kv>
          )}
          {rec.why && <Kv label="why now">{rec.why}</Kv>}
          {rec.idea != null && rec.idea !== '' && <Kv label="idea tested">#{rec.idea}</Kv>}
          {(s.reason || rec.reason || rec.end_reason) && (
            <Kv label="ended">
              <span className="text-warn">{s.reason || rec.reason || rec.end_reason}</span>
            </Kv>
          )}
        </div>
      </Section>

      {s.hypothesis && (
        <Section
          title={
            s.hypothesis_from === 'submission'
              ? 'Hypothesis (submitted rationale)'
              : s.hypothesis_from === 'said'
                ? 'Hypothesis (its first reply)'
                : 'Hypothesis (its first reasoning)'
          }
        >
          <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-dim">{s.hypothesis}</p>
        </Section>
      )}

      {rec.submissions.length > 0 && (
        <Section title="Submitted">
          <div className="space-y-1.5">
            {rec.submissions.map((sub, i) => (
              <div key={i} className="rounded-lg border border-seam bg-panel-hi px-3 py-2 text-[12px]">
                <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                  <span className="text-ink">#{sub.seq ?? '?'}</span>
                  <span className={sub.status === 'ok' ? 'text-good' : 'text-bad'}>{sub.status}</span>
                  {sub.in_sample_score != null && <span className="text-ink-dim">in-sample {fmtNum(sub.in_sample_score)}</span>}
                  {sub.lookahead && (
                    <span className={sub.lookahead === 'fail' ? 'text-bad' : 'text-ink-dim'}>
                      look-ahead {sub.lookahead.split(' -- ')[0]}
                    </span>
                  )}
                  {sub.rank != null && <span className="text-ink-dim">rank {sub.rank}</span>}
                  <span className="ml-auto flex items-center gap-2">
                    <span className="text-ink-faint">{clockTime(sub.at)}</span>
                    {sub.candidate_id && objectiveId && (
                      <button
                        onClick={() => onOpenCandidate(objectiveId, sub.candidate_id!)}
                        className="rounded-md border border-seam px-2 py-0.5 text-[10.5px] text-ink-dim hover:border-accent/60 hover:text-ink"
                      >
                        open candidate
                      </button>
                    )}
                  </span>
                </div>
                {sub.error && <div className="mt-1 text-[11px] text-bad">{sub.error}</div>}
                {sub.rationale && <div className="mt-1 text-[11.5px] text-ink-dim">{sub.rationale}</div>}
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section
        title={`Steps (${shown.length}${shown.length !== timeline.length ? ` of ${timeline.length}` : ''})`}
        right={
          <span className="flex items-center gap-1">
            {shown.length > 0 && (
              <CopyButton
                text={() => shown.map((e) => stepText(e, t0, doc.model)).join('\n\n----\n\n')}
                label={`copy the ${shown.length} step${shown.length === 1 ? '' : 's'} shown`}
              />
            )}
            {(
              [
                ['all', 'all'],
                ['tools', 'tool calls'],
                ['failures', `failures & follow-ups${failed ? ` · ${failed}` : ''}${recovered ? ` (+${recovered} recovered)` : ''}`],
                ...(refusedSkipped || filter === 'skipped' ? [['skipped', `refused / skipped · ${refusedSkipped}`]] : []),
              ] as [Filter, string][]
            ).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setFilter(k)}
                className={`rounded-md px-2 py-0.5 font-mono text-[10.5px] ${
                  filter === k ? 'bg-panel-hi text-ink shadow-[inset_0_0_0_1px_var(--color-seam)]' : 'text-ink-faint hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </span>
        }
      >
        {rec.timeline_dropped ? (
          <div className="mb-2 font-mono text-[10.5px] text-ink-faint">{rec.timeline_dropped} earlier steps were not kept by the runner</div>
        ) : null}
        {shown.length === 0 && <div className="text-[12px] text-ink-faint">Nothing here.</div>}
        <ol className="space-y-1.5">
          {shown.map((e, i) => (
            <Step key={i} e={e} t0={t0} self={doc.model} />
          ))}
        </ol>
      </Section>
    </Panel>
  )
}

function Step({ e, t0, self }: { e: Entry; t0: number; self: string }) {
  const at = <span className="w-[52px] shrink-0 text-ink-faint">+{duration(e.at - t0)}</span>
  const copy = <CopyButton text={() => stepText(e, t0, self)} label="copy this step" className="-my-0.5 shrink-0" />
  if (e.kind === 'asked') {
    return (
      <li className="flex gap-2 font-mono text-[10.5px] text-ink-faint">
        {at}
        <span>
          request {e.index + 1} sent{e.model && e.model !== self ? ` to ${e.model}` : ''}
        </span>
      </li>
    )
  }
  if (e.kind === 'message') {
    return (
      <li className="flex gap-2 font-mono text-[10.5px]">
        {at}
        <div className="min-w-0 flex-1">
          <span className="text-warn">runner → model (follow-up)</span>
          <Fold label="text" text={e.text} />
        </div>
        {copy}
      </li>
    )
  }
  if (e.kind === 'chat') {
    const toks =
      e.prompt_tokens != null ? `${e.prompt_tokens.toLocaleString()} → ${(e.completion_tokens ?? 0).toLocaleString()} tok` : ''
    const truncated = e.finish === 'length'
    if (e.soft) {
      // A side request the runner made (auto-repair, answer to feedback) that the busy model
      // could not take: skipped by design, not an error.
      return (
        <li className="flex gap-2 font-mono text-[10.5px]">
          {at}
          <div className="min-w-0 flex-1">
            <div className="text-ink-faint" title="A side request the runner made, skipped because the model was busy: not an error">
              {e.model && e.model !== self ? `${e.model} ` : 'model '}
              {e.soft_note || 'skipped (model busy)'} after {e.seconds}s — {purposeLabel(e.purpose)}
              {toks && ` · ${toks}`}
            </div>
            {e.soft_error && <Fold label="why" text={e.soft_error} />}
          </div>
          {copy}
        </li>
      )
    }
    return (
      <li className="flex gap-2 font-mono text-[10.5px]">
        {at}
        <div className="min-w-0 flex-1">
          <div className={e.error ? 'text-bad' : 'text-ink-faint'}>
            {e.model && e.model !== self ? `${e.model} ` : 'model '}
            {e.error ? `failed after ${e.seconds}s: ${e.error}` : `replied in ${e.seconds}s`}
            {toks && ` · ${toks}`}
            {truncated && <span className="text-warn"> · truncated (finish: length)</span>}
            {e.finish && !truncated && e.finish !== 'stop' && e.finish !== 'tool_calls' && ` · ${e.finish}`}
            {e.tool_calls.length > 0 && <span className="text-ink-dim"> · calls {e.tool_calls.join(', ')}</span>}
          </div>
          {e.said && <Fold label="said" text={e.said} />}
          {e.reasoning && <Fold label="reasoning" text={e.reasoning} />}
        </div>
        {copy}
      </li>
    )
  }
  const resultText = typeof e.result === 'string' ? e.result : JSON.stringify(e.result, null, 2)
  const summary = argsSummary(e.name, e.args)
  // A failure fixed later in the iteration (recovered) stays visible but reads as dealt with.
  const broken = e.failed && !e.recovered
  const repairs = e.repair_errors ?? []
  // A policy refusal (over budget, truncated code) and a soft runner step are neither ok nor errors.
  const refused = !!e.refused
  const soft = !refused && !!e.soft
  return (
    <li className="flex gap-2 font-mono text-[10.5px]">
      {at}
      <div
        className={`min-w-0 flex-1 rounded-lg border px-2.5 py-1.5 ${
          broken
            ? 'border-bad/40 bg-bad/[0.06]'
            : e.failed
              ? 'border-bad/20 bg-panel-hi'
              : refused
                ? 'border-warn/30 bg-panel-hi'
                : 'border-seam bg-panel-hi'
        }`}
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[11.5px] ${broken ? 'text-bad' : soft ? 'text-ink-dim' : 'text-ink'}`}>{e.name}</span>
          {refused ? (
            <span
              className="rounded border border-warn/35 bg-warn/10 px-1.5 text-[10px] text-warn"
              title={
                e.refusal === 'experiment_budget'
                  ? "The runner refused this run: the iteration's experiment budget was used up. Not an error, not an experiment."
                  : e.refusal === 'truncated'
                    ? 'The call arrived with its code cut off, so the runner refused it without running it. Not an error.'
                    : 'Refused by the runner on purpose: not an error.'
              }
            >
              refused · {refusalLabel(e.refusal)}
            </span>
          ) : soft ? (
            <span className="text-ink-faint" title="A runner step that failed softly: the iteration went on without it. Not an error.">
              skipped
            </span>
          ) : (
            <span className={e.failed ? (broken ? 'text-bad' : 'text-bad/70') : 'text-good'}>{e.failed ? 'error' : 'ok'}</span>
          )}
          {refused && e.recovered && (
            <span className="rounded border border-good/35 bg-good/10 px-1.5 text-[10px] text-good" title="The agent resent the call and it ran.">
              resent ok
            </span>
          )}
          {e.failed && e.recovered && (
            <span
              className="rounded border border-good/35 bg-good/10 px-1.5 text-[10px] text-good"
              title="A later call of this tool in the same iteration worked: the agent fixed it. Not counted as failed."
            >
              recovered
            </span>
          )}
          {e.auto_repaired && (
            <span
              className="rounded border border-accent/35 bg-accent/10 px-1.5 text-[10px] text-accent"
              title="The script crashed and the runner had it repaired and run again: counted as recovered, not failed."
            >
              auto-repaired{e.repair_attempts ? ` · ${e.repair_attempts} attempt${e.repair_attempts === 1 ? '' : 's'}` : ''}
            </span>
          )}
          <span className="text-ink-faint">{e.seconds}s</span>
          {summary && <span className="min-w-0 flex-1 truncate text-ink-faint">{summary}</span>}
          <span className="ml-auto">{copy}</span>
        </div>
        {e.failed && e.error_line && (
          <div className={`mt-1 whitespace-pre-wrap break-words text-[11px] ${broken ? 'text-bad' : 'text-bad/70'}`}>
            {e.error_line}
          </div>
        )}
        {e.failed && e.error_tail && <Fold label="traceback / error" text={e.error_tail} tone="bad" />}
        {refused && e.error_line && <div className="mt-1 whitespace-pre-wrap break-words text-[11px] text-warn/80">{e.error_line}</div>}
        {refused && e.error_tail && <Fold label="runner's message" text={e.error_tail} tone="warn" />}
        {soft && e.soft_error && <div className="mt-1 whitespace-pre-wrap break-words text-[11px] text-ink-faint">{e.soft_error}</div>}
        {e.auto_repaired && repairs.length > 0 && (
          <Fold label={`repaired error${repairs.length === 1 ? '' : 's'}`} text={repairs.join('\n')} tone="warn" />
        )}
        <Fold label="inputs" text={JSON.stringify(e.args ?? {}, null, 2)} />
        <Fold label="result" text={resultText} />
      </div>
    </li>
  )
}

const SEGMENTS = ['in_sample', 'holdout', 'full'] as const
const METRIC_KEYS = ['sharpe', 'sortino', 'total_return', 'max_drawdown', 'calmar', 'win_rate', 'active_days', 'days', 'smoothness']

/** One candidate in full: rationale, scores by segment, look-ahead and audit, output, code. */
export function CandidateDetail({
  id,
  onClose,
  onLoaded,
  onOpenCandidate,
}: {
  id: string
  onClose: () => void
  onLoaded?: () => void
  onOpenCandidate: (objectiveId: string, candidateId: string) => void
}) {
  const [c, setC] = useState<CandidateDoc | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const onLoadedRef = useRef(onLoaded)
  onLoadedRef.current = onLoaded

  useEffect(() => {
    let cancelled = false
    setC(null)
    workApi
      .candidate(id)
      .then((d) => {
        if (cancelled) return
        setC(d)
        setErr(null)
        onLoadedRef.current?.()
      })
      .catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      cancelled = true
    }
  }, [id])

  if (!c) {
    return <Panel className="p-5 text-[13px] text-ink-faint">{err ? <span className="text-bad">{err}</span> : 'Loading…'}</Panel>
  }

  const seg = (k: string) => (c.metrics[k] ?? null) as Record<string, number | null> | null
  const segs = SEGMENTS.filter((k) => seg(k))
  const keys = METRIC_KEYS.filter((k) => segs.some((s) => typeof seg(s)?.[k] === 'number'))

  return (
    <Panel className="space-y-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[11px] text-ink-faint">
            <span>candidate {c.id}</span>
            <span>·</span>
            <span>{c.objective_title}</span>
          </div>
          <div className="text-[16px] font-medium leading-snug text-ink">
            #{c.seq} <span className="text-ink-dim">· {c.model ?? '—'}</span>
            {c.mode && <span className="text-ink-dim"> · {c.mode}</span>}
          </div>
        </div>
        <span className="flex shrink-0 items-center gap-2">
          <Button tone="ghost" className="px-3 py-1 text-[12px]" onClick={() => onOpenCandidate(c.objective_id, c.id)}>
            Open candidate view
          </Button>
          <button onClick={onClose} className="rounded-lg border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:text-ink">
            close
          </button>
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11.5px] sm:grid-cols-4">
        {[
          ['status', c.status],
          ['score', fmtNum(c.score)],
          ['in-sample', fmtNum(c.is_score)],
          ['holdout', fmtNum(c.holdout)],
          ['look-ahead', c.lookahead ?? '—'],
          ['audit', c.audit ?? '—'],
          ['eval', c.eval_seconds != null ? `${c.eval_seconds}s` : '—'],
          ['parent', c.parent_seq != null ? `#${c.parent_seq}` : '—'],
          ['submitted', stamp(c.created_at)],
          ['idea', c.idea != null ? `#${c.idea}` : '—'],
        ].map(([k, v]) => (
          <div key={k} className="min-w-0 truncate">
            <span className="text-ink-faint">{k} </span>
            <span
              className={
                (k === 'status' && v === 'error') || (k === 'look-ahead' && (v === 'fail' || v === 'error'))
                  ? 'text-bad'
                  : k === 'status'
                    ? 'text-good'
                    : 'text-ink'
              }
            >
              {v}
            </span>
          </div>
        ))}
      </div>

      {c.score_note && (
        <div className={`font-mono text-[11.5px] ${c.status === 'error' ? 'text-bad' : 'text-warn'}`}>{c.score_note}</div>
      )}

      <Section title="Rationale">
        <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-dim">{c.rationale || '—'}</p>
      </Section>

      {c.status === 'error' && (
        <Section title="Error">
          {c.error_line && <div className="mb-2 font-mono text-[12px] text-bad">{c.error_line}</div>}
          <CodeBlock text={c.stderr} empty="No stderr." tone="bad" />
        </Section>
      )}

      {segs.length > 0 && keys.length > 0 && (
        <Section title="Metrics">
          <div className="overflow-x-auto">
            <table className="font-mono text-[11.5px]">
              <thead>
                <tr className="text-ink-faint">
                  <th className="pr-6 text-left font-normal" />
                  {segs.map((s) => (
                    <th key={s} className="pr-6 text-right font-normal">
                      {s.replace('_', '-')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k}>
                    <td className="pr-6 text-ink-faint">{k.replace('_', ' ')}</td>
                    {segs.map((s) => (
                      <td key={s} className="pr-6 text-right text-ink">
                        {fmtNum(seg(s)?.[k] ?? null)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {(c.lookahead_detail || c.audit_notes) && (
        <Section title="Look-ahead and audit">
          {c.lookahead_detail && <Fold label="look-ahead detail" text={c.lookahead_detail} />}
          {c.audit_notes && <Fold label="audit notes" text={c.audit_notes} />}
        </Section>
      )}

      <Section title="Output (tail)">
        <CodeBlock text={c.stdout} empty="No stdout." />
        {c.status !== 'error' && c.stderr && (
          <div className="mt-2">
            <CodeBlock text={c.stderr} empty="" />
          </div>
        )}
      </Section>

      <Section title="Code">
        <details>
          <summary className="cursor-pointer font-mono text-[11px] text-ink-dim hover:text-ink">
            {c.code.split('\n').length} lines
          </summary>
          <div className="mt-2">
            <CodeBlock text={c.code} empty="No code." />
          </div>
        </details>
      </Section>

      {c.iterations.length > 0 && (
        <div className="font-mono text-[10.5px] text-ink-faint">submitted by iteration {c.iterations.join(', ')}</div>
      )}
    </Panel>
  )
}

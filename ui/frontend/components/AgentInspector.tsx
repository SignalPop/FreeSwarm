'use client'

import { useEffect, useRef, useState } from 'react'
import {
  agentActivity,
  type ActivityRecord,
  type AgentActivity,
  type AskedPrompt,
  type ForecastRequest,
  type ModelActivity,
  type PeerCall,
  type TimelineEntry,
} from '@/lib/agents'
import { clockTime, duration } from '@/lib/format'
import CopyButton from '@/components/CopyButton'
import PromptAnatomy from '@/components/PromptAnatomy'
import ForecastInputsTimeline, { type ForecastInputs } from '@/components/ForecastInputsTimeline'
import { Pill } from '@/components/ui'

const REFRESH_MS = 2000

/**
 * The agent inspector: click a model in the Swarm page's Resources panel to see what its
 * agents are actually working on -- the assignment, the exact system and iteration prompts
 * they were sent, every tool call with its inputs and a result snippet, what they submitted
 * and the tokens it took. For a forecaster, the recent forecast requests and who made them.
 * Live while open.
 */
export default function AgentInspector({
  model,
  kind,
  projectId,
  onClose,
}: {
  model: string
  kind: 'llm' | 'forecaster'
  projectId: string | null
  onClose: () => void
}) {
  const [doc, setDoc] = useState<ModelActivity | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [agentIdx, setAgentIdx] = useState(0)
  // 'current' = the iteration in progress; 'last' = the most recent finished one (its outputs
  // and the inputs that produced them); anything else = an older record's id.
  const [recordId, setRecordId] = useState<string | null>(null)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const d = await agentActivity.forModel(model, projectId)
        if (alive) {
          setDoc(d)
          setErr(null)
        }
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      }
    }
    void load()
    const t = setInterval(load, REFRESH_MS)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [model, projectId])

  // The parent re-renders on its own live poll, handing a new onClose each time: keep the
  // latest in a ref so focus is taken once, on open, and not stolen back every poll.
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current()
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      opener?.focus?.()
    }
  }, [])

  const agents = doc?.agents ?? []
  const agent: AgentActivity | undefined = agents[Math.min(agentIdx, Math.max(0, agents.length - 1))]
  const records = agent ? [...agent.records].reverse() : []
  const current = records[0]?.status === 'running' ? records[0] : undefined
  const last = records.find((r) => r.status !== 'running')
  const older = records.filter((r) => r !== current && r !== last)
  const tab = recordId ?? (current ? 'current' : 'last')
  const rec = tab === 'current' ? current : tab === 'last' ? last : records.find((r) => r.id === tab)
  const now = doc?.now ?? Date.now() / 1000

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`What ${model} is working on`}
        className="flex h-full w-full max-w-[920px] flex-col overflow-hidden border-l border-seam bg-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-seam px-5 py-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="truncate text-[15px] font-medium text-ink">{model}</span>
              <Pill tone="neutral">{kind === 'forecaster' ? 'forecaster' : 'agent inspector'}</Pill>
              {rec?.status === 'running' && (
                <Pill tone="good" pulse>
                  working
                </Pill>
              )}
            </div>
            <div className="mt-1.5 font-mono text-[10.5px] text-ink-faint">
              {kind === 'forecaster'
                ? 'recent forecast requests: who asked, for which series, with which inputs'
                : 'what it was asked, what it used as inputs, and what came of it'}
              {' · live'}
              {agent && ` · last update ${duration(agent.seconds_since_update)} ago`}
            </div>
          </div>
          <button
            ref={closeRef}
            onClick={onClose}
            className="shrink-0 rounded-md border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:border-accent/50 hover:text-ink"
          >
            close
          </button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
          {err && <div className="rounded-lg border border-bad/35 bg-bad/10 p-3 text-[12px] text-bad">{err}</div>}
          {!doc && !err && <div className="text-[12px] text-ink-faint">Loading…</div>}

          {doc && kind === 'forecaster' && <Forecasts rows={doc.forecasts} now={now} />}

          {doc && kind === 'llm' && agents.length === 0 && (
            <div className="rounded-lg border border-seam bg-panel-hi p-3 text-[12px] leading-relaxed text-ink-dim">
              Nothing recorded for this model yet. Each agent reports what it is asked from the start of its next
              iteration. If the swarm runner was started before the inspector existed, restart it
              (<span className="font-mono">run-swarm.bat</span>) to begin recording.
            </div>
          )}

          {agents.length > 1 && (
            <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Agents on this model">
              {agents.map((a, i) => (
                <button
                  key={a.agent}
                  role="tab"
                  aria-selected={i === agentIdx}
                  onClick={() => {
                    setAgentIdx(i)
                    setRecordId(null)
                  }}
                  className={`rounded-md border px-2.5 py-1 font-mono text-[11px] ${
                    i === agentIdx ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'
                  }`}
                >
                  {a.agent}
                  {a.role !== 'search' && <span className="ml-1 text-ink-faint">({a.role})</span>}
                </button>
              ))}
            </div>
          )}

          {records.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 border-b border-seam pb-2" role="tablist" aria-label="Iterations">
              {(
                [
                  ['current', 'Current', current ? `${current.mode} · ${statusText(current)}` : 'idle'],
                  ['last', 'Last', last ? `${clockTime(last.started_at)} · ${last.mode} · ${statusText(last)}` : 'none yet'],
                ] as const
              ).map(([key, label, sub]) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={tab === key}
                  onClick={() => setRecordId(key)}
                  className={`rounded-md border px-3 py-1 text-left ${
                    tab === key ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'
                  }`}
                >
                  <span className="text-[12px] font-medium">{label}</span>
                  <span className="ml-2 font-mono text-[10.5px] opacity-80">{sub}</span>
                </button>
              ))}
              {older.length > 0 && (
                <span className="ml-2 flex flex-wrap items-center gap-1.5">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">earlier</span>
                  {older.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => setRecordId(r.id)}
                      aria-pressed={tab === r.id}
                      title={r.objective?.title ?? undefined}
                      className={`rounded-md border px-2 py-0.5 font-mono text-[10.5px] ${
                        tab === r.id ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'
                      }`}
                    >
                      {clockTime(r.started_at)} · {r.mode} · {statusText(r)}
                    </button>
                  ))}
                </span>
              )}
            </div>
          )}

          {tab === 'current' && !current && records.length > 0 && (
            <div className="rounded-lg border border-seam bg-panel-hi p-3 text-[12px] text-ink-dim">
              Not working on anything right now. <button className="text-accent hover:underline" onClick={() => setRecordId('last')}>
                See its last iteration
              </button>{' '}
              -- what it produced and the inputs that produced it.
            </div>
          )}
          {rec && <RecordView rec={rec} now={now} self={model} />}

          {doc && doc.peer_calls.length > 0 && <PeerCalls rows={doc.peer_calls} />}

          {doc && kind === 'llm' && doc.forecasts.length > 0 && <Forecasts rows={doc.forecasts} now={now} />}
        </div>
      </div>
    </div>
  )
}

/** What the model produced in this iteration: every reply that said something, newest first,
 *  with the tools it chose -- the "outputs" beside the prompt and tool inputs that led to them. */
function Outputs({ rec }: { rec: ActivityRecord }) {
  const replies = rec.timeline.filter(
    (e): e is Extract<TimelineEntry, { kind: 'chat' }> => e.kind === 'chat' && (!!e.said || e.tool_calls.length > 0),
  )
  if (!replies.length) return null
  const newest = [...replies].reverse()
  return (
    <Section title={`Outputs (${replies.length} repl${replies.length === 1 ? 'y' : 'ies'})`}>
      <div className="space-y-1.5">
        {newest.map((e, i) => (
          <details key={i} open={i === 0} className="rounded-lg border border-seam bg-panel-hi px-3 py-2">
            <summary className="flex cursor-pointer flex-wrap items-center gap-2 font-mono text-[11px] text-ink-dim">
              <span className={i === 0 ? 'text-accent' : 'text-ink'}>{i === 0 ? 'last reply' : `reply ${replies.length - i}`}</span>
              <span className="text-ink-faint">+{duration(e.at - rec.started_at)}</span>
              {e.tool_calls.length > 0 && <span>→ {e.tool_calls.join(', ')}</span>}
              {e.completion_tokens != null && <span className="text-ink-faint">{e.completion_tokens.toLocaleString()} tok</span>}
              {e.finish && e.finish !== 'stop' && e.finish !== 'tool_calls' && (
                <span className="text-warn">finish: {e.finish}</span>
              )}
            </summary>
            {e.said ? (
              <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-ink-dim">
                {e.said}
              </pre>
            ) : (
              <div className="mt-1 text-[11.5px] text-ink-faint">(no text -- only tool calls; their inputs are under Inputs and steps)</div>
            )}
          </details>
        ))}
      </div>
    </Section>
  )
}

function statusText(r: ActivityRecord): string {
  if (r.status === 'running') return r.pending ? (r.pending.kind === 'tool' ? r.pending.name : 'thinking') : 'running'
  return r.outcome ?? r.status
}

function statusTone(r: ActivityRecord): 'good' | 'warn' | 'bad' | 'accent' | 'neutral' {
  if (r.status === 'running') return 'good'
  if (r.status === 'submitted') return r.submissions.at(-1)?.status === 'ok' ? 'accent' : 'warn'
  if (r.status === 'interrupted' || r.status === 'no submission') return 'warn'
  return 'neutral'
}

const MODE_LABEL: Record<string, string> = {
  explore: 'EXPLORE — a new idea',
  improve: 'IMPROVE — mutate a parent',
  build: 'BUILD — a reusable library module',
  audit: 'AUDIT — check a would-be champion',
  consolidate: 'CONSOLIDATE — merge the lessons',
  practices: 'PRACTICES — rewrite the team playbook',
  mentor: 'MENTOR — directions for the team',
  task: 'one-off task',
  chat: 'chat',
}

function RecordView({ rec, now, self }: { rec: ActivityRecord; now: number; self: string }) {
  const end = rec.ended_at ?? now
  const tok = rec.tokens ?? { prompt: 0, completion: 0, chats: 0 }
  const tools = rec.timeline.filter((e) => e.kind === 'tool').length
  return (
    <div className="space-y-5">
      <Section title="Assignment">
        <div className="space-y-2 rounded-lg border border-seam bg-panel-hi p-3 text-[12px]">
          <div className="flex flex-wrap items-center gap-2">
            <Pill tone="accent">{MODE_LABEL[rec.mode] ?? rec.mode}</Pill>
            <Pill tone={statusTone(rec)} pulse={rec.status === 'running'}>
              {statusText(rec)}
            </Pill>
            <span className="font-mono text-[10.5px] text-ink-faint">
              started {clockTime(rec.started_at)} · {duration(end - rec.started_at)}
              {rec.ended_at ? '' : ' so far'} · {tools} tool call{tools === 1 ? '' : 's'}
            </span>
          </div>
          <Kv label="objective">{rec.objective?.title ?? rec.task?.title ?? '—'}</Kv>
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
              {rec.audit_of.score != null && ` · score ${fmtNum(rec.audit_of.score)}`}
            </Kv>
          )}
          {rec.why && <Kv label="why now">{rec.why}</Kv>}
          <Kv label="idea tested">{rec.idea != null ? `#${rec.idea}` : '—'}</Kv>
          {rec.context && (
            <Kv label="context">
              <span className="font-mono text-[11px] text-ink-dim">
                {Object.entries(rec.context)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => `${k.replace('_', ' ')} ${v}`)
                  .join(' · ')}
              </span>
            </Kv>
          )}
          {rec.ideas_offered && rec.ideas_offered.length > 0 && (
            <Kv label="ideas offered">
              <ul className="space-y-1">
                {rec.ideas_offered.map((i) => (
                  <li key={i.id} className="text-ink-dim">
                    <span className="font-mono text-accent">#{i.id}</span>{' '}
                    <span className="font-mono text-[10.5px] text-ink-faint">
                      ({i.model}, tried {i.tried}×)
                    </span>{' '}
                    {i.text}
                  </li>
                ))}
              </ul>
            </Kv>
          )}
          {rec.pending && (
            <Kv label="right now">
              <span className="text-good">
                {rec.pending.kind === 'tool'
                  ? `running ${rec.pending.name}`
                  : `waiting on ${rec.pending.model === self ? 'the model' : rec.pending.model ?? 'the model'}`}{' '}
                for {duration(now - rec.pending.since)}
              </span>
            </Kv>
          )}
          <Kv label="tokens">
            <span className="font-mono text-[11px]">
              {tok.prompt.toLocaleString()} prompt · {tok.completion.toLocaleString()} completion · {tok.chats} request
              {tok.chats === 1 ? '' : 's'}
            </span>
          </Kv>
        </div>
      </Section>

      <Outputs rec={rec} />

      {rec.submissions.length > 0 && (
        <Section title="Submitted">
          <div className="space-y-1.5">
            {rec.submissions.map((s, i) => (
              <div key={i} className="rounded-lg border border-seam bg-panel-hi px-3 py-2 text-[12px]">
                <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                  <span className="text-ink">#{s.seq ?? '?'}</span>
                  <span className={s.status === 'ok' ? 'text-good' : 'text-bad'}>{s.status}</span>
                  {s.in_sample_score != null && <span className="text-ink-dim">in-sample {fmtNum(s.in_sample_score)}</span>}
                  {s.lookahead && (
                    <span className={s.lookahead === 'fail' ? 'text-bad' : 'text-ink-dim'}>look-ahead {s.lookahead}</span>
                  )}
                  {s.rank != null && <span className="text-ink-dim">rank {s.rank}</span>}
                  {s.judge_score != null && <span className="text-ink-dim">judge {s.judge_score}</span>}
                  {s.contender_for_best && <span className="text-accent">contender for best</span>}
                  {s.idea != null && s.idea !== '' && <span className="text-accent">idea #{s.idea}</span>}
                  <span className="ml-auto text-ink-faint">{clockTime(s.at)}</span>
                </div>
                {s.not_ranked && <div className="mt-1 text-[11px] text-warn">not ranked: {s.not_ranked}</div>}
                {s.error && <div className="mt-1 text-[11px] text-bad">{s.error}</div>}
                {s.rationale && <div className="mt-1 text-[11.5px] text-ink-dim">{s.rationale}</div>}
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section title="What it was asked">
        {rec.asked.length === 0 && <div className="text-[12px] text-ink-faint">No request sent yet.</div>}
        <div className="space-y-3">
          {rec.asked.map((a, i) => (
            <Asked key={i} a={a} n={i} self={self} many={rec.asked.length > 1} />
          ))}
        </div>
      </Section>

      <Section title={`Inputs and steps (${rec.timeline.length})`}>
        {rec.timeline_dropped ? (
          <div className="mb-2 font-mono text-[10.5px] text-ink-faint">{rec.timeline_dropped} earlier steps not kept</div>
        ) : null}
        {rec.timeline.length === 0 && <div className="text-[12px] text-ink-faint">Nothing yet.</div>}
        <ol className="space-y-1.5">
          {rec.timeline.map((e, i) => (
            <Step key={i} e={e} t0={rec.started_at} self={self} />
          ))}
        </ol>
      </Section>
    </div>
  )
}

function Asked({ a, n, self, many }: { a: AskedPrompt; n: number; self: string; many: boolean }) {
  return (
    <div className="space-y-2">
      {(many || a.model !== self) && (
        <div className="font-mono text-[10.5px] text-ink-faint">
          request {n + 1} · {clockTime(a.at)}
          {a.model && a.model !== self && <span className="text-remote"> · sent to {a.model}</span>}
        </div>
      )}
      <PromptAnatomy a={a} />
      {a.tools.length > 0 && (
        <div className="font-mono text-[10.5px] leading-relaxed text-ink-faint">
          tools offered: <span className="text-ink-dim">{a.tools.join(', ')}</span>
        </div>
      )}
    </div>
  )
}

/** A long prompt or result: collapsible, copyable, monospace. */
function TextBlock({ label, text, chars, open = false }: { label: string; text: string; chars?: number; open?: boolean }) {
  const [shown, setShown] = useState(open)
  const cut = chars != null && chars > text.length
  return (
    <div className="rounded-lg border border-seam">
      <div className="flex items-center gap-2 px-3 py-1.5">
        <button
          onClick={() => setShown((v) => !v)}
          aria-expanded={shown}
          className="flex flex-1 items-center gap-2 text-left font-mono text-[11px] text-ink-dim hover:text-ink"
        >
          <span className="inline-block w-3 text-ink-faint">{shown ? '▾' : '▸'}</span>
          {label}
          <span className="text-ink-faint">
            {(chars ?? text.length).toLocaleString()} chars{cut ? ' (middle cut)' : ''}
          </span>
        </button>
        <CopyButton text={text} label={`copy ${label.toLowerCase()}`} />
      </div>
      {shown && (
        <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap border-t border-seam bg-panel-hi p-3 font-mono text-[11.5px] text-ink-dim">
          {text}
        </pre>
      )}
    </div>
  )
}

const FORECAST_KEYS = ['model', 'column', 'columns', 'covariates', 'calendar', 'horizon', 'every', 'context', 'bar', 'dataset', 'name']

function Step({ e, t0, self }: { e: TimelineEntry; t0: number; self: string }) {
  const at = <span className="w-[52px] shrink-0 text-ink-faint">+{duration(e.at - t0)}</span>
  if (e.kind === 'asked') {
    return (
      <li className="flex gap-2 font-mono text-[10.5px] text-ink-faint">
        {at}
        <span>
          request {e.index + 1} sent{e.model && e.model !== self ? ` to ${e.model}` : ''} (see above)
        </span>
      </li>
    )
  }
  if (e.kind === 'message') {
    return (
      <li className="flex gap-2 font-mono text-[10.5px]">
        {at}
        <div className="min-w-0 flex-1">
          <span className="text-warn">runner → model: </span>
          <span className="whitespace-pre-wrap text-ink-dim">{e.text}</span>
        </div>
      </li>
    )
  }
  if (e.kind === 'chat') {
    const toks =
      e.prompt_tokens != null ? `${e.prompt_tokens.toLocaleString()} → ${(e.completion_tokens ?? 0).toLocaleString()} tok` : ''
    return (
      <li className="flex gap-2 font-mono text-[10.5px]">
        {at}
        <div className="min-w-0 flex-1 space-y-1">
          <div className={e.error ? 'text-bad' : 'text-ink-faint'}>
            {e.model && e.model !== self ? `${e.model} ` : 'model '}
            {e.error ? `failed after ${e.seconds}s: ${e.error}` : `replied in ${e.seconds}s`}
            {toks && ` · ${toks}`}
            {e.finish && e.finish !== 'stop' && e.finish !== 'tool_calls' && ` · ${e.finish}`}
            {e.tool_calls.length > 0 && <span className="text-ink-dim"> · calls {e.tool_calls.join(', ')}</span>}
          </div>
          {e.said && <div className="whitespace-pre-wrap font-sans text-[11.5px] text-ink-dim">{e.said}</div>}
          {e.reasoning && !e.said && (
            <details>
              <summary className="cursor-pointer text-ink-faint hover:text-ink-dim">reasoning</summary>
              <div className="mt-1 whitespace-pre-wrap text-ink-faint">{e.reasoning}</div>
            </details>
          )}
        </div>
      </li>
    )
  }
  const args = (e.args ?? {}) as Record<string, unknown>
  const isForecast = e.name === 'forecast' || e.name === 'forecast_feature'
  const argText = JSON.stringify(args, null, 2)
  const resultText = typeof e.result === 'string' ? e.result : JSON.stringify(e.result, null, 2)
  return (
    <li className="flex gap-2 font-mono text-[10.5px]">
      {at}
      <div className="min-w-0 flex-1 rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[11.5px] ${e.ok ? 'text-ink' : 'text-bad'}`}>{e.name}</span>
          <span className={e.ok ? 'text-good' : 'text-bad'}>{e.ok ? 'ok' : 'error'}</span>
          <span className="text-ink-faint">{e.seconds}s</span>
        </div>
        {isForecast && (
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-ink-dim">
            {FORECAST_KEYS.filter((k) => args[k] != null && args[k] !== '' && !(Array.isArray(args[k]) && !(args[k] as unknown[]).length)).map(
              (k) => (
                <span key={k}>
                  <span className="text-ink-faint">{k}</span> {Array.isArray(args[k]) ? (args[k] as unknown[]).join(', ') : String(args[k])}
                </span>
              ),
            )}
          </div>
        )}
        <Fold label="inputs" text={argText} />
        <Fold label="result" text={resultText} tone={e.ok ? undefined : 'bad'} />
      </div>
    </li>
  )
}

function Fold({ label, text, tone }: { label: string; text: string; tone?: 'bad' }) {
  const [shown, setShown] = useState(false)
  if (!text || text === '{}') return null
  return (
    <div className="mt-1">
      <div className="flex items-center gap-1">
        <button
          onClick={() => setShown((v) => !v)}
          aria-expanded={shown}
          className="text-ink-faint hover:text-ink-dim"
        >
          {shown ? '▾' : '▸'} {label}
          {!shown && <span className="ml-1.5 text-ink-faint/80">{text.replace(/\s+/g, ' ').slice(0, 110)}</span>}
        </button>
        {shown && <CopyButton text={text} label={`copy ${label}`} />}
      </div>
      {shown && (
        <pre
          className={`mt-1 max-h-[320px] overflow-auto whitespace-pre-wrap rounded bg-panel p-2 font-mono text-[11px] ${
            tone === 'bad' ? 'text-bad/90' : 'text-ink-dim'
          }`}
        >
          {text}
        </pre>
      )}
    </div>
  )
}

function PeerCalls({ rows }: { rows: PeerCall[] }) {
  return (
    <Section title="Used by other agents">
      <div className="mb-2 text-[11.5px] text-ink-faint">
        Audits and judging go to a different model than the one that wrote the code; these are requests other
        agents sent to this one.
      </div>
      <div className="space-y-2">
        {rows.map((p) => (
          <div key={p.record_id} className="rounded-lg border border-seam bg-panel-hi px-3 py-2">
            <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
              <span className="text-ink">{p.agent}</span>
              <span className="text-accent">{p.mode}</span>
              {p.objective && <span className="truncate text-ink-dim">{p.objective.title}</span>}
              <span className="ml-auto text-ink-faint">
                {clockTime(p.started_at)} · {p.chats} request{p.chats === 1 ? '' : 's'} ·{' '}
                {p.prompt_tokens.toLocaleString()} → {p.completion_tokens.toLocaleString()} tok
              </span>
            </div>
            {p.asked.map((a, i) => (
              <div key={i} className="mt-2">
                <PromptAnatomy a={a} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </Section>
  )
}

function Forecasts({ rows, now }: { rows: ForecastRequest[]; now: number }) {
  return (
    <Section title={`Forecast requests (${rows.length})`}>
      {rows.length === 0 && (
        <div className="text-[12px] text-ink-faint">
          No forecasts since the control plane started. Requests are listed here as agents (or the Forecast Lab) make
          them.
        </div>
      )}
      <div className="space-y-2">
        {rows.map((f) => {
          const r = f.request ?? {}
          const s = f.shape
          const active = now - f.last_at < 5
          return (
            <div key={String(f.rid)} className="rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[10.5px]">
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className={f.agent ? 'text-ink' : 'text-ink-dim'}>{f.agent ?? 'operator / API'}</span>
                <span className="text-accent">{f.via}</span>
                {f.objective_id && <span className="text-ink-faint">objective {f.objective_id}</span>}
                {f.model && <span className="text-ink-faint">{f.model}</span>}
                <span className={`ml-auto ${active ? 'text-good' : 'text-ink-faint'}`}>
                  {clockTime(f.first_at)} · {f.calls} call{f.calls === 1 ? '' : 's'} · {f.seconds.toFixed(1)}s
                </span>
              </div>
              {Object.keys(r).length > 0 && (
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-ink-dim">
                  {Object.entries(r).map(([k, v]) => (
                    <span key={k}>
                      <span className="text-ink-faint">{k}</span> {Array.isArray(v) ? v.join(', ') : String(v)}
                    </span>
                  ))}
                </div>
              )}
              <div className="mt-1 text-ink-faint">
                {s.kind ?? 'forecast'} · horizon {s.horizon ?? '?'} · {f.anchors.toLocaleString()} anchor
                {f.anchors === 1 ? '' : 's'}
                {s.context != null && ` × ${s.context} bars of context`}
                {s.targets && s.targets > 1 ? ` · ${s.targets} targets` : ''}
                {s.past_covariates?.length ? ` · inputs ${s.past_covariates.join(', ')}` : ''}
                {s.future_covariates?.length ? ` · known ahead ${s.future_covariates.join(', ')}` : ''}
                {s.samples ? ` · ${s.samples} samples` : ''}
              </div>
              {f.errors > 0 && (
                <div className="mt-1 text-bad">
                  {f.errors} failed{f.error ? `: ${f.error}` : ''}
                </div>
              )}
              {/* The input streams sent, with dates, on one time axis (objectives.input_streams). */}
              <ForecastInputsTimeline details={(f as { inputs?: ForecastInputs[] }).inputs} />
            </div>
          )
        })}
      </div>
    </Section>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{title}</h3>
      {children}
    </section>
  )
}

function Kv({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="w-[92px] shrink-0 font-mono text-[10.5px] text-ink-faint">{label}</span>
      <div className="min-w-0 flex-1 text-ink">{children}</div>
    </div>
  )
}

function fmtNum(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3)
}

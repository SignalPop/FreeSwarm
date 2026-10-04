'use client'

import { useRouter } from 'next/navigation'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, EmptyState, PageHeader, Panel, Pill } from '@/components/ui'
import CandidateView from '@/components/objective/CandidateView'
import {
  CandidateDetail,
  fmtNum,
  IterationDetail,
  OUTCOME_LABEL,
  OUTCOME_TONE,
  stamp,
  TONE_TEXT,
} from '@/components/WorkDetail'
import { duration } from '@/lib/format'
import { objectives, type ObjectiveDetail } from '@/lib/objectives'
import { projects } from '@/lib/projects'
import { usePoll } from '@/lib/usePoll'
import {
  refusalsTitle,
  workApi,
  type CandidateRow,
  type IterationRow,
  type WorkDoc,
  type WorkFilters,
  type WorkRow,
} from '@/lib/work'

const REFRESH_MS = 15_000
const PAGE = 150

const WINDOWS: { label: string; hours: number }[] = [
  { label: 'last 6h', hours: 6 },
  { label: 'last 12h', hours: 12 },
  { label: 'last 24h', hours: 24 },
  { label: 'last 3d', hours: 72 },
  { label: 'last 7d', hours: 168 },
]

const OUTCOMES: { key: NonNullable<WorkFilters['outcome']>; label: string }[] = [
  { key: '', label: 'any outcome' },
  { key: 'error', label: 'errors only' },
  { key: 'tool_errors', label: 'any failure (tool, chat, candidate)' },
  { key: 'no_submission', label: 'no submission only' },
  { key: 'interrupted', label: 'interrupted' },
  { key: 'ok', label: 'ok only' },
  { key: 'running', label: 'running' },
]

const selectCls =
  'rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent disabled:opacity-40'

function ago(ts: number | null | undefined): string {
  return ts ? `${duration(Date.now() / 1000 - ts)} ago` : '—'
}

/** "14:03" today, "Tue 14:03" this week, a date before that. */
function when(ts: number): string {
  const d = new Date(ts * 1000)
  const now = new Date()
  const hm = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false })
  if (d.toDateString() === now.toDateString()) return hm
  if (now.getTime() - d.getTime() < 6 * 86400_000) return `${d.toLocaleDateString(undefined, { weekday: 'short' })} ${hm}`
  return `${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${hm}`
}

function readHours(): number {
  try {
    const v = Number(localStorage.getItem('ft-work-hours'))
    return WINDOWS.some((w) => w.hours === v) ? v : 24
  } catch {
    return 24
  }
}

function Tile({
  label,
  value,
  sub,
  tone,
  title,
}: {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
  tone?: keyof typeof TONE_TEXT
  title?: string
}) {
  return (
    <Panel className="min-w-0 px-4 py-3">
      <div title={title}>
        <div className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">{label}</div>
        <div className={`mt-1.5 truncate font-mono text-[22px] leading-none ${tone ? TONE_TEXT[tone] : 'text-ink'}`}>{value}</div>
        {sub && <div className="mt-1.5 truncate font-mono text-[11px] text-ink-faint">{sub}</div>}
      </div>
    </Panel>
  )
}

function IterationCells({ it }: { it: IterationRow }) {
  const tone = OUTCOME_TONE[it.outcome]
  // Rows carry only the errors that stayed broken; ones fixed later in the iteration are counted apart.
  const lastErr = it.errors.filter((e) => (e.state ?? 'failed') === 'failed').at(-1)
  const recovered = it.tool_errors_recovered ?? 0
  const repaired = it.tool_auto_repaired ?? 0
  // Policy refusals and side requests skipped (model busy) are not failures: counted apart.
  const refused = it.refusals ?? it.refused_count ?? 0
  const skipped = it.chat_errors_soft ?? it.soft_count ?? 0
  return (
    <>
      <span className="font-mono text-[11px]">
        <span className="block text-ink-dim">iteration</span>
        <span className="block text-accent">{it.mode}</span>
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[13px] leading-snug text-ink">
          {it.agent}
          {it.parent_seq != null && <span className="text-ink-faint"> · from #{it.parent_seq}</span>}
          {it.objective_title && <span className="text-ink-faint"> · {it.objective_title}</span>}
        </span>
        {it.hypothesis && (
          <span className="line-clamp-2 break-words text-[12px] leading-snug text-ink-dim" title={it.hypothesis}>
            {it.hypothesis}
          </span>
        )}
        {lastErr && (
          <span className="block truncate font-mono text-[10.5px] text-bad" title={lastErr.line}>
            {lastErr.tool}: {lastErr.line}
            {it.error_count > 1 && <span className="text-ink-faint"> (+{it.error_count - 1} more)</span>}
          </span>
        )}
        {it.reason && !lastErr && <span className="block truncate font-mono text-[10.5px] text-warn">{it.reason}</span>}
      </span>
      <span className={`font-mono text-[11.5px] ${TONE_TEXT[tone]}`}>
        {it.outcome_text && (it.outcome === 'ok' || it.outcome === 'error') ? it.outcome_text : OUTCOME_LABEL[it.outcome]}
        {it.outcome === 'running' && <span className="ml-1 inline-block h-1.5 w-1.5 animate-dot rounded-full bg-accent" />}
      </span>
      <span className="font-mono text-[11px] leading-snug text-ink-dim">
        <span className="block">
          {it.tool_calls} tool{it.tool_calls === 1 ? '' : 's'}
          {it.tool_errors > 0 && <span className="text-bad"> · {it.tool_errors} failed</span>}
          {recovered > 0 && (
            <span
              className="text-ink-faint"
              title={`${recovered} failed call${recovered === 1 ? '' : 's'} fixed later in this iteration${
                repaired ? ` (${repaired} auto-repaired)` : ''
              }: not counted as failed`}
            >
              {' '}
              · {recovered} recovered
            </span>
          )}
          {refused > 0 && (
            <span className="text-warn" title={refusalsTitle(refused, it.refusals_by_kind, it.refusals_recovered ?? 0)}>
              {' '}
              · {refused} refused
            </span>
          )}
        </span>
        <span className="block text-ink-faint">
          {it.experiments} py{it.experiments_failed > 0 && <span className="text-bad"> ({it.experiments_failed} ✗)</span>}
          {it.chat_errors > 0 && <span className="text-bad"> · {it.chat_errors} chat err</span>}
          {skipped > 0 && (
            <span title={`${skipped} side request${skipped === 1 ? '' : 's'} skipped, model busy: not errors`}> · {skipped} skipped</span>
          )}
          {it.truncations > 0 && <span className="text-warn"> · {it.truncations} cut</span>}
          {it.followups > 0 && <span> · {it.followups} nudge{it.followups === 1 ? '' : 's'}</span>}
        </span>
      </span>
      <span className="text-right font-mono text-[11px] text-ink-faint">
        {it.ended ? duration(it.ended - it.at) : it.outcome === 'running' ? `${duration(Date.now() / 1000 - it.at)}…` : '—'}
      </span>
    </>
  )
}

function CandidateCells({ c }: { c: CandidateRow }) {
  const leak = c.lookahead === 'fail' || c.lookahead === 'error'
  return (
    <>
      <span className="font-mono text-[11px]">
        <span className="block text-ink-dim">candidate</span>
        <span className="block text-accent">{c.mode ?? '—'}</span>
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[13px] leading-snug text-ink">
          #{c.seq} · {c.agent ?? c.model ?? '—'}
          {c.parent_seq != null && <span className="text-ink-faint"> · from #{c.parent_seq}</span>}
          {c.objective_title && <span className="text-ink-faint"> · {c.objective_title}</span>}
        </span>
        {c.rationale && (
          <span className="line-clamp-2 break-words text-[12px] leading-snug text-ink-dim" title={c.rationale}>
            {c.rationale}
          </span>
        )}
        {c.error_line ? (
          <span
            className={`block truncate font-mono text-[10.5px] ${c.recovered ? 'text-ink-faint' : 'text-bad'}`}
            title={c.error_line}
          >
            {c.error_line}
          </span>
        ) : (
          c.score_note && <span className="block truncate font-mono text-[10.5px] text-warn">{c.score_note}</span>
        )}
      </span>
      <span className="font-mono text-[11.5px]">
        <span className={`block ${c.status === 'ok' ? 'text-good' : c.recovered ? 'text-ink-dim' : 'text-bad'}`}>
          #{c.seq} {c.status}
        </span>
        {c.recovered && (
          <span
            className="block text-[10.5px] text-ink-faint"
            title="Its iteration made it good: a later submission ran, or the runner auto-repaired the script"
          >
            recovered
          </span>
        )}
        {c.lookahead && c.lookahead !== 'skipped' && (
          <span className={`block text-[10.5px] ${leak ? 'text-bad' : 'text-ink-faint'}`}>look-ahead {c.lookahead}</span>
        )}
      </span>
      <span className="font-mono text-[11px] leading-snug text-ink-dim">
        {c.status === 'ok' ? (
          <>
            <span className="block">
              score <span className="text-ink">{fmtNum(c.score)}</span>
            </span>
            <span className="block text-ink-faint">
              IS {fmtNum(c.is_score)} · HO {fmtNum(c.holdout)}
            </span>
          </>
        ) : (
          <span className="text-ink-faint">—</span>
        )}
      </span>
      <span className="text-right font-mono text-[11px] text-ink-faint">{c.eval_seconds != null ? `${c.eval_seconds}s` : '—'}</span>
    </>
  )
}

const rowKey = (r: WorkRow) => `${r.kind}:${r.id}`

export default function WorkPage() {
  const router = useRouter()
  const [projectId, setProjectId] = useState<string | null | undefined>(undefined)
  const [hours, setHours] = useState(24)
  const [kind, setKind] = useState<'' | 'iteration' | 'candidate'>('')
  const [agent, setAgent] = useState('')
  const [outcome, setOutcome] = useState<NonNullable<WorkFilters['outcome']>>('')
  const [q, setQ] = useState('')
  const [limit, setLimit] = useState(PAGE)
  const [selected, setSelected] = useState<string | null>(null)
  const [showAllErrors, setShowAllErrors] = useState(false)
  const [openCand, setOpenCand] = useState<{ objective: ObjectiveDetail; id: string } | null>(null)
  const [candErr, setCandErr] = useState<string | null>(null)
  // Clear is a view marker kept server-side per project; "show all" looks past it for this view.
  const [includeCleared, setIncludeCleared] = useState(false)
  const [clearBusy, setClearBusy] = useState(false)
  const [clearErr, setClearErr] = useState<string | null>(null)

  useEffect(() => {
    setHours(readHours())
    projects
      .list()
      .then((r) => setProjectId(r.active))
      .catch(() => setProjectId(null))
  }, [])

  function chooseHours(h: number) {
    setHours(h)
    try {
      localStorage.setItem('ft-work-hours', String(h))
    } catch {
      /* private window */
    }
  }

  const list = usePoll<WorkDoc | null>(
    () =>
      projectId
        ? workApi.list(projectId, { hours, limit, agent, outcome, kind, q: q.trim(), includeCleared })
        : Promise.resolve(null),
    REFRESH_MS,
  )
  const { refresh } = list

  useEffect(() => {
    const t = setTimeout(refresh, 250) // debounce typing
    return () => clearTimeout(t)
  }, [projectId, hours, limit, agent, outcome, kind, q, includeCleared, refresh])

  // A filter change starts the list over at one page.
  useEffect(() => setLimit(PAGE), [hours, agent, outcome, kind, q, includeCleared])

  async function setClear(on: boolean) {
    if (!projectId) return
    setClearBusy(true)
    setClearErr(null)
    try {
      await (on ? workApi.clear(projectId) : workApi.unclear(projectId))
      setIncludeCleared(false)
      setSelected(null)
      refresh()
    } catch (e) {
      setClearErr(e instanceof Error ? e.message : String(e))
    } finally {
      setClearBusy(false)
    }
  }

  const doc = list.data
  const summary = doc?.summary
  const filter = `${hours}|${agent}|${outcome}|${kind}|${q.trim()}|${includeCleared}`
  const fetched: WorkRow[] = useMemo(() => doc?.items ?? [], [doc])

  // The open row keeps its place while the filter is unchanged, even if the next poll drops it
  // (a running iteration that just finished and no longer matches "running", say).
  const [pinned, setPinned] = useState<{ row: WorkRow; index: number; filter: string } | null>(null)
  useEffect(() => {
    if (selected === null) return
    const index = fetched.findIndex((r) => rowKey(r) === selected)
    if (index >= 0) setPinned({ row: fetched[index], index, filter })
  }, [fetched, selected, filter])

  const rows: WorkRow[] = useMemo(() => {
    if (selected === null || !pinned || rowKey(pinned.row) !== selected || pinned.filter !== filter) return fetched
    if (fetched.some((r) => rowKey(r) === selected)) return fetched
    const at = Math.min(pinned.index, fetched.length)
    return [...fetched.slice(0, at), pinned.row, ...fetched.slice(at)]
  }, [fetched, selected, pinned, filter])

  // Bring the expanded row into view when it opens below the fold. Tall details align to the top.
  const openRow = useRef<HTMLLIElement | null>(null)
  const reveal = useCallback(() => {
    requestAnimationFrame(() => {
      const el = openRow.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const vh = window.innerHeight
      if (r.top >= 0 && r.bottom <= vh) return
      el.scrollIntoView({ behavior: 'smooth', block: r.height > vh || r.top < 0 ? 'start' : 'nearest' })
    })
  }, [])
  const selectedShown = selected !== null && rows.some((r) => rowKey(r) === selected)
  useEffect(() => {
    if (selectedShown) reveal()
  }, [selected, selectedShown, reveal])

  async function openCandidate(objectiveId: string, candidateId: string) {
    setCandErr(null)
    try {
      setOpenCand({ objective: await objectives.get(objectiveId), id: candidateId })
    } catch (e) {
      setCandErr(e instanceof Error ? e.message : String(e))
    }
  }

  const counts = doc?.counts ?? { all: 0, iteration: 0, candidate: 0 }
  const tabs: { key: '' | 'iteration' | 'candidate'; label: string; n: number }[] = [
    { key: '', label: 'All', n: counts.all },
    { key: 'iteration', label: 'Iterations', n: counts.iteration },
    { key: 'candidate', label: 'Candidates', n: counts.candidate },
  ]
  const by = summary?.by_outcome ?? {}
  const errors = summary?.errors ?? []
  const toolRecovered = summary?.tool_errors_recovered ?? 0
  const toolRepaired = summary?.tool_auto_repaired ?? 0
  const expRecovered = summary?.experiments_recovered ?? 0
  const candRecovered = summary?.candidates_error_recovered ?? 0
  const recoveredHidden = toolRecovered + candRecovered
  const refusedAll = summary?.refusals ?? 0
  const expRefused = summary?.experiments_refused ?? 0
  const softChats = summary?.chat_errors_soft ?? 0
  const refusedTitle = refusalsTitle(refusedAll, summary?.refusals_by_kind, summary?.refusals_recovered ?? 0)
  const shownErrors = showAllErrors ? errors : errors.slice(0, 6)
  const best = summary?.best ?? []
  const clearedAt = doc?.cleared_at ?? null
  const hiding = !!doc?.hiding_cleared
  // The marker only matters when it falls inside the window (an older one hides nothing).
  const clearInWindow = doc != null && clearedAt != null && clearedAt > doc.window_since
  const windowLabel =
    hiding && clearedAt ? `since ${when(clearedAt)}` : (WINDOWS.find((w) => w.hours === hours)?.label ?? `last ${hours}h`)

  return (
    <div className="mx-auto max-w-[1440px] px-8 py-8">
      <PageHeader
        title="Work"
        subtitle="Everything the swarm tried, newest first: every iteration with its tool calls and errors, every candidate and how it scored or failed. Kept for 30 days."
        right={
          <div className="flex flex-wrap items-center justify-end gap-2">
            {doc && (
              <Pill tone={list.error ? 'warn' : 'neutral'}>
                {list.error ? 'refresh failed — showing the last good copy' : `updated ${ago(doc.now)}`}
              </Pill>
            )}
            <select className={selectCls} value={hours} onChange={(e) => chooseHours(Number(e.target.value))} aria-label="Time window">
              {WINDOWS.map((w) => (
                <option key={w.hours} value={w.hours}>
                  {w.label}
                </option>
              ))}
            </select>
            <Button tone="ghost" onClick={refresh}>
              Refresh
            </Button>
            <span title="Hide the work so far, so the page shows only what is new from now on. Nothing is deleted; Undo clear brings it back.">
              <Button tone="ghost" onClick={() => setClear(true)} disabled={!projectId || clearBusy}>
                Clear
              </Button>
            </span>
          </div>
        }
      />

      {projectId === null && (
        <EmptyState title="No active project" hint="Pick a project in the sidebar; the work log is kept per project." />
      )}

      {clearInWindow && clearedAt != null && (
        <div className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[11.5px] text-ink-faint">
          <span>
            {hiding ? (
              <>
                Showing work since <span className="text-ink-dim" title={stamp(clearedAt)}>{when(clearedAt)}</span> (cleared)
              </>
            ) : (
              <>
                Showing all work, including before the clear at{' '}
                <span className="text-ink-dim" title={stamp(clearedAt)}>{when(clearedAt)}</span>
              </>
            )}
          </span>
          <span>·</span>
          <button onClick={() => setIncludeCleared((v) => !v)} className="text-ink-dim hover:text-ink">
            {hiding ? 'Show all' : 'Show only new'}
          </button>
          <span>·</span>
          <button onClick={() => setClear(false)} disabled={clearBusy} className="text-ink-dim hover:text-ink disabled:opacity-40">
            Undo clear
          </button>
        </div>
      )}
      {clearErr && <div className="mb-3 text-[12px] text-bad">could not change the clear marker: {clearErr}</div>}

      {summary && (
        <>
          <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Tile
              label="Iterations"
              value={summary.iterations}
              sub={`${by.running ?? 0} running · ${by.done ?? 0} chores`}
            />
            <Tile
              label="Candidates"
              value={summary.candidates}
              sub={
                <>
                  <span className="text-good">{summary.candidates_ok} ok</span> ·{' '}
                  <span className={summary.candidates_error ? 'text-bad' : ''}>{summary.candidates_error} error</span>
                  {candRecovered > 0 && <span> ({candRecovered} recovered)</span>}
                </>
              }
            />
            <Tile
              label="No submission"
              value={by.no_submission ?? 0}
              tone={by.no_submission ? 'warn' : undefined}
              sub={`${by.interrupted ?? 0} interrupted · ${by.error ?? 0} errored`}
            />
            <Tile
              label="Experiments"
              value={summary.experiments}
              sub={
                <>
                  <span className={summary.experiments_failed ? 'text-bad' : ''}>{summary.experiments_failed} failed</span>
                  {expRecovered > 0 && <span> · {expRecovered} recovered</span>}
                  {expRefused > 0 && (
                    <span className="text-warn" title="run_python calls the runner refused (over the experiment budget, or truncated): not run, not failed">
                      {' '}
                      · {expRefused} refused
                    </span>
                  )}{' '}
                  · {summary.tool_calls} tool calls
                </>
              }
            />
            <Tile
              label="Tool errors"
              value={summary.tool_errors}
              tone={summary.tool_errors ? 'bad' : undefined}
              title={
                `Failed tool calls the agent did not fix later in the same iteration. ${toolRecovered} more failed and ` +
                `were fixed (recovered)${toolRepaired ? `, ${toolRepaired} of them auto-repaired by the runner` : ''}.` +
                (refusedAll ? ` ${refusedTitle}.` : '') +
                (softChats ? ` ${softChats} side request${softChats === 1 ? '' : 's'} skipped (model busy): not chat errors.` : '')
              }
              sub={
                <>
                  {(toolRecovered > 0 || refusedAll > 0) && (
                    <span className="block truncate">
                      {toolRecovered > 0 && (
                        <>
                          {toolRecovered} recovered{toolRepaired > 0 && ` (${toolRepaired} auto-repaired)`}
                        </>
                      )}
                      {toolRecovered > 0 && refusedAll > 0 && ' · '}
                      {refusedAll > 0 && (
                        <span className="text-warn" title={refusedTitle}>
                          {refusedAll} refused
                        </span>
                      )}
                    </span>
                  )}
                  <span className="block truncate">
                    {summary.chat_errors} chat errors
                    {softChats > 0 && <span className="text-ink-faint/80"> ({softChats} soft (model busy))</span>} ·{' '}
                    {summary.truncations} truncated
                  </span>
                </>
              }
            />
            <Tile
              label={best.length > 1 ? `Best score (${best.length} objectives)` : 'Best score'}
              value={best[0] ? fmtNum(best[0].score) : '—'}
              tone={best[0] ? 'good' : undefined}
              sub={
                best[0]
                  ? `#${best[0].seq} · IS ${fmtNum(best[0].is_score)} · HO ${fmtNum(best[0].holdout)} · ${best[0].model ?? ''}`
                  : 'nothing scored in this window'
              }
            />
          </div>

          {best.length > 1 && (
            <div className="mb-3 space-y-0.5 font-mono text-[11px] text-ink-faint">
              {best.map((b) => (
                <div key={b.objective_id} className="truncate">
                  best on <span className="text-ink-dim">{b.objective_title}</span>: <span className="text-good">{fmtNum(b.score)}</span>{' '}
                  #{b.seq} · {b.model}
                </div>
              ))}
            </div>
          )}

          {errors.length > 0 && (
            <Panel className="mb-3 px-4 py-3">
              <div className="mb-2 flex items-center justify-between gap-3">
                <h3 className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-faint">
                  Most common errors · {windowLabel}
                  {recoveredHidden + refusedAll + softChats > 0 && (
                    <span
                      className="ml-2 normal-case tracking-normal text-ink-faint/80"
                      title="Errors the agent fixed itself later in the same iteration (or the runner auto-repaired), policy refusals, and side requests skipped while the model was busy are not counted here; each iteration's detail still shows them."
                    >
                      ·{' '}
                      {[
                        recoveredHidden > 0 && `${recoveredHidden} recovered`,
                        refusedAll > 0 && `${refusedAll} refused`,
                        softChats > 0 && `${softChats} skipped`,
                      ]
                        .filter(Boolean)
                        .join(' · ')}{' '}
                      not counted
                    </span>
                  )}
                </h3>
                {errors.length > 6 && (
                  <button onClick={() => setShowAllErrors((v) => !v)} className="font-mono text-[11px] text-ink-dim hover:text-ink">
                    {showAllErrors ? 'show fewer' : `show all ${errors.length}`}
                  </button>
                )}
              </div>
              <ol className="space-y-0.5">
                {shownErrors.map((g) => (
                  <li key={g.line}>
                    <button
                      onClick={() => setQ(q === g.line ? '' : g.line)}
                      title="Show the work that hit this error"
                      className={`grid w-full grid-cols-[44px_minmax(0,1fr)_auto] items-center gap-3 rounded-lg px-2 py-1 text-left font-mono text-[11.5px] transition-colors ${
                        q === g.line ? 'bg-panel-hi' : 'hover:bg-panel-hi/60'
                      }`}
                    >
                      <span className="text-right text-ink">{g.count}×</span>
                      <span className="truncate text-bad">{g.line}</span>
                      <span className="text-[10.5px] text-ink-faint">
                        {Object.entries(g.sources)
                          .sort((a, b) => b[1] - a[1])
                          .map(([k, n]) => `${k} ${n}`)
                          .join(' · ')}{' '}
                        · {ago(g.last_at)}
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            </Panel>
          )}
        </>
      )}

      {projectId && (
        <Panel className="mb-3 flex flex-wrap items-center gap-2 px-4 py-2.5">
          {tabs.map((t) => (
            <button
              key={t.key || 'all'}
              onClick={() => setKind(t.key)}
              className={`rounded-lg px-3 py-1.5 text-[12.5px] transition-colors ${
                kind === t.key ? 'bg-panel-hi text-ink shadow-[inset_0_0_0_1px_var(--color-seam)]' : 'text-ink-dim hover:text-ink'
              }`}
            >
              {t.label}
              <span className="ml-1.5 font-mono text-[11px] text-ink-faint">{t.n}</span>
            </button>
          ))}
          <select className={selectCls} value={agent} onChange={(e) => setAgent(e.target.value)} aria-label="Agent or model">
            <option value="">every agent</option>
            {(summary?.agents ?? []).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
            {agent && !(summary?.agents ?? []).includes(agent) && <option value={agent}>{agent}</option>}
          </select>
          <select
            className={selectCls}
            value={outcome}
            onChange={(e) => setOutcome(e.target.value as NonNullable<WorkFilters['outcome']>)}
            aria-label="Outcome"
          >
            {OUTCOMES.map((o) => (
              <option key={o.key || 'any'} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>
          <input
            className="ml-auto min-w-[260px] rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
            placeholder="Search hypothesis, error, tool, #seq…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </Panel>
      )}

      {list.error && <div className="mb-3 text-[12px] text-bad">{list.error}</div>}
      {candErr && <div className="mb-3 text-[12px] text-bad">could not open the candidate: {candErr}</div>}

      {projectId && (
        <div className="min-w-0">
          {rows.length === 0 && !list.loading && doc ? (
            <EmptyState
              title={
                doc.counts.all === 0
                  ? hiding && clearedAt
                    ? `No new work since you cleared at ${when(clearedAt)}`
                    : `No work in the ${windowLabel}`
                  : 'Nothing matches these filters'
              }
              hint={
                doc.counts.all === 0
                  ? hiding
                    ? 'New iterations and candidates show up here as the swarm runs them; "Show all" brings back the earlier work.'
                    : 'Iterations show up here as the swarm runs them; widen the window to look further back.'
                  : 'Clear the search or pick another outcome or agent.'
              }
            />
          ) : (
            <Panel className="overflow-hidden">
              <div className="grid grid-cols-[72px_84px_minmax(0,1fr)_132px_184px_64px] gap-3 border-b border-seam px-4 py-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">
                <span>when</span>
                <span>kind</span>
                <span>what</span>
                <span>outcome</span>
                <span>stats</span>
                <span className="text-right">took</span>
              </div>
              <ol>
                {rows.map((r) => {
                  const key = rowKey(r)
                  const open = key === selected
                  return (
                    <li
                      key={key}
                      ref={open ? openRow : undefined}
                      className={`border-b border-seam/50 transition-colors last:border-0 ${open ? 'bg-panel-hi' : 'hover:bg-panel-hi/50'}`}
                    >
                      <button
                        onClick={() => setSelected(open ? null : key)}
                        aria-expanded={open}
                        aria-controls={`work-detail-${key}`}
                        className="grid w-full min-w-0 grid-cols-[72px_84px_minmax(0,1fr)_132px_184px_64px] items-center gap-3 px-4 py-2.5 text-left"
                      >
                        <span className="font-mono text-[11px] text-ink-dim" title={stamp(r.at)}>
                          {when(r.at)}
                        </span>
                        {r.kind === 'iteration' ? <IterationCells it={r} /> : <CandidateCells c={r} />}
                      </button>
                      {open && (
                        <div id={`work-detail-${key}`} className="px-3 pb-3">
                          {r.kind === 'iteration' ? (
                            <IterationDetail
                              id={r.id}
                              onClose={() => setSelected(null)}
                              onLoaded={reveal}
                              onOpenCandidate={openCandidate}
                            />
                          ) : (
                            <CandidateDetail
                              id={r.id}
                              onClose={() => setSelected(null)}
                              onLoaded={reveal}
                              onOpenCandidate={openCandidate}
                            />
                          )}
                        </div>
                      )}
                    </li>
                  )
                })}
              </ol>
              {doc && doc.total > rows.length && (
                <div className="flex items-center justify-between border-t border-seam px-4 py-2.5 font-mono text-[11px] text-ink-faint">
                  <span>
                    showing {rows.length} of {doc.total}
                  </span>
                  <Button tone="ghost" className="px-3 py-1 text-[12px]" onClick={() => setLimit((n) => n + PAGE)}>
                    Show {Math.min(PAGE, doc.total - rows.length)} more
                  </Button>
                </div>
              )}
            </Panel>
          )}
        </div>
      )}

      {openCand && (
        <CandidateView
          objective={openCand.objective}
          candidateId={openCand.id}
          onClose={() => setOpenCand(null)}
          onOpenCandidate={(id) => setOpenCand((o) => (o ? { ...o, id } : o))}
          onOpenChat={() => {
            setOpenCand(null)
            router.push('/chat')
          }}
        />
      )}
    </div>
  )
}

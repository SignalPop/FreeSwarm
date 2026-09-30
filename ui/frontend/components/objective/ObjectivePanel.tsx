'use client'

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { duration } from '@/lib/format'
import {
  ensembleEligible,
  fmtMetric,
  holdoutScore,
  isDisqualified,
  isRanked,
  objectives,
  robustRank,
  robustRanking,
  type Candidate,
  type DeleteCandidatesResult,
  type Direction,
  type Objective,
  type ObjectiveDetail,
  type RobustRank,
} from '@/lib/objectives'
import { Button, Panel, Pill } from '@/components/ui'
import { ProgressChart } from './Charts'
import CandidateView, { verdictTone } from './CandidateView'
import LibraryTab from './Library'
import PlaybookTab from './Playbook'
import IdeasTab from './Ideas'
import TeamMemory from './TeamMemory'
import ForecastsTab from './ForecastsTab'
import DeciPlots from './DeciPlots'
import RegimeLab from './RegimeLab'
import LookaheadRetest from './LookaheadRetest'
import TradeBoard from './TradeBoard'
import { DangerButton, DangerLink, DeleteAllRow, PickBox, RowDelete } from './Prune'
import CombineForm from './CombineForm'

const POLL_MS = 4000

/** The project's objectives and the selected one's live detail, polled together. */
export function useObjectives(projectId: string | null) {
  const [list, setList] = useState<Objective[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<ObjectiveDetail | null>(null)
  const [ranked, setRanked] = useState<Candidate[]>([])
  // Demoted / look-ahead-failed results. They cannot be crowned, but they are shown, so a
  // result the operator just disqualified is visibly gone rather than silently absent.
  const [disqualified, setDisqualified] = useState<Candidate[]>([])
  const [recent, setRecent] = useState<Candidate[]>([])
  // How much of the leaderboard to fetch. The top 8 is the normal view; pruning clones
  // means seeing (and ticking) the ranks below it too.
  const [rankLimit, setRankLimit] = useState(8)
  // How many of the newest candidates the Recent tab holds; scrolling to its end loads older ones.
  const [recentLimit, setRecentLimit] = useState(12)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((n) => n + 1), [])

  useEffect(() => {
    if (!projectId) return
    let alive = true
    async function load() {
      try {
        const l = await objectives.list(projectId as string)
        if (!alive) return
        setList(l.objectives)
        // Keep the operator's choice; otherwise follow the running objective.
        const keep = l.objectives.find((o) => o.id === selected)
        const pick = keep ?? l.objectives.find((o) => o.status === 'running') ?? l.objectives[0] ?? null
        if (!pick) {
          setDetail(null)
          setRanked([])
          setDisqualified([])
          setRecent([])
          if (selected !== null) setSelected(null)
          return
        }
        if (pick.id !== selected) setSelected(pick.id)
        const [d, r, rc] = await Promise.all([
          objectives.get(pick.id),
          objectives.candidates(pick.id, 'rank', rankLimit),
          objectives.candidates(pick.id, 'recent', recentLimit),
        ])
        if (!alive) return
        setDetail(d)
        setRanked(r.candidates)
        setDisqualified(r.disqualified ?? [])
        setRecent(rc.candidates)
        setError(null)
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    void load()
    const t = setInterval(load, POLL_MS)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [projectId, selected, tick, rankLimit, recentLimit])

  return {
    list, selected, setSelected, detail, ranked, disqualified, recent, error, refresh,
    rankLimit, setRankLimit, recentLimit, setRecentLimit,
  }
}

/** The foot of a paged list: how much of it is loaded, and the next 50 (also loaded by scrolling here). */
function ListEnd({ shown, total, what, onMore }: { shown: number; total: number; what: string; onMore: () => void }) {
  return (
    <div className="flex items-center gap-3 py-2 font-mono text-[10.5px] text-ink-faint">
      <span>
        showing {shown} of {total}
      </span>
      {shown < total && (
        <button type="button" onClick={onMore} className="hover:text-accent">
          scroll or click for 50 {what} ↓
        </button>
      )}
    </div>
  )
}

function ago(ts: number | null | undefined): string {
  return ts ? `${duration(Date.now() / 1000 - ts)} ago` : '—'
}

/**
 * A row of tabs that never spills out of its panel: when they do not all fit, the row scrolls
 * sideways behind a left and a right arrow, each greyed out once there is nothing more that way.
 * The selected tab is always scrolled into view.
 */
function TabStrip<K extends string>({
  tabs,
  active,
  onPick,
}: {
  tabs: readonly (readonly [K, string])[]
  active: K
  onPick: (k: K) => void
}) {
  const row = useRef<HTMLDivElement>(null)
  const [more, setMore] = useState({ left: false, right: false })
  const measure = useCallback(() => {
    const el = row.current
    if (el) setMore({ left: el.scrollLeft > 1, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 1 })
  }, [])
  const labels = tabs.map(([, label]) => label).join('|')
  useEffect(() => {
    measure()
    const el = row.current
    if (!el) return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [measure, labels])
  // Horizontal only: scrollIntoView would also scroll the page to the tabs.
  useEffect(() => {
    const el = row.current
    const btn = el?.querySelector<HTMLElement>('[data-active]')
    if (!el || !btn) return
    if (btn.offsetLeft < el.scrollLeft) el.scrollLeft = btn.offsetLeft
    else if (btn.offsetLeft + btn.offsetWidth > el.scrollLeft + el.clientWidth)
      el.scrollLeft = btn.offsetLeft + btn.offsetWidth - el.clientWidth
  }, [active])

  const arrow = (dir: -1 | 1) => (
    <button
      type="button"
      aria-label={dir < 0 ? 'scroll tabs left' : 'scroll tabs right'}
      disabled={dir < 0 ? !more.left : !more.right}
      onClick={() => row.current?.scrollBy({ left: dir * row.current.clientWidth * 0.6, behavior: 'smooth' })}
      className="grid w-6 shrink-0 place-items-center text-ink-dim hover:text-accent disabled:text-ink-faint/30"
    >
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d={dir < 0 ? 'M6.5 1.5 3 5l3.5 3.5' : 'M3.5 1.5 7 5 3.5 8.5'} />
      </svg>
    </button>
  )
  const scrolls = more.left || more.right
  return (
    <div className="mt-2 flex border-b border-seam">
      {scrolls && arrow(-1)}
      <div
        ref={row}
        onScroll={measure}
        className="flex min-w-0 flex-1 gap-1 overflow-x-auto scrollbar-none [&::-webkit-scrollbar]:hidden"
      >
        {tabs.map(([k, label]) => (
          <button
            key={k}
            data-active={k === active ? '' : undefined}
            onClick={() => onPick(k)}
            className={`shrink-0 whitespace-nowrap border-b-2 px-3 py-1.5 font-mono text-[11px] ${
              k === active ? 'border-accent text-accent' : 'border-transparent text-ink-faint hover:text-ink-dim'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {scrolls && arrow(1)}
    </div>
  )
}

/**
 * The objective the swarm is pursuing: how it is measured, the current champion, the shape
 * of the search so far, the leaderboard, and the team's accumulated lessons.
 */
export default function ObjectivePanel({
  state,
  onNew,
}: {
  state: ReturnType<typeof useObjectives>
  onNew: () => void
}) {
  const router = useRouter()
  const { list, detail: o, ranked, disqualified, recent, error, setSelected, refresh, rankLimit, setRankLimit, recentLimit, setRecentLimit } =
    state
  const [openId, setOpenId] = useState<string | null>(null)
  const [tab, setTab] = useState<
    'leaderboard' | 'trades' | 'recent' | 'memory' | 'forecasts' | 'deci' | 'regimes' | 'library' | 'playbook' | 'lessons' | 'steering' | 'ideas'
  >('leaderboard')
  const [busy, setBusy] = useState(false)
  const [rowErr, setRowErr] = useState<string | null>(null)

  if (!o) {
    return (
      <Panel className="mb-3 flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-[13px] text-ink">No objective</div>
          <div className="text-[12px] text-ink-faint">
            Give the swarm a standing goal and a way to measure it — it will keep producing better
            candidates until you stop it. {error && <span className="text-bad">{error}</span>}
          </div>
        </div>
        <Button tone="primary" onClick={onNew}>
          New objective
        </Button>
      </Panel>
    )
  }

  const kind = o.metric.kind
  const higher = o.metric.higher_is_better
  const best = o.best
  const hours = o.last_candidate_at && o.created_at ? Math.max(0.05, (o.last_candidate_at - o.created_at) / 3600) : 0
  const rate = hours && o.candidates ? (o.candidates / hours).toFixed(1) : '—'

  async function setDirection(d: Direction) {
    if (!o) return
    setBusy(true)
    try {
      await objectives.setDirection(o.id, d)
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function setIntraday(on: boolean) {
    if (!o) return
    if (
      !window.confirm(
        on
          ? 'Trade intraday only? Every position is forced flat at each day’s last bar. Every scored candidate is re-marked under the new rule in the background, best first (candidates whose positions were not kept are run once more), and the leaderboard changes as that goes.'
          : 'Allow holding positions overnight? Every scored candidate is re-marked under the new rule in the background.',
      )
    )
      return
    setBusy(true)
    try {
      await objectives.setIntraday(o.id, on)
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function setMinTrades(v: number) {
    if (!o) return
    const perDay = o.metric.min_trades_per_day ?? 0
    if (
      !window.confirm(
        (v
          ? `Require at least ${v} in-sample trades in all? Candidates with fewer are not ranked, and the swarm is told from its next brief.`
          : 'Remove the trade floor?') +
          (perDay ? ` The older ${perDay}/day quota is removed too.` : '') +
          ' Scored candidates are ranked again at once from their stored trade counts.',
      )
    )
      return
    setBusy(true)
    try {
      await objectives.setMinTrades(o.id, perDay ? { min_trades: v, min_trades_per_day: 0 } : { min_trades: v })
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function setTradeLimit(v: number) {
    if (!o) return
    if (
      !window.confirm(
        v
          ? `Allow at most ${v} trade${v > 1 ? 's' : ''} opened per day? Entries past the limit are ignored by the task server (exits and stops always go through). Every scored candidate is re-scored from its kept actions in the background, best first, and the swarm is told from its next brief.`
          : 'Remove the daily trade limit? Every scored candidate is re-scored in the background.',
      )
    )
      return
    setBusy(true)
    try {
      await objectives.setTradeLimit(o.id, v)
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function rescoreAll() {
    if (!o) return
    if (!window.confirm('Re-score every candidate from its kept actions under the task server’s current valuation (costs, fills, limits)? Runs in the background, best first.'))
      return
    setBusy(true)
    try {
      await objectives.rescore(o.id)
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function setSideShare(v: number) {
    if (!o) return
    if (
      !window.confirm(
        v
          ? `Require longs and shorts to each be at least ${Math.round(v * 100)}% of a candidate's in-sample trades? One-sided candidates go unranked. Every scored candidate is re-marked in the background to count its trades.`
          : 'Drop the both-sides requirement? Every scored candidate is re-marked in the background.',
      )
    )
      return
    setBusy(true)
    try {
      await objectives.setSideShare(o.id, v)
      refresh()
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function setStatus(s: Objective['status']) {
    if (!o) return
    if (s === 'stopped' && !window.confirm('Stop this objective? Its candidates and lessons are kept; you can resume it later.'))
      return
    setBusy(true)
    try {
      await objectives.setStatus(o.id, s)
      refresh()
    } finally {
      setBusy(false)
    }
  }

  async function remove() {
    if (!o) return
    if (!window.confirm(`Delete "${o.title}" with all ${o.candidates} candidates and its lessons? This cannot be undone.`))
      return
    await objectives.remove(o.id)
    setSelected(null)
    refresh()
  }

  /** Delete lessons or steering notes: one (the ×) or every one ("Delete all…"). */
  async function dropRows(kind: 'lessons' | 'notes', ids: number[] | 'all', question: string) {
    if (!o || !window.confirm(question)) return
    setRowErr(null)
    try {
      await objectives.deleteRows(o.id, kind, ids === 'all' ? { all: true } : { ids })
      refresh()
    } catch (e) {
      setRowErr(e instanceof Error ? e.message : String(e))
    }
  }

  // Counted from the chart's points (every candidate), not the rows on screen: "delete all
  // ranked" removes the whole leaderboard, including the ranks below the ones shown.
  const rankedCount = o.points.filter(isRanked).length
  const disqualifiedCount = o.points.filter(isDisqualified).length

  const describe = [
    o.metric_label + (higher ? '' : ' (lower is better)'),
    o.split_date
      ? robustRanking(o)
        ? `ranked on the weaker of in-sample and the holdout from ${o.split_date}, × equity-curve smoothness`
        : `ranked on the holdout from ${o.split_date}`
      : null,
    o.metric.price_column ? `positions priced on ${o.metric.price_column}, ${o.metric.cost_bps} bps` : null,
    o.lookahead_check ? 'look-ahead test' : null,
    o.require_audit ? 'audited' : null,
  ].filter(Boolean)

  return (
    <Panel className="mb-3 p-4">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] uppercase tracking-wide text-ink-faint">Objective</span>
            <Pill tone={o.status === 'running' ? 'good' : o.status === 'paused' ? 'warn' : 'neutral'} pulse={o.status === 'running'}>
              {o.status}
            </Pill>
            {o.evaluating > 0 && (
              // Hover: which candidates are being scored, whose they are, and for how long.
              <span
                tabIndex={0}
                className="cursor-help rounded-full outline-none focus-visible:ring-2 focus-visible:ring-accent"
                title={
                  o.points
                    .filter((p) => p.status === 'evaluating')
                    .map((p) => `#${p.seq} by ${p.model ?? 'unknown'} -- scoring for ${duration(Date.now() / 1000 - p.created_at)}`)
                    .join('\n') || `${o.evaluating} being scored`
                }
              >
                <Pill tone="accent" pulse>
                  {o.evaluating} evaluating
                </Pill>
              </span>
            )}
            {(o.lookahead_pending ?? 0) > 0 && (
              // Scored already -- its agent has moved on -- while the harness re-runs it with the
              // future cut away. It joins the leaderboard (or is disqualified) when this finishes.
              <span
                tabIndex={0}
                className="cursor-help rounded-full outline-none focus-visible:ring-2 focus-visible:ring-accent"
                title={
                  'Look-ahead tests running in the background (off the leaderboard until they pass):\n' +
                  o.points
                    .filter((p) => p.lookahead === 'pending')
                    .map((p) => `#${p.seq} by ${p.model ?? 'unknown'} -- submitted ${duration(Date.now() / 1000 - p.created_at)} ago`)
                    .join('\n')
                }
              >
                <Pill tone="warn" pulse>
                  {o.lookahead_pending} checking look-ahead
                </Pill>
              </span>
            )}
            {list.length > 1 && (
              <select
                className="ml-1 max-w-[260px] rounded-lg border border-seam bg-panel-hi px-2 py-0.5 font-mono text-[11px] text-ink outline-none"
                value={o.id}
                onChange={(e) => setSelected(e.target.value)}
              >
                {list.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.title.slice(0, 50)} · {x.status}
                  </option>
                ))}
              </select>
            )}
          </div>
          <div className="mt-1 text-[14px] font-medium leading-snug text-ink">{o.title}</div>
          <div className="mt-0.5 font-mono text-[10.5px] text-ink-faint">{describe.join(' · ')}</div>
          {o.metric.price_column && (
            <div className="mt-1.5 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-ink-faint">
              trades
              <span className="flex rounded-md border border-seam p-0.5">
                {(
                  [
                    ['long', 'long only'],
                    ['short', 'short only'],
                    ['both', 'long + short'],
                  ] as [Direction, string][]
                ).map(([d, l]) => (
                  <button
                    key={d}
                    type="button"
                    disabled={busy}
                    onClick={() => d !== (o.metric.direction ?? 'both') && setDirection(d)}
                    className={`rounded px-2 py-0.5 ${(o.metric.direction ?? 'both') === d ? 'bg-accent/15 text-accent' : 'hover:text-ink-dim'}`}
                  >
                    {l}
                  </button>
                ))}
              </span>
              <span title="Candidates already scored keep their scores; the swarm is told from its next brief and new candidates are priced under the new rule.">
                applies to new candidates
              </span>
              <span className="ml-2">holding</span>
              <span className="flex rounded-md border border-seam p-0.5">
                {(
                  [
                    [true, 'intraday only'],
                    [false, 'overnight ok'],
                  ] as [boolean, string][]
                ).map(([on, l]) => (
                  <button
                    key={l}
                    type="button"
                    disabled={busy}
                    title={on ? "Every position is forced flat at each day's last bar: all trades open and close the same day." : undefined}
                    onClick={() => on !== !!o.metric.intraday && setIntraday(on)}
                    className={`rounded px-2 py-0.5 ${!!o.metric.intraday === on ? 'bg-accent/15 text-accent' : 'hover:text-ink-dim'}`}
                  >
                    {l}
                  </button>
                ))}
              </span>
              {(o.metric.direction ?? 'both') === 'both' && (
                <>
                  <span className="ml-2">each side ≥</span>
                  <span className="flex rounded-md border border-seam p-0.5">
                    {[0, 0.1, 0.2, 0.3].map((v) => (
                      <button
                        key={v}
                        type="button"
                        disabled={busy}
                        title={v ? `Longs and shorts must each be at least ${v * 100}% of in-sample trades to rank.` : 'No requirement.'}
                        onClick={() => v !== (o.metric.min_side_share ?? 0) && setSideShare(v)}
                        className={`rounded px-2 py-0.5 ${(o.metric.min_side_share ?? 0) === v ? 'bg-accent/15 text-accent' : 'hover:text-ink-dim'}`}
                      >
                        {v ? `${v * 100}%` : 'off'}
                      </button>
                    ))}
                  </span>
                </>
              )}
              <RemarkProgress id={o.id} />
            </div>
          )}
          {o.metric.kind === 'task' && (
            <div className="mt-1.5 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-ink-faint">
              <span title="Trades OPENED per day; the task server ignores entries past the limit. Start loose so the swarm can find signals, tighten toward the goal as its entries get better.">
                max trades / day
              </span>
              <span className="flex rounded-md border border-seam p-0.5">
                {[0, 15, 6, 3, 2].map((v) => {
                  const cur = tradeLimitOf(o.metric.action_rule ?? o.metric.task_info?.action_rule)
                  return (
                    <button
                      key={v}
                      type="button"
                      disabled={busy}
                      onClick={() => v !== cur && setTradeLimit(v)}
                      className={`rounded px-2 py-0.5 ${cur === v ? 'bg-accent/15 text-accent' : 'hover:text-ink-dim'}`}
                    >
                      {v || 'off'}
                    </button>
                  )
                })}
              </span>
              <span
                className="ml-2"
                title="A floor on in-sample trades IN ALL: candidates below it are not ranked. Without it, when every strategy loses after costs the ranking drifts to ones that barely trade. Not a daily quota: a selective strategy that skips most days still ranks."
              >
                min trades
              </span>
              <span className="flex rounded-md border border-seam p-0.5">
                {[0, 20, 40, 80, 150].map((v) => (
                  <button
                    key={v}
                    type="button"
                    disabled={busy}
                    onClick={() => (v !== (o.metric.min_trades ?? 0) || o.metric.min_trades_per_day) && setMinTrades(v)}
                    className={`rounded px-2 py-0.5 ${(o.metric.min_trades ?? 0) === v ? 'bg-accent/15 text-accent' : 'hover:text-ink-dim'}`}
                  >
                    {v || 'off'}
                  </button>
                ))}
              </span>
              {!!o.metric.min_trades_per_day && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setMinTrades(o.metric.min_trades ?? 0)}
                  title="The older floor: a daily quota, which leaves selective strategies unranked. Click to remove it."
                  className="rounded-md border border-warn/40 px-2 py-0.5 text-warn hover:border-warn"
                >
                  also ≥ {o.metric.min_trades_per_day}/day ✕
                </button>
              )}
              <button
                type="button"
                disabled={busy}
                onClick={rescoreAll}
                title="Value every candidate again from its kept actions (after the task server's valuation changed)"
                className="rounded-md border border-seam px-2 py-0.5 hover:border-accent/40 hover:text-accent"
              >
                re-score all
              </button>
              <RemarkProgress id={o.id} />
            </div>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {o.status === 'running' ? (
            <Button tone="ghost" onClick={() => setStatus('paused')} disabled={busy}>
              Pause
            </Button>
          ) : (
            <Button tone="primary" onClick={() => setStatus('running')} disabled={busy}>
              {o.status === 'paused' ? 'Resume' : 'Restart'}
            </Button>
          )}
          {o.status !== 'stopped' ? (
            <Button tone="ghost" onClick={() => setStatus('stopped')} disabled={busy}>
              Stop
            </Button>
          ) : (
            <Button tone="danger" onClick={remove}>
              Delete
            </Button>
          )}
          <Button tone="ghost" onClick={onNew}>
            New
          </Button>
        </div>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <div
          className={`rounded-xl border p-3 sm:col-span-2 ${best ? 'cursor-pointer border-good/40 bg-good/5 hover:bg-good/10' : 'border-seam bg-panel-hi/40'}`}
          onClick={best ? () => setOpenId(best.id) : undefined}
        >
          <div className="text-[10.5px] uppercase tracking-wide text-ink-faint">
            Best so far · {o.metric_label}
            {o.split_date ? (robustRanking(o) ? ' (robust score)' : ' (holdout)') : ''}
          </div>
          {best ? (
            <>
              <div className="mt-0.5 font-mono text-[22px] text-good">{fmtMetric(kind, best.score)}</div>
              <div className="truncate font-mono text-[10.5px] text-ink-faint">
                #{best.seq} by {best.model} ·{' '}
                {robustRanking(o) && `holdout ${fmtMetric(kind, holdoutScore(best, kind))} · `}in-sample{' '}
                {fmtMetric(kind, best.is_score)} · {ago(best.champion_at)}
              </div>
            </>
          ) : (
            <div className="mt-1 text-[12px] text-ink-faint">
              {o.candidates ? 'No candidate has earned the title yet.' : 'Waiting for the first candidate…'}
            </div>
          )}
        </div>
        <div className="rounded-xl border border-seam bg-panel-hi/40 p-3">
          <div className="text-[10.5px] uppercase tracking-wide text-ink-faint">Candidates</div>
          <div className="mt-0.5 font-mono text-[18px] text-ink">{o.candidates}</div>
          <div className="font-mono text-[10.5px] text-ink-faint">
            {o.candidates_error} failed · {rate}/h
          </div>
        </div>
        <div className="rounded-xl border border-seam bg-panel-hi/40 p-3">
          <div className="text-[10.5px] uppercase tracking-wide text-ink-faint">Improvements</div>
          <div className="mt-0.5 font-mono text-[18px] text-ink">{o.improvements}</div>
          <div className="font-mono text-[10.5px] text-ink-faint">last {ago(o.last_improvement_at)}</div>
        </div>
      </div>

      <div className="mt-3">
        <ProgressChart points={o.points} kind={kind} higher={higher} onPick={setOpenId} storageKey={o.id} />
      </div>

      <TabStrip
        active={tab}
        onPick={setTab}
        tabs={
          [
            ['leaderboard', `Leaderboard`],
            // The trade book is kept for task objectives: their server lists every trade.
            ...(kind === 'task' ? ([['trades', 'Trade leaderboard']] as const) : []),
            ['recent', 'Recent'],
            ['memory', 'Team memory'],
            ['forecasts', 'Forecasts'],
            ['deci', 'Deci-plots'],
            ['regimes', 'Regimes'],
            ['library', 'Code library'],
            ['playbook', 'Playbook'],
            ['lessons', `Lessons (${o.lessons.length})`],
            ['steering', `Steering (${o.notes.length})`],
            ['ideas', 'Ideas'],
          ] as const
        }
      />

      {/* The Regime Lab is a page of charts: it gets the room, not a 480px scroll box. */}
      <div
        className={tab === 'regimes' ? 'pt-2' : tab === 'trades' ? 'max-h-[900px] overflow-y-auto pt-2' : 'max-h-[480px] overflow-y-auto pt-2'}
        onScroll={(e) => {
          // At the end of the list, the next 50 older (Recent) or lower-ranked (Leaderboard) rows.
          const el = e.currentTarget
          if (el.scrollTop + el.clientHeight < el.scrollHeight - 60) return
          if (tab === 'recent' && recent.length >= recentLimit && recentLimit < o.candidates) setRecentLimit(recentLimit + 50)
          if (tab === 'leaderboard' && rankLimit > 8 && ranked.length >= rankLimit && rankLimit < rankedCount)
            setRankLimit(rankLimit + 50)
        }}
      >
        {tab === 'leaderboard' && o.lookahead_check && <LookaheadRetest objectiveId={o.id} onChange={refresh} />}
        {(tab === 'leaderboard' || tab === 'recent') && (
          <PrunableCandidates
            key={tab}
            objective={o}
            rows={tab === 'leaderboard' ? ranked : recent}
            scope={tab === 'leaderboard' ? 'ranked' : null}
            scopeCount={rankedCount}
            onOpen={setOpenId}
            onChanged={refresh}
            extra={
              tab === 'leaderboard' && rankedCount > 8 ? (
                <button
                  onClick={() => setRankLimit(rankLimit > 8 ? 8 : Math.min(rankedCount, 58))}
                  className="font-mono text-[10.5px] text-ink-faint hover:text-accent"
                >
                  {rankLimit > 8 ? 'show top 8' : `show more (${rankedCount} ranked)`}
                </button>
              ) : null
            }
          />
        )}
        {((tab === 'leaderboard' && rankLimit > 8) || tab === 'recent') && (
          <ListEnd
            shown={tab === 'recent' ? recent.length : ranked.length}
            total={tab === 'recent' ? o.candidates : rankedCount}
            what={tab === 'recent' ? 'older candidates' : 'lower ranks'}
            onMore={() => (tab === 'recent' ? setRecentLimit(recentLimit + 50) : setRankLimit(rankLimit + 50))}
          />
        )}
        {/* Disqualified results stay on screen, struck through and labelled, so a demoted
            score is visibly dead instead of quietly missing -- and so nobody, operator or
            agent, keeps improving a signal that was already thrown out. */}
        {tab === 'leaderboard' && disqualified.length > 0 && (
          <div className="mt-4 border-t border-bad/25 pt-2">
            <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-bad">
              disqualified — do not build on these ({disqualified.length})
            </div>
            <PrunableCandidates
              objective={o}
              rows={disqualified}
              scope="disqualified"
              scopeCount={disqualifiedCount}
              onOpen={setOpenId}
              onChanged={refresh}
            />
          </div>
        )}
        {tab === 'trades' && kind === 'task' && <TradeBoard objectiveId={o.id} onOpen={setOpenId} />}
        {tab === 'playbook' && <PlaybookTab projectId={o.project_id} />}
        {tab === 'ideas' && <IdeasTab objectiveId={o.id} />}
        {tab === 'memory' && <TeamMemory objectiveId={o.id} />}
        {tab === 'forecasts' && <ForecastsTab projectId={o.project_id} objectiveId={o.id} />}
        {tab === 'deci' && <DeciPlots objectiveId={o.id} />}
        {tab === 'regimes' && <RegimeLab objectiveId={o.id} />}
        {tab === 'library' && <LibraryTab projectId={o.project_id} objectiveId={o.id} kind={kind} />}
        {(tab === 'lessons' || tab === 'steering') && rowErr && <div className="mb-1 text-[12px] text-bad">✗ {rowErr}</div>}
        {tab === 'lessons' &&
          (o.lessons.length ? (
            <ul className="space-y-1.5">
              <DeleteAllRow
                label={`Delete all ${o.lessons.length} lessons…`}
                onClick={() =>
                  dropRows(
                    'lessons',
                    'all',
                    `Delete all ${o.lessons.length} active lessons? Agents stop reading them from their next iteration. This cannot be undone.`,
                  )
                }
              />
              {o.lessons.map((l) => (
                <li key={l.id} className="group text-[12px] leading-snug text-ink-dim">
                  <span
                    className={`mr-1.5 font-mono text-[10px] ${
                      l.text.startsWith('AVOID') ? 'text-bad' : l.text.startsWith('KEEP') ? 'text-good' : 'text-accent'
                    }`}
                  >
                    ●
                  </span>
                  {l.text}
                  <span className="ml-2 font-mono text-[10px] text-ink-faint">{l.model}</span>
                  <RowDelete
                    title="Delete this lesson"
                    onClick={() => dropRows('lessons', [l.id], `Delete this lesson?\n\n"${l.text.slice(0, 300)}"`)}
                  />
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-[12px] text-ink-faint">
              After every attempt the agent records one lesson (KEEP / AVOID / TRY). Every later
              iteration reads them, and they are periodically consolidated.
            </div>
          ))}
        {tab === 'steering' &&
          (o.notes.length ? (
            <ul className="space-y-1.5">
              <DeleteAllRow
                label={`Delete all ${o.notes.length} steering notes…`}
                onClick={() =>
                  dropRows(
                    'notes',
                    'all',
                    'Delete every steering note for this objective? Agents stop reading them from their next iteration. This cannot be undone.',
                  )
                }
              />
              {o.notes.map((n) => (
                <li key={n.id} className="group text-[12px] leading-snug text-ink-dim">
                  <span className="mr-2 font-mono text-[10px] text-ink-faint">{ago(n.ts)}</span>
                  {n.text}
                  <RowDelete
                    title="Delete this steering note"
                    onClick={() => dropRows('notes', [n.id], `Delete this steering note?\n\n"${n.text.slice(0, 300)}"`)}
                  />
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-[12px] text-ink-faint">
              Messages you send below while this objective is selected steer it: every agent reads
              them at the start of its next iteration.
            </div>
          ))}
      </div>

      {openId && (
        <CandidateView
          key={openId}
          objective={o}
          candidateId={openId}
          onClose={() => setOpenId(null)}
          onDemoted={refresh}
          onOpenCandidate={setOpenId}
          onOpenChat={() => {
            setOpenId(null)
            router.push('/chat')
          }}
        />
      )}
    </Panel>
  )
}

/** One line summing up a deletion: how many went, who holds the title now, what was left. */
function describeDeletion(r: DeleteCandidatesResult): string {
  const parts = [`Deleted ${r.deleted.length}`]
  if (r.recrowned !== null) parts.push(`#${r.recrowned} is now the best`)
  else if (r.lost_best) parts.push('no ranked candidate is left to take the title')
  if (r.skipped.length)
    parts.push(`skipped ${r.skipped.map((x) => `${x.seq !== null ? `#${x.seq}` : x.id} (${x.reason})`).join(', ')}`)
  return parts.join(' · ')
}

const DELETE_TAIL =
  '\n\nAgents are told on #results so they stop building on them. Candidates still evaluating are skipped. This cannot be undone.'

/**
 * A candidate table the operator can prune. Tick rows (the header box ticks every row shown)
 * and delete them, or delete a whole scope -- every ranked or every disqualified candidate,
 * including those below the rows on screen. Deleting is for pollution nobody should learn
 * from (clones, a leaky champion); demoting is for a result whose reason should stay on record.
 */
function PrunableCandidates({
  objective,
  rows,
  scope,
  scopeCount,
  onOpen,
  onChanged,
  extra,
}: {
  objective: ObjectiveDetail
  rows: Candidate[]
  /** What "delete all" removes; null = no "delete all" (the Recent tab). */
  scope: 'ranked' | 'disqualified' | null
  scopeCount: number
  onOpen: (id: string) => void
  onChanged: () => void
  extra?: ReactNode
}) {
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  // Only what is on screen now counts: the table is polled, and a ticked row that has since
  // dropped out of view must not be deleted unseen.
  const chosen = rows.filter((c) => picked.has(c.id))
  const bestId = objective.best_id
  // Combining the ticked rows into an ensemble: only verified, non-ensemble rows can join.
  const [combining, setCombining] = useState(false)
  const combinable = chosen.filter(ensembleEligible)
  const canCombine = combinable.length >= 2 && combinable.length <= 8

  function pick(ids: string[], on: boolean) {
    setPicked((prev) => {
      const next = new Set(prev)
      for (const id of ids) {
        if (on) next.add(id)
        else next.delete(id)
      }
      return next
    })
  }

  async function run(body: { ids?: string[]; scope?: 'ranked' | 'disqualified' }, question: string) {
    if (!window.confirm(question)) return
    setBusy(true)
    setMsg(null)
    try {
      const r = await objectives.deleteCandidates(objective.id, body)
      setPicked(new Set())
      setMsg({ ok: true, text: describeDeletion(r) })
      onChanged()
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) })
    } finally {
      setBusy(false)
    }
  }

  function deleteSelected() {
    const seqs = chosen.map((c) => c.seq).sort((a, b) => a - b)
    const list = seqs.length > 20 ? `${seqs.slice(0, 20).join(', #')} …` : seqs.join(', #')
    const best = chosen.find((c) => c.id === bestId)
    void run(
      { ids: chosen.map((c) => c.id) },
      `Delete ${seqs.length} candidate${seqs.length === 1 ? '' : 's'} (#${list}) for good?` +
        (best ? `\n\n#${best.seq} is the current best: the next ranked candidate will be crowned in its place.` : '') +
        DELETE_TAIL,
    )
  }

  function deleteScope() {
    if (!scope) return
    void run(
      { scope },
      `Delete ALL ${scopeCount} ${scope} candidates — not only the ${rows.length} shown?` +
        (scope === 'ranked' && bestId
          ? '\n\nThe current best is among them. With nothing ranked left there is no best until a new candidate scores.'
          : '') +
        DELETE_TAIL,
    )
  }

  const toolbar = chosen.length > 0 || (scope && scopeCount > 0) || extra || msg
  return (
    <div>
      {toolbar && (
        <div className="mb-1 flex flex-wrap items-center gap-2">
          {chosen.length > 0 && (
            <>
              <DangerButton onClick={deleteSelected} disabled={busy}>
                {busy ? 'deleting…' : `Delete ${chosen.length} selected`}
              </DangerButton>
              <button onClick={() => setPicked(new Set())} className="font-mono text-[10.5px] text-ink-faint hover:text-ink-dim">
                clear
              </button>
              {canCombine && (
                <button
                  onClick={() => setCombining((v) => !v)}
                  title={
                    combinable.length < chosen.length
                      ? `${chosen.length - combinable.length} ticked row(s) cannot join (not verified, or an ensemble) and are left out`
                      : 'Combine the ticked candidates into one weighted ensemble candidate'
                  }
                  className="rounded-md border border-accent/40 px-2 py-0.5 font-mono text-[10.5px] text-accent hover:bg-accent/10"
                >
                  combine selected ({combinable.length})
                </button>
              )}
            </>
          )}
          {msg && (
            <span className={`min-w-0 truncate font-mono text-[10.5px] ${msg.ok ? 'text-ink-dim' : 'text-bad'}`} title={msg.text}>
              {msg.ok ? '' : '✗ '}
              {msg.text}
            </span>
          )}
          <span className="ml-auto flex items-center gap-3">
            {extra}
            {scope && scopeCount > 0 && (
              <DangerLink onClick={deleteScope} disabled={busy}>
                Delete all {scope} ({scopeCount})…
              </DangerLink>
            )}
          </span>
        </div>
      )}
      {combining && canCombine && (
        <CombineForm
          objectiveId={objective.id}
          members={combinable}
          onCancel={() => setCombining(false)}
          onCreated={(id) => {
            setCombining(false)
            setPicked(new Set())
            onChanged()
            onOpen(id)
          }}
        />
      )}
      <CandidateTable
        rows={rows}
        kind={objective.metric.kind}
        robust={robustRanking(objective)}
        bestId={bestId}
        ranked={scope === 'ranked'}
        disqualified={scope === 'disqualified'}
        onOpen={onOpen}
        picked={picked}
        onPick={pick}
      />
    </div>
  )
}

function CandidateTable({
  rows,
  kind,
  robust,
  bestId,
  ranked,
  disqualified = false,
  onOpen,
  picked,
  onPick,
}: {
  rows: Candidate[]
  kind: Objective['metric']['kind']
  /** Ranked on the robust score: show it beside the plain holdout figure. */
  robust: boolean
  bestId: string | null
  ranked: boolean
  /** Rendered as thrown-out: struck through, red, and never marked champion. */
  disqualified?: boolean
  onOpen: (id: string) => void
  /** With onPick, a checkbox column for selecting rows to delete. */
  picked?: Set<string>
  onPick?: (ids: string[], on: boolean) => void
}) {
  if (!rows.length) {
    return <div className="text-[12px] text-ink-faint">{ranked ? 'Nothing ranked yet.' : 'No candidates yet.'}</div>
  }
  const sel = picked && onPick ? { picked, onPick } : null
  const nPicked = sel ? rows.filter((c) => sel.picked.has(c.id)).length : 0
  return (
    <table className="w-full table-fixed font-mono text-[11px]">
      <thead>
        <tr className="text-left text-ink-faint">
          {sel && (
            <th className="w-[20px] py-1 font-normal">
              <PickBox
                title="Select every row shown"
                checked={nPicked === rows.length}
                indeterminate={nPicked > 0 && nPicked < rows.length}
                onChange={(on) => sel.onPick(rows.map((c) => c.id), on)}
              />
            </th>
          )}
          <th className="w-[34px] py-1 font-normal">{ranked ? 'rank' : ''}</th>
          <th className="w-[44px] py-1 font-normal">#</th>
          {robust && (
            <th className="w-[56px] py-1 text-right font-normal" title="Ranking score: the weaker of in-sample and holdout, × R² of the whole equity curve">
              score
            </th>
          )}
          <th className="w-[70px] py-1 text-right font-normal">holdout</th>
          <th className="w-[70px] py-1 text-right font-normal">in-sample</th>
          <th className="w-[74px] py-1 pl-3 font-normal">checks</th>
          <th className="py-1 pl-2 font-normal">hypothesis</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((c, i) => (
          <tr
            key={c.id}
            onClick={() => onOpen(c.id)}
            className={`cursor-pointer border-t border-seam/60 hover:bg-panel-hi ${
              disqualified ? 'text-bad/70 line-through' : c.id === bestId ? 'text-good' : 'text-ink-dim'
            } ${sel?.picked.has(c.id) ? 'bg-bad/5' : ''}`}
          >
            {sel && (
              // The whole cell swallows the click, so a near-miss on the box never opens the row.
              <td className="py-1" onClick={(e) => e.stopPropagation()}>
                <PickBox checked={sel.picked.has(c.id)} onChange={(on) => sel.onPick([c.id], on)} />
              </td>
            )}
            <td className="py-1">{ranked ? i + 1 : ''}</td>
            <td className="py-1">
              {c.seq}
              {disqualified ? '' : c.id === bestId ? ' ★' : ''}
              {c.liked ? <span className="text-good" title={c.liked_note || 'liked'}> ♥</span> : ''}
            </td>
            {robust && (
              <td
                className={`py-1 text-right ${disqualified ? '' : 'text-ink'}`}
                title={rankTitle(robustRank(c.metrics))}
              >
                {c.status === 'ok' ? fmtMetric(kind, c.score) : ''}
              </td>
            )}
            <td className={`py-1 text-right ${disqualified || robust ? '' : 'text-ink'}`}>
              {c.status === 'error' ? (
                <span className="text-bad">error</span>
              ) : c.status === 'evaluating' ? (
                '…'
              ) : (
                fmtMetric(kind, robust ? holdoutScore(c, kind) : c.score)
              )}
            </td>
            <td className="py-1 text-right">{fmtMetric(kind, c.is_score)}</td>
            <td className="py-1 pl-3">
              <span title={`look-ahead: ${c.lookahead}`}>
                <Dot tone={c.lookahead === 'error' ? 'error' : verdictTone(c.lookahead)} />
              </span>{' '}
              <span title={`audit: ${c.audit}`}>
                <Dot tone={c.audit === 'none' ? 'neutral' : verdictTone(c.audit)} />
              </span>
            </td>
            <td className="truncate py-1 pl-2 font-sans text-[11.5px]" title={c.rationale}>
              <span className="text-ink-faint">{c.model?.split('/').pop()} · </span>
              {c.mode === 'ensemble' && (
                <span className="text-accent">
                  ensemble of {c.metrics?.ensemble?.members.map((x) => `#${x.seq}`).join('+') ?? '?'} ·{' '}
                </span>
              )}
              {c.rationale || c.score_note}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function rankTitle(r: RobustRank | null): string | undefined {
  if (!r) return undefined
  const smooth = typeof r.smoothness === 'number' ? `R² ${r.smoothness.toFixed(2)} · ` : ''
  return `${smooth}weaker: ${r.weaker.replace('_', '-')}`
}

/** `error`: the check could not run (orange), apart from `warn`: still pending (amber). */
function Dot({ tone }: { tone: 'good' | 'bad' | 'warn' | 'error' | 'neutral' }) {
  const cls =
    tone === 'good' ? 'bg-good' : tone === 'bad' ? 'bg-bad' : tone === 'warn' ? 'bg-warn' : tone === 'error' ? 'bg-[#f97316]' : 'bg-ink-faint/40'
  return <span className={`inline-block h-2 w-2 rounded-full ${cls}`} />
}

/** How far re-marking the scored candidates under a changed holding rule has got; nothing once done. */
/** The daily trade limit in a task action rule ("intraday+max_3_trades" -> 3; none -> 0). */
function tradeLimitOf(rule: string | undefined | null): number {
  const m = /(?:^|\+)max_(\d+)_trades?/.exec(rule ?? '')
  return m ? +m[1] : 0
}

function RemarkProgress({ id }: { id: string }) {
  const [p, setP] = useState<{ done?: number; failed?: number; total?: number; running?: boolean } | null>(null)
  useEffect(() => {
    let live = true
    let timer: ReturnType<typeof setTimeout>
    const tick = () =>
      objectives
        .remarkProgress(id)
        .then((r) => live && setP(r))
        .catch(() => {})
        .finally(() => {
          if (live) timer = setTimeout(tick, 5000)
        })
    tick()
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [id])
  if (!p?.running) return null
  return (
    <span className="text-warn" title="Scored candidates are marked to market again under the new holding rule, best first; the leaderboard updates as it goes.">
      re-marking {(p.done ?? 0) + (p.failed ?? 0)}/{p.total ?? 0}
      {p.failed ? ` · ${p.failed} failed` : ''}
    </span>
  )
}

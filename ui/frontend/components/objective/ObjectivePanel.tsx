'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { duration } from '@/lib/format'
import {
  fmtMetric,
  objectives,
  type Candidate,
  type Objective,
  type ObjectiveDetail,
} from '@/lib/objectives'
import { Button, Panel, Pill } from '@/components/ui'
import { ProgressChart } from './Charts'
import CandidateView, { verdictTone } from './CandidateView'
import LibraryTab from './Library'
import PlaybookTab from './Playbook'
import IdeasTab from './Ideas'
import LookaheadRetest from './LookaheadRetest'

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
          objectives.candidates(pick.id, 'rank', 8),
          objectives.candidates(pick.id, 'recent', 12),
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
  }, [projectId, selected, tick])

  return { list, selected, setSelected, detail, ranked, disqualified, recent, error, refresh }
}

function ago(ts: number | null | undefined): string {
  return ts ? `${duration(Date.now() / 1000 - ts)} ago` : '—'
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
  const { list, detail: o, ranked, disqualified, recent, error, setSelected, refresh } = state
  const [openId, setOpenId] = useState<string | null>(null)
  const [tab, setTab] = useState<'leaderboard' | 'recent' | 'library' | 'playbook' | 'lessons' | 'steering' | 'ideas'>('leaderboard')
  const [busy, setBusy] = useState(false)

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

  const describe = [
    o.metric_label + (higher ? '' : ' (lower is better)'),
    o.split_date ? `ranked on the holdout from ${o.split_date}` : null,
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
            {o.evaluating > 0 && <Pill tone="accent" pulse>{o.evaluating} evaluating</Pill>}
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
            {o.split_date ? ' (holdout)' : ''}
          </div>
          {best ? (
            <>
              <div className="mt-0.5 font-mono text-[22px] text-good">{fmtMetric(kind, best.score)}</div>
              <div className="truncate font-mono text-[10.5px] text-ink-faint">
                #{best.seq} by {best.model} · in-sample {fmtMetric(kind, best.is_score)} · {ago(best.champion_at)}
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
        <ProgressChart points={o.points} kind={kind} higher={higher} onPick={setOpenId} />
      </div>

      <div className="mt-2 flex gap-1 border-b border-seam">
        {(
          [
            ['leaderboard', `Leaderboard`],
            ['recent', 'Recent'],
            ['library', 'Code library'],
            ['playbook', 'Playbook'],
            ['lessons', `Lessons (${o.lessons.length})`],
            ['steering', `Steering (${o.notes.length})`],
            ['ideas', 'Ideas when stuck'],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={`-mb-px border-b-2 px-3 py-1.5 font-mono text-[11px] ${
              tab === k ? 'border-accent text-accent' : 'border-transparent text-ink-faint hover:text-ink-dim'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="max-h-[480px] overflow-y-auto pt-2">
        {tab === 'leaderboard' && o.lookahead_check && <LookaheadRetest objectiveId={o.id} onChange={refresh} />}
        {(tab === 'leaderboard' || tab === 'recent') && (
          <CandidateTable
            rows={tab === 'leaderboard' ? ranked : recent}
            kind={kind}
            bestId={o.best_id}
            ranked={tab === 'leaderboard'}
            onOpen={setOpenId}
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
            <CandidateTable
              rows={disqualified}
              kind={kind}
              bestId={o.best_id}
              ranked={false}
              disqualified
              onOpen={setOpenId}
            />
          </div>
        )}
        {tab === 'playbook' && <PlaybookTab projectId={o.project_id} />}
        {tab === 'ideas' && <IdeasTab objectiveId={o.id} />}
        {tab === 'library' && <LibraryTab projectId={o.project_id} objectiveId={o.id} kind={kind} />}
        {tab === 'lessons' &&
          (o.lessons.length ? (
            <ul className="space-y-1.5">
              {o.lessons.map((l) => (
                <li key={l.id} className="text-[12px] leading-snug text-ink-dim">
                  <span
                    className={`mr-1.5 font-mono text-[10px] ${
                      l.text.startsWith('AVOID') ? 'text-bad' : l.text.startsWith('KEEP') ? 'text-good' : 'text-accent'
                    }`}
                  >
                    ●
                  </span>
                  {l.text}
                  <span className="ml-2 font-mono text-[10px] text-ink-faint">{l.model}</span>
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
              {o.notes.map((n) => (
                <li key={n.id} className="text-[12px] leading-snug text-ink-dim">
                  <span className="mr-2 font-mono text-[10px] text-ink-faint">{ago(n.ts)}</span>
                  {n.text}
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
          objective={o}
          candidateId={openId}
          onClose={() => setOpenId(null)}
          onDemoted={refresh}
          onOpenChat={() => {
            setOpenId(null)
            router.push('/chat')
          }}
        />
      )}
    </Panel>
  )
}

function CandidateTable({
  rows,
  kind,
  bestId,
  ranked,
  disqualified = false,
  onOpen,
}: {
  rows: Candidate[]
  kind: Objective['metric']['kind']
  bestId: string | null
  ranked: boolean
  /** Rendered as thrown-out: struck through, red, and never marked champion. */
  disqualified?: boolean
  onOpen: (id: string) => void
}) {
  if (!rows.length) {
    return <div className="text-[12px] text-ink-faint">{ranked ? 'Nothing ranked yet.' : 'No candidates yet.'}</div>
  }
  return (
    <table className="w-full table-fixed font-mono text-[11px]">
      <thead>
        <tr className="text-left text-ink-faint">
          <th className="w-[34px] py-1 font-normal">{ranked ? 'rank' : ''}</th>
          <th className="w-[44px] py-1 font-normal">#</th>
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
            }`}
          >
            <td className="py-1">{ranked ? i + 1 : ''}</td>
            <td className="py-1">
              {c.seq}
              {disqualified ? '' : c.id === bestId ? ' ★' : ''}
            </td>
            <td className={`py-1 text-right ${disqualified ? '' : 'text-ink'}`}>
              {c.status === 'error' ? <span className="text-bad">error</span> : c.status === 'evaluating' ? '…' : fmtMetric(kind, c.score)}
            </td>
            <td className="py-1 text-right">{fmtMetric(kind, c.is_score)}</td>
            <td className="py-1 pl-3">
              <span title={`look-ahead: ${c.lookahead}`}>
                <Dot tone={verdictTone(c.lookahead)} />
              </span>{' '}
              <span title={`audit: ${c.audit}`}>
                <Dot tone={c.audit === 'none' ? 'neutral' : verdictTone(c.audit)} />
              </span>
            </td>
            <td className="truncate py-1 pl-2 font-sans text-[11.5px]" title={c.rationale}>
              <span className="text-ink-faint">{c.model?.split('/').pop()} · </span>
              {c.rationale || c.score_note}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Dot({ tone }: { tone: 'good' | 'bad' | 'warn' | 'neutral' }) {
  const cls = tone === 'good' ? 'bg-good' : tone === 'bad' ? 'bg-bad' : tone === 'warn' ? 'bg-warn' : 'bg-ink-faint/40'
  return <span className={`inline-block h-2 w-2 rounded-full ${cls}`} />
}

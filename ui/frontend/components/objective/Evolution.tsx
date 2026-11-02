'use client'

import { useCallback, useEffect, useState } from 'react'
import { objectives, type EvolutionArm, type EvolutionView, type SealedRun, type SealedView } from '@/lib/objectives'

const num = (v: number | null | undefined, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '—')
const pct = (v: number | null | undefined, d = 0) => (typeof v === 'number' ? `${(v * 100).toFixed(d)}%` : '—')
const Head = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{children}</div>
)
const VERDICT: Record<string, { label: string; tone: string }> = {
  holds: { label: 'HOLDS UP', tone: 'text-good' },
  weakens: { label: 'WEAKENS', tone: 'text-warn' },
  fails: { label: 'FAILS', tone: 'text-bad' },
  not_comparable: { label: 'NOT COMPARABLE', tone: 'text-warn' },
  no_dates: { label: 'NO NEW DATES', tone: 'text-ink-dim' },
}

function SealedRunRow({ run }: { run: SealedRun }) {
  const s = run.summary
  const verdict = s.verdict ? VERDICT[s.verdict] : null
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded border border-seam p-2">
      <div className="flex flex-wrap items-baseline gap-2 font-mono text-[11px]">
        <span className="text-ink-faint">{new Date(run.started_at * 1000).toLocaleString()}</span>
        <span className="text-ink-dim">{run.source || run.rows_path}</span>
        {verdict ? (
          <span className={`font-semibold ${verdict.tone}`}>{verdict.label}</span>
        ) : (
          <span className={run.state === 'running' ? 'text-accent' : 'text-bad'}>{run.state}</span>
        )}
        {run.state === 'running' && (
          <span className="text-ink-faint">
            {run.phase} · {run.results.length} of {run.total}
          </span>
        )}
        <button onClick={() => setOpen(!open)} className="ml-auto text-accent hover:underline">
          {open ? 'hide' : `${run.results.length} candidates`}
        </button>
      </div>
      {run.error && <div className="mt-1 text-[11.5px] text-bad">{run.error}</div>}
      {s.reading && <div className="mt-1 text-[12px] text-ink-dim">{s.reading}</div>}
      {s.replayed !== undefined && s.verdict !== 'no_dates' && (
        <div className="mt-1 flex flex-wrap gap-x-4 font-mono text-[11px] text-ink-dim">
          <span>{s.first} to {s.last}, {s.days} days</span>
          <span title="median sealed Sharpe of the board's top, against the median of the rest">top {num(s.median_sharpe_top)} · rest {num(s.median_sharpe_rest)}</span>
          <span title="median ranking score of the top">their score {num(s.median_score_top)}</span>
          <span title="does the board's order predict the sealed result? 1 = perfectly, 0 = not at all">rank predicts: {num(s.rank_rho)}</span>
          <span>profitable {pct(s.positive_share)}</span>
          <span title="how alike the replay and the scored run are on the days both have">overlap ρ {num(s.median_overlap_rho)}</span>
          <span title="the top's median Sharpe on the dates it was built on: where it was scored, then on these rows">
            built-on dates {num(s.same_dates_built_on_scored)} → {num(s.same_dates_built_on_replay)}
          </span>
          <span title="the top's median Sharpe on the objective's holdout dates: where it was scored, then on these rows">
            holdout dates {num(s.same_dates_holdout_scored)} → {num(s.same_dates_holdout_replay)}
          </span>
          {s.failed ? <span className="text-bad">{s.failed} failed</span> : null}
          {s.missing_forecasts?.length ? <span className="text-bad">{s.missing_forecasts.length} forecasts not rebuilt</span> : null}
        </div>
      )}
      {open && (
        <table className="mt-2 w-full font-mono text-[11px]">
          <thead>
            <tr className="text-ink-faint">
              <th className="py-1 text-left font-normal">rank</th>
              <th className="py-1 text-left font-normal">candidate</th>
              <th className="py-1 text-right font-normal">score</th>
              <th className="py-1 text-right font-normal">sealed Sharpe</th>
              <th className="py-1 text-right font-normal">return</th>
              <th className="py-1 text-right font-normal">drawdown</th>
              <th className="py-1 text-right font-normal">active days</th>
              <th className="py-1 text-right font-normal">overlap ρ</th>
              <th className="py-1 text-right font-normal" title="Sharpe on the dates it was built on, on these rows">built-on</th>
              <th className="py-1 text-right font-normal" title="Sharpe on the objective's holdout dates, on these rows">holdout</th>
            </tr>
          </thead>
          <tbody>
            {run.results.map((r) => (
              <tr key={r.id} className="border-t border-seam/60">
                <td className="py-1 text-ink-faint">{r.rank ?? '—'}</td>
                <td className="py-1 text-ink">#{r.seq}</td>
                {r.status === 'ok' ? (
                  <>
                    <td className="py-1 text-right text-ink-dim">{num(r.score)}</td>
                    <td className={`py-1 text-right ${(r.sharpe ?? 0) > 0 ? 'text-good' : 'text-bad'}`}>{num(r.sharpe)}</td>
                    <td className="py-1 text-right text-ink-dim">{pct(r.total_return, 1)}</td>
                    <td className="py-1 text-right text-ink-dim">{pct(r.max_drawdown, 1)}</td>
                    <td className="py-1 text-right text-ink-dim">{r.active_days ?? '—'}</td>
                    <td className="py-1 text-right text-ink-dim">{num(r.overlap_rho)}</td>
                    <td className="py-1 text-right text-ink-dim">{num(r.same_dates?.built_on.replay)}</td>
                    <td className="py-1 text-right text-ink-dim">{num(r.same_dates?.holdout.replay)}</td>
                  </>
                ) : (
                  <td colSpan={8} className="py-1 pl-3 text-bad" title={r.error}>
                    {(r.error ?? 'failed').slice(0, 110)}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function Sealed({ objectiveId, data, onChange }: { objectiveId: string; data: SealedView; onChange: () => void }) {
  const [choice, setChoice] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const options = data.task ? data.sources : data.rows_offered
  const pick = choice || options[0] || ''
  const running = data.runs.some((r) => r.state === 'running')
  const i = data.integrity

  const start = () => {
    const looks = data.runs.length
    if (!window.confirm(`Open the seal on ${pick}? This is look number ${looks + 1}. Each look you act on uses it up, so choosing a strategy from these results makes them one more thing it was fitted to.`)) return
    objectives
      .startSealed(objectiveId, data.task ? { source: pick } : { rows_path: pick })
      .then(() => (setErr(null), onChange()))
      .catch((e: Error) => setErr(e.message))
  }

  return (
    <section>
      <Head>The sealed test · dates nothing in the search has touched · never shown to agents</Head>
      {i && (
        <div className="mb-2 text-[12px] text-ink-dim">
          After <b className="text-ink">{i.trials.toLocaleString()}</b> tries, the best result pure luck would reach is a Sharpe near{' '}
          <b className="text-ink">{num(i.luck_bar_sharpe, 1)}</b> ({num(i.luck_bar_distinct, 1)} counting only the {i.distinct_strategies} distinct strategies). Chance the leader #{i.leader} is
          better than luck:{' '}
          <b className={(i.deflated_all_trials ?? 0) >= 0.95 ? 'text-good' : (i.deflated_all_trials ?? 0) >= 0.5 ? 'text-warn' : 'text-bad'}>{pct(i.deflated_all_trials)}</b> by all tries,{' '}
          {pct(i.deflated_distinct)} by distinct strategies. Its record is {i.trades ?? '?'} entries, active on {i.active_days.in_sample ?? '?'} in-sample and {i.active_days.holdout ?? '?'} holdout days.
        </div>
      )}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        {options.length > 0 ? (
          <>
            <select value={pick} onChange={(e) => setChoice(e.target.value)} className="rounded border border-seam bg-transparent px-1.5 py-0.5 font-mono text-[11px] text-ink">
              {options.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            <button disabled={running} onClick={start} className="rounded border border-accent/50 px-2 py-0.5 text-[11.5px] text-accent disabled:opacity-40">
              {running ? 'Running…' : 'Run sealed test'}
            </button>
          </>
        ) : (
          <span className="text-[12px] text-ink-faint">No other rows for this objective: there is nothing past its last date to test on yet.</span>
        )}
        <span className="text-[11px] text-ink-faint">
          {data.runs.length} look{data.runs.length === 1 ? '' : 's'} so far
        </span>
      </div>
      {err && <div className="mb-1 text-[12px] text-bad">✗ {err}</div>}
      <div className="space-y-2">
        {data.runs.map((r) => (
          <SealedRunRow key={r.id} run={r} />
        ))}
      </div>
    </section>
  )
}

function ArmTable({ rows, first }: { rows: EvolutionArm[]; first: string }) {
  if (!rows.length) return <div className="text-[12px] text-ink-faint">Nothing measured yet: these fill in as the agents work.</div>
  return (
    <table className="w-full font-mono text-[11px]">
      <thead>
        <tr className="text-ink-faint">
          <th className="py-1 text-left font-normal">{first}</th>
          <th className="py-1 text-right font-normal">tried</th>
          <th className="py-1 text-right font-normal" title="a jump over the parent, or for an explorer a new strategy that holds up">wins</th>
          <th className="py-1 text-right font-normal">beat parent</th>
          <th className="py-1 text-right font-normal">median gain</th>
          <th className="py-1 text-right font-normal">median change</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.arm} className="border-t border-seam/60">
            <td className="py-1 text-ink-dim">{r.arm}</td>
            <td className="py-1 text-right text-ink">{r.n}{r.failed ? <span className="text-bad"> ({r.failed}✗)</span> : ''}</td>
            <td className="py-1 text-right text-ink">{r.wins} <span className="text-ink-faint">({pct(r.win_rate)})</span></td>
            <td className="py-1 text-right text-ink-dim">{pct(r.beat_parent)}</td>
            <td className="py-1 text-right text-ink-dim">{num(r.median_gain)}</td>
            <td className="py-1 text-right text-ink-dim">{pct(r.median_change)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/**
 * Whether the search is learning, and how far its score can be trusted: the sealed test and the
 * deflated score, how many strategies the candidates really are, where on the map they sit, and
 * the measured record of every instruction, parent route and lens. The operator's view: agents see
 * only the in-sample scorecard and never anything sealed.
 */
export default function Evolution({ objectiveId }: { objectiveId: string }) {
  const [view, setView] = useState<EvolutionView | null>(null)
  const [sealed, setSealed] = useState<SealedView | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(() => {
    objectives
      .evolution(objectiveId)
      .then((d) => (setView(d), setErr(null)))
      .catch((e: Error) => setErr(e.message))
    objectives
      .sealed(objectiveId)
      .then(setSealed)
      .catch((e: Error) => setErr(e.message))
  }, [objectiveId])

  useEffect(() => {
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [load])

  if (err) return <div className="text-[12px] text-bad">✗ {err}</div>
  if (!view) return <div className="text-[12px] text-ink-faint">loading…</div>
  if (!view.applicable) return <div className="text-[12px] text-ink-faint">This objective has no daily results to compare strategies by.</div>
  const card = view.scorecard
  const top = card ? card.top_same_strategy / Math.max(card.top, 1) : 0

  return (
    <div className="space-y-5">
      {sealed && <Sealed objectiveId={objectiveId} data={sealed} onChange={load} />}

      <section>
        <Head>Distinct strategies · candidates whose in-sample days correlate above 0.9 are one</Head>
        {card && (
          <div className="mb-2 text-[12px] text-ink-dim">
            {card.candidates} usable candidates are <b className="text-ink">{card.strategies}</b> strategies.{' '}
            {card.top_same_strategy > 1 && (
              <span className={top >= 0.5 ? 'text-warn' : ''}>
                {card.top_same_strategy} of the top {card.top} are one strategy
                {top >= 0.5 ? ': the score differences between them are the same trades reshuffled.' : '.'}
              </span>
            )}
          </div>
        )}
        <table className="w-full font-mono text-[11px]">
          <thead>
            <tr className="text-ink-faint">
              <th className="py-1 text-left font-normal">best</th>
              <th className="py-1 text-right font-normal">candidates</th>
              <th className="py-1 text-right font-normal">in-sample</th>
              <th className="py-1 text-right font-normal">ranked</th>
              <th className="py-1 text-right font-normal" title="correlation of its days with the leader's">ρ leader</th>
              <th className="py-1 text-right font-normal">trades/day</th>
              <th className="py-1 text-right font-normal">active</th>
              <th className="py-1 text-right font-normal" title="children in the last 60 candidates">recent work</th>
            </tr>
          </thead>
          <tbody>
            {view.islands.slice(0, 15).map((i) => (
              <tr key={i.island} className="border-t border-seam/60" title={i.niche ?? ''}>
                <td className="py-1 text-ink">#{i.best_seq}</td>
                <td className="py-1 text-right text-ink-dim">{i.size}</td>
                <td className="py-1 text-right text-ink">{num(i.best_in_sample)}</td>
                <td className="py-1 text-right text-ink-dim">{num(i.best_ranked)}</td>
                <td className="py-1 text-right text-ink-dim">{num(i.rho_leader)}</td>
                <td className="py-1 text-right text-ink-dim">{num(i.trades_per_day)}</td>
                <td className="py-1 text-right text-ink-dim">{pct(i.active_share)}</td>
                <td className="py-1 text-right text-ink-dim">{i.recent_children}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section>
        <Head>The map · where strategies exist, best in-sample score · explorers are sent to the empty places</Head>
        <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
          {view.niches
            .slice()
            .sort((a, b) => b.strategies - a.strategies)
            .map((n) => (
              <span key={n.cell.join('|')} className="rounded border border-seam px-2 py-0.5 text-ink-dim" title={`best: #${n.best_seq}`}>
                {n.label} · <span className="text-ink">{n.strategies}</span> · {num(n.best_in_sample)}
              </span>
            ))}
        </div>
      </section>

      {card && (
        <section>
          <Head>Several definitions of good · the best by each (in-sample)</Head>
          <ul className="space-y-0.5 text-[12px] text-ink-dim">
            {Object.entries(card.leaders).map(([k, l]) => (
              <li key={k}>
                <span className="font-mono text-[11px] text-accent">{k}</span> #{l.seq} ({num(l.in_sample_score)}{l.trades ? `, ${l.trades} trades` : ''}): {l.about}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <Head>Instructions under test · the words of the assignment are an experiment</Head>
        <ArmTable
          first="instruction"
          rows={view.experiments.map((v) => ({
            ...v,
            arm: `${v.mode} · ${v.name}${v.control ? ' (control)' : ''}${v.status !== 'active' ? ' — retired' : ''}${v.author !== 'seed' ? ` · ${v.author.split('/').pop()}` : ''}`,
          }))}
        />
      </section>

      <section>
        <Head>Parent route · the archive has to beat the old tournament, its control ({pct(view.share)} of IMPROVE iterations)</Head>
        <ArmTable first="route" rows={view.parent_arms} />
      </section>

      <section>
        <Head>Lens · which definition of good the parent was chosen by</Head>
        <ArmTable first="lens" rows={view.lenses} />
      </section>

      {view.practices.length > 0 && (
        <section>
          <Head>Team practices · does each rewrite help?</Head>
          <ArmTable first="version" rows={view.practices} />
        </section>
      )}

      <section>
        <Head>Size of the change · measured since sizes were recorded</Head>
        <ArmTable first="change from parent" rows={view.change_sizes} />
      </section>

      <section>
        <Head>Evaluation · last {view.stages.recent} submissions</Head>
        <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
          <span className="rounded border border-seam px-2 py-0.5 text-ink-dim">scored {view.stages.ok}</span>
          <span className="rounded border border-seam px-2 py-0.5 text-ink-dim">failed in the full run {view.stages.full}</span>
          <span className="rounded border border-seam px-2 py-0.5 text-ink-dim">stopped by the static check {view.stages.static}</span>
          <span className="rounded border border-seam px-2 py-0.5 text-ink-dim">stopped by a smoke run {view.stages.smoke}</span>
          <span className="rounded border border-seam px-2 py-0.5 text-ink-dim">
            look-ahead: {view.stages.lookahead_full} full, {view.stages.lookahead_short} short ({view.stages.cuts_saved} runs saved)
          </span>
        </div>
      </section>
    </div>
  )
}

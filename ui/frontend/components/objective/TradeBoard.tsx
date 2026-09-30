'use client'

import { Fragment, useEffect, useState } from 'react'
import { objectives, type TradeBook, type TradeClass, type TradeRow } from '@/lib/objectives'
import TaskDayChart from './TaskDayChart'

const CLASSES: { key: TradeClass; label: string; tone: string; hint: string }[] = [
  { key: 'win', label: 'Big winners', tone: 'text-good', hint: 'at least +threshold per unit of size, after costs -- the only trades the swarm optimises for' },
  { key: 'loss', label: 'Big losers', tone: 'text-bad', hint: 'at most -threshold per unit of size, after costs' },
  { key: 'scratch', label: 'Scratch', tone: 'text-ink-dim', hint: 'flat, small wins and small losses: costs and noise' },
]
const PAGE = 100

const toMs = (iso: string) => Date.parse(iso.endsWith('Z') ? iso : `${iso}Z`) // the server speaks UTC

/**
 * The trade leaderboard of a task objective: every trade of every eligible candidate (one row per
 * distinct trade, with every candidate that took it), as BIG WINNERS, BIG LOSERS or SCRATCH --
 * classed on the result per unit of size after costs. A row opens its day on the candle chart with
 * that trade picked out among the day's others; "what the big winners have in common" is the
 * review the agents are briefed with (in-sample).
 */
export default function TradeBoard({ objectiveId, onOpen }: { objectiveId: string; onOpen: (candidateId: string) => void }) {
  const [cls, setCls] = useState<TradeClass>('win')
  const [side, setSide] = useState<'all' | 'long' | 'short'>('all')
  const [segment, setSegment] = useState<'all' | 'in_sample' | 'holdout'>('all')
  const [limit, setLimit] = useState(PAGE)
  const [book, setBook] = useState<TradeBook | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [pick, setPick] = useState<{ key: string; taker: string } | null>(null)
  const [review, setReview] = useState<string | null>(null)
  const [showReview, setShowReview] = useState(false)
  const [thr, setThr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const building = !!book?.building?.running
  useEffect(() => {
    let alive = true
    const load = () =>
      objectives
        .trades(objectiveId, cls, side, segment, limit)
        .then((d) => alive && (setBook(d), setErr(null)))
        .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    void load()
    const t = setInterval(load, building ? 5000 : 30000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [objectiveId, cls, side, segment, limit, building, tick])

  useEffect(() => {
    if (!showReview) return
    let alive = true
    setReview(null)
    objectives
      .tradeReview(objectiveId)
      .then((r) => alive && setReview(r.text))
      .catch((e) => alive && setReview(`✗ ${e instanceof Error ? e.message : String(e)}`))
    return () => {
      alive = false
    }
  }, [objectiveId, showReview, tick, book?.indexed_trades])

  if (err && !book) return <div className="text-[12px] text-bad">✗ {err}</div>
  if (!book) return <div className="text-[12px] text-ink-faint">loading the trade book…</div>

  const add = book.additive
  const tz = book.display_tz || 'UTC'
  const u = (v: number) => (add ? `${v >= 0 ? '+' : ''}${v.toFixed(3)}` : `${v >= 0 ? '+' : ''}${(v * 1e4).toFixed(1)} bps`)
  const thrText = add ? book.threshold.toPrecision(3) : `${(book.threshold * 1e4).toFixed(1)} bps`
  const segs: ('in_sample' | 'holdout')[] = segment === 'all' ? ['in_sample', 'holdout'] : [segment]
  const count = (k: TradeClass) => segs.reduce((n, s) => n + (book.counts[s]?.[k] ?? 0), 0)
  const total = (k: TradeClass) => segs.reduce((n, s) => n + (book.counts[s]?.[`${k}_total` as const] ?? 0), 0)
  const all = CLASSES.reduce((n, c) => n + count(c.key), 0)
  const time = (iso: string) =>
    new Intl.DateTimeFormat('en-GB', { timeZone: tz, hour: '2-digit', minute: '2-digit' }).format(toMs(iso))
  const date = (iso: string) => new Intl.DateTimeFormat('en-CA', { timeZone: tz }).format(toMs(iso))
  const chip = (on: boolean) =>
    `rounded border px-2 py-0.5 ${on ? 'border-accent text-ink' : 'border-seam text-ink-dim hover:border-ink-faint hover:text-ink'}`
  const rowKey = (r: TradeRow) => `${r.entry}|${r.exit}|${r.side}`

  async function saveThreshold(value: number | null) {
    try {
      await objectives.setTradeThreshold(objectiveId, value)
      setThr(null)
      setTick((n) => n + 1)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="space-y-3">
      <div className="text-[11.5px] text-ink-dim">
        Every trade of every candidate, one row per distinct trade (several candidates often take the same one), classed on
        its result per unit of size after costs. The swarm optimises for <span className="text-good">big winners</span> only:
        agents are briefed with what those have in common at entry (in-sample) and told to skip everything else.
      </div>

      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
        {CLASSES.map((c) => (
          <button key={c.key} type="button" title={c.hint} className={chip(cls === c.key)} onClick={() => (setCls(c.key), setLimit(PAGE), setPick(null))}>
            <span className={c.tone}>{c.label}</span> {count(c.key)}
            {all > 0 && <span className="text-ink-faint"> · {((count(c.key) / all) * 100).toFixed(0)}%</span>}
            <span className={`ml-1 ${total(c.key) >= 0 ? 'text-good' : 'text-bad'}`}>{u(total(c.key))}</span>
          </button>
        ))}
        <span className="mx-1 text-seam">|</span>
        {(['all', 'in_sample', 'holdout'] as const).map((s) => (
          <button key={s} type="button" className={chip(segment === s)} onClick={() => (setSegment(s), setLimit(PAGE), setPick(null))}>
            {s === 'all' ? 'both periods' : s === 'in_sample' ? 'in-sample' : 'holdout'}
          </button>
        ))}
        <span className="mx-1 text-seam">|</span>
        {(['all', 'long', 'short'] as const).map((s) => (
          <button key={s} type="button" className={chip(side === s)} onClick={() => (setSide(s), setLimit(PAGE), setPick(null))}>
            {s === 'all' ? 'long + short' : s}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ink-dim">
        <span>
          big = ±{thrText} per unit{' '}
          <span className="text-ink-faint">
            ({book.threshold_source === 'set' ? 'set by you' : book.threshold_source === 'auto' ? 'auto: top 20% of |result|' : 'floor: 8× the cost'})
          </span>
        </span>
        {thr === null ? (
          <button type="button" className="text-accent hover:underline" onClick={() => setThr(add ? String(book.threshold) : (book.threshold * 1e4).toFixed(1))}>
            change
          </button>
        ) : (
          <span className="flex items-center gap-1">
            <input
              value={thr}
              onChange={(e) => setThr(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && Number(thr) > 0 && void saveThreshold(Number(thr))}
              className="w-16 rounded border border-seam bg-panel px-1 py-0.5 text-ink"
              aria-label="threshold"
            />
            <span className="text-ink-faint">{add ? 'units' : 'bps'}</span>
            <button type="button" className="text-accent hover:underline disabled:opacity-40" disabled={!(Number(thr) > 0)} onClick={() => void saveThreshold(Number(thr))}>
              set
            </button>
            <button type="button" className="text-accent hover:underline" onClick={() => void saveThreshold(null)}>
              auto
            </button>
            <button type="button" className="text-ink-faint hover:text-ink" onClick={() => setThr(null)}>
              cancel
            </button>
          </span>
        )}
        <span className="text-ink-faint">
          {book.indexed_trades.toLocaleString()} trades from {book.indexed} of {book.candidates} candidates
          {book.index_errors > 0 && <span className="text-warn"> · {book.index_errors} without trades</span>}
        </span>
        {building && (
          <span className="text-warn">
            indexing {book.building?.done ?? 0}/{book.building?.total ?? 0}…
          </span>
        )}
        <button
          type="button"
          className="text-ink-faint hover:text-ink"
          title="index every candidate's trades again from its kept actions"
          onClick={() => void objectives.reindexTrades(objectiveId).then(() => setTick((n) => n + 1))}
        >
          rebuild
        </button>
        <button type="button" className="ml-auto text-accent hover:underline" onClick={() => setShowReview((v) => !v)}>
          {showReview ? 'hide' : 'what the big winners have in common'}
        </button>
      </div>

      {showReview && (
        <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap rounded-lg border border-seam bg-panel-hi/30 p-3 font-mono text-[10.5px] leading-relaxed text-ink-dim">
          {review ?? 'reviewing the book…'}
        </pre>
      )}

      {book.rows.length === 0 ? (
        <div className="text-[12px] text-ink-faint">
          {building ? 'Indexing the candidates’ trades…' : `No ${CLASSES.find((c) => c.key === cls)?.label.toLowerCase()} here.`}
        </div>
      ) : (
        <table className="w-full font-mono text-[11px]">
          <thead className="text-ink-faint">
            <tr>
              <th className="py-1 text-left font-normal">#</th>
              <th className="py-1 text-left font-normal">day</th>
              <th className="py-1 text-left font-normal">entry → exit</th>
              <th className="py-1 text-left font-normal">side</th>
              <th className="py-1 text-right font-normal">per unit</th>
              <th className="py-1 text-right font-normal">net</th>
              <th className="py-1 text-right font-normal">size</th>
              <th className="py-1 text-right font-normal">bars</th>
              <th className="py-1 pl-3 text-left font-normal">period</th>
              <th className="py-1 pl-3 text-left font-normal">taken by</th>
            </tr>
          </thead>
          <tbody>
            {book.rows.map((r, i) => {
              const key = rowKey(r)
              const open = pick?.key === key
              return (
                <Fragment key={key}>
                  <tr
                    className={`cursor-pointer border-t border-seam/60 hover:bg-panel-hi/40 ${open ? 'bg-panel-hi/50' : ''}`}
                    onClick={() => setPick(open ? null : { key, taker: r.takers[0]?.id ?? '' })}
                    title="open this day's candles with the trade picked out"
                  >
                    <td className="py-1 text-ink-faint">{i + 1}</td>
                    <td className="py-1 text-ink">{date(r.entry)}</td>
                    <td className="py-1 tabular-nums text-ink-dim">
                      {time(r.entry)} → {time(r.exit)}
                    </td>
                    <td className={`py-1 ${r.side === 'long' ? 'text-good' : 'text-accent'}`}>{r.side}</td>
                    <td className={`py-1 text-right tabular-nums ${r.unit >= 0 ? 'text-good' : 'text-bad'}`}>{u(r.unit)}</td>
                    <td className={`py-1 text-right tabular-nums ${r.net >= 0 ? 'text-good' : 'text-bad'}`}>{u(r.net)}</td>
                    <td className="py-1 text-right tabular-nums text-ink-dim">{r.size.toFixed(2)}</td>
                    <td className="py-1 text-right tabular-nums text-ink-dim">{r.bars}</td>
                    <td className="py-1 pl-3">
                      {r.holdout ? <span className="text-warn">holdout</span> : <span className="text-ink-faint">in-sample</span>}
                    </td>
                    <td className="py-1 pl-3">
                      <span className="flex flex-wrap gap-1">
                        {r.takers.slice(0, 6).map((t) => (
                          <button
                            key={t.id}
                            type="button"
                            title={`open candidate #${t.seq}`}
                            className="rounded border border-seam px-1 text-ink-dim hover:border-accent hover:text-ink"
                            onClick={(e) => (e.stopPropagation(), onOpen(t.id))}
                          >
                            #{t.seq}
                          </button>
                        ))}
                        {r.takers.length > 6 && <span className="text-ink-faint">+{r.takers.length - 6}</span>}
                      </span>
                    </td>
                  </tr>
                  {open && pick && (
                    <tr>
                      <td colSpan={10} className="pb-2">
                        {r.takers.length > 1 && (
                          <div className="mt-1 flex flex-wrap items-center gap-1 font-mono text-[10.5px] text-ink-faint">
                            day as traded by
                            {r.takers.map((t) => (
                              <button key={t.id} type="button" className={chip(pick.taker === t.id)} onClick={() => setPick({ key, taker: t.id })}>
                                #{t.seq}
                              </button>
                            ))}
                          </div>
                        )}
                        <TaskDayChart
                          objectiveId={objectiveId}
                          candidateId={pick.taker}
                          day={r.entry.slice(0, 10)}
                          days={[r.entry.slice(0, 10)]}
                          dayValue={undefined}
                          additive={add}
                          onDay={() => undefined}
                          onClose={() => setPick(null)}
                          highlight={r.entry}
                        />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      )}
      {book.rows.length >= limit && (
        <button type="button" className="font-mono text-[11px] text-accent hover:underline" onClick={() => setLimit(limit + PAGE)}>
          show {PAGE} more
        </button>
      )}
    </div>
  )
}

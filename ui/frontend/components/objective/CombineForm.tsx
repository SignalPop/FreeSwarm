'use client'

import { useEffect, useState } from 'react'
import { fmtMetric, objectives, type Candidate, type CorrelationReport } from '@/lib/objectives'
import { CorrelationMatrix } from './EnsembleView'

/**
 * Combine the picked leaderboard rows into one ensemble candidate: choose the weighting, see
 * the members' in-sample correlations first, then create it and open it. The backend checks
 * eligibility again and says why a member cannot join.
 */
export default function CombineForm({
  objectiveId,
  members,
  onCancel,
  onCreated,
}: {
  objectiveId: string
  /** Eligible picked rows, 2..8. */
  members: Candidate[]
  onCancel: () => void
  onCreated: (id: string) => void
}) {
  const [weighting, setWeighting] = useState<'equal' | 'inverse_vol'>('inverse_vol')
  const [lookback, setLookback] = useState(20)
  const [rationale, setRationale] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [corr, setCorr] = useState<CorrelationReport | null>(null)
  const seqs = [...members].sort((a, b) => a.seq - b.seq).map((c) => c.seq)
  const key = seqs.join(',')

  useEffect(() => {
    let alive = true
    setCorr(null)
    objectives
      .correlations(objectiveId, key.split(',').map(Number))
      .then((r) => alive && setCorr(r))
      .catch(() => alive && setCorr(null))
    return () => {
      alive = false
    }
  }, [objectiveId, key])

  async function create() {
    setBusy(true)
    setErr(null)
    try {
      const r = await objectives.combine(objectiveId, {
        members: seqs,
        weighting,
        lookback_days: lookback,
        rationale: rationale.trim(),
        model: 'operator',
      })
      onCreated(r.id ?? r.candidate_id)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const offDiag = corr ? corr.matrix.flatMap((row, i) => row.filter((v, j) => j > i && v !== null) as number[]) : []
  const avgAbs = offDiag.length ? offDiag.reduce((s, v) => s + Math.abs(v), 0) / offDiag.length : null

  return (
    <div className="mb-2 rounded-xl border border-accent/40 bg-accent/5 p-3">
      <div className="text-[12.5px] font-medium text-ink">Combine #{seqs.join(' + #')} into an ensemble</div>
      <div className="mt-0.5 text-[11.5px] leading-relaxed text-ink-dim">
        A portfolio of the members&apos; own net daily returns, weighted each day (no netting of positions, each
        member&apos;s costs as it paid them). It is scored, audited and ranked like any candidate.
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3 font-mono text-[11px] text-ink-dim">
        <label className="flex items-center gap-1.5">
          weighting
          <select
            value={weighting}
            onChange={(e) => setWeighting(e.target.value as 'equal' | 'inverse_vol')}
            className="rounded-md border border-seam bg-panel px-1.5 py-0.5 text-ink outline-none"
          >
            <option value="inverse_vol">inverse volatility</option>
            <option value="equal">equal</option>
          </select>
        </label>
        {weighting === 'inverse_vol' && (
          <label className="flex items-center gap-1.5">
            lookback days
            <input
              type="number"
              min={5}
              max={250}
              value={lookback}
              onChange={(e) => setLookback(Math.max(5, Math.min(250, Number(e.target.value) || 20)))}
              className="w-16 rounded-md border border-seam bg-panel px-1.5 py-0.5 text-ink outline-none"
            />
          </label>
        )}
        <input
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="why these members (optional)"
          className="min-w-[220px] flex-1 rounded-md border border-seam bg-panel px-2 py-0.5 font-sans text-[12px] text-ink outline-none focus:border-accent/60"
        />
      </div>
      <div className="mt-2">
        {corr ? (
          <>
            <CorrelationMatrix seqs={corr.seqs} matrix={corr.matrix} />
            <div className="mt-1 font-mono text-[10.5px] text-ink-faint">
              average |ρ| {avgAbs === null ? '—' : avgAbs.toFixed(2)} over {corr.days} in-sample days · in-sample Sharpe{' '}
              {corr.candidates.map((c) => `#${c.seq} ${fmtMetric('sharpe', c.in_sample_sharpe)}`).join(' · ')}
            </div>
          </>
        ) : (
          <div className="font-mono text-[10.5px] text-ink-faint">loading in-sample correlations…</div>
        )}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <button
          onClick={create}
          disabled={busy}
          className="rounded-lg border border-accent/50 bg-accent/10 px-3 py-1 font-mono text-[11.5px] text-accent disabled:opacity-40"
        >
          {busy ? 'combining…' : 'Create ensemble'}
        </button>
        <button onClick={onCancel} className="font-mono text-[11px] text-ink-faint hover:text-ink-dim">
          cancel
        </button>
        {err && <span className="font-mono text-[11px] text-bad">✗ {err}</span>}
      </div>
    </div>
  )
}

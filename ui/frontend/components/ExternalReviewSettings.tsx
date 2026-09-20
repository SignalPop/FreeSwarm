'use client'

import { useEffect, useState } from 'react'
import { review, type ReviewConfig } from '@/lib/review'
import { Panel, Pill } from '@/components/ui'

/**
 * External review: the operator's Anthropic credentials, which model reviews, and — the
 * consequential one — whether the reviewer may disqualify a result on its own.
 *
 * The key is write-only. It is stored beside the other secrets on the backend and never sent
 * back to the browser; all the console ever learns is whether one is set.
 */
export default function ExternalReviewSettings() {
  const [cfg, setCfg] = useState<ReviewConfig | null>(null)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [tested, setTested] = useState<string | null>(null)

  useEffect(() => {
    review.config().then(setCfg).catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [])

  async function save(patch: Parameters<typeof review.save>[0]) {
    setBusy(true)
    setErr(null)
    setTested(null)
    try {
      setCfg(await review.save(patch))
      if (patch.api_key !== undefined) setKey('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function test() {
    setBusy(true)
    setErr(null)
    setTested(null)
    try {
      const r = await review.test()
      setTested(`${r.model} replied "${r.reply}"`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  if (!cfg) {
    return (
      <Panel className="mb-6 p-5">
        <div className="mb-2 text-[15px] font-medium text-ink">External review</div>
        <div className="text-[12px] text-ink-faint">{err ?? 'Loading…'}</div>
      </Panel>
    )
  }

  return (
    <Panel className="mb-6 p-5">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-[15px] font-medium text-ink">External review</span>
        <Pill tone={cfg.enabled && cfg.key_set ? 'good' : 'neutral'}>
          {cfg.enabled && cfg.key_set ? 'ready' : cfg.key_set ? 'off' : 'no key'}
        </Pill>
        <label className="ml-auto flex items-center gap-2 text-[12.5px] text-ink-dim">
          <input type="checkbox" checked={cfg.enabled} disabled={busy}
            onChange={(ev) => save({ enabled: ev.target.checked })} />
          Enabled
        </label>
      </div>
      <p className="mb-4 text-[12px] leading-relaxed text-ink-faint">
        Sends a candidate — its hypothesis, its code, and the source of every project library module it imports — to a
        Claude model, and asks specifically about look-ahead bias, overfitting and whether the score can be trusted.
        The local look-ahead test only proves positions don&apos;t change when future rows are deleted; it cannot see a
        position computed correctly and then mis-aligned onto earlier bars. Every review is posted to the swarm&apos;s
        message board, and anything it disqualifies becomes a team lesson and a project pitfall.
      </p>

      {err && <div className="mb-3 rounded-lg border border-bad/35 bg-bad/5 px-3 py-2 text-[12px] text-bad">{err}</div>}
      {tested && <div className="mb-3 rounded-lg border border-good/35 bg-good/5 px-3 py-2 text-[12px] text-good">{tested}</div>}

      {/* ---- Credentials ---- */}
      <div className="text-[11px] uppercase tracking-wide text-ink-faint">Anthropic API key</div>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <input
          type="password"
          value={key}
          onChange={(ev) => setKey(ev.target.value)}
          placeholder={cfg.key_set ? '•••••••••••••••• (stored)' : 'sk-ant-…'}
          className="min-w-0 flex-1 rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
        />
        <button
          disabled={busy || !key.trim()}
          onClick={() => save({ api_key: key.trim() })}
          className="rounded-lg border border-accent/45 px-3 py-1.5 font-mono text-[12px] text-accent disabled:opacity-40"
        >
          Save key
        </button>
        {cfg.key_set && (
          <>
            <button disabled={busy} onClick={test}
              className="rounded-lg border border-seam px-3 py-1.5 font-mono text-[12px] text-ink-dim hover:text-ink">
              Test
            </button>
            <button disabled={busy} onClick={() => save({ api_key: '' })}
              className="font-mono text-[11.5px] text-ink-faint hover:text-bad">
              remove
            </button>
          </>
        )}
      </div>
      <div className="mt-1 font-mono text-[10.5px] text-ink-faint">
        Stored in ui\backend\auth\secrets.json with the other secrets — owner-only, git-ignored, never sent to this page.
      </div>

      {/* ---- Model ---- */}
      <div className="mt-4 text-[11px] uppercase tracking-wide text-ink-faint">Reviewer</div>
      <div className="mt-1 space-y-1.5">
        {cfg.models.map((m) => (
          <label key={m.id} className="flex cursor-pointer items-start gap-2.5">
            <input type="radio" name="review-model" className="mt-1" checked={cfg.model === m.id} disabled={busy}
              onChange={() => save({ model: m.id })} />
            <span className="min-w-0">
              <span className="font-mono text-[12.5px] text-ink">{m.label}</span>
              <span className="block font-mono text-[10.5px] text-ink-faint">{m.note}</span>
            </span>
          </label>
        ))}
      </div>

      {/* ---- Autonomy: the consequential setting, so it is spelled out ---- */}
      <div className="mt-4 text-[11px] uppercase tracking-wide text-ink-faint">
        When the reviewer says a result should not stand
      </div>
      <div className="mt-1 space-y-1.5">
        <label className="flex cursor-pointer items-start gap-2.5">
          <input type="radio" name="review-autonomy" className="mt-1" checked={cfg.autonomy === 'ask'} disabled={busy}
            onChange={() => save({ autonomy: 'ask' })} />
          <span className="min-w-0">
            <span className="text-[12.5px] text-ink">Ask me first</span>
            <span className="block text-[11.5px] text-ink-faint">
              The verdict is shown with the demotion pre-filled; nothing changes until you confirm.
            </span>
          </span>
        </label>
        <label className="flex cursor-pointer items-start gap-2.5">
          <input type="radio" name="review-autonomy" className="mt-1" checked={cfg.autonomy === 'auto'} disabled={busy}
            onChange={() => save({ autonomy: 'auto' })} />
          <span className="min-w-0">
            <span className="text-[12.5px] text-ink">Demote automatically</span>
            <span className="block text-[11.5px] text-ink-faint">
              The reviewer disqualifies on its own — the result drops off the leaderboard, is dethroned if it held the
              title, and the reason becomes a lesson and a pitfall. Use this for an unattended swarm, so it stops
              building on a leak overnight. A wrong verdict demotes good work; you can see every one on the board.
            </span>
          </span>
        </label>
      </div>

      <label className="mt-4 flex items-start gap-2.5">
        <input type="checkbox" className="mt-1" checked={cfg.auto_review_champions} disabled={busy}
          onChange={(ev) => save({ auto_review_champions: ev.target.checked })} />
        <span className="min-w-0">
          <span className="text-[12.5px] text-ink">Review every new champion</span>
          <span className="block text-[11.5px] text-ink-faint">
            Off by default: this costs real money per candidate that takes the title, and a busy objective crowns often.
            Leave it off and review by hand from the candidate panel.
          </span>
        </span>
      </label>
    </Panel>
  )
}

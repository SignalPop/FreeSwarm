'use client'

import { useCallback, useEffect, useState } from 'react'
import { fmtMetric, objectives, type Objective } from '@/lib/objectives'
import NewObjective from '@/components/objective/NewObjective'
import { Button, Pill } from '@/components/ui'

/**
 * The project's objectives -- defined here, next to the data/action MCP whose settings score them;
 * the Swarm page shows and runs them. `initialText` opens the form pre-filled (e.g. a message the
 * operator wanted to turn into an objective from the Swarm page).
 */
export default function ProjectObjectives({ projectId, initialText }: { projectId: string; initialText?: string | null }) {
  const [list, setList] = useState<Objective[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [form, setForm] = useState<string | null>(initialText ?? null)
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await objectives.list(projectId)
      setList(r.objectives)
      setErr(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [projectId])

  useEffect(() => {
    setList(null)
    void load()
  }, [load])
  useEffect(() => {
    if (initialText != null) setForm(initialText)
  }, [initialText])

  const tone = (s: Objective['status']) => (s === 'running' ? 'good' : s === 'paused' ? 'warn' : 'neutral')

  /** Pause (the swarm stops picking it; an iteration already on it finishes) or resume an objective. */
  async function toggle(o: Objective) {
    setBusy(o.id)
    try {
      await objectives.setStatus(o.id, o.status === 'running' ? 'paused' : 'running')
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  async function remove(o: Objective) {
    if (
      !window.confirm(
        `Delete "${o.title}" with all ${o.candidates} candidates, its lessons, ideas and trade book? This cannot be undone.` +
          (o.status === 'running' ? '\n\nIt is running: an agent iteration on it now will fail to submit.' : ''),
      )
    )
      return
    setBusy(o.id)
    try {
      await objectives.remove(o.id)
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-[15px] font-medium text-ink">Objectives</div>
          <p className="mt-1 text-[12px] text-ink-faint">
            What this project&apos;s swarm pursues and how a result is measured -- by the data/action MCP&apos;s settings
            above when the project has one. Define them here; the Swarm page shows and runs them.
          </p>
        </div>
        <Button tone="primary" onClick={() => setForm('')}>
          New objective
        </Button>
      </div>
      {err && <div className="mt-2 text-[12px] text-bad">✗ {err}</div>}
      <div className="mt-3 space-y-2">
        {list === null && !err && <div className="text-[12px] text-ink-faint">Loading…</div>}
        {list?.length === 0 && <div className="text-[12px] text-ink-faint">No objectives yet.</div>}
        {list?.map((o) => (
          <div
            key={o.id}
            className={`flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-seam px-3 py-2 hover:border-ink-faint ${
              busy === o.id ? 'opacity-50' : ''
            }`}
          >
            <Pill tone={tone(o.status)}>{o.status}</Pill>
            <a href="/agents" title="open on the Swarm page" className="min-w-0 flex-1 truncate text-[13px] text-ink hover:underline">
              {o.title}
            </a>
            <span className="font-mono text-[11px] text-ink-faint">
              {o.metric.kind === 'task'
                ? `${o.metric.task_server}/${o.metric.task}${o.metric.source ? ` (source ${o.metric.source})` : ''} · ${o.metric.value_function ?? o.metric_label}`
                : o.metric_label}
              {' · '}
              {o.candidates} candidates
              {o.best ? ` · best ${fmtMetric(o.metric.kind, o.best.score)} (#${o.best.seq})` : ''}
            </span>
            <span className="flex gap-1">
              <button
                type="button"
                disabled={busy === o.id}
                onClick={() => void toggle(o)}
                title={
                  o.status === 'running'
                    ? 'Pause: the swarm stops working on it (an iteration already on it finishes). Resume any time.'
                    : 'Resume: the swarm works on it again, taking turns with the other running objectives.'
                }
                className="rounded-md border border-seam px-2 py-0.5 font-mono text-[11px] text-ink-dim hover:border-ink-faint hover:text-ink disabled:opacity-40"
              >
                {o.status === 'running' ? 'pause' : 'resume'}
              </button>
              <button
                type="button"
                disabled={busy === o.id}
                onClick={() => void remove(o)}
                title="Delete it with its candidates, lessons, ideas and trade book"
                className="rounded-md border border-seam px-2 py-0.5 font-mono text-[11px] text-bad hover:border-bad/60 disabled:opacity-40"
              >
                delete
              </button>
            </span>
          </div>
        ))}
      </div>
      {form !== null && (
        <NewObjective
          projectId={projectId}
          initialText={form}
          onClose={() => setForm(null)}
          onCreated={() => {
            setForm(null)
            void load()
          }}
        />
      )}
    </div>
  )
}

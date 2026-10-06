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
          <a
            key={o.id}
            href="/agents"
            className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-seam px-3 py-2 hover:border-ink-faint"
            title="open on the Swarm page"
          >
            <Pill tone={tone(o.status)}>{o.status}</Pill>
            <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{o.title}</span>
            <span className="font-mono text-[11px] text-ink-faint">
              {o.metric.kind === 'task'
                ? `${o.metric.task_server}/${o.metric.task}${o.metric.source ? ` (source ${o.metric.source})` : ''} · ${o.metric.value_function ?? o.metric_label}`
                : o.metric_label}
              {' · '}
              {o.candidates} candidates
              {o.best ? ` · best ${fmtMetric(o.metric.kind, o.best.score)} (#${o.best.seq})` : ''}
            </span>
          </a>
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

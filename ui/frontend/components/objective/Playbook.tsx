'use client'

import { useEffect, useState } from 'react'
import { duration } from '@/lib/format'
import Markdown from '@/components/Markdown'
import { Button, Pill } from '@/components/ui'

type Playbook = {
  charter: string
  practices: string
  practices_version: number
  practices_at: number
  practices_every: number
  candidates_since_practices: number
  history: { id: number; part: 'charter' | 'practices'; version: number; author: string | null; ts: number; note: string; chars: number }[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const b = await res.json()
      if (b?.detail) detail = String(b.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const ago = (ts: number) => `${duration(Date.now() / 1000 - ts)} ago`

/**
 * The playbook every agent reads at the start of every iteration: the operator's charter
 * (edit it here) and the team practices the agents rewrite themselves from their results --
 * the recursive part. Every version of both is kept and can be viewed or restored.
 */
export default function PlaybookTab({ projectId }: { projectId: string }) {
  const [pb, setPb] = useState<Playbook | null>(null)
  const [editing, setEditing] = useState<'charter' | 'practices' | null>(null)
  const [draft, setDraft] = useState('')
  const [viewing, setViewing] = useState<{ title: string; text: string; part: 'charter' | 'practices' } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const base = `/api/projects/${encodeURIComponent(projectId)}/playbook`

  useEffect(() => {
    let alive = true
    const load = () =>
      req<Playbook>(base)
        .then((d) => alive && setPb(d))
        .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    void load()
    const t = setInterval(load, 10_000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [base, tick])

  async function save(part: 'charter' | 'practices', text: string, note: string) {
    setErr(null)
    try {
      await req(base, { method: 'PUT', body: JSON.stringify({ part, text, author: 'operator', note }) })
      setEditing(null)
      setViewing(null)
      setTick((n) => n + 1)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  async function open(id: number, title: string) {
    try {
      const h = await req<{ text: string; part: 'charter' | 'practices' }>(`${base}/history/${id}`)
      setViewing({ title, text: h.text, part: h.part })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  if (!pb) return <div className="text-[12px] text-ink-faint">{err ?? 'Loading…'}</div>

  const section = (part: 'charter' | 'practices', title: string, hint: string, text: string, meta: React.ReactNode) => (
    <div className="rounded-lg border border-seam p-3">
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <span className="text-[12.5px] font-medium text-ink">{title}</span>
        {meta}
        {editing !== part && (
          <button
            onClick={() => {
              setEditing(part)
              setDraft(text)
            }}
            className="ml-auto font-mono text-[11px] text-accent hover:opacity-80"
          >
            edit
          </button>
        )}
      </div>
      <div className="mb-2 text-[11px] text-ink-faint">{hint}</div>
      {editing === part ? (
        <>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="h-[320px] w-full resize-y rounded-lg border border-seam bg-panel-hi p-2 font-mono text-[12px] text-ink outline-none focus:border-accent"
          />
          <div className="mt-1.5 flex gap-2">
            <Button tone="primary" onClick={() => save(part, draft, 'edited by the operator')}>
              Save — agents use it next iteration
            </Button>
            <Button tone="ghost" onClick={() => setEditing(null)}>
              Cancel
            </Button>
          </div>
        </>
      ) : text.trim() ? (
        <div className="max-h-[260px] overflow-y-auto text-[12.5px]">
          <Markdown source={text} />
        </div>
      ) : (
        <div className="text-[12px] text-ink-faint">Nothing yet.</div>
      )}
    </div>
  )

  return (
    <div className="space-y-3">
      {err && <div className="text-[12px] text-bad">{err}</div>}
      {section(
        'charter',
        'Charter',
        'Your standing instructions to every agent: how to coordinate, what to review, how to work. Read at the start of every iteration.',
        pb.charter,
        <Pill tone="accent">operator</Pill>,
      )}
      {section(
        'practices',
        'Team practices',
        `Written by the agents from their own results and rewritten every ${pb.practices_every} candidates — how the team should work, learned from what worked. ${pb.candidates_since_practices}/${pb.practices_every} candidates toward the next rewrite.`,
        pb.practices,
        pb.practices_version ? (
          <Pill tone="good">
            v{pb.practices_version} · {ago(pb.practices_at)}
          </Pill>
        ) : (
          <Pill tone="neutral">not written yet</Pill>
        ),
      )}
      <div>
        <div className="mb-1 text-[10.5px] uppercase tracking-wide text-ink-faint">History</div>
        <ul className="space-y-0.5">
          {pb.history.map((h) => (
            <li key={h.id} className="flex items-center gap-2 font-mono text-[11px] text-ink-dim">
              <span className={h.part === 'charter' ? 'text-accent' : 'text-good'}>{h.part}</span>
              <span>v{h.version}</span>
              <span className="text-ink-faint">{h.author}</span>
              <span className="truncate text-ink-faint">{h.note}</span>
              <span className="ml-auto shrink-0 text-ink-faint">{ago(h.ts)}</span>
              <button onClick={() => open(h.id, `${h.part} v${h.version} by ${h.author}`)} className="shrink-0 text-accent hover:opacity-80">
                view
              </button>
            </li>
          ))}
        </ul>
      </div>
      {viewing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setViewing(null)}>
          <div className="flex max-h-[85vh] w-full max-w-[760px] flex-col overflow-hidden rounded-2xl border border-seam bg-panel" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-3 border-b border-seam px-5 py-3">
              <span className="text-[14px] text-ink">{viewing.title}</span>
              <Button tone="primary" className="ml-auto" onClick={() => save(viewing.part, viewing.text, `restored: ${viewing.title}`)}>
                Restore this version
              </Button>
              <button onClick={() => setViewing(null)} className="font-mono text-[11px] text-ink-faint hover:text-accent">
                close
              </button>
            </div>
            <div className="min-h-0 overflow-y-auto p-5 text-[12.5px]">
              <Markdown source={viewing.text} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

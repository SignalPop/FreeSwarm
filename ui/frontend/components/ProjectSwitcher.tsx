'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { projects, type Project } from '@/lib/projects'

/**
 * Active-project selector for the sidebar.
 *
 * Switching here changes the *server's* active project, not just this tab's view. That is
 * deliberate: an agent that posts to the message board without naming a project resolves
 * to the same one, so the console and the swarm never disagree about which work is current.
 *
 * `router.refresh()` afterwards because every other panel (Swarm, Connectors) reads
 * project-scoped data and would otherwise keep showing the previous project's.
 */
export default function ProjectSwitcher({ collapsed }: { collapsed: boolean }) {
  const router = useRouter()
  const [items, setItems] = useState<Project[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)

  async function load() {
    try {
      const r = await projects.list()
      setItems(r.projects)
      setActive(r.active)
    } catch {
      // Control plane down; the page's own error panels explain it.
    }
  }

  useEffect(() => {
    load()
  }, [])

  const current = items.find((p) => p.id === active)

  async function choose(id: string) {
    setOpen(false)
    if (id === active) return
    setBusy(true)
    try {
      await projects.activate(id)
      setActive(id)
      router.refresh()
      // Panels poll on their own interval; a reload is the simplest way to make every
      // one of them re-read at once without threading a context through the whole app.
      window.location.reload()
    } finally {
      setBusy(false)
    }
  }

  if (items.length === 0) return null

  if (collapsed) {
    return (
      <div
        className="mx-3 mb-2 grid h-8 place-items-center rounded-lg border border-seam bg-panel-hi/40 font-mono text-[11px] text-accent"
        title={current?.name ?? 'project'}
      >
        {(current?.name ?? '?').slice(0, 2).toUpperCase()}
      </div>
    )
  }

  return (
    <div className="relative mx-3 mb-2">
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={busy}
        className="flex w-full items-center gap-2 rounded-lg border border-seam bg-panel-hi/40 px-3 py-2 text-left transition-colors hover:bg-panel-hi"
      >
        <span className="min-w-0 flex-1">
          <span className="block text-[10px] uppercase tracking-[0.14em] text-ink-faint">
            Project
          </span>
          <span className="block truncate text-[12.5px] text-ink">
            {current?.name ?? 'none'}
          </span>
        </span>
        <svg
          viewBox="0 0 24 24"
          className={`h-3.5 w-3.5 shrink-0 text-ink-faint transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div className="absolute left-0 right-0 top-full z-20 mt-1 overflow-hidden rounded-lg border border-seam bg-panel shadow-lg">
          {items.map((p) => (
            <button
              key={p.id}
              onClick={() => choose(p.id)}
              className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[12.5px] transition-colors ${
                p.id === active ? 'bg-accent/10 text-accent' : 'text-ink-dim hover:bg-panel-hi'
              }`}
            >
              <span className="min-w-0 flex-1 truncate">{p.name}</span>
              <span className="shrink-0 font-mono text-[10px] text-ink-faint">
                {p.connectors_active.length}c
              </span>
            </button>
          ))}
          <button
            onClick={() => {
              setOpen(false)
              router.push('/projects')
            }}
            className="w-full border-t border-seam px-3 py-2 text-left text-[12px] text-ink-faint transition-colors hover:bg-panel-hi hover:text-ink"
          >
            Manage projects…
          </button>
        </div>
      )}
    </div>
  )
}

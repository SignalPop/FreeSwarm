'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { api, type ConsoleDoc } from '@/lib/api'
import { gib } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import ProjectSwitcher from '@/components/ProjectSwitcher'

type NavItem = { href: string; label: string; icon: React.ReactNode; badge?: number }

function Icon({ d, filled = false }: { d: string; filled?: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-[18px] w-[18px] shrink-0"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={d} />
    </svg>
  )
}

const ICONS = {
  models: 'M4 6h2v12H4zM9 4h2v16H9zM14 8h2v10h-2zM19 6h2v12h-2z',
  console: 'M12 20a8 8 0 1 0-8-8m8-4v4l3 2',
  chat: 'M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z',
  apps: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
  logs: 'M5 3h10l4 4v14H5zM15 3v4h4M8 12h8M8 16h5',
  plug: 'M9 3v6M15 3v6M6 9h12v3a6 6 0 0 1-12 0zM12 18v3',
  lab: 'M3 17l5-6 4 4 5-8 4 5M3 21h18',
  network: 'M12 3a3 3 0 1 1 0 6 3 3 0 0 1 0-6zM5 15a3 3 0 1 1 0 6 3 3 0 0 1 0-6zM19 15a3 3 0 1 1 0 6 3 3 0 0 1 0-6zM12 9v3M12 12l-5.5 3.5M12 12l5.5 3.5',
  folder: 'M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z',
  settings: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H1a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 2.6 7a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H7a1.6 1.6 0 0 0 1-1.5V1a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V7a1.6 1.6 0 0 0 1.5 1H23a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z',
}

/** Sidebar gauge: one labelled bar, used for aggregate VRAM and host RAM. */
function Gauge({ label, used, total }: { label: string; used: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0
  // Green under half, amber past two-thirds, red when nearly full -- the point of the bar
  // is to be readable at a glance from across the room.
  const tone = pct > 90 ? 'bg-bad' : pct > 66 ? 'bg-warn' : 'bg-good'
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="text-ink-dim">{label}</span>
        <span className="font-mono text-ink">
          {gib(used)} <span className="text-ink-faint">/ {gib(total)} GiB</span>
        </span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-seam">
        <div className={`h-full rounded-full ${tone} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [modelCount, setModelCount] = useState<number | null>(null)
  // The running control plane's version -- not the one the page was built with -- so a stale
  // tab or an un-restarted backend shows up as a mismatch instead of passing as current.
  const [appVersion, setAppVersion] = useState<string | null>(null)
  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d?.app_version && setAppVersion(String(d.app_version)))
      .catch(() => {})
  }, [])

  // The sidebar's own poll is slow (5s): it only needs the resource gauges and the engine
  // pill, not the 1 Hz cadence the console view uses.
  const { data } = usePoll<ConsoleDoc>(api.console, 5000)

  useEffect(() => {
    const stored = (localStorage.getItem('ft-theme') as 'dark' | 'light') ?? 'dark'
    setTheme(stored)
    api
      .models()
      .then((r) => setModelCount(r.models.length))
      .catch(() => setModelCount(null))
  }, [])

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try {
      localStorage.setItem('ft-theme', next)
    } catch {
      /* private window */
    }
    document.documentElement.setAttribute('data-theme', next)
  }

  const nav: NavItem[] = [
    { href: '/projects', label: 'Projects', icon: <Icon d={ICONS.folder} /> },
    { href: '/models', label: 'Models', icon: <Icon d={ICONS.models} filled />, badge: modelCount ?? undefined },
    { href: '/', label: 'Console', icon: <Icon d={ICONS.console} /> },
    { href: '/chat', label: 'Chat', icon: <Icon d={ICONS.chat} /> },
    { href: '/agents', label: 'Swarm', icon: <Icon d={ICONS.apps} /> },
    { href: '/forecast-lab', label: 'Forecast Lab', icon: <Icon d={ICONS.lab} /> },
    { href: '/connectors', label: 'Connectors', icon: <Icon d={ICONS.plug} /> },
    { href: '/network', label: 'Network', icon: <Icon d={ICONS.network} /> },
    { href: '/logs', label: 'Logs', icon: <Icon d={ICONS.logs} /> },
    { href: '/settings', label: 'Settings', icon: <Icon d={ICONS.settings} /> },
  ]

  const gpus = data?.gpus ?? []
  const vramUsed = gpus.reduce((a, g) => a + g.memory_used_bytes, 0)
  const vramTotal = gpus.reduce((a, g) => a + g.memory_total_bytes, 0)
  const ram = data?.host_memory
  const engineState = data?.engine.state ?? 'stopped'
  const port = data?.engine.engine_url?.split(':').pop() ?? '1919'

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      {/* ---- Sidebar ---- */}
      <aside
        className={`flex shrink-0 flex-col border-r border-seam bg-panel transition-[width] duration-200 ${
          collapsed ? 'w-[68px]' : 'w-[232px]'
        }`}
      >
        <div className="flex h-14 items-center justify-end px-3">
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="grid h-8 w-8 place-items-center rounded-lg border border-seam text-ink-dim transition-colors hover:bg-panel-hi hover:text-ink"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d={collapsed ? 'M9 18l6-6-6-6' : 'M15 18l-6-6 6-6'} />
            </svg>
          </button>
        </div>

        <ProjectSwitcher collapsed={collapsed} />

        <nav className="flex-1 space-y-1 px-3">
          {nav.map((item) => {
            const active = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                title={collapsed ? item.label : undefined}
                className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${
                  active
                    ? 'bg-panel-hi text-ink shadow-[inset_0_0_0_1px_var(--color-seam)]'
                    : 'text-ink-dim hover:bg-panel-hi/60 hover:text-ink'
                }`}
              >
                {item.icon}
                {!collapsed && (
                  <>
                    <span className="flex-1">{item.label}</span>
                    {item.badge !== undefined && (
                      <span className="font-mono text-xs text-ink-faint">{item.badge}</span>
                    )}
                  </>
                )}
              </Link>
            )
          })}
        </nav>

        {!collapsed && (
          <div className="m-3 space-y-3 rounded-xl border border-seam bg-panel-hi/40 p-3">
            <Gauge label="System VRAM" used={vramUsed} total={vramTotal} />
            <Gauge label="System RAM" used={(ram?.total_bytes ?? 0) - (ram?.available_bytes ?? 0)} total={ram?.total_bytes ?? 0} />
            <div className="border-t border-seam pt-2 font-mono text-[10px] leading-relaxed text-ink-faint">
              {gpus.length > 0
                ? Array.from(new Set(gpus.map((g) => g.name))).map((n) => <div key={n}>{n}</div>)
                : 'no GPUs detected'}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between px-4 pb-4 pt-1">
          <button
            onClick={toggleTheme}
            className="grid h-9 w-9 place-items-center rounded-xl border border-seam text-ink-dim transition-colors hover:bg-panel-hi hover:text-ink"
            aria-label="Toggle theme"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
              {theme === 'dark' ? (
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              ) : (
                <>
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
                </>
              )}
            </svg>
          </button>
          {!collapsed && (
            <span className="font-mono text-[10px] text-ink-faint" title="FreeToken version of the running control plane">
              v{appVersion ?? '?'}
            </span>
          )}
        </div>
      </aside>

      {/* ---- Main ---- */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-center gap-3 border-b border-seam px-6">
          <span className="flex items-center gap-2 font-mono text-[13px] text-ink-dim">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                engineState === 'running'
                  ? 'bg-good animate-dot'
                  : engineState === 'starting'
                    ? 'bg-warn animate-dot'
                    : engineState === 'error'
                      ? 'bg-bad'
                      : 'bg-ink-faint'
              }`}
            />
            FreeSwarm Console
          </span>
          <span className="ml-auto font-mono text-[12px] text-ink-faint">API :{port}</span>
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  )
}

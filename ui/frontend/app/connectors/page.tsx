'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button, EmptyState, PageHeader, Panel, Pill } from '@/components/ui'

type AuthState = {
  authorised: boolean
  scope?: string | null
  has_refresh_token?: boolean
  expires_at?: number | null
}

type ServerSpec = {
  name: string
  transport: 'stdio' | 'http' | 'sse'
  command: string | null
  args: string[]
  cwd: string | null
  url: string | null
  enabled: boolean
  oauth: boolean
  auth: AuthState
}

type ProbeResult = {
  name: string
  ok: boolean
  error: string | null
  tools: { function: { name: string; description: string; parameters: unknown } }[]
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export default function ConnectorsPage() {
  const [servers, setServers] = useState<ServerSpec[]>([])
  const [configPath, setConfigPath] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [callResult, setCallResult] = useState<Record<string, string>>({})
  const [err, setErr] = useState<string | null>(null)
  const [connecting, setConnecting] = useState<string | null>(null)

  // Each server is probed on its OWN request. One combined request meant a single server that
  // never answered (e.g. `docker run` while Docker was down) kept every card on "probing..."
  // forever; now each result lands as soon as it is ready and a slow one delays only itself.
  const [results, setResults] = useState<Record<string, ProbeResult>>({})
  const [pending, setPending] = useState<Set<string>>(new Set())
  const probe = { servers: Object.values(results) }
  const loading = pending.size > 0

  const probeAll = useCallback(async (names: string[]) => {
    setPending(new Set(names))
    await Promise.all(
      names.map(async (name) => {
        try {
          const r = await get<{ servers: ProbeResult[] }>(
            `/api/mcp/probe?server=${encodeURIComponent(name)}`,
          )
          const hit = r.servers[0]
          if (hit) setResults((x) => ({ ...x, [name]: hit }))
        } catch (e) {
          setResults((x) => ({
            ...x,
            [name]: { name, ok: false, tools: [], error: e instanceof Error ? e.message : String(e) },
          }))
        } finally {
          setPending((p) => {
            const n = new Set(p)
            n.delete(name)
            return n
          })
        }
      }),
    )
  }, [])

  const refresh = useCallback(() => {
    void probeAll(servers.filter((s) => s.enabled).map((s) => s.name))
  }, [probeAll, servers])

  // Probe once the server list is known, then slowly (probing spawns a process per server).
  useEffect(() => {
    if (!servers.length) return
    refresh()
    const t = setInterval(refresh, 60000)
    return () => clearInterval(t)
  }, [servers, refresh])

  async function load() {
    try {
      const r = await get<{ servers: ServerSpec[]; config_path: string }>('/api/mcp/servers')
      setServers(r.servers)
      setConfigPath(r.config_path)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    load()
  }, [])

  async function initConfig() {
    try {
      await fetch('/api/mcp/servers/init', { method: 'POST' })
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  async function connect(name: string) {
    setConnecting(name)
    setErr(null)
    try {
      const res = await fetch('/api/mcp/oauth/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server: name }),
      })
      const body = await res.json()
      if (!res.ok) throw new Error(body?.detail ?? `${res.status} ${res.statusText}`)

      // Open the provider's consent page in a new tab. It redirects back to the control
      // plane's loopback callback, which completes the exchange server-side.
      window.open(body.authorization_url, '_blank', 'noopener,noreferrer')

      // Poll until the token lands (or the operator gives up).
      const deadline = Date.now() + 5 * 60 * 1000
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 1500))
        const st = await fetch(
          `/api/mcp/oauth/status?server=${encodeURIComponent(name)}&state=${encodeURIComponent(body.state)}`,
        ).then((r) => r.json())
        if (st.authorised) break
        if (st.error) throw new Error(st.error)
      }
      await load()
      refresh()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setConnecting(null)
    }
  }

  async function disconnect(name: string) {
    if (!window.confirm(`Forget the stored credentials for "${name}"?`)) return
    try {
      await fetch('/api/mcp/oauth/disconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server: name }),
      })
      await load()
      refresh()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  async function invoke(tool: string) {
    setCallResult((p) => ({ ...p, [tool]: 'running…' }))
    try {
      const res = await fetch('/api/mcp/call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // No-argument smoke test. Tools with required parameters will report the
        // validation error, which is itself useful feedback about the schema.
        body: JSON.stringify({ tool, arguments: {} }),
      })
      const body = await res.json()
      setCallResult((p) => ({
        ...p,
        [tool]: body.content ?? body.detail ?? JSON.stringify(body).slice(0, 2000),
      }))
    } catch (e) {
      setCallResult((p) => ({ ...p, [tool]: e instanceof Error ? e.message : String(e) }))
    }
  }

  const probes = probe?.servers ?? []
  const totalTools = probes.reduce((a, s) => a + s.tools.length, 0)

  return (
    <div className="mx-auto max-w-[1000px] px-8 py-8">
      <PageHeader
        title="Connectors"
        subtitle={`${servers.filter((s) => s.enabled).length} enabled · ${totalTools} tools available to models`}
        right={
          <div className="flex items-center gap-2">
            <Pill tone={loading ? 'warn' : 'good'} pulse={loading}>
              {loading ? 'Probing…' : 'Ready'}
            </Pill>
            <Button tone="ghost" onClick={refresh}>
              Re-probe
            </Button>
          </div>
        }
      />

      <Panel className="mb-6 border-accent/25 bg-accent/[0.04] p-4">
        <div className="text-[13px] leading-relaxed text-ink-dim">
          Servers are declared on disk, never through this API — a stdio MCP server is an
          arbitrary command line, so accepting one over HTTP would make this a
          remote-code-execution endpoint.
        </div>
        {configPath && (
          <div className="mt-2 break-all font-mono text-[11px] text-ink-faint">{configPath}</div>
        )}
        {servers.length === 0 && (
          <Button tone="ghost" className="mt-3" onClick={initConfig}>
            Create a starter config
          </Button>
        )}
      </Panel>

      {err && (
        <Panel className="mb-6 border-bad/35 bg-bad/5 p-4 font-mono text-[12px] text-bad">{err}</Panel>
      )}

      {servers.length === 0 ? (
        <EmptyState
          title="No connectors declared"
          hint="Add a server to mcp_servers.json. Both the native format and the Claude-Desktop mcpServers format are accepted."
        />
      ) : (
        <div className="space-y-4">
          {servers.map((s) => {
            const p = probes.find((x) => x.name === s.name)
            return (
              <Panel key={s.name} className="p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-[15px] text-ink">{s.name}</span>
                  <Pill>{s.transport}</Pill>
                  {s.oauth && (
                    <Pill tone={s.auth?.authorised ? 'good' : 'warn'}>
                      {s.auth?.authorised ? 'authorised' : 'needs sign-in'}
                    </Pill>
                  )}
                  {!s.enabled ? (
                    <Pill>disabled</Pill>
                  ) : p ? (
                    <Pill tone={p.ok ? 'good' : 'bad'} pulse={p.ok}>
                      {p.ok ? `${p.tools.length} tools` : 'failed'}
                    </Pill>
                  ) : (
                    <Pill tone="warn">probing…</Pill>
                  )}
                  <span className="ml-auto flex items-center gap-2">
                    {s.oauth && !s.auth?.authorised && (
                      <Button
                        tone="primary"
                        onClick={() => connect(s.name)}
                        disabled={connecting !== null}
                      >
                        {connecting === s.name ? 'Waiting for sign-in…' : 'Connect'}
                      </Button>
                    )}
                    {s.oauth && s.auth?.authorised && (
                      <Button tone="ghost" onClick={() => disconnect(s.name)}>
                        Disconnect
                      </Button>
                    )}
                    {p && p.tools.length > 0 && (
                      <Button
                        tone="ghost"
                        onClick={() => setExpanded(expanded === s.name ? null : s.name)}
                      >
                        {expanded === s.name ? 'Hide tools' : 'Show tools'}
                      </Button>
                    )}
                  </span>
                </div>

                <div className="mt-2 break-all font-mono text-[11px] text-ink-faint">
                  {s.transport === 'stdio'
                    ? `${s.command ?? '?'} ${s.args.join(' ')}${s.cwd ? `   (cwd ${s.cwd})` : ''}`
                    : s.url}
                </div>
                {s.oauth && s.auth?.authorised && (
                  <div className="mt-1 font-mono text-[10.5px] text-ink-faint">
                    OAuth
                    {s.auth.scope ? ` · scope ${s.auth.scope}` : ''}
                    {s.auth.has_refresh_token ? ' · auto-refresh' : ''}
                    {s.auth.expires_at
                      ? ` · expires ${new Date(s.auth.expires_at * 1000).toLocaleString()}`
                      : ''}
                  </div>
                )}

                {p && !p.ok && p.error && (
                  <div className="mt-3 rounded-xl border border-bad/35 bg-bad/5 p-3 font-mono text-[11.5px] leading-relaxed text-bad">
                    {p.error}
                  </div>
                )}

                {expanded === s.name && p && (
                  <div className="mt-4 space-y-2">
                    {p.tools.map((t) => {
                      const name = t.function.name
                      return (
                        <div key={name} className="rounded-xl border border-seam bg-panel-hi/40 p-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-[12px] text-accent">{name}</span>
                            <Button tone="ghost" className="ml-auto" onClick={() => invoke(name)}>
                              Test
                            </Button>
                          </div>
                          <div className="mt-1.5 text-[12px] leading-relaxed text-ink-dim">
                            {t.function.description || '(no description)'}
                          </div>
                          {callResult[name] && (
                            <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg border border-seam bg-canvas p-2 font-mono text-[11px] text-ink-dim">
                              {callResult[name]}
                            </pre>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </Panel>
            )
          })}
        </div>
      )}
    </div>
  )
}

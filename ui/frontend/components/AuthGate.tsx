'use client'

import { useEffect, useState } from 'react'
import { authStatus, installAuthFetch, login, refreshTokens, tokens, type AuthStatus } from '@/lib/auth'
import { Button, Panel } from '@/components/ui'

/**
 * Renders a login form when the control plane requires authentication and we hold no
 * usable token; otherwise renders the app.
 *
 * Authentication is presence-based on the server (any account => required), so the common
 * loopback install never sees this screen at all. That is intentional: the security
 * boundary is the bind address, and `Settings.validate()` refuses to expose an
 * unauthenticated server off loopback.
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus | null>(null)
  const [authed, setAuthed] = useState(false)
  const [checking, setChecking] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    installAuthFetch()
    ;(async () => {
      try {
        const s = await authStatus()
        setStatus(s)
        if (!s.auth_required) {
          setAuthed(true)
        } else if (tokens.access()) {
          // A stored access token may have expired while the tab was closed; the refresh
          // token usually has not. Verify rather than assuming.
          const me = await fetch('/api/auth/me')
          setAuthed(me.ok ? true : await refreshTokens())
        }
      } catch {
        // Control plane down. Let the app render so its own error panels explain it --
        // a login form would be misleading here.
        setAuthed(true)
      } finally {
        setChecking(false)
      }
    })()
  }, [])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    try {
      await login(username, password)
      setAuthed(true)
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : String(e2))
    } finally {
      setBusy(false)
    }
  }

  if (checking) {
    return (
      <div className="grid h-screen place-items-center bg-canvas">
        <span className="font-mono text-[13px] text-ink-faint">Connecting…</span>
      </div>
    )
  }

  if (authed) return <>{children}</>

  return (
    <div className="grid h-screen place-items-center bg-canvas px-6">
      <Panel className="w-full max-w-sm p-7">
        <div className="text-[20px] font-semibold tracking-tight text-ink">FreeSwarm Console</div>
        <p className="mt-1.5 text-[12.5px] text-ink-dim">
          Sign in to reach the control plane.
        </p>

        <form onSubmit={submit} className="mt-6 space-y-3">
          <label className="block">
            <span className="text-[12px] text-ink-dim">Username</span>
            <input
              className="mt-1.5 w-full rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[13px] text-ink outline-none focus:border-accent"
              value={username}
              autoFocus
              autoComplete="username"
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label className="block">
            <span className="text-[12px] text-ink-dim">Password</span>
            <input
              type="password"
              className="mt-1.5 w-full rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[13px] text-ink outline-none focus:border-accent"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {err && <div className="font-mono text-[12px] text-bad">{err}</div>}

          <Button tone="primary" type="submit" className="w-full" disabled={busy || !username || !password}>
            {busy ? 'Signing in…' : 'Sign in'}
          </Button>
        </form>

        {status && !status.tls && status.bind !== '127.0.0.1' && (
          <p className="mt-5 border-t border-seam pt-4 text-[11.5px] leading-relaxed text-warn">
            This server is bound to {status.bind} without TLS. Your token will cross the
            network in clear text — prefer an SSH tunnel to loopback.
          </p>
        )}

        <p className="mt-5 border-t border-seam pt-4 text-[11.5px] leading-relaxed text-ink-faint">
          Accounts are created on the server:{' '}
          <code className="font-mono">python -m app.usercli add &lt;name&gt;</code>
        </p>
      </Panel>
    </div>
  )
}

'use client'

// Bearer-token plumbing for the console.
//
// The token lives in sessionStorage, not localStorage: it is a credential, and scoping it
// to the tab means closing the tab ends the session. It is deliberately NOT a cookie --
// no cookie means no CSRF surface on the control plane, which is a process launcher.

const ACCESS_KEY = 'ft-access-token'
const REFRESH_KEY = 'ft-refresh-token'

function read(key: string): string | null {
  try {
    return sessionStorage.getItem(key)
  } catch {
    return null // private window / blocked storage
  }
}

function write(key: string, value: string | null) {
  try {
    if (value === null) sessionStorage.removeItem(key)
    else sessionStorage.setItem(key, value)
  } catch {
    /* non-fatal: the session just will not survive a reload */
  }
}

export const tokens = {
  access: () => read(ACCESS_KEY),
  refresh: () => read(REFRESH_KEY),
  set(access: string, refreshToken: string) {
    write(ACCESS_KEY, access)
    write(REFRESH_KEY, refreshToken)
  },
  clear() {
    write(ACCESS_KEY, null)
    write(REFRESH_KEY, null)
  },
}

export type AuthStatus = {
  auth_required: boolean
  user_count: number
  tls: boolean
  bind: string
}

export async function authStatus(): Promise<AuthStatus> {
  const res = await fetch('/api/auth/status')
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function login(username: string, password: string): Promise<void> {
  // OAuth2 password grant is form-encoded, not JSON -- that is the spec, and FastAPI's
  // OAuth2PasswordRequestForm expects exactly these field names.
  const body = new URLSearchParams({ username, password, grant_type: 'password' })
  const res = await fetch('/api/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) detail = String(j.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  const data = await res.json()
  tokens.set(data.access_token, data.refresh_token)
}

/** Exchange the refresh token for a new pair. Returns false when it is gone or expired. */
export async function refreshTokens(): Promise<boolean> {
  const rt = tokens.refresh()
  if (!rt) return false
  const res = await fetch('/api/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: rt }),
  })
  if (!res.ok) {
    tokens.clear()
    return false
  }
  const data = await res.json()
  tokens.set(data.access_token, data.refresh_token)
  return true
}

/**
 * Install a global fetch wrapper that attaches the bearer token and retries once on 401.
 *
 * Patching `window.fetch` rather than threading a client through every component keeps
 * lib/api.ts and lib/board.ts unaware of auth entirely -- they issue plain fetches and
 * this adds the header. The single retry handles the common case of an access token that
 * expired mid-session while the refresh token is still good.
 */
let installed = false

export function installAuthFetch() {
  if (installed || typeof window === 'undefined') return
  installed = true

  const original = window.fetch.bind(window)

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const isOurApi = url.startsWith('/api/') || url.startsWith('/mb/')
    // Never attach a token to the auth endpoints themselves: /api/auth/token is how you
    // get one, and sending a stale bearer there only invites confusion.
    const isAuthRoute = url.startsWith('/api/auth/')

    if (!isOurApi || isAuthRoute) return original(input, init)

    const withAuth = (token: string | null): RequestInit => {
      if (!token) return init ?? {}
      const headers = new Headers(init?.headers)
      headers.set('Authorization', `Bearer ${token}`)
      return { ...init, headers }
    }

    let res = await original(input, withAuth(tokens.access()))
    if (res.status === 401 && tokens.refresh()) {
      if (await refreshTokens()) {
        res = await original(input, withAuth(tokens.access()))
      }
    }
    return res
  }
}

// LAN federation: share this computer's models, and use models shared by other computers.
// Backend: ui/backend/app/federation.py (console routes under /api/federation).

export type RemoteModel = { name: string; ready: boolean; context: number | null; decode_tps: number | null; active: number | null }

export type FederationDoc = {
  node: {
    node_id: string
    name: string
    fingerprint: string
    addresses: string[]
    port: number
    discovery_port: number
    app_version: string
    protocol: number
    min_protocol: number
  }
  sharing: {
    enabled: boolean
    gateway_running: boolean
    error: string | null
    shared_models: string[]
    available_models: { name: string; ready: boolean }[]
  }
  requests: {
    user_code: string
    client_name: string
    address: string
    client_fingerprint: string
    expires_in: number
    app_version: string | null
  }[]
  clients: {
    client_id: string
    name: string
    address: string
    created: number
    last_seen: number | null
    requests: number
    scope: string
    app_version: string | null
  }[]
  discovered: {
    node_id: string
    name: string
    address: string
    port: number
    fingerprint: string
    last_seen: number
    app_version: string | null
    protocol: number | null
    compatible: boolean
    version_note: string | null
  }[]
  peers: {
    node_id: string
    name: string
    slug: string
    address: string
    port: number
    fingerprint: string
    connected_at: number | null
    enabled: boolean
    status: string
    error: string | null
    models: RemoteModel[]
    app_version: string | null
    version_note: string | null
  }[]
  firewall: {
    platform: string
    tcp_rule: boolean | null
    udp_rule: boolean | null
    tcp_port: number
    udp_port: number
    networks: { name: string; category: string }[]
  }
  /** Other computers announcing THIS computer's node id -- a copied federation.json. */
  identity_conflicts: { name: string; address: string; fingerprint: string; last_seen: number }[]
  pairings: {
    id: string
    node_id: string
    name: string
    address: string
    port: number
    fingerprint: string
    user_code: string
    expires: number
    status: 'waiting' | 'connected' | 'denied' | 'failed' | 'expired'
    error: string | null
  }[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const b = await res.json()
      if (b?.detail) detail = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const federation = {
  get: () => req<FederationDoc>('/api/federation'),
  setSharing: (body: { enabled?: boolean; shared_models?: string[]; name?: string }) =>
    req<FederationDoc>('/api/federation/sharing', { method: 'PUT', body: JSON.stringify(body) }),
  decide: (code: string, approve: boolean) =>
    req<FederationDoc>(`/api/federation/requests/${encodeURIComponent(code)}`, { method: 'POST', body: JSON.stringify({ approve }) }),
  revokeClient: (id: string) => req<FederationDoc>(`/api/federation/clients/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  connect: (body: { node_id?: string; address?: string }) =>
    req<FederationDoc['pairings'][number]>('/api/federation/peers', { method: 'POST', body: JSON.stringify(body) }),
  disconnect: (nodeId: string) => req<FederationDoc>(`/api/federation/peers/${encodeURIComponent(nodeId)}`, { method: 'DELETE' }),
}

/** Models served by another computer are named `model@computer`; local ids never contain '@'. */
export function isRemoteModel(name: string | null | undefined): boolean {
  return !!name && name.includes('@')
}

export function remoteParts(name: string): { model: string; node: string } {
  const i = name.lastIndexOf('@')
  return { model: name.slice(0, i), node: name.slice(i + 1) }
}

/** First 16 hex digits of a fingerprint, grouped -- what an operator compares by eye. */
export function shortFp(fp: string): string {
  return fp.replace(/:/g, '').slice(0, 16).replace(/(.{4})/g, '$1 ').trim()
}

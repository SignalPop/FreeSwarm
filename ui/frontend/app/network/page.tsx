'use client'

import { useEffect, useState } from 'react'
import { duration } from '@/lib/format'
import { federation, shortFp, type FederationDoc } from '@/lib/federation'
import { Button, PageHeader, Panel, Pill } from '@/components/ui'

/** Windows Firewall: the one TCP port a sharing computer must open, the optional UDP port for
 *  discovery, and whether each network is Private (the rules are scoped to Private only). */
function FirewallPanel({ fw, sharing }: { fw: FederationDoc['firewall']; sharing: boolean }) {
  if (fw.platform !== 'win32') return null
  const pub = fw.networks.filter((n) => n.category === 'Public')
  const mark = (v: boolean | null) =>
    v === null ? <span className="text-ink-faint">unknown</span> : v ? <span className="text-good">open</span> : <span className="text-warn">not open</span>
  return (
    <Panel className="p-4">
      <div className="mb-2 text-[14px] font-medium text-ink">Windows Firewall</div>
      <div className="space-y-0.5 font-mono text-[11.5px] text-ink-dim">
        <div>
          TCP {fw.tcp_port} (gateway, needed to share): {mark(fw.tcp_rule)}
        </div>
        <div>
          UDP {fw.udp_port} (discovery, optional): {mark(fw.udp_rule)}
        </div>
        {fw.networks.map((n) => (
          <div key={n.name}>
            network &quot;{n.name}&quot;: <span className={n.category === 'Public' ? 'text-warn' : 'text-good'}>{n.category}</span>
          </div>
        ))}
      </div>
      {(!fw.tcp_rule || !fw.udp_rule) && (
        <div className="mt-2 text-[12px] leading-relaxed text-ink-dim">
          Run <code className="font-mono text-ink">ui\federation-firewall.cmd</code> as Administrator on each computer. It opens
          only these two ports, only for the Private network profile and only to your local subnet
          {sharing ? '' : ' — the TCP port matters only on a computer that shares'}.
        </div>
      )}
      {pub.length > 0 && (
        <div className="mt-2 text-[12px] leading-relaxed text-warn">
          {pub.map((n) => `"${n.name}"`).join(', ')} {pub.length === 1 ? 'is' : 'are'} Public, so the rules do not apply
          there (by design). If it is your trusted home or office network: Settings → Network &amp; internet → your
          connection → Private network.
        </div>
      )}
      {fw.networks.some((n) => /tailscale/i.test(n.name)) && (
        <div className="mt-2 text-[12px] leading-relaxed text-ink-faint">
          Tailscale is available: pairing over it works between any of your computers, even on different networks,
          with its own encryption on top of this TLS. Discovery does not cross Tailscale — use Connect by address with
          the other computer&apos;s Tailscale IP (100.x.y.z) or name.
        </div>
      )}
    </Panel>
  )
}

/** The other computer's release and whether it can work with this one. */
function VersionLine({ version, compatible, note }: { version: string | null; compatible: boolean; note: string | null }) {
  return (
    <div className="mt-0.5 flex flex-wrap items-center gap-1.5 font-mono text-[10.5px]">
      <span className="text-ink-dim">FreeToken {version ?? '(unknown version)'}</span>
      <Pill tone={!compatible ? 'bad' : note ? 'warn' : 'good'}>{!compatible ? 'incompatible' : note ? 'compatible, differs' : 'same version'}</Pill>
      {note && <span className={compatible ? 'text-ink-faint' : 'text-bad'}>{note}</span>}
    </div>
  )
}

const ago = (ts: number | null | undefined) => (ts ? `${duration(Date.now() / 1000 - ts)} ago` : '—')

/**
 * Other FreeToken computers on the local network: share this computer's models with them,
 * and use theirs. Pairing is OAuth 2.0 device authorization -- the request must be approved on
 * the computer that owns the model -- over TLS with the certificate pinned at pairing.
 */
export default function NetworkPage() {
  const [doc, setDoc] = useState<FederationDoc | null>(null)
  // Two kinds of failure, kept apart: what an action reported (sticky -- the operator has to be
  // able to read why Connect failed) and what the 2.5s refresh hit (cleared as soon as it works).
  const [err, setErr] = useState<string | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)
  const [addr, setAddr] = useState('')
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState<string | null>(null)

  async function load() {
    try {
      setDoc(await federation.get())
      setLoadErr(null)
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e))
    }
  }
  useEffect(() => {
    void load()
    const t = setInterval(load, 2500)
    return () => clearInterval(t)
  }, [])

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setErr(null)
    try {
      await fn()
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  if (!doc) {
    return (
      <div className="mx-auto max-w-[1180px] px-8 py-8">
        <PageHeader title="Network" subtitle="Other FreeToken computers on this network" />
        <Panel className="p-4 text-[13px] text-ink-faint">{loadErr ?? err ?? 'Loading…'}</Panel>
      </div>
    )
  }

  const s = doc.sharing
  const shared = new Set(s.shared_models)
  const waiting = doc.pairings.filter((p) => p.status !== 'connected')

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Network"
        subtitle={`FreeToken ${doc.node.app_version} · federation protocol ${doc.node.protocol} · ${doc.peers.length} connected · ${doc.discovered.length} discovered · sharing ${s.enabled ? 'on' : 'off'}`}
        right={<Pill tone={s.gateway_running ? 'good' : 'neutral'} pulse={s.gateway_running}>{s.gateway_running ? `sharing on :${doc.node.port}` : 'not sharing'}</Pill>}
      />
      {err && (
        <Panel className="mb-4 flex items-start gap-3 border-bad/35 bg-bad/5 p-3 text-[12.5px] leading-relaxed text-bad">
          <div className="min-w-0 flex-1">{err}</div>
          <button onClick={() => setErr(null)} className="shrink-0 font-mono text-[11px] text-ink-faint hover:text-bad">
            dismiss
          </button>
        </Panel>
      )}
      {loadErr && <Panel className="mb-4 border-warn/35 bg-warn/5 p-3 text-[12.5px] text-warn">{loadErr}</Panel>}

      {/* ---- Two computers sharing one identity file: nothing can pair until it is fixed ---- */}
      {doc.identity_conflicts.length > 0 && (
        <Panel className="mb-6 border-bad/50 bg-bad/5 p-4">
          <div className="mb-2 text-[14px] font-medium text-bad">Another computer is using this computer&apos;s identity</div>
          {doc.identity_conflicts.map((c) => (
            <div key={c.fingerprint} className="border-t border-seam/60 py-1.5 font-mono text-[11.5px] text-ink-dim first:border-t-0">
              {c.name} at {c.address} · cert {shortFp(c.fingerprint)} · node id {doc.node.node_id.slice(0, 8)}… (same as this one)
            </div>
          ))}
          <div className="mt-2 text-[12.5px] leading-relaxed text-ink-dim">
            Its <code className="font-mono text-ink">ui\backend\auth\federation.json</code> was copied from this computer, so both
            claim the same node id. Neither can discover or pair with the other: a request sent from here shows up there for
            approval, but approving it can never finish. On <span className="text-ink">one</span> of the two computers, stop
            FreeToken, delete that file, and start it again — a fresh node id and name are written on startup.
          </div>
        </Panel>
      )}

      {/* ---- Requests waiting for THIS computer's approval: the most urgent thing on the page ---- */}
      {doc.requests.length > 0 && (
        <Panel className="mb-6 border-warn/50 bg-warn/5 p-4">
          <div className="mb-2 text-[14px] font-medium text-ink">Requests to use this computer</div>
          {doc.requests.map((r) => (
            <div key={r.user_code} className="flex flex-wrap items-center gap-3 border-t border-seam/60 py-2 first:border-t-0">
              <span className="font-mono text-[20px] tracking-widest text-warn">{r.user_code}</span>
              <div className="min-w-0 flex-1 text-[12.5px] text-ink-dim">
                <span className="text-ink">{r.client_name}</span> at {r.address}
                {r.app_version ? ` (FreeToken ${r.app_version})` : ''} wants to use the models you share.
                <div className="font-mono text-[10.5px] text-ink-faint">
                  Approve only if the same code is showing on {r.client_name}. Expires in {Math.max(0, Math.round(r.expires_in / 60))} min.
                </div>
              </div>
              <Button tone="primary" disabled={busy} onClick={() => act(() => federation.decide(r.user_code, true))}>
                Approve
              </Button>
              <Button tone="ghost" disabled={busy} onClick={() => act(() => federation.decide(r.user_code, false))}>
                Deny
              </Button>
            </div>
          ))}
        </Panel>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* ---- Using other computers ---- */}
        <div className="space-y-4">
          <Panel className="p-4">
            <div className="mb-1 text-[14px] font-medium text-ink">Other computers</div>
            <div className="mb-3 text-[12px] text-ink-faint">
              Computers announcing FreeToken sharing on this network. Connect, then approve the request on that computer.
              Their models then appear here as <span className="font-mono text-remote">model@computer</span>, in violet.
            </div>
            {doc.discovered.length === 0 && (
              <div className="text-[12px] text-ink-faint">
                None heard yet. Turn on sharing on the other computer; it announces itself every few seconds (UDP{' '}
                {doc.node.discovery_port}). If Windows Firewall blocks the broadcast, connect by address below.
              </div>
            )}
            {doc.discovered.map((d) => (
              <div key={d.node_id} className="flex items-center gap-3 border-t border-seam/60 py-2 first:border-t-0">
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-[12.5px] text-ink">{d.name}</div>
                  <div className="font-mono text-[10.5px] text-ink-faint">
                    {d.address}:{d.port} · cert {shortFp(d.fingerprint)} · seen {ago(d.last_seen)}
                  </div>
                  <VersionLine version={d.app_version} compatible={d.compatible} note={d.version_note} />
                </div>
                <Button tone="primary" disabled={busy || !d.compatible} onClick={() => act(() => federation.connect({ node_id: d.node_id }))}>
                  {d.compatible ? 'Connect' : 'Update needed'}
                </Button>
              </div>
            ))}
            <div className="mt-3 flex gap-2">
              <input
                value={addr}
                onChange={(e) => setAddr(e.target.value)}
                placeholder={`address[:port]   e.g. 192.168.1.20:${doc.node.port}`}
                className="min-w-0 flex-1 rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
              />
              <Button tone="ghost" disabled={busy || !addr.trim()} onClick={() => act(() => federation.connect({ address: addr.trim() }))}>
                Connect by address
              </Button>
            </div>
          </Panel>

          {waiting.length > 0 && (
            <Panel className="p-4">
              <div className="mb-2 text-[14px] font-medium text-ink">Pairing</div>
              {waiting.map((p) => (
                <div key={p.id} className="border-t border-seam/60 py-2 first:border-t-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[12.5px] text-ink">{p.name}</span>
                    <Pill tone={p.status === 'waiting' ? 'warn' : 'bad'} pulse={p.status === 'waiting'}>{p.status}</Pill>
                  </div>
                  {p.status === 'waiting' ? (
                    <div className="mt-1 text-[12.5px] text-ink-dim">
                      On <span className="text-ink">{p.name}</span>, open FreeToken → Network and approve code{' '}
                      <span className="font-mono text-[18px] tracking-widest text-warn">{p.user_code}</span>
                      <div className="font-mono text-[10.5px] text-ink-faint">
                        Its certificate: {shortFp(p.fingerprint)} — it should match the one shown under "This computer" there.
                      </div>
                      {/* Still 'waiting', but the last poll failed: say so, rather than showing a
                          code that will never be answered. */}
                      {p.error && <div className="mt-1 text-[11.5px] text-warn">Last check failed: {p.error}</div>}
                    </div>
                  ) : (
                    <div className="mt-1 text-[12px] text-bad">{p.error ?? p.status}</div>
                  )}
                </div>
              ))}
            </Panel>
          )}

          <Panel className="p-4">
            <div className="mb-2 text-[14px] font-medium text-ink">Connected computers</div>
            {doc.peers.length === 0 && <div className="text-[12px] text-ink-faint">None yet.</div>}
            {doc.peers.map((p) => (
              <div key={p.node_id} className="border-t border-seam/60 py-2.5 first:border-t-0">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-remote" />
                  <span className="font-mono text-[12.5px] text-remote">{p.name}</span>
                  <Pill tone={p.status === 'ok' ? 'good' : p.status === 'checking' ? 'neutral' : 'bad'}>{p.status}</Pill>
                  <button
                    onClick={() => window.confirm(`Disconnect ${p.name}? Its models disappear here and the access is revoked there.`) && act(() => federation.disconnect(p.node_id))}
                    className="ml-auto font-mono text-[11px] text-ink-faint hover:text-bad"
                  >
                    disconnect
                  </button>
                </div>
                <div className="font-mono text-[10.5px] text-ink-faint">
                  {p.address}:{p.port} · pinned cert {shortFp(p.fingerprint)} · connected {ago(p.connected_at)}
                </div>
                <VersionLine version={p.app_version} compatible={p.status !== 'incompatible version'} note={p.version_note} />
                {p.error && <div className="mt-0.5 text-[11.5px] text-bad">{p.error}</div>}
                {p.models.length > 0 ? (
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {p.models.map((m) => (
                      <span key={m.name} className="rounded-md border border-remote/40 bg-remote/10 px-2 py-0.5 font-mono text-[11px] text-remote">
                        {m.name}@{p.slug}
                        {m.context ? ` · ${Math.round(m.context / 1024)}K` : ''}
                        {m.decode_tps ? ` · ${m.decode_tps.toFixed(0)} tok/s` : ''}
                      </span>
                    ))}
                  </div>
                ) : (
                  p.status === 'ok' && <div className="mt-1 text-[11.5px] text-ink-faint">It shares no loaded models right now.</div>
                )}
              </div>
            ))}
          </Panel>
        </div>

        {/* ---- Sharing this computer ---- */}
        <div className="space-y-4">
          <Panel className="p-4">
            <div className="mb-2 flex items-center gap-2">
              <span className="text-[14px] font-medium text-ink">This computer</span>
              <label className="ml-auto flex items-center gap-2 text-[12.5px] text-ink-dim">
                <input type="checkbox" checked={s.enabled} disabled={busy}
                  onChange={(e) => act(() => federation.setSharing({ enabled: e.target.checked }))} />
                Share models on this network
              </label>
            </div>
            <div className="space-y-1 font-mono text-[11.5px] text-ink-dim">
              <div className="flex items-center gap-2">
                name
                {name === null ? (
                  <>
                    <span className="text-ink">{doc.node.name}</span>
                    <button onClick={() => setName(doc.node.name)} className="text-accent hover:opacity-80">rename</button>
                  </>
                ) : (
                  <>
                    <input value={name} onChange={(e) => setName(e.target.value)}
                      className="rounded border border-seam bg-panel-hi px-1.5 py-0.5 text-ink outline-none focus:border-accent" />
                    <button onClick={() => act(async () => { await federation.setSharing({ name: name.trim() }); setName(null) })}
                      className="text-accent">save</button>
                  </>
                )}
              </div>
              <div>addresses {doc.node.addresses.join(', ') || '(none found)'} · port {doc.node.port} (TLS)</div>
              <div>certificate {shortFp(doc.node.fingerprint)}</div>
              <div>
                version {doc.node.app_version} · protocol {doc.node.protocol}
                {doc.node.min_protocol !== doc.node.protocol ? ` (accepts ${doc.node.min_protocol}+)` : ''}
              </div>
              {s.error && <div className="text-bad">{s.error}</div>}
            </div>
            <div className="mt-3 text-[10.5px] uppercase tracking-wide text-ink-faint">Models to share (only while loaded)</div>
            {s.available_models.length === 0 && s.shared_models.length === 0 && (
              <div className="mt-1 text-[12px] text-ink-faint">No models are loaded on this computer.</div>
            )}
            <div className="mt-1 flex flex-wrap gap-2">
              {[...new Set([...s.available_models.map((m) => m.name), ...s.shared_models])].map((m) => {
                const on = shared.has(m)
                const loaded = s.available_models.some((x) => x.name === m)
                return (
                  <button
                    key={m}
                    disabled={busy}
                    onClick={() =>
                      act(() => federation.setSharing({ shared_models: on ? s.shared_models.filter((x) => x !== m) : [...s.shared_models, m] }))
                    }
                    className={`rounded-lg border px-2.5 py-1 font-mono text-[11.5px] ${
                      on ? 'border-accent/50 bg-accent/[0.08] text-ink' : 'border-seam text-ink-faint hover:text-ink-dim'
                    }`}
                  >
                    {on ? '✓ ' : ''}
                    {m}
                    {!loaded && <span className="text-ink-faint"> (not loaded)</span>}
                  </button>
                )
              })}
            </div>
            <div className="mt-3 text-[11.5px] leading-relaxed text-ink-faint">
              Another computer can only use what you tick, only for chat completions, only after you approve its request
              here, and only from a private network address. Traffic is TLS; tokens expire hourly and rotate; revoke any
              computer below at any time.
            </div>
          </Panel>

          <FirewallPanel fw={doc.firewall} sharing={s.enabled} />

          <Panel className="p-4">
            <div className="mb-2 text-[14px] font-medium text-ink">Computers using this one</div>
            {doc.clients.length === 0 && <div className="text-[12px] text-ink-faint">None approved.</div>}
            {doc.clients.map((c) => (
              <div key={c.client_id} className="flex items-center gap-3 border-t border-seam/60 py-2 first:border-t-0">
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-[12.5px] text-ink">{c.name}</div>
                  <div className="font-mono text-[10.5px] text-ink-faint">
                    {c.address} · {c.app_version ? `v${c.app_version} · ` : ''}approved {ago(c.created)} · last seen {ago(c.last_seen)} · {c.requests} requests · scope {c.scope}
                  </div>
                </div>
                <Button tone="danger" disabled={busy}
                  onClick={() => window.confirm(`Revoke ${c.name}? Its access stops immediately.`) && act(() => federation.revokeClient(c.client_id))}>
                  Revoke
                </Button>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  )
}

'use client'

import { useEffect, useState } from 'react'
import {
  projectResources as res,
  type Project,
  type SqlVerification,
} from '@/lib/projects'
import { Button, Panel, Pill } from '@/components/ui'
import QueryBox from '@/components/QueryBox'
import SqlExport from '@/components/SqlExport'

/**
 * A project's SQL Server connection -- read-only, and only the tables ticked here.
 *
 * "Set up" is the one privileged step: it runs as YOUR Windows login and creates a login for
 * this project (freetoken_<slug>) that SQL Server lets SELECT from exactly these tables and
 * nothing else. Agents only ever connect as that login, so even a hostile query is refused
 * by the server. The verification badge is SQL Server's own answer to "what can it do?".
 */
export default function ProjectSql({
  project,
  onChanged,
  onExported = () => {},
}: {
  project: Project
  onChanged: () => void
  /** Called when an export finishes, so the data panel can show the new table. */
  onExported?: () => void
}) {
  const cfg = project.sql
  const [editing, setEditing] = useState(!cfg)
  const [server, setServer] = useState(cfg?.server ?? 'localhost')
  const [database, setDatabase] = useState(cfg?.database ?? '')
  const [databases, setDatabases] = useState<string[] | null>(null)
  const [tables, setTables] = useState<{ table: string; rows: number | null }[] | null>(null)
  const [chosen, setChosen] = useState<Set<string>>(new Set(cfg?.tables ?? []))
  const [filter, setFilter] = useState('')
  const [verify, setVerify] = useState<SqlVerification | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // Reset when switching projects.
  useEffect(() => {
    setEditing(!project.sql)
    setServer(project.sql?.server ?? 'localhost')
    setDatabase(project.sql?.database ?? '')
    setChosen(new Set(project.sql?.tables ?? []))
    setDatabases(null)
    setTables(null)
    setVerify(null)
    setErr(null)
  }, [project.id, project.sql])

  // Ask SQL Server, as the project's login, what it can actually do.
  useEffect(() => {
    if (!project.sql || editing) return
    let alive = true
    res
      .sqlVerify(project.id)
      .then((v) => alive && setVerify(v))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [project.id, project.sql, editing])

  async function step<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(label)
    setErr(null)
    try {
      return await fn()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      return undefined
    } finally {
      setBusy(null)
    }
  }

  async function listDatabases() {
    const r = await step('databases', () => res.sqlDatabases(server))
    if (r) setDatabases(r.databases)
  }

  async function listTables(db: string) {
    setDatabase(db)
    setTables(null)
    const r = await step('tables', () => res.sqlTables(server, db))
    if (r) setTables(r.tables)
  }

  async function setup() {
    const r = await step('setup', () => res.sqlSetup(project.id, server, database, [...chosen]))
    if (r) {
      setVerify(r.verification)
      setEditing(false)
      onChanged()
    }
  }

  async function remove() {
    if (!window.confirm(`Remove SQL access for ${project.name}? This drops its SQL Server login.`)) return
    const r = await step('remove', () => res.sqlRemove(project.id))
    if (r) onChanged()
  }

  const shown = (tables ?? []).filter((t) => t.table.toLowerCase().includes(filter.toLowerCase()))

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[15px] font-medium text-ink">SQL Server</span>
        {cfg && !editing && verify && (
          <Pill tone={verify.read_only && verify.can_read_all ? 'good' : 'bad'}>
            {verify.read_only
              ? verify.can_read_all
                ? 'verified read-only'
                : 'read-only · some tables unreadable'
              : 'NOT read-only'}
          </Pill>
        )}
        {cfg && !editing && (
          <span className="ml-auto flex gap-2">
            <Button tone="default" onClick={() => setEditing(true)}>
              Change tables
            </Button>
            <Button tone="danger" onClick={remove} disabled={busy === 'remove'}>
              Remove
            </Button>
          </span>
        )}
      </div>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
        The swarm queries through its own login, <code className="font-mono">freetoken_{project.slug}</code>,
        which SQL Server allows to read <strong className="text-ink-dim">only the tables you tick</strong> —
        nothing can be written, and no other table or database is visible. Setup runs once as
        your Windows login.
      </p>

      {cfg && !editing && (
        <div className="mt-3 space-y-3">
          <div className="font-mono text-[11.5px] text-ink-dim">
            {cfg.server} / {cfg.database} · {cfg.tables.length} table
            {cfg.tables.length === 1 ? '' : 's'}: {cfg.tables.join(', ')}
          </div>
          <QueryBox
            run={(sql) => res.sqlQuery(project.id, sql)}
            placeholder={`SELECT TOP 10 * FROM ${cfg.tables[0]}`}
            initial={`SELECT TOP 10 * FROM ${cfg.tables[0]}`}
          />
          <SqlExport project={project} onExported={onExported} />
        </div>
      )}

      {editing && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={server}
              onChange={(e) => setServer(e.target.value)}
              placeholder="server (localhost, .\\SQLEXPRESS, host\\instance)"
              className="w-72 rounded-lg border border-seam bg-canvas px-2.5 py-1.5 font-mono text-[12px] text-ink outline-none"
            />
            <Button tone="default" onClick={listDatabases} disabled={!!busy || !server.trim()}>
              {busy === 'databases' ? 'connecting…' : 'List databases'}
            </Button>
            {databases && (
              <select
                value={database}
                onChange={(e) => e.target.value && void listTables(e.target.value)}
                className="rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink outline-none"
              >
                <option value="">choose a database…</option>
                {databases.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            )}
            {cfg && (
              <Button tone="ghost" onClick={() => setEditing(false)}>
                Cancel
              </Button>
            )}
          </div>

          {busy === 'tables' && <div className="font-mono text-[12px] text-ink-faint">listing tables…</div>}
          {tables && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="filter tables"
                  className="w-56 rounded-lg border border-seam bg-canvas px-2.5 py-1 font-mono text-[12px] text-ink outline-none"
                />
                <span className="font-mono text-[11px] text-ink-faint">
                  {chosen.size} of {tables.length} chosen
                </span>
              </div>
              <div className="max-h-64 space-y-1 overflow-y-auto rounded-lg border border-seam p-2">
                {shown.map((t) => {
                  const on = chosen.has(t.table)
                  return (
                    <label key={t.table} className="flex cursor-pointer items-center gap-2 px-1 py-0.5 text-[12px]">
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() =>
                          setChosen((s) => {
                            const n = new Set(s)
                            if (on) n.delete(t.table)
                            else n.add(t.table)
                            return n
                          })
                        }
                      />
                      <span className="font-mono text-ink">{t.table}</span>
                      <span className="ml-auto font-mono text-[10.5px] text-ink-faint">
                        {t.rows == null ? 'view' : `${t.rows.toLocaleString()} rows`}
                      </span>
                    </label>
                  )
                })}
              </div>
              <Button tone="primary" onClick={setup} disabled={!!busy || chosen.size === 0}>
                {busy === 'setup' ? 'setting up…' : `Set up read-only access to ${chosen.size} table${chosen.size === 1 ? '' : 's'}`}
              </Button>
            </div>
          )}
        </div>
      )}

      {verify && !verify.read_only && (
        <div className="mt-3 rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 text-[12px] text-ink-dim">
          SQL Server reports this login can write
          {verify.sysadmin ? ' (it is sysadmin)' : verify.can_create_tables ? ' (it can create tables)' : ''}.
          Do not let agents use it — run Change tables to rebuild the login.
        </div>
      )}
      {err && (
        <div className="mt-3 rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 text-[12px] text-ink-dim">{err}</div>
      )}
    </Panel>
  )
}

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { projects, type FileEntry, type Project } from '@/lib/projects'
import { bytesLabel, clockTime } from '@/lib/format'
import { Button, EmptyState, PageHeader, Panel, Pill } from '@/components/ui'
import ProjectSql from '@/components/ProjectSql'
import { ProjectDataQuery, ProjectModels } from '@/components/ProjectResources'

type ConnectorRow = { name: string; enabled: boolean; transport: string }

const inputCls =
  'w-full rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[13px] text-ink outline-none focus:border-accent'

export default function ProjectsPage() {
  const [items, setItems] = useState<Project[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [defaultRoot, setDefaultRoot] = useState('')
  const [connectors, setConnectors] = useState<ConnectorRow[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // new-project form
  const [newName, setNewName] = useState('')
  const [newDir, setNewDir] = useState('')

  // file browser
  const [subdir, setSubdir] = useState('')
  const [files, setFiles] = useState<FileEntry[]>([])
  const [fileRoot, setFileRoot] = useState('')
  const [totalBytes, setTotalBytes] = useState(0)
  const [preview, setPreview] = useState<{ path: string; text: string } | null>(null)
  const uploadRef = useRef<HTMLInputElement>(null)

  const current = items.find((p) => p.id === selected) ?? null
  // Bumped when a SQL export finishes, so the data panel re-reads its catalog.
  const [dataVersion, setDataVersion] = useState(0)
  const bumpData = useCallback(() => setDataVersion((v) => v + 1), [])

  const load = useCallback(async () => {
    try {
      const r = await projects.list()
      setItems(r.projects)
      setActive(r.active)
      setDefaultRoot(r.default_root)
      setSelected((prev) => prev ?? r.active ?? r.projects[0]?.id ?? null)
      setErr(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    load()
    // The connector catalog is global (Settings / mcp_servers.json); a project picks from it.
    fetch('/api/mcp/servers')
      .then((r) => r.json())
      .then((d) =>
        setConnectors(
          (d.servers ?? []).map((s: ConnectorRow) => ({
            name: s.name,
            enabled: s.enabled,
            transport: s.transport,
          })),
        ),
      )
      .catch(() => setConnectors([]))
  }, [load])

  const loadFiles = useCallback(
    async (id: string, dir: string) => {
      try {
        const r = await projects.files(id, dir)
        setFiles(r.entries)
        setFileRoot(r.root)
        setTotalBytes(r.total_bytes)
        setErr(null)
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e))
      }
    },
    [],
  )

  useEffect(() => {
    if (selected) loadFiles(selected, subdir)
  }, [selected, subdir, loadFiles])

  async function run(fn: () => Promise<unknown>) {
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

  async function toggleConnector(name: string) {
    if (!current) return
    const next = current.connectors.includes(name)
      ? current.connectors.filter((c) => c !== name)
      : [...current.connectors, name]
    await run(() => projects.update(current.id, { connectors: next }))
  }

  const crumbs = subdir ? subdir.split('/').filter(Boolean) : []

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Projects"
        subtitle={
          items.length
            ? `${items.length} project${items.length === 1 ? '' : 's'} · each with its own message board, connectors and data`
            : 'No projects yet'
        }
        right={
          active && (
            <Pill tone="good" pulse>
              active: {items.find((p) => p.id === active)?.name ?? '—'}
            </Pill>
          )
        }
      />

      {err && (
        <Panel className="mb-6 border-bad/35 bg-bad/5 p-4 font-mono text-[12px] text-bad">
          {err}
        </Panel>
      )}

      {/* ---- create ---- */}
      <Panel className="mb-6 p-5">
        <div className="text-[15px] font-medium text-ink">New project</div>
        <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1.4fr_auto]">
          <input
            className={inputCls}
            placeholder="Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <input
            className={inputCls}
            placeholder={`Data directory (blank = ${defaultRoot}\\<slug>\\data)`}
            value={newDir}
            onChange={(e) => setNewDir(e.target.value)}
          />
          <Button
            tone="primary"
            disabled={!newName.trim() || busy}
            onClick={() =>
              run(async () => {
                await projects.create(newName.trim(), newDir.trim() || undefined)
                setNewName('')
                setNewDir('')
              })
            }
          >
            Create
          </Button>
        </div>
      </Panel>

      {items.length === 0 ? (
        <EmptyState
          title="No projects"
          hint="A project scopes its own message board, the connectors its agents may use, and a data directory."
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
          {/* ---- list ---- */}
          <div className="space-y-2">
            {items.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  setSelected(p.id)
                  setSubdir('')
                  setPreview(null)
                }}
                className={`w-full rounded-xl border p-3 text-left transition-colors ${
                  selected === p.id
                    ? 'border-accent/50 bg-accent/[0.06]'
                    : 'border-seam bg-panel hover:bg-panel-hi/60'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-[14px] text-ink">{p.name}</span>
                  {p.id === active && <Pill tone="good">active</Pill>}
                </div>
                <div className="mt-1 font-mono text-[10.5px] text-ink-faint">
                  {p.connectors_active.length} connector
                  {p.connectors_active.length === 1 ? '' : 's'}
                  {!p.data_dir_exists && ' · data dir missing'}
                </div>
              </button>
            ))}
          </div>

          {/* ---- detail ---- */}
          {current && (
            <div className="space-y-6">
              <Panel className="p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-[16px] font-medium text-ink">{current.name}</span>
                  <span className="font-mono text-[11px] text-ink-faint">{current.slug}</span>
                  <span className="ml-auto flex gap-2">
                    {current.id !== active && (
                      <Button
                        tone="primary"
                        disabled={busy}
                        onClick={() => run(() => projects.activate(current.id))}
                      >
                        Make active
                      </Button>
                    )}
                    <Button
                      tone="danger"
                      disabled={busy}
                      onClick={() => {
                        if (!window.confirm(`Delete project "${current.name}"?`)) return
                        const wipe = window.confirm(
                          'Also delete its data directory?\n\nOK = delete files, Cancel = keep them.\n' +
                            '(Files are only removed if they live under the managed projects root.)',
                        )
                        run(async () => {
                          await projects.remove(current.id, wipe)
                          setSelected(null)
                        })
                      }}
                    >
                      Delete
                    </Button>
                  </span>
                </div>

                <div className="mt-4 space-y-3">
                  <label className="block">
                    <span className="text-[12px] text-ink-dim">Data directory</span>
                    <div className="mt-1.5 flex gap-2">
                      <input
                        className={inputCls}
                        defaultValue={current.data_dir}
                        key={current.id}
                        onBlur={(e) => {
                          const v = e.target.value.trim()
                          if (v && v !== current.data_dir)
                            run(() => projects.update(current.id, { data_dir: v }))
                        }}
                      />
                    </div>
                    <div className="mt-1 text-[11px] text-ink-faint">
                      Changing this points the project at a new directory — existing files are
                      not moved.
                    </div>
                  </label>
                </div>
              </Panel>

              {/* ---- connectors ---- */}
              <Panel className="p-5">
                <div className="text-[15px] font-medium text-ink">Connectors</div>
                <p className="mt-1 text-[12px] text-ink-faint">
                  Declared globally in <code className="font-mono">mcp_servers.json</code>; enabled
                  per project here. A connector must be on in both places for this project&apos;s
                  agents to reach it.
                </p>
                <div className="mt-4 space-y-2">
                  {connectors.length === 0 && (
                    <div className="text-[12px] text-ink-faint">
                      No connectors declared. Add them on the Connectors page.
                    </div>
                  )}
                  {connectors.map((c) => {
                    const on = current.connectors.includes(c.name)
                    return (
                      <button
                        key={c.name}
                        onClick={() => toggleConnector(c.name)}
                        disabled={busy}
                        className={`flex w-full items-center gap-3 rounded-xl border p-3 text-left transition-colors ${
                          on
                            ? 'border-accent/45 bg-accent/[0.07]'
                            : 'border-seam bg-panel-hi/30 opacity-70 hover:opacity-100'
                        }`}
                      >
                        <span
                          className={`grid h-5 w-5 shrink-0 place-items-center rounded-md border ${
                            on ? 'border-accent bg-accent text-white' : 'border-seam text-transparent'
                          }`}
                        >
                          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                            <path d="M5 13l4 4L19 7" />
                          </svg>
                        </span>
                        <span className="flex-1 font-mono text-[12px] text-ink">{c.name}</span>
                        <Pill>{c.transport}</Pill>
                        {!c.enabled && <Pill tone="warn">off globally</Pill>}
                      </button>
                    )
                  })}
                </div>
                {current.connectors_unavailable.length > 0 && (
                  <div className="mt-3 rounded-xl border border-warn/35 bg-warn/5 p-3 text-[12px] text-warn">
                    Selected but switched off globally:{' '}
                    {current.connectors_unavailable.join(', ')} — these are inactive until
                    enabled in mcp_servers.json.
                  </div>
                )}
              </Panel>

              {/* ---- what the swarm may use ---- */}
              <ProjectModels project={current} onChanged={load} />
              <ProjectSql project={current} onChanged={load} onExported={bumpData} />
              <ProjectDataQuery project={current} refreshKey={dataVersion} />

              {/* ---- data ---- */}
              <Panel className="p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-[15px] font-medium text-ink">Data</span>
                  <span className="font-mono text-[11px] text-ink-faint">
                    {bytesLabel(totalBytes)} total
                  </span>
                  <span className="ml-auto flex gap-2">
                    <input
                      ref={uploadRef}
                      type="file"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0]
                        if (!f) return
                        run(async () => {
                          await projects.upload(current.id, f, subdir)
                          await loadFiles(current.id, subdir)
                        })
                        e.target.value = ''
                      }}
                    />
                    <Button tone="ghost" onClick={() => uploadRef.current?.click()} disabled={busy}>
                      Upload
                    </Button>
                    <Button
                      tone="ghost"
                      disabled={busy}
                      onClick={() => {
                        const name = window.prompt('New folder name')
                        if (!name) return
                        run(async () => {
                          await projects.mkdir(current.id, subdir ? `${subdir}/${name}` : name)
                          await loadFiles(current.id, subdir)
                        })
                      }}
                    >
                      New folder
                    </Button>
                  </span>
                </div>

                <div className="mt-2 break-all font-mono text-[10.5px] text-ink-faint">
                  {fileRoot}
                </div>

                {/* breadcrumbs */}
                <div className="mt-3 flex flex-wrap items-center gap-1 font-mono text-[11px]">
                  <button
                    className={subdir ? 'text-accent hover:underline' : 'text-ink-faint'}
                    onClick={() => setSubdir('')}
                  >
                    /
                  </button>
                  {crumbs.map((c, i) => (
                    <span key={i} className="flex items-center gap-1">
                      <span className="text-ink-faint">/</span>
                      <button
                        className="text-accent hover:underline"
                        onClick={() => setSubdir(crumbs.slice(0, i + 1).join('/'))}
                      >
                        {c}
                      </button>
                    </span>
                  ))}
                </div>

                <div className="mt-3 space-y-1.5">
                  {files.length === 0 && (
                    <div className="py-4 text-center text-[12px] text-ink-faint">
                      Empty. Upload documents or data the agents in this project should see.
                    </div>
                  )}
                  {files.map((f) => (
                    <div
                      key={f.path}
                      className="flex items-center gap-3 rounded-lg border border-seam/60 bg-panel-hi/30 px-3 py-2"
                    >
                      <span className="w-4 shrink-0 text-center text-ink-faint">
                        {f.is_dir ? '▸' : '·'}
                      </span>
                      {f.is_dir ? (
                        <button
                          className="min-w-0 flex-1 truncate text-left font-mono text-[12px] text-accent hover:underline"
                          onClick={() => setSubdir(f.path)}
                        >
                          {f.name}
                        </button>
                      ) : (
                        <button
                          className="min-w-0 flex-1 truncate text-left font-mono text-[12px] text-ink hover:text-accent"
                          onClick={() =>
                            projects
                              .readFile(current.id, f.path)
                              .then((r) => setPreview({ path: r.path, text: r.text }))
                              .catch((e) => setErr(String(e)))
                          }
                        >
                          {f.name}
                        </button>
                      )}
                      <span className="shrink-0 font-mono text-[10.5px] text-ink-faint">
                        {f.is_dir ? '' : bytesLabel(f.size_bytes)} · {clockTime(f.modified_at)}
                      </span>
                      <button
                        className="shrink-0 font-mono text-[10.5px] text-ink-faint hover:text-bad"
                        onClick={() => {
                          if (!window.confirm(`Delete ${f.name}?`)) return
                          run(async () => {
                            await projects.deleteFile(current.id, f.path)
                            await loadFiles(current.id, subdir)
                          })
                        }}
                      >
                        delete
                      </button>
                    </div>
                  ))}
                </div>

                {preview && (
                  <div className="mt-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] text-ink-dim">{preview.path}</span>
                      <button
                        className="ml-auto font-mono text-[11px] text-ink-faint hover:text-ink"
                        onClick={() => setPreview(null)}
                      >
                        close
                      </button>
                    </div>
                    <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg border border-seam bg-canvas p-3 font-mono text-[11.5px] leading-relaxed text-ink-dim">
                      {preview.text}
                    </pre>
                  </div>
                )}
              </Panel>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

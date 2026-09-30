'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  research,
  type Chunk,
  type DocDetail,
  type DocIdea,
  type DocStatus,
  type ResearchDoc,
  type ResearchSettings,
  type ResearchStatus,
  type SearchHit,
  type SearchKind,
} from '@/lib/research'
import { projects, type Project } from '@/lib/projects'
import { objectives, type Objective } from '@/lib/objectives'
import { api } from '@/lib/api'
import { bytesLabel } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Button, PageHeader, Panel, Pill } from '@/components/ui'
import CopyButton from '@/components/CopyButton'
import Markdown from '@/components/Markdown'

const field = 'rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent'
const label = 'font-mono text-[10.5px] uppercase tracking-wide text-ink-faint'
const link = 'font-mono text-[11px] text-accent hover:opacity-80 disabled:opacity-40'

const ACCEPT = '.pdf,.html,.htm,.md,.markdown,.txt'
const ACCEPT_RE = /\.(pdf|html?|md|markdown|txt)$/i
const SEARCH_KINDS: SearchKind[] = ['idea', 'text', 'table', 'figure', 'code']
type Tab = 'ideas' | 'code' | 'tables' | 'figures' | 'text' | 'outline'
/** The tab a search hit of each kind opens. */
const TAB_OF: Record<SearchKind, Tab> = { idea: 'ideas', code: 'code', table: 'tables', figure: 'figures', text: 'text' }

const err2s = (e: unknown) => (e instanceof Error ? e.message : String(e))
const when = (ts: number) => new Date(ts * 1000).toLocaleString()
const done = (s: DocStatus) => s === 'ready' || s === 'error'

function statusTone(s: DocStatus): 'good' | 'bad' | 'neutral' | 'accent' {
  return s === 'ready' ? 'good' : s === 'error' ? 'bad' : s === 'queued' ? 'neutral' : 'accent'
}

type Upload = { name: string; state: 'uploading' | 'created' | 'exists' | 'error'; detail?: string }

/**
 * The research library (app/research.py): drop strategy reports in, and they are parsed into
 * prose, tables, figures and code, embedded for search, and read by a model for testable
 * trading ideas. Relevant ideas feed each running objective's idea stream (trigger
 * 'research'); agents search the library and import the documents' Python code in the sandbox.
 */
export default function ResearchPage() {
  const [projectList, setProjectList] = useState<Project[]>([])
  // undefined until the project list has loaded, so the first poll is already scoped.
  const [projectId, setProjectId] = useState<string | null | undefined>(undefined)
  const [shared, setShared] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [objs, setObjs] = useState<Objective[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('ideas')
  const [focusChunk, setFocusChunk] = useState<number | null>(null)
  const [fast, setFast] = useState(true)

  useEffect(() => {
    projects
      .list()
      .then((r) => {
        setProjectList(r.projects)
        setProjectId(r.active ?? r.projects[0]?.id ?? null)
      })
      .catch(() => setProjectId(null))
    api
      .engines()
      .then((r) => setModels([...new Set(r.loaded.filter((m) => m.ready && m.model).map((m) => m.model as string))].sort()))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!projectId) return setObjs([])
    objectives.list(projectId).then((r) => setObjs(r.objectives)).catch(() => setObjs([]))
  }, [projectId])

  // Status and documents in one poll: fast while anything is being processed, slow at rest.
  const poll = usePoll<{ status: ResearchStatus; docs: ResearchDoc[] } | null>(async () => {
    if (projectId === undefined) return null
    const [status, d] = await Promise.all([research.status(), research.docs(projectId)])
    setFast(d.docs.some((x) => !done(x.status)) || status.queued > 0)
    return { status, docs: d.docs }
  }, fast ? 3000 : 20_000)
  const { refresh } = poll
  useEffect(() => refresh(), [projectId, refresh])
  const docs = useMemo(() => poll.data?.docs ?? [], [poll.data])
  const status = poll.data?.status ?? null

  const current = docs.find((d) => d.id === selected) ?? null
  const running = objs.filter((o) => o.status === 'running')

  function open(docId: string, t?: Tab, chunk?: number) {
    setSelected(docId)
    if (t) setTab(t)
    setFocusChunk(chunk ?? null)
  }

  return (
    <div className="mx-auto max-w-[1400px] px-8 py-8">
      <PageHeader
        title="Research"
        subtitle="Strategy reports and papers, parsed and searchable — their testable ideas feed the objectives' idea streams"
        right={
          projectList.length > 0 && (
            <label className="flex items-center gap-2 text-[12px] text-ink-dim">
              Project
              <select className={field} value={projectId ?? ''} onChange={(e) => setProjectId(e.target.value || null)}>
                {projectList.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
          )
        }
      />

      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <DropZone projectId={projectId ?? ''} shared={shared || !projectId} setShared={setShared} canScope={!!projectId}
          onDone={refresh} />
        <StatusStrip status={status} docs={docs} models={models} error={poll.error} />
      </div>

      <SearchPanel projectId={projectId ?? null} onOpen={open} />

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,8fr)]">
        <DocList docs={docs} projectId={projectId ?? null} projectList={projectList} models={models} selected={selected}
          onSelect={(id) => open(id)} onChanged={refresh} loading={poll.loading} />
        {current ? (
          <DocView key={current.id} doc={current} tab={tab} setTab={setTab} focusChunk={focusChunk}
            setFocusChunk={setFocusChunk} running={running} objs={objs} />
        ) : (
          <Panel className="grid place-items-center px-6 py-16 text-center text-[13px] text-ink-faint">
            Select a document to see its ideas, code, tables and figures.
          </Panel>
        )}
      </div>
    </div>
  )
}

/** Drop (or browse for) documents; each file is uploaded on its own so one bad file does not sink the rest. */
function DropZone({ projectId, shared, setShared, canScope, onDone }: {
  projectId: string
  shared: boolean
  setShared: (v: boolean) => void
  canScope: boolean
  onDone: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [uploads, setUploads] = useState<Upload[]>([])

  async function send(list: FileList | null) {
    const files = Array.from(list ?? [])
    if (!files.length) return
    setUploads(files.map((f) => ({ name: f.name, state: 'uploading' })))
    for (const [n, f] of files.entries()) {
      const set = (u: Partial<Upload>) => setUploads((us) => us.map((x, i) => (i === n ? { ...x, ...u } : x)))
      if (!ACCEPT_RE.test(f.name)) {
        set({ state: 'error', detail: 'only PDF, HTML, Markdown and text documents' })
        continue
      }
      try {
        const r = await research.upload([f], shared ? '' : projectId)
        const d = r.docs[0]
        set(d?.created ? { state: 'created', detail: d.title } : { state: 'exists', detail: d?.title })
      } catch (e) {
        set({ state: 'error', detail: err2s(e) })
      }
    }
    if (input.current) input.current.value = ''
    onDone()
  }

  return (
    <Panel className="p-4">
      <div
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          void send(e.dataTransfer.files)
        }}
        className={`grid cursor-pointer place-items-center rounded-xl border border-dashed px-4 py-8 text-center transition-colors ${
          over ? 'border-accent bg-accent/[0.06]' : 'border-seam hover:bg-panel-hi/60'}`}
      >
        <div className="text-[13px] text-ink">Drop documents here, or click to browse</div>
        <div className="mt-1 text-[11.5px] text-ink-faint">PDF, HTML, Markdown or text · up to 100 MB each</div>
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => void send(e.target.files)} />
      </div>
      <label className="mt-3 flex items-center gap-1.5 text-[12px] text-ink-dim"
        title={canScope ? 'Uploads are visible to every project, not only the one picked above' : 'No project yet: uploads are shared'}>
        <input type="checkbox" checked={shared} disabled={!canScope} onChange={(e) => setShared(e.target.checked)} />
        Shared with all projects
      </label>
      {uploads.length > 0 && (
        <ul className="mt-2 space-y-1 font-mono text-[11.5px]">
          {uploads.map((u, i) => (
            <li key={i} className="flex gap-2">
              <span className={u.state === 'error' ? 'text-bad' : u.state === 'created' ? 'text-good' : 'text-ink-dim'}>
                {{ uploading: '…', created: '✓', exists: '=', error: '✗' }[u.state]}
              </span>
              <span className="truncate text-ink">{u.name}</span>
              <span className="text-ink-faint">
                {u.state === 'created' ? 'added, processing' : u.state === 'exists' ? 'already in the library' : u.state === 'error' ? u.detail : 'uploading'}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

/** The worker, the embedding model and the library's settings (PUT /api/research/settings). */
function StatusStrip({ status, docs, models, error }: {
  status: ResearchStatus | null
  docs: ResearchDoc[]
  models: string[]
  error: string | null
}) {
  const [draft, setDraft] = useState<ResearchSettings | null>(null)
  const [saved, setSaved] = useState<ResearchSettings | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  // Seed the form once; later polls must not overwrite what is being edited.
  useEffect(() => {
    if (status && !saved) {
      setSaved(status.settings)
      setDraft(status.settings)
    }
  }, [status, saved])

  const dirty = !!draft && !!saved && (Object.keys(draft) as (keyof ResearchSettings)[]).some((k) => draft[k] !== saved[k])

  async function save() {
    if (!draft || !saved) return
    setBusy(true)
    setErr(null)
    try {
      const patch = Object.fromEntries((Object.keys(draft) as (keyof ResearchSettings)[]).filter((k) => draft[k] !== saved[k]).map((k) => [k, draft[k]]))
      const next = await research.saveSettings(patch)
      setSaved(next)
      setDraft(next)
    } catch (e) {
      setErr(err2s(e))
    } finally {
      setBusy(false)
    }
  }

  if (!status || !draft) return <Panel className="p-4 text-[12px] text-ink-faint">{error ?? 'Loading…'}</Panel>
  const emb = status.embedding
  const titles = Object.fromEntries(docs.map((d) => [d.id, d.title]))
  const set = <K extends keyof ResearchSettings>(k: K, v: ResearchSettings[K]) => setDraft((d) => (d ? { ...d, [k]: v } : d))
  const options = (value: string) => (value && !models.includes(value) ? [value, ...models] : models)
  const num = (k: 'push_on_ingest' | 'push_gap_minutes' | 'min_relevance' | 'min_overlap', min: number, max: number, step: number, tip: string) => (
    <input type="number" min={min} max={max} step={step} value={draft[k]} title={tip}
      onChange={(e) => set(k, Number(e.target.value))} className={`${field} w-20`} />
  )

  return (
    <Panel className="space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-ink-dim">
        {!status.running && <Pill tone="bad">worker not running</Pill>}
        <Pill tone={emb.loaded ? 'good' : emb.error ? 'warn' : 'neutral'}>
          <span title={emb.error ?? undefined}>
            {emb.loaded ? `embeddings: ${emb.model}` : emb.error ? 'lexical search only' : `embeddings: ${emb.model} (loads on first use)`}
          </span>
        </Pill>
        {emb.device && emb.loaded && <span className="font-mono text-[11px] text-ink-faint">on {emb.device}</span>}
        <span className="ml-auto font-mono text-[11px] text-ink-faint">
          {status.queued} queued
          {Object.entries(status.active).map(([id, stage]) => ` · ${titles[id] ?? id}: ${stage}`)}
        </span>
      </div>
      {emb.error && <div className="text-[11.5px] text-warn" title={emb.error}>Embedding model unavailable — search is lexical (BM25) only. {emb.error.slice(0, 160)}</div>}
      {error && <div className="text-[11.5px] text-bad">✗ {error}</div>}

      <div className="grid gap-x-4 gap-y-2 text-[12px] text-ink-dim sm:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className={label}>Idea model</span>
          <select className={field} value={draft.idea_model} onChange={(e) => set('idea_model', e.target.value)}>
            <option value="">project&apos;s first idea rung (auto)</option>
            {options(draft.idea_model).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className={label}>Figure model (vision)</span>
          <select className={field} value={draft.figure_model} onChange={(e) => set('figure_model', e.target.value)}>
            <option value="">off (captions only)</option>
            {options(draft.figure_model).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px] text-ink-dim">
        <label className="flex items-center gap-1.5" title="Send relevant ideas to running objectives' idea streams automatically">
          <input type="checkbox" checked={draft.auto_push} onChange={(e) => set('auto_push', e.target.checked)} /> auto-send ideas
        </label>
        <label className="flex items-center gap-1.5">
          {num('push_on_ingest', 0, 10, 1, 'Ideas sent to each running objective when a document finishes')} on arrival
        </label>
        <label className="flex items-center gap-1.5">
          then one per {num('push_gap_minutes', 5, 10_080, 5, 'At most one more idea per objective this often')} min
        </label>
        <label className="flex items-center gap-1.5">
          min relevance {num('min_relevance', 0, 1, 0.05, 'Least cosine between an idea and the objective to send it automatically')}
        </label>
        <label className="flex items-center gap-1.5">
          min overlap {num('min_overlap', 0, 1, 0.01, 'The same floor on token overlap, used when no embedding model is loaded')}
        </label>
        <span className="ml-auto flex items-center gap-2">
          {err && <span className="text-bad">✗ {err}</span>}
          <Button tone="ghost" disabled={!dirty || busy} onClick={save}>{busy ? 'Saving…' : 'Save'}</Button>
        </span>
      </div>
    </Panel>
  )
}

/** Hybrid (vector + BM25) search over the project's documents and ideas. */
function SearchPanel({ projectId, onOpen }: { projectId: string | null; onOpen: (docId: string, tab: Tab, chunk?: number) => void }) {
  const [query, setQuery] = useState('')
  const [kinds, setKinds] = useState<SearchKind[]>([])
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [dense, setDense] = useState(true)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function run() {
    if (!query.trim()) return
    setBusy(true)
    setErr(null)
    try {
      const r = await research.search({ query: query.trim(), project_id: projectId, kinds: kinds.length ? kinds : undefined, k: 20 })
      setHits(r.hits)
      setDense(r.dense)
    } catch (e) {
      setErr(err2s(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel className="p-4">
      <form className="flex flex-wrap items-center gap-2" onSubmit={(e) => {
        e.preventDefault()
        void run()
      }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search the library — findings, tables, figures, code, ideas…"
          className="min-w-[260px] flex-1 rounded-lg border border-seam bg-panel-hi px-3 py-1.5 text-[12.5px] text-ink outline-none focus:border-accent" />
        {SEARCH_KINDS.map((k) => {
          const on = kinds.includes(k)
          return (
            <button key={k} type="button" onClick={() => setKinds((ks) => (on ? ks.filter((x) => x !== k) : [...ks, k]))}
              className={`rounded-lg border px-2.5 py-1 font-mono text-[11px] ${on ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'}`}>
              {k}
            </button>
          )
        })}
        <Button type="submit" tone="ghost" disabled={busy || !query.trim()}>{busy ? 'Searching…' : 'Search'}</Button>
        {hits && <button type="button" className={link} onClick={() => setHits(null)}>clear</button>}
      </form>
      {err && <div className="mt-2 text-[12px] text-bad">✗ {err}</div>}
      {hits && (
        <div className="mt-3">
          <div className="mb-2 font-mono text-[10.5px] text-ink-faint">
            {hits.length} result{hits.length === 1 ? '' : 's'}{dense ? '' : ' · lexical search only (no embedding model)'}
          </div>
          <ul className="space-y-2">
            {hits.map((h, i) => (
              <li key={`${h.chunk_id ?? 'i' + h.idea_id}-${i}`} onClick={() => onOpen(h.doc_id, TAB_OF[h.kind], h.chunk_id)}
                className="flex cursor-pointer gap-3 rounded-lg border border-seam p-2 hover:bg-panel-hi/60">
                {h.image_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={h.image_url} alt={h.label || 'figure'} className="h-16 w-24 shrink-0 rounded border border-seam bg-white object-contain" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
                    <Pill tone={h.kind === 'idea' ? 'accent' : 'neutral'}>{h.kind}</Pill>
                    <span className="text-[12px] text-ink">{h.doc}</span>
                    <span className="font-mono">
                      {[h.label, h.section, h.page != null ? `p. ${h.page}` : ''].filter(Boolean).join(' · ')}
                    </span>
                    <span className="ml-auto font-mono" title={h.cosine != null ? `cosine ${h.cosine.toFixed(3)}` : 'lexical match'}>
                      {h.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="line-clamp-4 whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{h.text}</div>
                  {h.import && <ImportLine text={h.import} />}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

/** The library's documents with their processing state and actions. */
function DocList({ docs, projectId, projectList, models, selected, onSelect, onChanged, loading }: {
  docs: ResearchDoc[]
  projectId: string | null
  projectList: Project[]
  models: string[]
  selected: string | null
  onSelect: (id: string) => void
  onChanged: () => void
  loading: boolean
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [rowErr, setRowErr] = useState<Record<string, string>>({})
  const [reextract, setReextract] = useState<string | null>(null)
  const [model, setModel] = useState('')
  const names = Object.fromEntries(projectList.map((p) => [p.id, p.name]))

  async function act(d: ResearchDoc, fn: () => Promise<unknown>, question?: string) {
    if (question && !window.confirm(question)) return
    setBusy(d.id)
    setRowErr((m) => ({ ...m, [d.id]: '' }))
    try {
      await fn()
      onChanged()
    } catch (e) {
      setRowErr((m) => ({ ...m, [d.id]: err2s(e) }))
    } finally {
      setBusy(null)
    }
  }

  if (!docs.length) {
    return (
      <Panel className="grid place-items-center self-start px-6 py-16 text-center text-[13px] text-ink-faint">
        {loading ? 'Loading…' : 'No documents yet — drop a report above.'}
      </Panel>
    )
  }

  return (
    <Panel className="self-start p-2">
      <ul className="space-y-1">
        {docs.map((d) => {
          const c = d.chunks ?? {}
          const working = !done(d.status)
          return (
            <li key={d.id} onClick={() => onSelect(d.id)}
              className={`cursor-pointer rounded-xl border p-3 ${selected === d.id ? 'border-accent/50 bg-accent/[0.05]' : 'border-transparent hover:bg-panel-hi/60'}`}>
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] text-ink" title={d.filename}>{d.title || d.filename}</div>
                  <div className="font-mono text-[10.5px] text-ink-faint">
                    {d.kind}{d.pages ? ` · ${d.pages} pp` : ''} · {bytesLabel(d.size_bytes)} ·{' '}
                    {d.project_id ? names[d.project_id] ?? d.project_id : 'shared by all projects'}
                  </div>
                </div>
                <Pill tone={statusTone(d.status)} pulse={working && d.status !== 'queued'}>{d.stage ?? d.status}</Pill>
              </div>
              <div className="mt-1.5 font-mono text-[11px] text-ink-dim">
                {c.text ?? 0} text · {c.table ?? 0} tables · {c.figure ?? 0} figures · {c.code ?? 0} code ·{' '}
                <span className="text-ink">{d.ideas} ideas</span> · {d.pushed} sent
              </div>
              {d.detail && <div className="mt-1 text-[11.5px] text-ink-faint">{d.detail}</div>}
              {d.error && <div className="mt-1 text-[11.5px] text-bad">✗ {d.error}</div>}
              {rowErr[d.id] && <div className="mt-1 text-[11.5px] text-bad">✗ {rowErr[d.id]}</div>}
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1" onClick={(e) => e.stopPropagation()}>
                <a href={research.sourceUrl(d.id)} target="_blank" rel="noopener noreferrer" className={link}>source ↗</a>
                <button className={link} disabled={busy !== null || working}
                  onClick={() => setReextract((r) => (r === d.id ? null : d.id))}>re-extract ideas…</button>
                <button className={link} disabled={busy !== null || working}
                  onClick={() => act(d, () => research.reprocess(d.id, 'embed'))}>re-embed</button>
                <button className={link} disabled={busy !== null || working}
                  onClick={() => act(d, () => research.reprocess(d.id, 'all'),
                    `Reparse "${d.title}" from its file?\n\nChunks, vectors and unsent ideas are rebuilt; ideas already sent to objectives are kept.`)}>
                  reparse all
                </button>
                {d.project_id ? (
                  <button className={link} disabled={busy !== null} onClick={() => act(d, () => research.scope(d.id, ''))}>share with all</button>
                ) : (
                  projectId && (
                    <button className={link} disabled={busy !== null} onClick={() => act(d, () => research.scope(d.id, projectId))}>
                      unshare (only {names[projectId] ?? projectId})
                    </button>
                  )
                )}
                <button className="font-mono text-[11px] text-bad hover:opacity-80 disabled:opacity-40" disabled={busy !== null}
                  onClick={() => act(d, () => research.remove(d.id),
                    `Delete "${d.title}" from the library?\n\nIts chunks, figures and ideas go; ideas already sent stay in the objectives' idea streams.`)}>
                  delete
                </button>
                {busy === d.id && <span className="font-mono text-[11px] text-ink-faint">…</span>}
              </div>
              {reextract === d.id && (
                <div className="mt-2 flex flex-wrap items-center gap-2" onClick={(e) => e.stopPropagation()}>
                  <select className={field} value={model} onChange={(e) => setModel(e.target.value)}>
                    <option value="">the settings&apos; idea model</option>
                    {models.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <Button tone="ghost" disabled={busy !== null} onClick={() => {
                    setReextract(null)
                    void act(d, () => research.reprocess(d.id, 'ideas', model))
                  }}>Re-extract</Button>
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}

/** One document: its ideas, code, tables, figures, prose and outline. */
function DocView({ doc, tab, setTab, focusChunk, setFocusChunk, running, objs }: {
  doc: ResearchDoc
  tab: Tab
  setTab: (t: Tab) => void
  focusChunk: number | null
  setFocusChunk: (id: number | null) => void
  running: Objective[]
  objs: Objective[]
}) {
  const [detail, setDetail] = useState<DocDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setDetail(await research.doc(doc.id))
      setErr(null)
    } catch (e) {
      setErr(err2s(e))
    }
  }, [doc.id])
  // Re-read whenever the list shows the document changed (a stage finished, ideas re-extracted).
  useEffect(() => {
    void load()
  }, [load, doc.updated_at, doc.status, doc.pushed])

  // A search hit or an idea's code link: bring that chunk into view once its tab renders.
  useEffect(() => {
    if (focusChunk == null || !detail) return
    const t = setTimeout(() => document.getElementById(`chunk-${focusChunk}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
    return () => clearTimeout(t)
  }, [focusChunk, tab, detail])

  const byKind = useMemo(() => {
    const out: Record<string, Chunk[]> = { text: [], table: [], figure: [], code: [] }
    for (const c of detail?.chunks ?? []) (out[c.kind] ??= []).push(c)
    return out
  }, [detail])

  if (!detail) return <Panel className="p-4 text-[12px] text-ink-faint">{err ?? 'Loading…'}</Panel>
  const tabs: [Tab, string, number][] = [
    ['ideas', 'Ideas', detail.ideas.length],
    ['code', 'Code', byKind.code.length],
    ['tables', 'Tables', byKind.table.length],
    ['figures', 'Figures', byKind.figure.length],
    ['text', 'Text', byKind.text.length],
    ['outline', 'Outline', detail.outline.length],
  ]
  const chunkById = Object.fromEntries(detail.chunks.map((c) => [c.id, c]))
  const goChunk = (id: number) => {
    setTab('code')
    setFocusChunk(id)
  }

  return (
    <Panel className="min-w-0 self-start p-4">
      <div className="mb-3 flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-[15px] font-medium text-ink">{detail.title || detail.filename}</div>
          <div className="font-mono text-[10.5px] text-ink-faint">
            {detail.filename} · added {when(detail.created_at)}
            {detail.package ? ` · from research.${detail.package} import …` : ''}
            {detail.idea_model ? ` · ideas by ${detail.idea_model}` : ''}
            {detail.embed_model ? ` · embedded with ${detail.embed_model}` : ''}
          </div>
        </div>
        <a href={research.sourceUrl(detail.id)} target="_blank" rel="noopener noreferrer" className={link}>source ↗</a>
      </div>
      {err && <div className="mb-2 text-[12px] text-bad">✗ {err}</div>}

      <div className="mb-3 flex flex-wrap gap-1.5">
        {tabs.map(([t, name, n]) => (
          <button key={t} onClick={() => setTab(t)}
            className={`rounded-lg border px-3 py-1 text-[12px] ${tab === t ? 'border-accent/50 bg-accent/[0.08] text-ink' : 'border-seam text-ink-dim hover:text-ink'}`}>
            {name} <span className="font-mono text-[10.5px] text-ink-faint">{n}</span>
          </button>
        ))}
      </div>

      {tab === 'ideas' && (
        detail.ideas.length ? (
          <ul className="space-y-3">
            {detail.ideas.map((i) => (
              <IdeaCard key={i.id} idea={i} chunkById={chunkById} running={running} objs={objs} onCode={goChunk} onChanged={load} />
            ))}
          </ul>
        ) : (
          <Empty>{done(doc.status) ? 'No ideas extracted — see the document detail, or re-extract with another model.' : 'Ideas are extracted after parsing and embedding.'}</Empty>
        )
      )}

      {tab === 'code' && (
        byKind.code.length ? (
          <ul className="space-y-3">
            {byKind.code.map((c) => (
              <li key={c.id} id={`chunk-${c.id}`} className={`rounded-lg border p-2 ${focusChunk === c.id ? 'border-accent/60' : 'border-seam'}`}>
                <div className="mb-1 flex items-center gap-2 font-mono text-[10.5px] text-ink-faint">
                  <span className="text-[11.5px] text-ink">{c.label || `listing ${c.seq}`}</span>
                  <span>{[c.meta.language, c.meta.lines != null ? `${c.meta.lines} lines` : '', c.page != null ? `p. ${c.page}` : '', `chunk ${c.id}`].filter(Boolean).join(' · ')}</span>
                  <span className="ml-auto"><CopyButton text={c.text} label="copy code" /></span>
                </div>
                {c.import && <ImportLine text={c.import} />}
                <pre className="mt-1 max-h-96 overflow-auto rounded-md border border-seam bg-panel-hi p-2 font-mono text-[11.5px] leading-relaxed text-ink">{c.text}</pre>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No code listings found.</Empty>
        )
      )}

      {tab === 'tables' && (
        byKind.table.length ? (
          <ul className="space-y-3">
            {byKind.table.map((c) => (
              <li key={c.id} id={`chunk-${c.id}`} className={`rounded-lg border p-2 ${focusChunk === c.id ? 'border-accent/60' : 'border-seam'}`}>
                <ChunkHead c={c} extra={c.meta.rows != null ? `${c.meta.rows} rows` : ''} />
                <div className="overflow-x-auto text-[12px]">
                  <Markdown source={c.text} sandboxRun={false} />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No tables found.</Empty>
        )
      )}

      {tab === 'figures' && (
        byKind.figure.length ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {byKind.figure.map((c) => (
              <figure key={c.id} id={`chunk-${c.id}`} className={`rounded-lg border p-2 ${focusChunk === c.id ? 'border-accent/60' : 'border-seam'}`}>
                {c.image_url ? (
                  <a href={c.image_url} target="_blank" rel="noopener noreferrer">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={c.image_url} alt={c.label || 'figure'} loading="lazy" className="max-h-72 w-full rounded border border-seam bg-white object-contain" />
                  </a>
                ) : (
                  <div className="grid h-24 place-items-center rounded border border-dashed border-seam text-[11px] text-ink-faint">no image extracted</div>
                )}
                <figcaption className="mt-1.5">
                  <ChunkHead c={c} extra={c.meta.described_by ? `described by ${c.meta.described_by}` : ''} />
                  <div className="max-h-40 overflow-auto whitespace-pre-wrap text-[11.5px] leading-snug text-ink-dim">{c.text}</div>
                </figcaption>
              </figure>
            ))}
          </div>
        ) : (
          <Empty>No figures found.</Empty>
        )
      )}

      {tab === 'text' && (
        byKind.text.length ? (
          <ul className="space-y-2">
            {byKind.text.map((c) => (
              <li key={c.id} id={`chunk-${c.id}`} className={`rounded-lg border p-2 ${focusChunk === c.id ? 'border-accent/60' : 'border-seam'}`}>
                <ChunkHead c={c} />
                <div className="whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{c.text}</div>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No text parsed yet.</Empty>
        )
      )}

      {tab === 'outline' && (
        detail.outline.length ? (
          <ul className="space-y-0.5 text-[12.5px]">
            {detail.outline.map((o, n) => (
              <li key={n} className="flex gap-2 text-ink-dim" style={{ paddingLeft: `${Math.max(0, o.level - 1) * 16}px` }}>
                <span className={o.level <= 1 ? 'text-ink' : ''}>{o.title}</span>
                {o.page != null && <span className="ml-auto font-mono text-[10.5px] text-ink-faint">p. {o.page}</span>}
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No outline — the document has no headings the parser recognised.</Empty>
        )
      )}
    </Panel>
  )
}

/** One extracted idea, with where it went and a hand-send to a running objective. */
function IdeaCard({ idea, chunkById, running, objs, onCode, onChanged }: {
  idea: DocIdea
  chunkById: Record<number, Chunk>
  running: Objective[]
  objs: Objective[]
  onCode: (chunkId: number) => void
  onChanged: () => Promise<void>
}) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const titles = Object.fromEntries(objs.map((o) => [o.id, o.title]))
  const sentTo = new Set(idea.pushes.map((p) => p.objective_id))

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setErr(null)
    try {
      await fn()
      await onChanged()
    } catch (e) {
      setErr(err2s(e))
    } finally {
      setBusy(false)
    }
  }

  const part = (name: string, text: string) =>
    text ? (
      <div>
        <span className={label}>{name}</span>
        <div className="whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{text}</div>
      </div>
    ) : null

  return (
    <li className={`space-y-1.5 rounded-lg border border-seam p-3 ${idea.dismissed ? 'opacity-50' : ''}`}>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-medium text-ink">{idea.title}</div>
          <div className="font-mono text-[10.5px] text-ink-faint">
            #{idea.id}{idea.horizon ? ` · horizon ${idea.horizon}` : ''} · {idea.model}
            {idea.dismissed ? ' · dismissed' : ''}
          </div>
        </div>
        <button className={link} disabled={busy} onClick={() => act(() => research.dismiss(idea.id, !idea.dismissed))}>
          {idea.dismissed ? 'restore' : 'dismiss'}
        </button>
      </div>
      {part('Summary', idea.summary)}
      {part('Rules', idea.rules)}
      {idea.fields.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className={`${label} mr-1`}>Fields</span>
          {idea.fields.map((f) => <span key={f} className="rounded border border-seam px-1.5 font-mono text-[10.5px] text-ink-dim">{f}</span>)}
        </div>
      )}
      {part('Evidence (the document’s own)', idea.evidence)}
      {part('Caveats', idea.caveats)}
      {part('Falsify with', idea.falsify)}
      {idea.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {idea.tags.map((t) => <span key={t} className="rounded-full bg-panel-hi px-2 font-mono text-[10.5px] text-ink-faint">#{t}</span>)}
        </div>
      )}
      {idea.code_refs.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className={label}>Code</span>
          {idea.code_refs.map((id) => (
            <button key={id} className={link} onClick={() => onCode(id)}>
              {chunkById[id]?.label || `chunk ${id}`}{chunkById[id] ? '' : ' (gone)'}
            </button>
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 border-t border-seam/60 pt-2 text-[11.5px] text-ink-faint">
        <span title={idea.pushes.map((p) => `${titles[p.objective_id] ?? p.objective_id} — ${when(p.ts)} by ${p.by}`).join('\n') || undefined}>
          {idea.pushes.length ? `sent to ${idea.pushes.length} objective${idea.pushes.length === 1 ? '' : 's'}` : 'not sent yet'}
        </span>
        <select className={`${field} ml-auto`} value="" disabled={busy || !!idea.dismissed || !running.length}
          title={idea.dismissed ? 'Restore the idea to send it' : running.length ? 'Write it into that objective\'s idea stream now' : 'No running objective in this project'}
          onChange={(e) => {
            const oid = e.target.value
            if (oid) void act(() => research.push(idea.id, oid))
          }}>
          <option value="">{running.length ? 'Send to objective ▾' : 'no running objective'}</option>
          {running.map((o) => (
            <option key={o.id} value={o.id}>{o.title}{sentTo.has(o.id) ? ' (sent)' : ''}</option>
          ))}
        </select>
      </div>
      {err && <div className="text-[11.5px] text-bad">✗ {err}</div>}
    </li>
  )
}

function ChunkHead({ c, extra = '' }: { c: Chunk; extra?: string }) {
  return (
    <div className="mb-1 font-mono text-[10.5px] text-ink-faint">
      {c.label && <span className="mr-1 text-[11.5px] text-ink">{c.label}</span>}
      {[c.section, c.page != null ? `p. ${c.page}` : '', extra].filter(Boolean).join(' · ')}
    </div>
  )
}

/** The line a sandbox script imports a document's code with. */
function ImportLine({ text }: { text: string }) {
  return (
    <div className="mt-1 flex items-center gap-1 font-mono text-[11px] text-accent" onClick={(e) => e.stopPropagation()}>
      <code className="truncate">{text}</code>
      <CopyButton text={text} label="copy import" />
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="py-6 text-center text-[12px] text-ink-faint">{children}</div>
}

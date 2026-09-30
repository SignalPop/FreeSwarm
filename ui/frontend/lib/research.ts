/** The research library -- documents dropped in, parsed, embedded and mined for trading
 *  ideas that feed the objectives' idea streams. See app/research.py. */

export type ChunkKind = 'text' | 'table' | 'figure' | 'code'
export type DocStatus = 'queued' | 'parsing' | 'figures' | 'embedding' | 'ideas' | 'ready' | 'error'

export type ResearchSettings = {
  /** Model that extracts ideas; '' = the project's first idea rung. */
  idea_model: string
  /** Vision model that describes figures; '' = captions only. */
  figure_model: string
  auto_push: boolean
  /** Ideas sent to each running objective when a document finishes. */
  push_on_ingest: number
  /** Then at most one more per objective this often. */
  push_gap_minutes: number
  /** Least cosine (idea vs objective) to push automatically, with an embedding model. */
  min_relevance: number
  /** The same floor on token overlap, when no embedding model loaded. */
  min_overlap: number
}

export type ResearchStatus = {
  running: boolean
  queued: number
  /** doc id -> stage, while processing. */
  active: Record<string, string>
  embedding: { model: string; device: string | null; loaded: boolean; error: string | null }
  settings: ResearchSettings
}

export type ResearchDoc = {
  id: string
  /** '' = shared by every project. */
  project_id: string
  title: string
  filename: string
  kind: 'pdf' | 'html' | 'md' | 'txt'
  size_bytes: number
  pages: number | null
  /** Python package its code ships as: `from research.<package> import <module>`. */
  package: string
  status: DocStatus
  detail: string
  error: string
  idea_model: string
  embed_model: string
  created_at: number
  updated_at: number
  chunks: Partial<Record<ChunkKind, number>>
  /** Ideas not dismissed. */
  ideas: number
  /** Idea pushes to objectives, all told. */
  pushed: number
  /** The stage being worked on now, or null. */
  stage: string | null
}

export type Chunk = {
  id: number
  seq: number
  kind: ChunkKind
  section: string
  page: number | null
  label: string
  text: string
  image: string
  image_url?: string
  meta: { language?: string; lines?: number; module?: string; rows?: number; described_by?: string }
  /** For a code chunk with a module: the line that imports it in a sandbox script. */
  import?: string
}

export type Push = { objective_id: string; doc_idea_id: number; idea_row_id: number | null; ts: number; relevance: number | null; by: string }

export type DocIdea = {
  id: number
  doc_id: string
  seq: number
  title: string
  summary: string
  rules: string
  horizon: string
  fields: string[]
  evidence: string
  caveats: string
  falsify: string
  tags: string[]
  /** Chunk ids of the code blocks that implement it. */
  code_refs: number[]
  model: string
  created_at: number
  dismissed: 0 | 1
  pushes: Push[]
}

export type DocDetail = Omit<ResearchDoc, 'chunks' | 'ideas' | 'pushed'> & {
  outline: { level: number; title: string; page: number | null }[]
  chunks: Chunk[]
  ideas: DocIdea[]
}

export type SearchKind = ChunkKind | 'idea'

export type SearchHit = {
  doc_id: string
  /** The document's title. */
  doc: string
  kind: SearchKind
  section: string
  page: number | null
  label: string
  score: number
  cosine: number | null
  text: string
  chunk_id?: number
  idea_id?: number
  image_url?: string
  import?: string
}

/** A research idea as one objective sees it (GET /api/objectives/{oid}/research). */
export type ObjectiveResearchIdea = Omit<DocIdea, 'pushes'> & {
  doc_title: string
  package: string
  relevance: number
  /** Relevance is a cosine (true) or a token overlap (false). */
  dense: boolean
  pushed: Push | null
  /** Relevant enough to be pushed automatically. */
  eligible: boolean
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) throw new Error(await failure(res))
  return res.json() as Promise<T>
}

async function failure(res: Response): Promise<string> {
  let detail = `${res.status} ${res.statusText}`
  try {
    const b = await res.json()
    if (b?.detail) detail = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)
  } catch {
    /* non-JSON */
  }
  return detail
}

const post = <T,>(path: string, body: unknown) => req<T>(path, { method: 'POST', body: JSON.stringify(body) })
const e = encodeURIComponent

export const research = {
  status: () => req<ResearchStatus>('/api/research/status'),
  docs: (projectId?: string | null) =>
    req<{ docs: ResearchDoc[] }>(`/api/research/docs${projectId ? `?project_id=${e(projectId)}` : ''}`),
  doc: (id: string) => req<DocDetail>(`/api/research/docs/${e(id)}`),
  sourceUrl: (id: string) => `/api/research/docs/${e(id)}/source`,

  // Multipart: no Content-Type header, the browser sets the boundary itself.
  upload: async (files: File[], projectId: string) => {
    const fd = new FormData()
    for (const f of files) fd.append('file', f)
    fd.append('project_id', projectId)
    const res = await fetch('/api/research/docs', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(await failure(res))
    return res.json() as Promise<{ docs: (Omit<ResearchDoc, 'chunks' | 'ideas' | 'pushed' | 'stage'> & { created: boolean })[] }>
  },

  remove: (id: string) => post<{ deleted: string }>(`/api/research/docs/${e(id)}/delete`, {}),
  /** 'all' reparses from the file; 'embed' re-vectorises; 'ideas' re-extracts (optionally with `model`). */
  reprocess: (id: string, stage: 'all' | 'embed' | 'ideas', model?: string) =>
    post<{ queued: string; stage: string }>(`/api/research/docs/${e(id)}/reprocess`, { stage, model: model || null }),
  /** Give the document to one project, or '' to share it with all. */
  scope: (id: string, projectId: string) => post<ResearchDoc>(`/api/research/docs/${e(id)}/scope`, { project_id: projectId }),

  search: (body: { query: string; project_id?: string | null; kinds?: SearchKind[]; doc_id?: string; k?: number }) =>
    post<{ hits: SearchHit[]; dense: boolean }>('/api/research/search', body),

  push: (ideaId: number, objectiveId: string) =>
    post<{ idea_row_id: number; objective_id: string; doc_idea_id: number }>(`/api/research/ideas/${ideaId}/push`, {
      objective_id: objectiveId,
    }),
  dismiss: (ideaId: number, dismissed: boolean) =>
    post<{ id: number; dismissed: boolean }>(`/api/research/ideas/${ideaId}/dismiss`, { dismissed }),

  forObjective: (oid: string, limit = 20) =>
    req<{ ideas: ObjectiveResearchIdea[]; settings: ResearchSettings }>(`/api/objectives/${e(oid)}/research?limit=${limit}`),
  saveSettings: (patch: Partial<ResearchSettings>) =>
    req<ResearchSettings>('/api/research/settings', { method: 'PUT', body: JSON.stringify(patch) }),
}


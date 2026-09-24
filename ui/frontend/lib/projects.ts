// Client for the projects API.
//
// The *active* project lives on the server, not in the browser: agents calling the message
// board without an explicit project_id resolve to the same one the console is showing, so
// switching projects in the UI switches what an unscoped agent sees too. That is the whole
// point -- two sources of truth here would be a bug factory.

export type Project = {
  id: string
  name: string
  slug: string
  data_dir: string
  connectors: string[]
  created_at: number
  data_dir_exists: boolean
  /** Selected by the project but switched off globally -- inactive, and worth showing. */
  connectors_unavailable: string[]
  connectors_active: string[]
  /** Models this project's swarm may use; null = every loaded model. */
  models: string[] | null
  /** Read-only SQL Server access, or null. Holds no secret. */
  sql: SqlConfig | null
  /** Whether this project has a swarm (agents) at all. */
  swarm_enabled: boolean
  /** Per-model swarm role (search / ideas / both); absent = automatic. */
  model_roles?: Record<string, 'search' | 'ideas' | 'both'>
}

export type SwarmResources = {
  project: { id: string; name: string; slug: string }
  enabled: boolean
  models_filter: string[] | null
  llms: {
    model: string
    loaded: boolean
    ready: boolean
    gpu: string | null
    /** Served by another computer on the network. */
    remote?: { node: string; decode_tps: number | null; active: number | null } | null
    /** A hosted, pay-per-token model (Groq / OpenRouter). */
    external?: { provider_label: string; price_blended: number | null } | null
    swe?: number | null
    aa?: number | null
  }[]
  forecasters: { model: string; gpu: string; state: string; family: string | null }[]
  data: { data_dir: string; count: number; files: DataFile[] }
  sql: SqlConfig | null
  connectors: { active: string[]; unavailable: string[] }
}

export type SqlConfig = { server: string; database: string; tables: string[]; login: string }

export type SqlVerification = {
  read_only: boolean
  can_read_all: boolean
  sysadmin: boolean
  can_create_tables: boolean
  tables: { table: string; select: boolean; write: boolean }[]
}

export type QueryResult = {
  columns: string[]
  rows: unknown[][]
  row_count: number
  truncated: boolean
  seconds: number
}

export type DataFile = {
  path: string
  view: string
  bytes: number
  format: string
  /** A folder of parquet parts exported from SQL, exposed as one table. */
  dataset?: boolean
  rows?: number | null
  source?: string | null
}

export type ExportJob = {
  id: string
  table: string
  relative: string
  where: string | null
  state: 'running' | 'done' | 'failed'
  rows: number
  total: number | null
  files: number
  started_at: number
  finished_at: number | null
  error: string | null
}

export type FileEntry = {
  name: string
  path: string
  is_dir: boolean
  size_bytes: number
  modified_at: number
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const b = await res.json()
      if (b?.detail) detail = String(b.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const post = <T,>(path: string, body: unknown) =>
  req<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const projectResources = {
  swarmResources: (id: string) => req<SwarmResources>(`/api/projects/${id}/swarm/resources`),
  setSwarm: (id: string, enabled: boolean) => post<Project>(`/api/projects/${id}/swarm`, { enabled }),

  setModels: (id: string, models: string[] | null) =>
    post<Project>(`/api/projects/${id}/models`, { models }),

  dataCatalog: (id: string) =>
    req<{ data_dir: string; files: DataFile[] }>(`/api/projects/${id}/data/catalog`),
  dataQuery: (id: string, sql: string) =>
    post<QueryResult>(`/api/projects/${id}/data/query`, { sql, max_rows: 200 }),

  sqlDatabases: (server: string) =>
    req<{ databases: string[] }>(`/api/sql/databases?server=${encodeURIComponent(server)}`),
  sqlTables: (server: string, database: string) =>
    req<{ tables: { table: string; rows: number | null }[] }>(
      `/api/sql/tables?server=${encodeURIComponent(server)}&database=${encodeURIComponent(database)}`,
    ),
  sqlSetup: (id: string, server: string, database: string, tables: string[]) =>
    post<{ config: SqlConfig; verification: SqlVerification }>(`/api/projects/${id}/sql/setup`, {
      server,
      database,
      tables,
    }),
  sqlVerify: (id: string) => post<SqlVerification>(`/api/projects/${id}/sql/verify`, {}),
  sqlRemove: (id: string) => post<{ ok: boolean }>(`/api/projects/${id}/sql/remove`, {}),
  sqlQuery: (id: string, sql: string) =>
    post<QueryResult>(`/api/projects/${id}/sql/query`, { sql, max_rows: 200 }),
  sqlExport: (
    id: string,
    body: { table: string; folder: string; where?: string; rows_per_file?: number },
  ) => post<ExportJob>(`/api/projects/${id}/sql/export`, body),
  sqlExports: (id: string) => req<{ jobs: ExportJob[] }>(`/api/projects/${id}/sql/exports`),
}

export const projects = {
  list: () =>
    req<{ projects: Project[]; active: string | null; default_root: string }>('/api/projects'),

  create: (name: string, data_dir?: string, connectors: string[] = []) =>
    req<Project>('/api/projects', {
      method: 'POST',
      body: JSON.stringify({ name, data_dir: data_dir || null, connectors }),
    }),

  update: (id: string, fields: Partial<Pick<Project, 'name' | 'data_dir' | 'connectors'>>) =>
    req<Project>(`/api/projects/${id}`, { method: 'POST', body: JSON.stringify(fields) }),

  activate: (id: string) => req<Project>(`/api/projects/${id}/activate`, { method: 'POST' }),

  remove: (id: string, removeFiles: boolean) =>
    req<{ deleted: boolean }>(
      `/api/projects/${id}/delete?remove_files=${removeFiles ? 'true' : 'false'}`,
      { method: 'POST' },
    ),

  files: (id: string, subdir = '') =>
    req<{ root: string; entries: FileEntry[]; total_bytes: number; exists: boolean }>(
      `/api/projects/${id}/files?subdir=${encodeURIComponent(subdir)}`,
    ),

  readFile: (id: string, path: string) =>
    req<{ path: string; size_bytes: number; truncated: boolean; text: string }>(
      `/api/projects/${id}/file?path=${encodeURIComponent(path)}`,
    ),

  mkdir: (id: string, path: string) =>
    req<{ created: string }>(`/api/projects/${id}/mkdir`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  deleteFile: (id: string, path: string) =>
    req<{ deleted: string }>(`/api/projects/${id}/files/delete`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  // Multipart: no Content-Type header, the browser sets the boundary itself.
  upload: async (id: string, file: File, subdir = '') => {
    const fd = new FormData()
    fd.append('file', file)
    if (subdir) fd.append('path', subdir)
    const res = await fetch(`/api/projects/${id}/upload`, { method: 'POST', body: fd })
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const b = await res.json()
        if (b?.detail) detail = String(b.detail)
      } catch {
        /* non-JSON */
      }
      throw new Error(detail)
    }
    return res.json() as Promise<{ path: string; size_bytes: number }>
  },
}

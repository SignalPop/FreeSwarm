'use client'

import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { api, type ConsoleDoc, type ModelEntry } from '@/lib/api'
import { bytesLabel } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Button, EmptyState, PageHeader, Panel, Pill } from '@/components/ui'
import TimeSeriesModels from '@/components/TimeSeriesModels'
import ModelDownloads from '@/components/ModelDownloads'

/** Launch options exposed in the UI. Every key here must exist in the backend's
 *  `_FLAG_SPEC` allow-list, which is what actually decides what reaches argv. */
type Options = {
  tp_size: number
  moe_backend: string
  attention_backend: string
  num_tokens: number
  max_running_requests: number
  memory_ratio: number
  expert_load: string
}

const DEFAULTS: Options = {
  tp_size: 1,
  moe_backend: 'fused',
  attention_backend: 'auto',
  // 128K tokens of context, stepped down per model and card by defaultContext() when it
  // does not fit. The old 8192 default was a trap: an agent with tools overflows 8K after two
  // or three calls (one real failure was a 15.6K-token prompt).
  num_tokens: 131072,
  max_running_requests: 4,
  memory_ratio: 0.9,
  expert_load: 'serial',
}

/**
 * Launch settings measured to work for a particular checkpoint, applied when it is
 * selected. These win over the size heuristic below, which infers a configuration from
 * the checkpoint's bytes and can only ever approximate what a model actually needs.
 *
 * Two things are deliberately NOT preset. The target GPU, because that depends on which
 * card is free at the time. And the context, because defaultContext() steps 128K down
 * when the chosen card cannot hold the KV cache -- pinning it here would turn selecting
 * the small card into an out-of-memory failure several minutes into a load.
 */
type Preset = Omit<Partial<Options>, 'num_tokens'>

const MODEL_PRESETS: { match: RegExp; opts: Preset }[] = [
  {
    // 146.6 GiB of experts, so offload is the only route on a single card, and serial
    // pinning is what the Windows page-file commit allows. Its sparse attention refuses
    // 'triton' outright ("valid backends: dsv4_sparse"), so the backend has to be the one
    // the engine resolves for itself.
    match: /^DeepSeek-V4-Flash/i,
    opts: {
      tp_size: 1,
      moe_backend: 'offload',
      attention_backend: 'auto',
      max_running_requests: 4,
      memory_ratio: 0.9,
      expert_load: 'serial',
    },
  },
]

function presetFor(modelId: string): Preset {
  return MODEL_PRESETS.find((p) => p.match.test(modelId))?.opts ?? {}
}

const CONTEXT_STEPS = [131072, 65536, 32768, 16384]
// What the engine needs on the card besides weights and KV cache (CUDA context, activations,
// graph workspace) -- the same slack the pinning/fit checks leave.
const ENGINE_SLACK_BYTES = 1.5 * 2 ** 30

/**
 * The largest context (up to 128K, and never past the model's own maximum) whose KV cache
 * still fits on the target card next to the weights that stay in VRAM.
 *
 * 128K is the default because a swarm iteration carries a parent candidate's code, the
 * lessons and the leaderboard before any tool output, and the KV cache is cheap on these
 * hybrid-attention models (2.5-4.5 GiB at 128K). But it is reserved at load, so on a small
 * card -- the 10 GiB 3080 that also drives the display -- it steps down rather than turning
 * a load that would have worked into an out-of-memory failure.
 */
function defaultContext(m: ModelEntry, budgetBytes: number, offload: boolean): number {
  const cap = m.max_position_embeddings || 131072
  const steps = CONTEXT_STEPS.filter((n) => n <= cap)
  if (!steps.length) return cap
  if (!m.kv_bytes_per_token) return Math.min(65536, steps[0])
  // With offload the experts live in host RAM; only the rest of the checkpoint is on the card.
  const resident = offload ? Math.max(0, m.size_bytes - (m.expert_bytes || 0)) : m.size_bytes * 1.06
  const room = budgetBytes - resident - ENGINE_SLACK_BYTES
  return steps.find((n) => n * (m.kv_bytes_per_token as number) <= room) ?? steps[steps.length - 1]
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <div className="text-[12px] text-ink-dim">{label}</div>
      {children}
      {hint && <div className="mt-1 text-[11px] text-ink-faint">{hint}</div>}
    </label>
  )
}

// Usable VRAM on the largest single card, less room for KV cache, CUDA graphs and
// activations. Measured: an A6000 reports 47.4 GiB free, and gpt-oss-20b took ~13.5 GiB
// of that for a 12.8 GiB checkpoint.
const SINGLE_GPU_BUDGET_BYTES = 40 * 2 ** 30

const inputCls =
  'mt-1.5 w-full rounded-lg border border-seam bg-panel-hi px-3 py-2 font-mono text-[13px] text-ink outline-none focus:border-accent'

export default function ModelsPage() {
  const router = useRouter()
  const [models, setModels] = useState<ModelEntry[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [opts, setOpts] = useState<Options>(DEFAULTS)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [targetGpu, setTargetGpu] = useState<string>('')

  const { data: console_, refresh } = usePoll<ConsoleDoc>(api.console, 2000)
  const consoleReady = console_ != null
  const engines = console_?.engines ?? []
  // Which GPUs already hold an engine -- the manager refuses a second one on the same card.
  const busyGpus = new Set(
    engines.filter((e) => e.state !== 'stopped' && e.gpus).map((e) => String(e.gpus)),
  )
  const pool = (console_?.gpus ?? []).map((g) => String(g.index))
  const freeGpus = pool.filter((g) => !busyGpus.has(g))
  const loadedIds = new Set(engines.map((e) => e.model_id).filter(Boolean) as string[])

  // Cards are not interchangeable here -- a 10 GiB display GPU sits in the same list as two
  // 48 GiB A6000s, and "GPU 1" alone does not say which is which. Labelling each with its
  // model and free VRAM turns picking a target from a guess into a decision, and flags the
  // one mistake that costs a long load: choosing a card the checkpoint cannot fit on.
  const selectedModel = models?.find((x) => x.id === selected) ?? null
  // With offload the experts live in host RAM, so VRAM only has to cover the rest; fused
  // needs the whole checkpoint resident. Anything else (cpu/hybrid) is lighter still.
  const needsFullVram = opts.moe_backend === 'fused'

  // Host RAM is the resource every offloaded engine SHARES -- VRAM is per card, but expert
  // banks are page-locked in RAM, so two big MoE models on two different GPUs still compete.
  // Catching it here turns "cudaHostRegister failed for 0.5 GiB" three minutes into a load
  // into a warning before the click. size_bytes over-estimates the expert share slightly,
  // which is the safe direction.
  const OFFLOAD_BACKENDS = new Set(['offload', 'cpu', 'hybrid'])
  const hostTotal = console_?.host_memory?.total_bytes ?? 0
  // NOT total RAM. Pinned expert banks are also GPU-mapped, and Windows/WDDM caps how much
  // system memory can be locked that way at about half of physical RAM -- measured at 69 GiB
  // on a 127.9 GiB machine, failing with `cudaHostRegister failed` while ~58 GiB was still
  // free. Budgeting against free RAM would pass launches that cannot physically pin.
  const PIN_FRACTION = 0.52
  const hostPinBudget =
    console_?.host_memory?.pin_budget_bytes ?? hostTotal * PIN_FRACTION
  const pinnedByOthers = engines
    .filter(
      (e) =>
        e.state !== 'stopped' &&
        OFFLOAD_BACKENDS.has(String((e.options as Record<string, unknown>)?.moe_backend ?? '')),
    )
    .reduce((sum, e) => {
      const m = models?.find((x) => x.id === e.model_id)
      // expert_bytes is what is actually pinned; size_bytes only as a fallback when the
      // checkpoint layout could not be read (0 means unknown, not "pins nothing").
      return sum + (m?.expert_bytes || m?.size_bytes || 0)
    }, 0)
  const hostBudgetLeft = Math.max(0, hostPinBudget - pinnedByOthers)
  // Only the experts are pinned, so a 67 GiB checkpoint with 61.5 GiB of experts fits under
  // a 66.5 GiB ceiling. Budgeting against size_bytes warned about models that actually load.
  const selectedPinned = selectedModel
    ? selectedModel.expert_bytes || selectedModel.size_bytes
    : 0
  const hostRamShort =
    hostTotal > 0 &&
    selectedModel != null &&
    OFFLOAD_BACKENDS.has(opts.moe_backend) &&
    selectedPinned > hostBudgetLeft

  /** Free VRAM on one card, less the slack the engine needs beyond the weights. */
  function gpuBudgetBytes(index: string): number {
    const gpu = (console_?.gpus ?? []).find((g) => String(g.index) === index)
    if (!gpu) return SINGLE_GPU_BUDGET_BYTES
    return Math.max(0, gpu.memory_total_bytes - gpu.memory_used_bytes) * 0.95
  }

  /** With auto, the engine takes the next free card -- so judge the fit against the
   *  SMALLEST free one, or a model could be configured for an A6000 and land on the 3080. */
  function autoGpuBudgetBytes(): number {
    const budgets = freeGpus.map(gpuBudgetBytes)
    return budgets.length ? Math.min(...budgets) : SINGLE_GPU_BUDGET_BYTES
  }

  function gpuOptionLabel(index: string): string {
    const gpu = (console_?.gpus ?? []).find((g) => String(g.index) === index)
    if (!gpu) return `GPU ${index}`
    const free = Math.max(0, gpu.memory_total_bytes - gpu.memory_used_bytes)
    // Trim the vendor prefix: "NVIDIA RTX A6000" -> "RTX A6000".
    const name = gpu.name.replace(/^NVIDIA\s+/i, '')
    const parts = [`GPU ${index}`, name, `${bytesLabel(free)} free`]
    let label = parts.join(' — ')
    if (busyGpus.has(index)) {
      const holder = engines.find((e) => String(e.gpus) === index)
      return `${label} — in use${holder?.model_id ? ` by ${holder.model_id}` : ''}`
    }
    if (selectedModel && needsFullVram && selectedModel.size_bytes * 1.06 > free * 0.95) {
      label += selectedModel.is_moe ? ' — needs offload' : ' — too small'
    }
    return label
  }

  useEffect(() => {
    api
      .models()
      .then((r) => setModels(r.models))
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [])

  // A checkpoint bigger than one card cannot be split across GPUs here: tensor
  // parallelism needs NCCL, which has no Windows build. The route that DOES work is
  // MoE offload -- experts stream from host RAM over PCIe - so default to that instead
  // of to --tp-size 2, which would fail several seconds into the load.
  //
  // The fit is judged against the card this will ACTUALLY land on, not a fixed budget.
  // A fixed 40 GiB rule silently assumes every GPU is an A6000: pick the 10 GiB 3080 for
  // a 12.8 GiB model and the backend stayed 'fused', so the launch was refused by the
  // server's preflight for a configuration the page had chosen itself. Small cards are
  // exactly where offload matters most -- gpt-oss-20b needs only ~3.4 GiB of non-expert
  // weights in VRAM once its experts are in host RAM.
  useEffect(() => {
    const m = models?.find((x) => x.id === selected)
    if (!m) return
    const budget = targetGpu ? gpuBudgetBytes(targetGpu) : autoGpuBudgetBytes()
    const tooBigForTheCard = m.size_bytes * 1.06 > budget
    const preset = presetFor(m.id)
    // Offload keyed on is_moe, NOT on num_experts being non-null. Multimodal configs
    // (Qwen3.6, Gemma-4) hide their expert count under text_config, so keying on the
    // count made an MoE model look dense and picked a backend that cannot fit.
    const moeBackend = preset.moe_backend ?? (tooBigForTheCard && m.is_moe ? 'offload' : 'fused')
    // The context has to be judged against the backend actually being used: with offload
    // the experts leave VRAM, which is the difference between 128K fitting and not.
    const offload = OFFLOAD_BACKENDS.has(moeBackend)
    setOpts((o) => ({
      ...o,
      tp_size: 1,
      expert_load: 'serial',
      ...preset,
      moe_backend: moeBackend,
      num_tokens: defaultContext(m, budget, offload),
    }))
    // Deliberately NOT keyed on console_. That object is replaced by the 2 s poll, so having
    // it here re-ran this effect every two seconds and silently reset moe_backend,
    // expert_load and tp_size -- any choice the user made in the form reverted a moment
    // later. The backend is a default picked when the MODEL or TARGET GPU changes; after
    // that it belongs to the user. It still reads the latest console_ at that moment.
    // consoleReady flips false -> true exactly once, so a model picked before the first poll
    // landed is re-judged against real VRAM, without re-firing on every later poll.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, models, targetGpu, consoleReady])

  async function start() {
    if (!selected) return
    setBusy(true)
    setErr(null)
    try {
      // /api/engines adds an engine alongside whatever is already resident, instead of
      // being a single global slot.
      await api.startEngine(
        selected,
        opts as unknown as Record<string, unknown>,
        targetGpu || undefined,
      )
      refresh()
      router.push('/')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Models"
        subtitle={
          models ? `${models.length} checkpoint${models.length === 1 ? '' : 's'} discovered locally` : 'Scanning…'
        }
        right={
          engines.length ? (
            <span className="flex flex-wrap items-center gap-2">
              {engines.map((e) => (
                <Pill key={e.instance_id} tone={e.state === 'running' ? 'good' : 'warn'} pulse>
                  {e.model_id} · GPU {e.gpus}
                </Pill>
              ))}
            </span>
          ) : undefined
        }
      />

      {err && (
        <Panel className="mb-6 border-bad/35 bg-bad/5 p-4 font-mono text-[12px] text-bad">{err}</Panel>
      )}

      <ModelDownloads
        onInstalled={() =>
          api
            .models()
            .then((r) => setModels(r.models))
            .catch(() => {})
        }
      />

      {models && models.length === 0 && (
        <EmptyState
          title="No checkpoints found"
          hint={
            <>
              Drop a HuggingFace checkpoint into <code className="font-mono">models\</code> in the
              repo root, or set <code className="font-mono">FREESWARM_MODELS_DIR</code>. The
              HuggingFace hub cache is scanned too.
            </>
          }
        />
      )}

      <div
        className="grid items-start gap-6"
        style={{ gridTemplateColumns: 'minmax(0,1fr) 340px' }}
      >
        <div className="space-y-3">
          {/* LLMs only: time-series checkpoints have their own section below, with their own
              load flow -- they are not engine models and must not reach the LLM launch panel. */}
          {models?.filter((m) => m.category !== 'timeseries').map((m) => {
            const active = selected === m.id
            const isRunning = loadedIds.has(m.id)
            return (
              <button
                key={m.path}
                onClick={() => m.supported && setSelected(m.id)}
                disabled={!m.supported}
                className={`w-full rounded-2xl border p-4 text-left transition-colors ${
                  active
                    ? 'border-accent/50 bg-accent/[0.06]'
                    : 'border-seam bg-panel hover:bg-panel-hi/60'
                }`}
              >
                <div className="flex flex-wrap items-center gap-3">
                  <span className="min-w-0 flex-1 truncate text-[15px] text-ink">{m.id}</span>
                  {isRunning && (
                    <Pill tone="good" pulse>
                      on GPU {engines.find((e) => e.model_id === m.id)?.gpus}
                    </Pill>
                  )}
                  {!m.supported && <Pill tone="bad">not supported</Pill>}
                  {m.supported && m.is_moe && <Pill tone="accent">MoE</Pill>}
                  {m.supported && m.size_bytes > SINGLE_GPU_BUDGET_BYTES && (
                    <Pill tone="warn">{m.is_moe ? 'needs offload' : 'too large'}</Pill>
                  )}
                  <span className="font-mono text-[12px] text-ink-dim">
                    {bytesLabel(m.size_bytes)}
                  </span>
                </div>
                {!m.supported && m.unsupported_reason && (
                  <div className="mt-2 rounded-lg border border-bad/30 bg-bad/5 p-2 text-[11.5px] leading-relaxed text-bad">
                    {m.unsupported_reason}
                  </div>
                )}
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-ink-faint">
                  {m.architecture && <span>{m.architecture}</span>}
                  {m.quantization && <span>quant {m.quantization}</span>}
                  {m.num_experts ? <span>{m.num_experts} experts</span> : null}
                  {m.max_position_embeddings ? (
                    <span>ctx {m.max_position_embeddings.toLocaleString()}</span>
                  ) : null}
                </div>
                <div className="mt-1.5 truncate font-mono text-[10px] text-ink-faint">{m.path}</div>
              </button>
            )
          })}

          <TimeSeriesModels
            models={(models ?? []).filter((m) => m.category === 'timeseries')}
            console_={console_}
          />
        </div>

        {/* ---- Launch options ---- */}
        <Panel className="h-fit p-5">
          <div className="text-[15px] font-medium text-ink">Launch options</div>
          <p className="mt-1 text-[12px] text-ink-faint">
            Only flags on the server&apos;s allow-list are accepted.
          </p>

          <div className="mt-5 space-y-4">
            <Field
              label="Tensor parallel size"
              hint="Multi-GPU needs NCCL, which has no Windows build. Use offload instead."
            >
              <select
                className={inputCls}
                value={opts.tp_size}
                onChange={(e) => setOpts({ ...opts, tp_size: Number(e.target.value) })}
              >
                <option value={1}>1 — single GPU</option>
                <option value={2} disabled>
                  2 — unsupported on Windows (no NCCL)
                </option>
              </select>
            </Field>

            <Field label="MoE backend" hint="fused = experts resident in VRAM; offload = host RAM + PCIe">
              <select
                className={inputCls}
                value={opts.moe_backend}
                onChange={(e) => setOpts({ ...opts, moe_backend: e.target.value })}
              >
                {['fused', 'offload', 'cpu', 'hybrid'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="Attention backend"
              hint="auto selects the backend required by the model (including DeepSeek-V4 sparse attention)"
            >
              <select
                className={inputCls}
                value={opts.attention_backend}
                onChange={(e) => setOpts({ ...opts, attention_backend: e.target.value })}
              >
                {['auto', 'triton', 'dsv4_sparse', 'dsa', 'm3_sparse', 'fi', 'fa', 'trtllm'].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="Context (tokens)"
              hint={
                selectedModel?.kv_bytes_per_token
                  ? `${opts.num_tokens.toLocaleString()} tokens ≈ ${bytesLabel(
                      selectedModel.kv_bytes_per_token * opts.num_tokens,
                    )} of VRAM for this model. Shared by all concurrent requests. Agents with tools need 32K+.`
                  : 'Total token capacity shared by all concurrent requests. Agents with tools need 32K+.'
              }
            >
              <div className="flex gap-1.5">
                <input
                  type="number"
                  className={inputCls}
                  value={opts.num_tokens}
                  min={512}
                  step={4096}
                  onChange={(e) => setOpts({ ...opts, num_tokens: Number(e.target.value) })}
                />
                {[16384, 32768, 65536, 131072].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setOpts({ ...opts, num_tokens: n })}
                    className={`shrink-0 rounded-md border px-1.5 font-mono text-[10.5px] transition-colors ${
                      opts.num_tokens === n ? 'border-accent/60 text-accent' : 'border-seam text-ink-faint hover:text-ink-dim'
                    }`}
                  >
                    {n / 1024}K
                  </button>
                ))}
              </div>
              {opts.num_tokens < 16384 && (
                <div className="mt-1 text-[11px] text-warn">
                  Under 16K: fine for chat, but a swarm agent will overflow after a few tool calls.
                </div>
              )}
            </Field>

            <Field
              label="Max running requests"
              hint="Concurrent slots. Raise for an agent swarm; KV pages must cover them all."
            >
              <input
                type="number"
                className={inputCls}
                value={opts.max_running_requests}
                min={1}
                max={256}
                onChange={(e) =>
                  setOpts({ ...opts, max_running_requests: Number(e.target.value) })
                }
              />
            </Field>

            <Field label="Memory ratio" hint="Fraction of free VRAM the engine may claim">
              <input
                type="number"
                className={inputCls}
                value={opts.memory_ratio}
                min={0.1}
                max={0.98}
                step={0.01}
                onChange={(e) => setOpts({ ...opts, memory_ratio: Number(e.target.value) })}
              />
            </Field>

            {opts.moe_backend === 'offload' && (
              <Field label="Expert load" hint="serial is required on Windows (page-file commit)">
                <select
                  className={inputCls}
                  value={opts.expert_load}
                  onChange={(e) => setOpts({ ...opts, expert_load: e.target.value })}
                >
                  <option value="serial">serial</option>
                  <option value="parallel">parallel</option>
                </select>
              </Field>
            )}
          </div>

          <div className="mt-6 space-y-3">
            <Field
              label="Target GPU"
              hint="One engine per card. Leave on auto to take the next free one."
            >
              <select
                className={inputCls}
                value={targetGpu}
                onChange={(e) => setTargetGpu(e.target.value)}
              >
                <option value="">auto — next free ({freeGpus.join(', ') || 'none'})</option>
                {pool.map((g) => (
                  <option key={g} value={g} disabled={busyGpus.has(g)}>
                    {gpuOptionLabel(g)}
                  </option>
                ))}
              </select>
            </Field>

            {hostRamShort && (
              <div className="rounded-lg border border-warn/40 bg-warn/[0.07] px-3 py-2 text-[12px] leading-relaxed text-ink-dim">
                <strong className="text-ink">Over the pinning limit.</strong> This
                checkpoint needs ~{bytesLabel(selectedPinned)} page-locked, but
                only {bytesLabel(hostBudgetLeft)} of this machine&rsquo;s{' '}
                {bytesLabel(hostPinBudget)} limit is left after the offloaded
                models already running. That ceiling is Windows/WDDM, not free RAM — the
                load would fail partway through however much RAM is free. Unload one first,
                or use a quantized checkpoint.
              </div>
            )}

            <Button
              tone="primary"
              className="w-full"
              onClick={start}
              disabled={
                !selected || busy || loadedIds.has(selected) || freeGpus.length === 0
              }
            >
              {busy
                ? 'Starting…'
                : selected && loadedIds.has(selected)
                  ? 'Already loaded'
                  : freeGpus.length === 0
                    ? 'No free GPU — stop an engine first'
                    : 'Load into a new engine'}
            </Button>
            {selected && loadedIds.has(selected) && (
              <Button
                tone="danger"
                className="w-full"
                onClick={async () => {
                  const inst = engines.find((e) => e.model_id === selected)
                  if (!inst) return
                  setBusy(true)
                  try {
                    await api.stopEngine(inst.instance_id)
                    refresh()
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                Stop this model&apos;s engine
              </Button>
            )}
          </div>

          {selected && (
            <div className="mt-4 space-y-2 text-[11px] leading-relaxed text-ink-faint">
              {opts.moe_backend === 'offload' && (
                <p className="text-ink-dim">
                  This checkpoint is larger than one GPU, so MoE backend is set to{' '}
                  <strong className="text-ink">offload</strong>: experts are pinned in host RAM
                  and streamed over PCIe. Windows cannot split a model across both cards (no
                  NCCL), so this is the route that works. Expect a long load — roughly 12
                  minutes for a 61 GiB model — most of it spent pinning experts.
                </p>
              )}
              {opts.moe_backend === 'fused' &&
                (models?.find((x) => x.id === selected)?.size_bytes ?? 0) >
                  SINGLE_GPU_BUDGET_BYTES && (
                  <p className="text-warn">
                    This checkpoint is larger than a single GPU and the fused backend keeps all
                    experts in VRAM — the launch will be refused. Switch MoE backend to
                    offload.
                  </p>
                )}
              <p>
                First launch of a checkpoint JIT-compiles Triton kernels — expect a slow first
                request, then full speed.
              </p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}

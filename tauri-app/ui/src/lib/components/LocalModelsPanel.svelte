<script lang="ts" module>
  /** One entry of autonomy_review_status.local_models (daemon pilot/system/model_health.py). */
  export type LocalModel = {
    name: string;
    state: string;
    reason?: string;
    device?: string | null;
    process?: string;
    retrying?: boolean;
    last_fault?: { at?: string; message?: string; kind?: string } | null;
  };
  type Gpu = {
    name: string;
    temperature_c: number | null;
  };
  export type GpuStatus = {
    machine?: { kind?: string; gpus?: Gpu[]; error?: string };
    max_temperature_c?: number;
    disabled?: boolean;
    disabled_reason?: string;
    faults_this_session?: number;
  };
  export type ReviewStatus = { local_models?: LocalModel[]; gpu?: GpuStatus };

  const STATE_WORDS: Record<string, string> = {
    loaded: "Ready",
    loading: "Loading",
    not_loaded: "Not loaded yet",
    unavailable: "Unavailable",
    off: "Off",
    not_supported: "Not on this computer",
  };
  const DEVICE_WORDS: Record<string, string> = {
    cuda: "graphics card",
    mps: "Apple graphics",
    cpu: "processor",
    system: "built into the system",
  };

  export function stateWords(model: LocalModel): string {
    return STATE_WORDS[model.state] ?? "Unknown";
  }

  /** "On the graphics card", "Runs in its own process", "Tries again by itself"... in one short line. */
  export function detail(model: LocalModel): string {
    const parts: string[] = [];
    if (model.device && DEVICE_WORDS[model.device]) {
      const where = DEVICE_WORDS[model.device];
      parts.push(model.device === "system" ? `Uses what is ${where}` : `On the ${where}`);
    }
    const reason = (model.reason ?? "").trim();
    if (reason) parts.push(reason.replace(/[.!?]$/, ""));
    if (model.retrying && !/tries again|restarting/i.test(reason)) parts.push("Tries again by itself");
    return parts.length ? `${parts.join(". ")}.` : "";
  }

  /** Amber only when something is wrong and the person may want to know; off or unsupported is just how it is. */
  export function tone(model: LocalModel): "ok" | "quiet" | "warn" {
    if (model.state === "unavailable") return "warn";
    if (model.state === "loaded" || model.state === "loading") return "ok";
    return "quiet";
  }

  export function gpuLine(status: GpuStatus | null): string {
    const machine = status?.machine;
    if (!machine) return "";
    if (machine.kind === "apple")
      return "Apple silicon: the graphics processor shares the system memory, so local models are limited by free memory.";
    if (machine.kind !== "nvidia") return "No NVIDIA graphics card: local models run on the processor.";
    if (machine.error) return `The graphics card did not report its state (${machine.error}).`;
    const card = machine.gpus?.[0];
    const limit = status?.max_temperature_c;
    const temp =
      card?.temperature_c != null
        ? `, ${Math.round(card.temperature_c)} °C${limit ? ` (models start below ${Math.round(limit)} °C)` : ""}`
        : "";
    const more = (machine.gpus?.length ?? 0) > 1 ? ` and ${(machine.gpus?.length ?? 1) - 1} more` : "";
    return `${card?.name ?? "NVIDIA graphics card"}${more}${temp}.`;
  }
</script>

<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call } from "../api/daemon";

  const REFRESH_MS = 15000;

  let models = $state<LocalModel[]>([]);
  let gpu = $state<GpuStatus | null>(null);
  let error = $state("");
  let loaded = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  async function load() {
    try {
      const result = await call<ReviewStatus>("autonomy_review_status");
      models = result.local_models ?? [];
      gpu = result.gpu ?? null;
      error = "";
    } catch (err) {
      error = err instanceof Error ? err.message : "The model status is not available.";
    } finally {
      loaded = true;
    }
  }

  onMount(() => {
    void load();
    timer = setInterval(() => void load(), REFRESH_MS);
  });
  onDestroy(() => {
    if (timer) clearInterval(timer);
  });
</script>

<section class="models-card" aria-labelledby="local-models-heading" data-testid="local-models-panel">
  <div class="section-title">
    <div>
      <span class="eyebrow">Health</span>
      <h3 id="local-models-heading">Models on this computer</h3>
      <p>What runs locally, where it runs, and why anything is not available right now.</p>
    </div>
    <button type="button" onclick={() => void load()}>Refresh</button>
  </div>

  {#if gpu}
    <div class="gpu" data-testid="gpu-line">
      <span>{gpuLine(gpu)}</span>
      {#if gpu.disabled}
        <span class="warn-text" role="status">{gpu.disabled_reason || "The graphics card is switched off."}</span>
      {/if}
    </div>
  {/if}

  {#if error}
    <div class="message" role="alert">{error}</div>
  {:else if loaded && models.length === 0}
    <p class="empty">No local models reported.</p>
  {:else}
    <ul class="rows">
      {#each models as model (model.name)}
        <li data-state={model.state}>
          <div class="row-text">
            <strong>{model.name}</strong>
            {#if detail(model)}<small>{detail(model)}</small>{/if}
            {#if model.last_fault?.message}
              <small class="fault"
                >Last problem{model.last_fault.at ? ` at ${model.last_fault.at.replace("T", " ")}` : ""}: {model
                  .last_fault.message}</small
              >
            {/if}
          </div>
          <span class="state {tone(model)}">{stateWords(model)}</span>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .models-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p,
  small {
    display: block;
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .eyebrow {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .gpu {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 11px;
  }
  .warn-text {
    color: var(--warning);
  }
  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 8%, transparent);
    font-size: 11px;
  }
  .empty {
    padding: 12px 20px 16px;
    border-top: 1px solid var(--border);
  }
  .rows {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .rows li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .row-text {
    min-width: 0;
  }
  .row-text strong {
    display: block;
    font-size: 12px;
  }
  .row-text small {
    overflow-wrap: anywhere;
  }
  .fault {
    font-family: var(--font-mono);
    font-size: 10px;
  }
  .state {
    flex-shrink: 0;
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 11px;
    white-space: nowrap;
  }
  .state.ok {
    color: var(--success);
  }
  .state.quiet {
    color: var(--text-secondary);
  }
  .state.warn {
    color: var(--warning);
  }
  button {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  @media (max-width: 760px) {
    .section-title,
    .rows li {
      padding-inline: 14px;
    }
    .gpu {
      padding-inline: 14px;
    }
  }
</style>

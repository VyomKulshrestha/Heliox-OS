<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireOkResult, type DaemonStatusResult } from "../api/daemon";

  interface MemoryFact {
    fact_id: string;
    subject: string;
    predicate: string;
    value: unknown;
    scope: string;
    status: "active" | "candidate";
    confidence: number;
    provenance: string;
    evidence_count: number;
    updated_at: string;
  }

  interface MemoryStatus {
    available: boolean;
    facts: MemoryFact[];
    counts: {
      facts: Record<string, number>;
      episodes: number;
      working_items: number;
    };
  }

  let memory = $state<MemoryStatus | null>(null);
  let loading = $state(true);
  let error = $state("");
  let confirmingFact = $state("");
  let forgettingFact = $state("");

  onMount(refresh);

  async function refresh() {
    loading = true;
    error = "";
    try {
      const result = requireOkResult(
        (await call("temporal_memory_status", { limit: 50 })) as MemoryStatus & DaemonStatusResult,
        "The daemon rejected the memory status request.",
      );
      if (!Array.isArray(result.facts) || !result.counts?.facts) {
        throw new Error("The daemon's memory status is incomplete.");
      }
      memory = result;
    } catch (cause) {
      memory = null;
      const message = cause instanceof Error ? cause.message : String(cause);
      error = /method not found.*temporal_memory_status/i.test(message)
        ? "This app is connected to an older daemon. Restart Heliox, then retry."
        : `Could not load memory: ${message}`;
    } finally {
      loading = false;
    }
  }

  async function forget(fact: MemoryFact) {
    if (confirmingFact !== fact.fact_id) {
      confirmingFact = fact.fact_id;
      return;
    }
    forgettingFact = fact.fact_id;
    error = "";
    try {
      requireOkResult(
        (await call("temporal_memory_retract", {
          fact_id: fact.fact_id,
          reason: "User retracted this memory from Settings",
        })) as DaemonStatusResult,
        "The daemon did not forget this memory.",
      );
      confirmingFact = "";
      await refresh();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not forget this memory.";
    } finally {
      forgettingFact = "";
    }
  }

  function displayValue(value: unknown): string {
    if (typeof value === "string") return value;
    return JSON.stringify(value);
  }

  function label(value: string): string {
    return value.replaceAll("_", " ");
  }
</script>

<div class="memory-panel">
  <div class="panel-header">
    <div>
      <h3>Memory &amp; Adaptation</h3>
      <span>Working, episodic, semantic, and temporal memory</span>
    </div>
    <button class="secondary" onclick={refresh} disabled={loading}>Refresh</button>
  </div>

  <p class="safety-note">
    Memory is advisory context, never permission. Inferred facts remain candidates until repeated evidence promotes
    them. You can inspect provenance, confidence, and forget any active or candidate fact.
  </p>

  {#if loading}
    <div class="empty">Loading memory status...</div>
  {:else if error && !memory}
    <div class="error" role="alert">
      <span>{error}</span>
      <button class="secondary" onclick={refresh}>Retry</button>
    </div>
  {:else if memory}
    <div class="status-grid">
      <div>
        <span>Active facts</span>
        <strong>{memory.counts.facts.active ?? 0}</strong>
      </div>
      <div>
        <span>Candidates</span>
        <strong>{memory.counts.facts.candidate ?? 0}</strong>
      </div>
      <div>
        <span>Verified episodes</span>
        <strong>{memory.counts.episodes}</strong>
      </div>
      <div>
        <span>Current task state</span>
        <strong>{memory.counts.working_items}</strong>
      </div>
    </div>

    <div class="facts-header">
      <strong>Reviewable facts</strong>
      <span>{memory.facts.length} shown</span>
    </div>
    {#if memory.facts.length === 0}
      <div class="empty">No learned facts yet. Verified task outcomes will still appear as episodic memory.</div>
    {:else}
      <div class="facts">
        {#each memory.facts as fact (fact.fact_id)}
          <article>
            <div class="fact-main">
              <div class="badges">
                <span class:active={fact.status === "active"} class:candidate={fact.status === "candidate"}>
                  {fact.status}
                </span>
                <span>{label(fact.provenance)}</span>
                <span>{Math.round(fact.confidence * 100)}% confidence</span>
                <span>{fact.evidence_count} evidence</span>
              </div>
              <strong>{label(fact.predicate)}</strong>
              <span class="value">{displayValue(fact.value)}</span>
              <small>{label(fact.scope)} memory · updated {new Date(fact.updated_at).toLocaleString()}</small>
            </div>
            <button
              class:confirm={confirmingFact === fact.fact_id}
              onclick={() => forget(fact)}
              disabled={forgettingFact === fact.fact_id}
              aria-label={`Forget ${label(fact.predicate)}`}
            >
              {forgettingFact === fact.fact_id
                ? "Forgetting..."
                : confirmingFact === fact.fact_id
                  ? "Confirm forget"
                  : "Forget"}
            </button>
          </article>
        {/each}
      </div>
    {/if}
    {#if error}
      <div class="error" role="alert">{error}</div>
    {/if}
  {/if}
</div>

<style>
  .memory-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .facts-header,
  article {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .panel-header {
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  h3 {
    margin: 0;
    font-size: 14px;
  }

  .panel-header span,
  .facts-header span,
  small {
    color: var(--text-muted);
    font-size: 11px;
  }

  .safety-note {
    margin: 0;
    padding: 10px 14px;
    color: var(--warning);
    background: color-mix(in srgb, var(--warning) 8%, transparent);
    font-size: 11px;
    line-height: 1.5;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    padding: 12px 14px;
  }

  .status-grid > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .status-grid span {
    color: var(--text-muted);
    font-size: 10px;
    text-transform: uppercase;
  }

  .facts-header {
    padding: 8px 14px;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }

  .facts {
    max-height: 300px;
    overflow: auto;
  }

  article {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .fact-main {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }

  .badges span {
    padding: 2px 5px;
    color: var(--text-muted);
    background: var(--bg-primary);
    border-radius: 3px;
    font-size: 9px;
    text-transform: uppercase;
  }

  .badges .active {
    color: var(--success);
  }

  .badges .candidate {
    color: var(--warning);
  }

  .value {
    overflow: hidden;
    color: var(--text-secondary);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  button {
    flex: 0 0 auto;
    padding: 6px 10px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 5px;
    cursor: pointer;
  }

  button:hover:not(:disabled) {
    border-color: var(--accent);
  }

  button.confirm {
    color: var(--danger);
    border-color: var(--danger);
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .empty,
  .error {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px;
    color: var(--text-muted);
  }

  .error {
    color: var(--danger);
  }

  @media (max-width: 800px) {
    .status-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>

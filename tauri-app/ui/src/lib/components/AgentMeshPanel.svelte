<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireOkResult, type DaemonStatusResult } from "../api/daemon";

  interface Specialist {
    agent_key: string;
    display_name: string;
    role: string;
    source: string;
    description: string;
    capabilities: string[];
    executable: boolean;
    handoff_contract: string;
    permissions: {
      action_types: string[];
      confirmation_actions: string[];
      filesystem_read: string[];
      filesystem_write: string[];
      network_domains: string[];
      process_names: string[];
      credential_names: string[];
      clipboard: string[];
      devices: string[];
      authority: string;
    };
    budget: {
      max_tokens_per_task: number;
      max_actions_per_task: number;
      max_latency_ms_per_action: number;
      max_concurrency: number;
    };
    performance: {
      attempts: number;
      successes: number;
      failures: number;
      quality_score: number;
      average_latency_ms: number;
      in_flight: number;
    };
  }

  interface MeshStatus {
    status?: string;
    enabled: boolean;
    total_specialists: number;
    executable_specialists: number;
    external_capability_providers: number;
    registered_action_types: number;
    available_action_types: number;
    coverage_complete: boolean;
    uncovered_action_types: string[];
    sources: Record<string, number>;
    delegation: {
      maximum_depth: number;
      maximum_fanout: number;
      cycle_detection: boolean;
      full_transcript_handoffs: boolean;
      cancellation_propagation: boolean;
      partial_result_recovery: boolean;
      parallel_only_when_explicitly_independent: boolean;
    };
    routing: {
      fixed_numeric_ceiling: boolean;
      selection: string;
      self_reported_success_authority: boolean;
    };
    specialists: Specialist[];
  }

  interface RouteMatch {
    agent_key: string;
    role: string;
    display_name: string;
    matches: number;
    quality_score: number;
  }

  let status = $state<MeshStatus | null>(null);
  let loading = $state(true);
  let error = $state("");
  let sourceFilter = $state("all");
  let routeInput = $state("");
  let routeMatches = $state<RouteMatch[]>([]);
  let routing = $state(false);

  let filtered = $derived(
    (status?.specialists ?? []).filter((specialist) => sourceFilter === "all" || specialist.source === sourceFilter),
  );

  onMount(load);

  async function load() {
    loading = true;
    error = "";
    try {
      const result = requireOkResult(
        (await call("agent_mesh_status")) as MeshStatus & DaemonStatusResult,
        "The daemon did not return the specialist mesh.",
      );
      if (!result.delegation || !result.sources || !Array.isArray(result.uncovered_action_types)) {
        throw new Error("The daemon's specialist mesh status is incomplete.");
      }
      status = result;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      error = /method not found.*agent_mesh/i.test(message)
        ? "This app is connected to an older Heliox daemon. Restart the daemon, then retry."
        : `Could not load the specialist mesh: ${message}`;
    } finally {
      loading = false;
    }
  }

  async function previewRoute() {
    if (!routeInput.trim()) return;
    routing = true;
    error = "";
    try {
      const result = (await call("agent_routing", { input: routeInput })) as {
        status?: string;
        message?: string;
        error?: string;
        orchestrator?: { assigned_specialists?: RouteMatch[] };
      };
      requireOkResult(result, "The daemon did not return a specialist routing preview.");
      if (!result.orchestrator) {
        throw new Error("Specialist routing is unavailable in the connected daemon.");
      }
      routeMatches = result.orchestrator?.assigned_specialists ?? [];
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not preview routing.";
    } finally {
      routing = false;
    }
  }

  function percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }

  function shortName(value: string): string {
    return value.replace("builtin:pilot.agents.", "").replace("plugin:", "");
  }

  function nonEmptyGrants(specialist: Specialist): string[] {
    const permissions = specialist.permissions;
    const grants: string[] = [];
    if (permissions.filesystem_read.length) grants.push(`Read: ${permissions.filesystem_read.join(", ")}`);
    if (permissions.filesystem_write.length) grants.push(`Write: ${permissions.filesystem_write.join(", ")}`);
    if (permissions.network_domains.length) grants.push(`Network: ${permissions.network_domains.join(", ")}`);
    if (permissions.process_names.length) grants.push(`Processes: ${permissions.process_names.join(", ")}`);
    if (permissions.credential_names.length) grants.push(`Credentials: ${permissions.credential_names.join(", ")}`);
    if (permissions.clipboard.length) grants.push(`Clipboard: ${permissions.clipboard.join(", ")}`);
    if (permissions.devices.length) grants.push(`Devices: ${permissions.devices.join(", ")}`);
    return grants;
  }
</script>

<div class="mesh-panel">
  <div class="panel-header">
    <div>
      <h3>Specialist Agent Mesh</h3>
      <span>Capability discovery, outcome-grounded routing, bounded delegation and per-agent budgets</span>
    </div>
    <button onclick={load} disabled={loading}>Refresh</button>
  </div>

  <p class="safety-note">
    Heliox has no fixed agent-count ceiling. A specialist joins only with a distinct capability contract and explicit
    authority. Verified results—not self-reported success—shape routing. Handoffs carry bounded references and partial
    results, never an uncontrolled copy of the full chat.
  </p>

  {#if loading}
    <div class="empty">Discovering specialist contracts...</div>
  {:else if error && !status}
    <div class="error" role="alert">{error}</div>
  {:else if status}
    <div class="summary">
      <div><strong>{status.total_specialists}</strong><span>Total contracts</span></div>
      <div><strong>{status.executable_specialists}</strong><span>Local specialists</span></div>
      <div><strong>{status.external_capability_providers}</strong><span>Guarded providers</span></div>
      <div>
        <strong>{status.registered_action_types} / {status.available_action_types}</strong>
        <span>Actions covered</span>
      </div>
    </div>

    <div class="guardrails">
      <span>Depth ≤ {status.delegation.maximum_depth}</span>
      <span>Fan-out ≤ {status.delegation.maximum_fanout}</span>
      <span>Cycle detection</span>
      <span>Cancellation propagation</span>
      <span>Partial-result recovery</span>
      <span>Explicit independence for parallel work</span>
      <span>No fixed numeric ceiling</span>
      <span class:warning={!status.coverage_complete}>
        {status.coverage_complete
          ? "Complete action coverage"
          : `${status.uncovered_action_types.length} actions uncovered`}
      </span>
    </div>

    {#if !status.coverage_complete}
      <div class="error" role="alert">
        Uncovered action contracts: {status.uncovered_action_types.join(", ")}
      </div>
    {/if}

    <div class="route-preview">
      <label>
        Preview capability routing
        <input
          bind:value={routeInput}
          maxlength="1000"
          placeholder="Example: inspect service logs, then email a summary"
          onkeydown={(event) => event.key === "Enter" && previewRoute()}
        />
      </label>
      <button class="primary" onclick={previewRoute} disabled={routing || !routeInput.trim()}>
        {routing ? "Analyzing..." : "Preview"}
      </button>
      {#if routeMatches.length}
        <div class="matches">
          {#each routeMatches as match}
            <span title={match.agent_key}>
              {match.display_name} · {match.matches} signals · quality {percent(match.quality_score)}
            </span>
          {/each}
        </div>
      {/if}
    </div>

    <div class="catalog-heading">
      <strong>Capability catalog</strong>
      <div class="filters">
        <button class:active={sourceFilter === "all"} onclick={() => (sourceFilter = "all")}>All</button>
        {#each Object.entries(status.sources) as [source, count]}
          <button class:active={sourceFilter === source} onclick={() => (sourceFilter = source)}>
            {source.replaceAll("_", " ")} ({count})
          </button>
        {/each}
      </div>
    </div>

    <div class="catalog">
      {#each filtered as specialist}
        <article>
          <div class="specialist-heading">
            <div>
              <strong>{specialist.display_name}</strong>
              <code title={specialist.agent_key}>{shortName(specialist.agent_key)}</code>
            </div>
            <span class:guarded={!specialist.executable}>
              {specialist.executable ? specialist.role : "guarded provider"}
            </span>
          </div>

          <p>{specialist.description || "No description supplied."}</p>

          <div class="metrics">
            <span>{specialist.capabilities.length} capabilities</span>
            <span>{specialist.performance.attempts} verified outcomes</span>
            <span>Quality {percent(specialist.performance.quality_score)}</span>
            <span>Avg {specialist.performance.average_latency_ms} ms</span>
          </div>

          <details>
            <summary>Contract and budgets</summary>
            <div class="contract">
              <div>
                <strong>Capabilities</strong>
                <p>{specialist.capabilities.join(", ") || "No direct actions"}</p>
              </div>
              <div>
                <strong>Authority</strong>
                <p>{specialist.permissions.authority}</p>
              </div>
              <div>
                <strong>Per-task budgets</strong>
                <p>
                  {specialist.budget.max_tokens_per_task.toLocaleString()} tokens ·
                  {specialist.budget.max_actions_per_task} actions ·
                  {Math.round(specialist.budget.max_latency_ms_per_action / 1000)}s/action ·
                  {specialist.budget.max_concurrency} concurrent
                </p>
              </div>
              <div>
                <strong>Handoff</strong>
                <p>{specialist.handoff_contract.replaceAll("_", " ")}</p>
              </div>
              {#if nonEmptyGrants(specialist).length}
                <div class="wide">
                  <strong>External grants</strong>
                  {#each nonEmptyGrants(specialist) as grant}<p>{grant}</p>{/each}
                </div>
              {/if}
            </div>
          </details>
        </article>
      {:else}
        <p class="empty">No contracts match this source.</p>
      {/each}
    </div>
  {/if}

  {#if error && status}<div class="error" role="alert">{error}</div>{/if}
</div>

<style>
  .mesh-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .specialist-heading,
  .catalog-heading,
  .filters,
  .metrics,
  .route-preview {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
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
  .metrics,
  code {
    color: var(--text-muted);
    font-size: 10px;
  }

  .safety-note {
    margin: 10px 14px 0;
    padding: 10px 12px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-sm);
    font-size: 11px;
    line-height: 1.45;
  }

  .summary {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    padding: 10px 14px;
  }

  .summary > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 9px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .summary strong {
    color: var(--accent-light);
    font-size: 15px;
  }

  .summary span {
    color: var(--text-muted);
    font-size: 10px;
  }

  .guardrails {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0 14px 10px;
  }

  .guardrails span,
  .specialist-heading > span,
  .matches span {
    padding: 3px 7px;
    color: var(--success);
    background: color-mix(in srgb, var(--success) 8%, var(--bg-primary));
    border: 1px solid color-mix(in srgb, var(--success) 35%, var(--border));
    border-radius: 999px;
    font-size: 9px;
    text-transform: capitalize;
  }

  .guardrails span.warning {
    color: var(--warning);
    background: color-mix(in srgb, var(--warning) 8%, var(--bg-primary));
    border-color: color-mix(in srgb, var(--warning) 35%, var(--border));
  }

  .route-preview {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    margin: 0 14px 10px;
    padding: 10px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    color: var(--text-muted);
    font-size: 10px;
  }

  input,
  button {
    box-sizing: border-box;
    padding: 6px 8px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font: inherit;
  }

  input {
    width: 100%;
  }

  button {
    cursor: pointer;
  }

  button.primary {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button:disabled {
    cursor: default;
    opacity: 0.5;
  }

  .matches {
    display: flex;
    grid-column: 1 / -1;
    flex-wrap: wrap;
    gap: 6px;
  }

  .catalog-heading {
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }

  .filters {
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .filters button {
    padding: 3px 7px;
    color: var(--text-muted);
    font-size: 9px;
  }

  .filters button.active {
    color: var(--accent-light);
    border-color: var(--accent);
  }

  .catalog {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    padding: 10px 14px 14px;
  }

  article {
    min-width: 0;
    padding: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .specialist-heading > div {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 2px;
  }

  .specialist-heading strong,
  .specialist-heading code {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .specialist-heading > span.guarded {
    color: var(--warning);
    background: color-mix(in srgb, var(--warning) 8%, var(--bg-primary));
    border-color: color-mix(in srgb, var(--warning) 35%, var(--border));
  }

  article > p,
  .contract p {
    margin: 7px 0;
    color: var(--text-secondary);
    font-size: 10px;
    line-height: 1.4;
  }

  .metrics {
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  details {
    margin-top: 8px;
    padding-top: 7px;
    border-top: 1px solid var(--border);
  }

  summary {
    color: var(--accent-light);
    cursor: pointer;
    font-size: 10px;
  }

  .contract {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px 12px;
    margin-top: 8px;
  }

  .contract strong {
    color: var(--text-muted);
    font-size: 9px;
    text-transform: uppercase;
  }

  .contract .wide {
    grid-column: 1 / -1;
  }

  .contract .wide p {
    margin: 3px 0;
  }

  .empty,
  .error {
    margin: 0;
    padding: 16px 14px;
    color: var(--text-muted);
    font-size: 11px;
  }

  .error {
    color: var(--danger);
  }

  @media (max-width: 760px) {
    .panel-header,
    .specialist-heading,
    .catalog-heading {
      align-items: flex-start;
      flex-direction: column;
    }

    .summary,
    .catalog {
      grid-template-columns: 1fr;
    }

    .route-preview {
      grid-template-columns: 1fr;
    }

    .contract {
      grid-template-columns: 1fr;
    }

    .contract .wide {
      grid-column: auto;
    }
  }
</style>

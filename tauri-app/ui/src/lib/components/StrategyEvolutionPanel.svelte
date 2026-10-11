<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireResultStatus, type DaemonStatusResult } from "../api/daemon";

  type Stage = "candidate" | "replay_passed" | "shadow" | "canary" | "promoted" | "rejected" | "rolled_back";

  interface Candidate {
    candidate_id: string;
    artifact_type: string;
    component: string;
    content?: string;
    rationale: string;
    stage: Stage;
    content_sha256: string;
    created_at: string;
    isolated_evaluation: Record<string, unknown>;
    shadow_evaluation: Record<string, unknown>;
    canary_evaluation: Record<string, unknown>;
  }

  interface StrategyStatus {
    enabled: boolean;
    algorithm: string;
    candidate_counts: Record<Stage, number>;
    pareto_front: string[];
    assignments: Record<
      string,
      { component: string; candidate_id: string; previous_candidate_id: string; version: number }
    >;
    promotion: {
      automatic: boolean;
      shadow_samples_required: number;
      canary_samples_required: number;
      exact_id_confirmation_required: boolean;
    };
  }

  const stages: Array<{ id: Stage; label: string }> = [
    { id: "candidate", label: "Candidate" },
    { id: "replay_passed", label: "Replay passed" },
    { id: "shadow", label: "Shadow" },
    { id: "canary", label: "Canary" },
    { id: "promoted", label: "Admin promoted" },
  ];

  const artifactTypes = [
    "planner_instruction",
    "tool_description",
    "recovery_strategy",
    "context_policy",
    "suggestion_wording",
    "decomposition_policy",
  ];

  let status = $state<StrategyStatus | null>(null);
  let candidates = $state<Candidate[]>([]);
  let loading = $state(true);
  let submitting = $state(false);
  let error = $state("");
  let notice = $state("");
  let showForm = $state(false);
  let artifactType = $state("planner_instruction");
  let component = $state("planner.primary");
  let content = $state("");
  let rationale = $state("");

  onMount(load);

  async function load() {
    loading = true;
    error = "";
    try {
      const [statusResult, candidateResult] = await Promise.all([
        call("strategy_evolution_status") as Promise<StrategyStatus>,
        call("strategy_candidates", { include_content: true, limit: 50 }) as Promise<{ candidates: Candidate[] }>,
      ]);
      if (!statusResult?.candidate_counts || !Array.isArray(statusResult.pareto_front)) {
        throw new Error(
          (statusResult as { message?: string })?.message || "The strategy evolution status is incomplete.",
        );
      }
      status = statusResult;
      candidates = candidateResult.candidates ?? [];
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      error = /method not found.*strategy_/i.test(message)
        ? "This app is connected to an older Heliox daemon. Restart the daemon, then retry."
        : `Could not load strategy evolution: ${message}`;
    } finally {
      loading = false;
    }
  }

  async function propose() {
    if (!content.trim() || !rationale.trim() || !component.trim()) return;
    submitting = true;
    error = "";
    notice = "";
    try {
      requireResultStatus(
        (await call("strategy_propose", {
          artifact_type: artifactType,
          component,
          content,
          rationale,
        })) as DaemonStatusResult,
        "candidate",
        "The daemon did not store the strategy candidate.",
      );
      notice = "Candidate stored in isolation. Production behavior has not changed.";
      content = "";
      rationale = "";
      showForm = false;
      await load();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not create candidate.";
    } finally {
      submitting = false;
    }
  }

  function shortId(value: string): string {
    return value.slice(0, 8);
  }

  function stageLabel(value: Stage): string {
    return (
      stages.find((stage) => stage.id === value)?.label ??
      value
        .split("_")
        .map((word) => word[0]?.toUpperCase() + word.slice(1))
        .join(" ")
    );
  }
</script>

<div class="strategy-panel">
  <div class="panel-header">
    <div>
      <h3>Reflective Strategy Evolution</h3>
      <span class="subtitle">GEPA-style trace reflection with Pareto selection and human-gated promotion</span>
    </div>
    <div class="header-actions">
      <button class="btn-secondary" onclick={load} disabled={loading}>Refresh</button>
      <button class="btn-primary" onclick={() => (showForm = !showForm)}>Propose candidate</button>
    </div>
  </div>

  <p class="safety-note">
    A generated planner instruction, tool description, recovery strategy, or wording change is never used immediately.
    It must pass isolated replay, regression and safety evaluation, shadow comparison, a consented canary, and exact-ID
    admin promotion. Rejected candidates remain audit evidence; rollback is always available.
  </p>

  {#if showForm}
    <div class="candidate-form">
      <label>
        Artifact type
        <select bind:value={artifactType}>
          {#each artifactTypes as artifact}
            <option value={artifact}>{artifact.replaceAll("_", " ")}</option>
          {/each}
        </select>
      </label>
      <label>
        Component
        <input bind:value={component} maxlength="128" placeholder="planner.primary" />
      </label>
      <label class="wide">
        Complete candidate text
        <textarea bind:value={content} rows="4" maxlength="12000" placeholder="Proposed replacement strategy"
        ></textarea>
      </label>
      <label class="wide">
        Trace-grounded rationale
        <textarea bind:value={rationale} rows="2" maxlength="4000" placeholder="Which failure does this address?"
        ></textarea>
      </label>
      <div class="form-actions wide">
        <span>Submitting creates an inert candidate only.</span>
        <button
          class="btn-primary"
          onclick={propose}
          disabled={submitting || !content.trim() || !rationale.trim() || !component.trim()}
        >
          {submitting ? "Storing..." : "Store candidate"}
        </button>
      </div>
    </div>
  {/if}

  {#if loading}
    <div class="empty">Loading strategy archive...</div>
  {:else if error && !status}
    <div class="unavailable" role="alert">
      <strong>Strategy archive unavailable</strong>
      <span>{error}</span>
    </div>
  {:else if status}
    <div class="pipeline" aria-label="Strategy promotion pipeline">
      {#each stages as stage, index}
        <div class="stage">
          <strong>{status.candidate_counts[stage.id] ?? 0}</strong>
          <span>{stage.label}</span>
        </div>
        {#if index < stages.length - 1}<span class="arrow">→</span>{/if}
      {/each}
    </div>

    <div class="status-strip">
      <span><strong>{status.pareto_front.length}</strong> Pareto candidates</span>
      <span><strong>{Object.keys(status.assignments).length}</strong> active assignments</span>
      <span><strong>{status.promotion.shadow_samples_required}</strong> shadow samples minimum</span>
      <span><strong>{status.promotion.canary_samples_required}</strong> canary samples minimum</span>
      <span class="safe">Automatic promotion: never</span>
    </div>

    <div class="candidate-list">
      <div class="list-header">
        <strong>Candidate archive</strong>
        <span>{candidates.length} shown</span>
      </div>
      {#if candidates.length}
        {#each candidates as candidate}
          <article>
            <div class="candidate-heading">
              <div>
                <code>{shortId(candidate.candidate_id)}</code>
                <strong>{candidate.component}</strong>
                <span>{candidate.artifact_type.replaceAll("_", " ")}</span>
              </div>
              <span class:promoted={candidate.stage === "promoted"} class:rejected={candidate.stage === "rejected"}>
                {stageLabel(candidate.stage)}
              </span>
            </div>
            <p>{candidate.rationale || "No rationale supplied."}</p>
            {#if candidate.content}<pre>{candidate.content}</pre>{/if}
            <div class="candidate-meta">
              <span>SHA-256 {candidate.content_sha256.slice(0, 12)}</span>
              {#if status.pareto_front.includes(candidate.candidate_id)}<span class="pareto">Pareto front</span>{/if}
              {#if status.assignments[candidate.component]?.candidate_id === candidate.candidate_id}
                <span class="active">Production v{status.assignments[candidate.component].version}</span>
              {/if}
            </div>
          </article>
        {/each}
      {:else}
        <p class="empty">No candidate has been proposed. Production uses its shipped strategies.</p>
      {/if}
    </div>
  {/if}

  {#if notice}<div class="notice" role="status">{notice}</div>{/if}
  {#if error && status}<div class="error" role="alert">{error}</div>{/if}
</div>

<style>
  .strategy-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .header-actions,
  .form-actions,
  .candidate-heading,
  .candidate-meta,
  .list-header {
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
    font-weight: 600;
  }

  .subtitle,
  .status-strip,
  .candidate-meta,
  .form-actions,
  .list-header span {
    color: var(--text-muted);
    font-size: 11px;
  }

  .safety-note {
    margin: 10px 14px 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-sm);
  }

  .candidate-form {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 12px 14px 0;
    padding: 12px;
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 5px;
    color: var(--text-muted);
    font-size: 11px;
  }

  .wide {
    grid-column: 1 / -1;
  }

  input,
  select,
  textarea {
    width: 100%;
    box-sizing: border-box;
    padding: 7px 8px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font: inherit;
  }

  textarea {
    resize: vertical;
  }

  .pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 14px;
  }

  .stage {
    display: flex;
    min-width: 92px;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .stage strong {
    color: var(--accent-light);
  }

  .stage span,
  .arrow {
    color: var(--text-muted);
    font-size: 10px;
  }

  .status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    padding: 0 14px 12px;
  }

  .status-strip strong {
    color: var(--text-primary);
  }

  .safe,
  .active,
  .promoted {
    color: var(--success) !important;
  }

  .candidate-list {
    margin: 0 14px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  .list-header {
    padding: 8px 10px;
    background: var(--bg-tertiary);
  }

  article {
    padding: 10px;
    border-top: 1px solid var(--border);
  }

  .candidate-heading > div {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }

  .candidate-heading span {
    color: var(--text-muted);
    font-size: 10px;
    text-transform: capitalize;
  }

  .rejected {
    color: var(--danger) !important;
  }

  article p {
    margin: 7px 0;
    color: var(--text-secondary);
    font-size: 11px;
  }

  pre {
    max-height: 120px;
    overflow: auto;
    margin: 7px 0;
    padding: 8px;
    white-space: pre-wrap;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    font-size: 10px;
  }

  code,
  .pareto {
    color: var(--accent-light);
  }

  .empty,
  .unavailable,
  .notice,
  .error {
    padding: 16px 14px;
    color: var(--text-muted);
    font-size: 12px;
  }

  .unavailable,
  .error {
    color: var(--danger);
  }

  .unavailable {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .notice {
    color: var(--success);
  }

  .btn-secondary,
  .btn-primary {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 5px 10px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    cursor: pointer;
  }

  .btn-primary {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  @media (max-width: 760px) {
    .panel-header,
    .candidate-heading {
      align-items: flex-start;
      flex-direction: column;
    }

    .candidate-form {
      grid-template-columns: 1fr;
    }

    .wide {
      grid-column: auto;
    }

    .pipeline {
      align-items: stretch;
      flex-direction: column;
    }

    .arrow {
      transform: rotate(90deg);
      align-self: center;
    }
  }
</style>

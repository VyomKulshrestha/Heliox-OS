<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireResultStatus, type DaemonStatusResult } from "../api/daemon";

  type RunState = "collecting" | "evaluating" | "evaluated" | "promotion_requested" | "rejected";
  type CandidateState = "proposed" | "eligible" | "rejected";

  interface EvolutionRun {
    run_id: string;
    problem: string;
    base_commit: string;
    profile: string;
    state: RunState;
    baseline_evaluation: { score?: number; passed?: boolean };
    created_at: string;
  }

  interface Candidate {
    candidate_id: string;
    run_id: string;
    title: string;
    rationale: string;
    patch_sha256: string;
    touched_files: string[];
    diversity_score: number;
    state: CandidateState;
    deterministic_evaluation: { score?: number; passed?: boolean };
    agent_evaluation: { quality_score?: number; risk_level?: string };
  }

  interface HarnessStatus {
    enabled: boolean;
    runner: {
      available: boolean;
      backend: string;
      image: string;
      image_id?: string;
      reason?: string;
      fallback?: string;
    };
    profiles: Record<string, unknown[]>;
    run_counts: Record<RunState, number>;
    candidate_counts: Record<CandidateState, number>;
    restrictions: Record<string, boolean | number>;
  }

  let status = $state<HarnessStatus | null>(null);
  let runs = $state<EvolutionRun[]>([]);
  let candidates = $state<Candidate[]>([]);
  let selectedRunId = $state("");
  let problem = $state("");
  let candidateCount = $state(3);
  let actor = $state("");
  let confirmation = $state("");
  let loading = $state(true);
  let busy = $state("");
  let error = $state("");
  let notice = $state("");

  let selectedRun = $derived(runs.find((run) => run.run_id === selectedRunId) ?? null);
  let selectedCandidates = $derived(candidates.filter((candidate) => candidate.run_id === selectedRunId));

  onMount(load);

  async function load() {
    loading = true;
    error = "";
    try {
      const [statusResult, runResult] = await Promise.all([
        call("evolution_status") as Promise<HarnessStatus>,
        call("evolution_runs", { limit: 50 }) as Promise<{ runs: EvolutionRun[] }>,
      ]);
      if (!statusResult?.runner) {
        throw new Error((statusResult as { message?: string })?.message || "The evolution status is incomplete.");
      }
      status = statusResult;
      runs = runResult.runs ?? [];
      if (!selectedRunId && runs.length) selectedRunId = runs[0].run_id;
      await loadCandidates();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      error = /method not found.*evolution_/i.test(message)
        ? "This app is connected to an older Heliox daemon. Restart the daemon, then retry."
        : `Could not load the evolution harness: ${message}`;
    } finally {
      loading = false;
    }
  }

  async function loadCandidates() {
    const result = (await call("evolution_candidates", {
      run_id: selectedRunId || undefined,
      include_patch: false,
      limit: 100,
    })) as { candidates: Candidate[] };
    candidates = result.candidates ?? [];
  }

  async function chooseRun(runId: string) {
    selectedRunId = runId;
    confirmation = "";
    error = "";
    try {
      await loadCandidates();
    } catch (cause) {
      candidates = [];
      error = `Could not load candidates: ${cause instanceof Error ? cause.message : String(cause)}`;
    }
  }

  async function createRun() {
    if (!problem.trim()) return;
    busy = "create";
    error = "";
    notice = "";
    try {
      const result = requireResultStatus(
        (await call("evolution_create_run", {
          problem,
          profile: "python",
        })) as DaemonStatusResult & { run?: EvolutionRun },
        "collecting",
        "The daemon did not create the evolution run.",
      );
      if (!result.run?.run_id) throw new Error("The daemon returned an incomplete evolution run.");
      selectedRunId = result.run.run_id;
      problem = "";
      notice = "Run created at the current Git commit. No code was changed.";
      await load();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not create the run.";
    } finally {
      busy = "";
    }
  }

  async function generate() {
    if (!selectedRunId) return;
    busy = "generate";
    error = "";
    notice = "";
    try {
      requireResultStatus(
        (await call("evolution_generate_candidates", {
          run_id: selectedRunId,
          count: candidateCount,
        })) as DaemonStatusResult,
        "collecting",
        "The daemon did not generate the candidates.",
      );
      notice = "Diverse patches were archived as inert candidates. Nothing has executed.";
      await load();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Candidate generation failed.";
    } finally {
      busy = "";
    }
  }

  async function evaluate() {
    if (!selectedRunId || !status?.runner.available || selectedCandidates.length < 2) return;
    busy = "evaluate";
    error = "";
    notice = "";
    try {
      requireResultStatus(
        (await call("evolution_evaluate", { run_id: selectedRunId })) as DaemonStatusResult,
        "evaluated",
        "The daemon did not complete the isolated evaluation.",
      );
      notice = "Baseline and candidates were evaluated in disposable, no-network containers.";
      await load();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Isolated evaluation failed.";
    } finally {
      busy = "";
    }
  }

  async function requestPromotion(candidateId: string) {
    if (!actor.trim() || confirmation !== candidateId) return;
    busy = candidateId;
    error = "";
    notice = "";
    try {
      requireResultStatus(
        (await call("evolution_request_promotion", {
          candidate_id: candidateId,
          actor,
          confirmation,
        })) as DaemonStatusResult,
        "pending_external_review",
        "The daemon did not archive the promotion request.",
      );
      confirmation = "";
      notice = "Evidence archived for external review. No merge, push, release, or live change occurred.";
      await load();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not archive the promotion request.";
    } finally {
      busy = "";
    }
  }

  function shortId(value: string): string {
    return value.slice(0, 8);
  }
</script>

<div class="harness-panel">
  <div class="panel-header">
    <div>
      <h3>Evolutionary Engineering Harness</h3>
      <span>Generate diverse patches, evaluate in locked containers, and archive evidence for human review</span>
    </div>
    <button onclick={load} disabled={loading || !!busy}>Refresh</button>
  </div>

  <p class="safety-note">
    Candidates never edit the installed app. They run only in disposable Git worktrees with networking disabled, a
    read-only container root, no inherited credentials, and bounded CPU, memory and processes. Heliox cannot merge,
    push, release, or promote them automatically.
  </p>

  {#if loading}
    <div class="empty">Checking the isolation engine and archive...</div>
  {:else if error && !status}
    <div class="error" role="alert">{error}</div>
  {:else if status}
    <div class:ready={status.runner.available} class="runner">
      <div>
        <strong>{status.runner.available ? "Isolation ready" : "Isolation unavailable"}</strong>
        <span>{status.runner.backend} · {status.runner.image}</span>
      </div>
      <code>{status.runner.available ? shortId(status.runner.image_id ?? "") : status.runner.reason}</code>
    </div>

    <div class="guardrails">
      <span>No network</span>
      <span>No credentials</span>
      <span>No host fallback</span>
      <span>No automatic promotion</span>
      <span>2–8 candidates</span>
    </div>

    <div class="create-form">
      <label>
        Bounded failure or opportunity
        <textarea
          bind:value={problem}
          rows="2"
          maxlength="8000"
          placeholder="Describe the observed failure, expected behavior, and evidence..."></textarea>
      </label>
      <button class="primary" onclick={createRun} disabled={!problem.trim() || !!busy}>
        {busy === "create" ? "Creating..." : "Create inert run"}
      </button>
    </div>

    <div class="workspace">
      <aside>
        <div class="list-title"><strong>Run archive</strong><span>{runs.length}</span></div>
        {#each runs as run}
          <button class:active={run.run_id === selectedRunId} class="run" onclick={() => chooseRun(run.run_id)}>
            <span>{run.problem}</span>
            <small>{shortId(run.run_id)} · {run.state.replaceAll("_", " ")}</small>
          </button>
        {:else}
          <p class="empty">No evolution runs yet.</p>
        {/each}
      </aside>

      <div class="details">
        {#if selectedRun}
          <div class="run-heading">
            <div>
              <strong>{selectedRun.problem}</strong>
              <span>Base {shortId(selectedRun.base_commit)} · {selectedRun.profile}</span>
            </div>
            <span class="state">{selectedRun.state.replaceAll("_", " ")}</span>
          </div>

          {#if selectedRun.state === "collecting"}
            <div class="actions">
              <label>
                Candidates
                <input type="number" min="2" max="8" bind:value={candidateCount} />
              </label>
              <button onclick={generate} disabled={!!busy}>
                {busy === "generate" ? "Generating..." : "Generate candidates"}
              </button>
              <button
                class="primary"
                onclick={evaluate}
                disabled={!!busy || !status.runner.available || selectedCandidates.length < 2}
              >
                {busy === "evaluate" ? "Evaluating..." : "Evaluate in isolation"}
              </button>
            </div>
          {/if}

          <div class="candidate-list">
            {#each selectedCandidates as candidate}
              <article>
                <div class="candidate-heading">
                  <div>
                    <code>{shortId(candidate.candidate_id)}</code>
                    <strong>{candidate.title}</strong>
                  </div>
                  <span class:eligible={candidate.state === "eligible"} class:rejected={candidate.state === "rejected"}>
                    {candidate.state}
                  </span>
                </div>
                <p>{candidate.rationale}</p>
                <div class="meta">
                  <span>Diversity {Math.round(candidate.diversity_score * 100)}%</span>
                  <span>Checks {Math.round((candidate.deterministic_evaluation.score ?? 0) * 100)}%</span>
                  <span>Risk {candidate.agent_evaluation.risk_level ?? "not reviewed"}</span>
                  <span>{candidate.touched_files.length} files</span>
                </div>
                {#if candidate.state === "eligible"}
                  <div class="promotion">
                    <input bind:value={actor} maxlength="128" placeholder="Reviewer name" />
                    <input bind:value={confirmation} placeholder={`Type ${candidate.candidate_id}`} />
                    <button
                      onclick={() => requestPromotion(candidate.candidate_id)}
                      disabled={busy === candidate.candidate_id ||
                        !actor.trim() ||
                        confirmation !== candidate.candidate_id}
                    >
                      {busy === candidate.candidate_id ? "Archiving..." : "Request external review"}
                    </button>
                  </div>
                {/if}
              </article>
            {:else}
              <p class="empty">No candidates in this run.</p>
            {/each}
          </div>
        {:else}
          <p class="empty">Create or select a run to inspect its evidence.</p>
        {/if}
      </div>
    </div>
  {/if}

  {#if notice}<div class="notice" role="status">{notice}</div>{/if}
  {#if error && status}<div class="error" role="alert">{error}</div>{/if}
</div>

<style>
  .harness-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .runner,
  .run-heading,
  .candidate-heading,
  .meta,
  .actions,
  .promotion,
  .list-title {
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
  .runner span,
  .run-heading span,
  .meta,
  small,
  .list-title span {
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

  .runner {
    margin: 10px 14px 0;
    padding: 9px 11px;
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 8%, var(--bg-primary));
    border: 1px solid color-mix(in srgb, var(--danger) 40%, var(--border));
    border-radius: var(--radius-sm);
  }

  .runner.ready {
    color: var(--success);
    background: color-mix(in srgb, var(--success) 8%, var(--bg-primary));
    border-color: color-mix(in srgb, var(--success) 40%, var(--border));
  }

  .runner > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .guardrails {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    padding: 9px 14px;
  }

  .guardrails span,
  .state {
    padding: 3px 7px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 10px;
    text-transform: capitalize;
  }

  .create-form {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 10px;
    padding: 0 14px 12px;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    color: var(--text-muted);
    font-size: 10px;
  }

  input,
  textarea,
  button {
    box-sizing: border-box;
    padding: 6px 8px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font: inherit;
  }

  textarea {
    width: 100%;
    resize: vertical;
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

  .workspace {
    display: grid;
    grid-template-columns: minmax(180px, 0.3fr) minmax(0, 1fr);
    margin: 0 14px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  aside {
    min-width: 0;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
  }

  .list-title,
  .run {
    width: 100%;
    padding: 8px 10px;
    border: 0;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    text-align: left;
  }

  .run {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 3px;
  }

  .run span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run.active {
    background: color-mix(in srgb, var(--accent) 12%, var(--bg-primary));
    border-left: 2px solid var(--accent);
  }

  .details {
    min-width: 0;
  }

  .run-heading,
  .actions,
  article {
    padding: 10px;
    border-bottom: 1px solid var(--border);
  }

  .run-heading > div {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 3px;
  }

  .actions {
    justify-content: flex-start;
  }

  .actions input {
    width: 76px;
  }

  .candidate-heading > div {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  article p {
    margin: 7px 0;
    color: var(--text-secondary);
    font-size: 11px;
  }

  code {
    color: var(--accent-light);
  }

  .eligible {
    color: var(--success);
  }

  .rejected,
  .error {
    color: var(--danger);
  }

  .meta {
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  .promotion {
    margin-top: 9px;
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  .promotion input {
    min-width: 160px;
    flex: 1;
  }

  .promotion input:nth-child(2) {
    min-width: 260px;
  }

  .empty,
  .error,
  .notice {
    margin: 0;
    padding: 15px 14px;
    color: var(--text-muted);
    font-size: 11px;
  }

  .notice {
    color: var(--success);
  }

  @media (max-width: 760px) {
    .panel-header,
    .runner,
    .run-heading,
    .candidate-heading {
      align-items: flex-start;
      flex-direction: column;
    }

    .create-form,
    .workspace {
      grid-template-columns: 1fr;
    }

    aside {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }
  }
</style>

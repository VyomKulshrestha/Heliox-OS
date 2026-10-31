<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import {
    call,
    onNotification,
    offNotification,
    requireOkResult,
    requireResultStatus,
    type DaemonStatusResult,
  } from "../api/daemon";

  interface WorkflowStep {
    index: number;
    title: string;
    description: string;
    status: string;
    output: string;
    error: string;
  }

  interface Workflow {
    workflow_id: string;
    goal: string;
    invocation_source: string;
    steps: WorkflowStep[];
    current_step: number;
    state: string;
    updated_at: string;
  }

  let workflows = $state<Workflow[]>([]);
  let loading = $state(true);
  let goal = $state("");
  let source = $state<"voice" | "gesture">("voice");
  let includeTerminal = $state(false);
  let submitting = $state(false);
  let error = $state("");
  let notificationHandler: ((method: string, params: unknown) => void) | null = null;

  onMount(() => {
    loadWorkflows();
    notificationHandler = (method) => {
      if (method === "voice_gesture_workflow_state") loadWorkflows();
    };
    onNotification(notificationHandler);
    return () => {
      if (notificationHandler) offNotification(notificationHandler);
    };
  });

  async function loadWorkflows() {
    try {
      const result = requireOkResult(
        (await call("voice_gesture_workflow_list", { include_terminal: includeTerminal })) as DaemonStatusResult & {
          workflows: Workflow[];
        },
        "Voice/gesture workflows are unavailable.",
      ) as DaemonStatusResult & {
        workflows: Workflow[];
      };
      workflows = result.workflows ?? [];
      error = "";
    } catch (cause) {
      workflows = [];
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      loading = false;
    }
  }

  async function submitWorkflow() {
    const trimmedGoal = goal.trim();
    if (!trimmedGoal || submitting) return;
    submitting = true;
    error = "";
    try {
      requireResultStatus(
        (await call("voice_gesture_workflow_submit", {
          goal: trimmedGoal,
          invocation_source: source,
        })) as DaemonStatusResult,
        "submitted",
        "Workflow was not submitted",
      );
      goal = "";
      await loadWorkflows();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      submitting = false;
    }
  }

  async function controlWorkflow(method: string, id: string, fallbackMessage: string) {
    error = "";
    try {
      requireOkResult((await call(method, { workflow_id: id })) as DaemonStatusResult, fallbackMessage);
      await loadWorkflows();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : fallbackMessage;
    }
  }

  function pause(id: string) {
    return controlWorkflow("voice_gesture_workflow_pause", id, "Workflow could not be paused.");
  }

  function resume(id: string) {
    return controlWorkflow("voice_gesture_workflow_resume", id, "Workflow could not be resumed.");
  }

  function cancel(id: string) {
    return controlWorkflow("voice_gesture_workflow_cancel", id, "Workflow could not be cancelled.");
  }

  function stateClass(state: string): string {
    if (state === "paused" || state === "waiting_for_trigger") return "state-paused";
    if (state === "running" || state === "decomposing" || state === "pending") return "state-running";
    return "state-other";
  }

  function canCancel(state: string): boolean {
    return ["pending", "decomposing", "running", "paused", "waiting_for_trigger"].includes(state);
  }
</script>

<div class="workflow-status">
  <div class="status-header">
    <h3>{$_("settings.voice_gesture_workflows")}</h3>
    <span class="count">{workflows.length}</span>
  </div>

  <p class="workflow-note">{$_("settings.voice_gesture_workflows_desc")}</p>

  <form
    class="workflow-create"
    onsubmit={(event) => {
      event.preventDefault();
      void submitWorkflow();
    }}
  >
    <input
      class="workflow-input"
      bind:value={goal}
      placeholder={$_("settings.voice_gesture_workflows_placeholder")}
      aria-label={$_("settings.voice_gesture_workflows_placeholder")}
    />
    <select class="source-select" bind:value={source} aria-label="Workflow source">
      <option value="voice">Voice</option>
      <option value="gesture">Gesture</option>
    </select>
    <button class="btn-save" type="submit" disabled={!goal.trim() || submitting}>
      {submitting ? $_("settings.saving") : $_("settings.voice_gesture_workflows_start")}
    </button>
  </form>

  <label class="history-toggle">
    <input type="checkbox" bind:checked={includeTerminal} onchange={loadWorkflows} />
    <span>{$_("settings.voice_gesture_workflows_history")}</span>
  </label>

  {#if error}
    <div class="panel-error" role="alert">{error}</div>
  {/if}

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if workflows.length === 0}
    <div class="empty">{$_("settings.voice_gesture_workflows_empty")}</div>
  {:else}
    <div class="workflow-list">
      {#each workflows as wf}
        <div class="workflow-card">
          <div class="workflow-card-header">
            <span class="state-tag {stateClass(wf.state)}">{wf.state}</span>
            <span class="source-tag">{wf.invocation_source}</span>
          </div>
          <div class="workflow-goal">{wf.goal}</div>
          <div class="workflow-progress">
            {wf.steps.filter((s) => s.status === "success").length} / {wf.steps.length}
            {$_("settings.voice_gesture_workflows_steps")}
          </div>
          {#if wf.steps[wf.current_step]}
            <div class="current-step">{wf.steps[wf.current_step].title}: {wf.steps[wf.current_step].status}</div>
          {/if}
          <div class="workflow-actions">
            {#if wf.state === "running" || wf.state === "pending" || wf.state === "decomposing"}
              <button class="btn-save" onclick={() => pause(wf.workflow_id)}>{$_("settings.pause")}</button>
            {/if}
            {#if wf.state === "paused" || wf.state === "waiting_for_trigger"}
              <button class="btn-save" onclick={() => resume(wf.workflow_id)}>{$_("settings.resume")}</button>
            {/if}
            {#if canCancel(wf.state)}
              <button class="btn-remove" onclick={() => cancel(wf.workflow_id)}>{$_("settings.cancel")}</button>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .workflow-status {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .status-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .empty {
    padding: 12px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }

  .workflow-note {
    margin: 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .workflow-create {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) auto auto;
    gap: 8px;
  }

  .workflow-input,
  .source-select {
    min-width: 0;
    padding: 7px 9px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 12px;
  }

  .history-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
  }

  .panel-error {
    padding: 8px 10px;
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 6px;
    background: rgba(248, 113, 113, 0.08);
    color: var(--danger);
    font-size: 11px;
  }

  .btn-save {
    padding: 6px 14px;
    border: 0;
    border-radius: 6px;
    background: var(--accent);
    color: var(--on-accent);
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
    cursor: pointer;
  }

  .btn-save:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .workflow-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .workflow-card {
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 13px;
  }

  .workflow-card-header {
    display: flex;
    gap: 6px;
    margin-bottom: 4px;
  }

  .state-tag,
  .source-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .state-paused {
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .state-running {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .workflow-goal {
    font-weight: 500;
    margin-bottom: 2px;
  }

  .workflow-progress {
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .current-step {
    margin-bottom: 6px;
    color: var(--text-secondary);
    font-size: 11px;
  }

  .workflow-actions {
    display: flex;
    gap: 6px;
  }

  .btn-remove {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--danger);
    cursor: pointer;
  }

  @media (max-width: 720px) {
    .workflow-create {
      grid-template-columns: 1fr;
    }
  }
</style>

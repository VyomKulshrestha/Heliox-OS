<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call, onNotification, offNotification } from "../api/daemon";

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
      const result = (await call("voice_gesture_workflow_list", { include_terminal: false })) as {
        workflows: Workflow[];
      };
      workflows = result.workflows ?? [];
    } catch {
      workflows = [];
    } finally {
      loading = false;
    }
  }

  async function pause(id: string) {
    await call("voice_gesture_workflow_pause", { workflow_id: id });
    await loadWorkflows();
  }

  async function resume(id: string) {
    await call("voice_gesture_workflow_resume", { workflow_id: id });
    await loadWorkflows();
  }

  async function cancel(id: string) {
    await call("voice_gesture_workflow_cancel", { workflow_id: id });
    await loadWorkflows();
  }

  function stateClass(state: string): string {
    if (state === "paused" || state === "waiting_for_trigger") return "state-paused";
    if (state === "running" || state === "decomposing" || state === "pending") return "state-running";
    return "state-other";
  }
</script>

<div class="workflow-status">
  <div class="status-header">
    <h3>{$_('settings.voice_gesture_workflows')}</h3>
    <span class="count">{workflows.length}</span>
  </div>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if workflows.length === 0}
    <div class="empty">{$_('settings.voice_gesture_workflows_empty')}</div>
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
            {wf.steps.filter((s) => s.status === 'success').length} / {wf.steps.length} {$_('settings.voice_gesture_workflows_steps')}
          </div>
          <div class="workflow-actions">
            {#if wf.state === "running" || wf.state === "pending" || wf.state === "decomposing"}
              <button class="btn-save" onclick={() => pause(wf.workflow_id)}>{$_('settings.pause')}</button>
            {/if}
            {#if wf.state === "paused" || wf.state === "waiting_for_trigger"}
              <button class="btn-save" onclick={() => resume(wf.workflow_id)}>{$_('settings.resume')}</button>
            {/if}
            <button class="btn-remove" onclick={() => cancel(wf.workflow_id)}>{$_('settings.cancel')}</button>
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
    font-size: 14px;
    font-weight: 600;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
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
</style>

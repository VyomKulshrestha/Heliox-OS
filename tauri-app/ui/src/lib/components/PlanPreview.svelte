<script lang="ts">
  import type { Plan } from "../stores/session";

  interface Props {
    plan: Plan;
  }

  let { plan }: Props = $props();

  function tierLabel(action: { requires_root: boolean; destructive: boolean }): string {
    if (action.requires_root) return "ROOT";
    if (action.destructive) return "DESTRUCTIVE";
    return "SAFE";
  }

  function actionLabel(action: { action_type: string; dry_run?: boolean }, planDryRun = false): string {
    return action.dry_run || planDryRun ? `${action.action_type} (dry run)` : action.action_type;
  }

  function tierClass(action: { requires_root: boolean; destructive: boolean }): string {
    if (action.requires_root) return "tier-root";
    if (action.destructive) return "tier-destructive";
    return "tier-safe";
  }
</script>

<div class="plan-preview">
  <div class="plan-header">
    <span class="plan-icon">&#9654;</span>
    <span class="plan-title">Execution Plan</span>
  </div>

  {#if plan.explanation}
    <p class="explanation">{plan.explanation}</p>
  {/if}

  <div class="action-list">
    {#each plan.actions as action, i}
      <div class="action-item">
        <span class="action-index">{i + 1}</span>
        <div class="action-detail">
        <span class="action-type">{actionLabel(action, Boolean(plan.dry_run))}</span>
          <span class="action-target">{action.target}</span>
        </div>
        <span class="tier-badge {tierClass(action)}">{tierLabel(action)}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .plan-preview {
    margin: 12px 16px 0;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .plan-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }

  .plan-icon {
    color: var(--accent);
    font-size: 10px;
  }

  .plan-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
  }

  .explanation {
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.4;
    border-bottom: 1px solid var(--border);
  }

  .action-list {
    padding: 6px 0;
  }

  .action-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 14px;
  }

  .action-item:hover {
    background: var(--bg-hover);
  }

  .action-index {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-primary);
    border-radius: 50%;
    flex-shrink: 0;
  }

  .action-detail {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .action-type {
    font-size: 13px;
    font-weight: 500;
    font-family: var(--font-mono);
    color: var(--text-primary);
  }

  .action-target {
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tier-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
  }

  .tier-safe {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .tier-destructive {
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .tier-root {
    background: var(--danger-bg);
    color: var(--danger);
  }
</style>

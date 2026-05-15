<script lang="ts">
  import type { PlanAction } from "../stores/session";

  interface Props {
    actions: PlanAction[];
    onconfirm: () => void;
    ondeny: () => void;
  }

  let { actions, onconfirm, ondeny }: Props = $props();
</script>

<div class="confirm-overlay">
  <div class="confirm-dialog">
    <div class="confirm-header">
      <span class="warn-icon">&#9888;</span>
      <span>Confirmation Required</span>
    </div>

    <p class="confirm-body">
      The following actions require your approval before execution:
    </p>

    <ul class="confirm-list">
      {#each actions as action}
        <li>
          <strong>{action.action_type}</strong> on
          <code>{action.target}</code>
          {#if action.requires_root}
            <span class="root-tag">ROOT</span>
          {/if}
          {#if action.destructive}
            <span class="destructive-tag">DESTRUCTIVE</span>
          {/if}
        </li>
      {/each}
    </ul>

    <div class="confirm-actions">
      <button class="btn-deny" title="Deny" onclick={ondeny}>Deny</button>
      <button class="btn-confirm" title="Confirm" onclick={onconfirm}>Approve & Execute</button>
    </div>
  </div>
</div>

<style>
  .confirm-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .confirm-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 480px;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .confirm-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .warn-icon {
    color: var(--warning);
    font-size: 20px;
  }

  .confirm-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 14px;
    line-height: 1.5;
  }

  .confirm-list {
    list-style: none;
    padding: 0;
    margin-bottom: 20px;
  }

  .confirm-list li {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .confirm-list code {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
  }

  .root-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--danger-bg);
    color: var(--danger);
  }

  .destructive-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .btn-deny {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-deny:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-confirm {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .btn-confirm:hover {
    background: var(--accent-hover);
  }
</style>

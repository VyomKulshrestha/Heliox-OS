<script lang="ts">
  import type { PlanAction } from "../stores/session";
  import { _ } from "svelte-i18n";

  interface Props {
    actions: PlanAction[];
    onconfirm: (approvedIndices: number[]) => void;
    ondeny: () => void;
    submitting?: boolean;
    error?: string;
  }

  let { actions, onconfirm, ondeny, submitting = false, error = "" }: Props = $props();

  let approved = $state<boolean[]>([]);
  let anyApproved = $derived(approved.some((v) => v));

  $effect(() => {
    if (approved.length !== actions.length) {
      approved = actions.map(() => true);
    }
  });

  function toggle(i: number) {
    if (submitting) return;
    approved[i] = !approved[i];
  }

  function handleConfirm() {
    if (submitting) return;
    const approvedIndices = actions
      .map((a, i) => (approved[i] ? (a.index ?? i) : -1))
      .filter((i) => i !== -1);
    onconfirm(approvedIndices);
  }
</script>

<div class="confirm-overlay">
  <div class="confirm-dialog">
    <div class="confirm-header">
      <span class="warn-icon">&#9888;</span>
      <span>{$_('confirm.title')}</span>
    </div>

    <p class="confirm-body">
      {$_('confirm.body')}
    </p>

    <ul class="confirm-list">
      {#each actions as action, i}
        <li>
          <label class="confirm-checkbox">
            <input type="checkbox" checked={approved[i]} disabled={submitting} onchange={() => toggle(i)} />
          </label>
          <strong>{action.action_type}</strong> on
          <code>{action.target}</code>
          {#if action.requires_root}
            <span class="root-tag">{$_('tier.root')}</span>
          {/if}
          {#if action.destructive}
            <span class="destructive-tag">{$_('tier.destructive')}</span>
          {/if}
          {#if action.irreversible}
            <span class="irreversible-tag">{$_('tier.irreversible')}</span>
          {/if}
        </li>
      {/each}
    </ul>

    {#if error}
      <p class="confirm-error" role="alert">{error}</p>
    {/if}

    <div class="confirm-actions">
      <button class="btn-deny" title={$_('confirm.deny')} disabled={submitting} onclick={ondeny}>{$_('confirm.deny')}</button>
      <button class="btn-confirm" title={$_('confirm.approve')} disabled={!anyApproved || submitting} onclick={handleConfirm}>
        {submitting ? $_('confirm.submitting') : $_('confirm.approve')}
      </button>
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

  .irreversible-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--danger);
    color: white;
  }

  .confirm-checkbox {
    display: flex;
    align-items: center;
    margin-right: 2px;
  }

  .confirm-checkbox input {
    width: 15px;
    height: 15px;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .confirm-error {
    margin: -8px 0 16px;
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    background: var(--danger-bg);
    color: var(--danger);
    font-size: 12px;
    line-height: 1.4;
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

  .btn-confirm:disabled {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    cursor: not-allowed;
  }
</style>

<script lang="ts">
  /**
   * SupervisionAlertDialog — advisory-only modal shown when
   * UserSupervisionEngine detects a risk pattern or a sustained cognitive
   * check-in signal from the user's OWN independent activity. Unlike
   * InterruptDialog.svelte, there is nothing to approve/deny here -- Heliox
   * cannot block the user's own OS-level input, it only observed a copy via
   * the hook. Dismiss is purely local UI state, no RPC call.
   *
   * Self-guarding, same pattern as InterruptDialog.svelte/
   * BudgetExceededDialog.svelte: always mounted, renders nothing unless
   * $supervision.active.
   */

  import { _ } from "svelte-i18n";
  import { supervision } from "../stores/supervision";
</script>

{#if $supervision.active}
  <div class="supervision-overlay" role="dialog" aria-modal="true" aria-labelledby="supervision-dialog-title">
    <div class="supervision-dialog">
      <div class="supervision-header">
        <span class="warn-icon">&#9888;</span>
        <span id="supervision-dialog-title">
          {$supervision.kind === "risk" ? $_('supervision_alert.risk_title') : $_('supervision_alert.coaching_title')}
        </span>
      </div>

      <p class="supervision-body">{$supervision.message || $_('supervision_alert.body')}</p>

      <div class="supervision-actions">
        <button class="btn-dismiss" onclick={() => supervision.dismiss()}>{$_('supervision_alert.dismiss')}</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .supervision-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .supervision-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--warning, rgba(251, 191, 36, 0.5));
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 480px;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .supervision-header {
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

  .supervision-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .supervision-actions {
    display: flex;
    justify-content: flex-end;
  }

  .btn-dismiss {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .btn-dismiss:hover {
    background: var(--accent-hover);
  }
</style>

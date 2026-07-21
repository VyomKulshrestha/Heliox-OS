<script lang="ts">
  /**
   * InterruptDialog — modal shown when the Live Execution Narrator
   * pre-emptively pauses a plan/action on a risk signal (a WARN-level
   * Agent Gateway critic verdict, or a dom_diff.assess_target problem)
   * and is waiting for the user to say whether to continue or stop.
   *
   * Self-subscribes to the narration store and renders conditionally,
   * same self-guarding pattern as BudgetExceededDialog.svelte. The spoken
   * interjection is triggered by narration.ts itself the moment the
   * "execution_interrupt" notification arrives, so this dialog and the
   * voice interruption always appear together.
   */

  import { _ } from "svelte-i18n";
  import { narration } from "../stores/narration";
</script>

{#if $narration.active}
  <div class="interrupt-overlay" role="dialog" aria-modal="true" aria-labelledby="interrupt-dialog-title">
    <div class="interrupt-dialog">
      <div class="interrupt-header">
        <span class="warn-icon">&#9888;</span>
        <span id="interrupt-dialog-title">{$_('interrupt.title')}</span>
      </div>

      <p class="interrupt-body">{$narration.reason || $_('interrupt.body')}</p>

      <div class="interrupt-actions">
        <button class="btn-deny" onclick={() => narration.respond(false)}>{$_('interrupt.stop')}</button>
        <button class="btn-confirm" onclick={() => narration.respond(true)}>{$_('interrupt.continue')}</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .interrupt-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .interrupt-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--warning, rgba(251, 191, 36, 0.5));
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 480px;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .interrupt-header {
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

  .interrupt-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .interrupt-actions {
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

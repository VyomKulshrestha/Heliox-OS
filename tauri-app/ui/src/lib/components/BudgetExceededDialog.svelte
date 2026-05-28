<script lang="ts">
  /**
   * BudgetExceededDialog — modal shown when a budget halt or circuit-breaker
   * trip fires from the orchestrator.
   *
   * Self-subscribes to the session store and renders conditionally based on
   * session.budget. The acknowledge button clears the event from session
   * state so a subsequent halt re-shows the dialog.
   *
   * The "Open settings" button opens the Settings panel; parent passes an
   * optional onopensettings callback. If absent, that button is hidden.
   */

  import { session } from "../stores/session";

  interface Props {
    onopensettings?: () => void;
  }

  let { onopensettings }: Props = $props();

  function handleAcknowledge() {
    session.acknowledgeBudgetEvent();
  }

  function handleOpenSettings() {
    session.acknowledgeBudgetEvent();
    onopensettings?.();
  }

  function titleFor(errorType: string): string {
    switch (errorType) {
      case "CircuitBreakerOpenError":
        return "Circuit breaker tripped";
      case "ActionBudgetExceededError":
        return "Per-action token limit reached";
      case "TaskBudgetExceededError":
        return "Task budget exhausted";
      case "BudgetExceededError":
        return "Monthly budget exceeded";
      default:
        return "Budget halt";
    }
  }

  function descFor(errorType: string): string {
    switch (errorType) {
      case "CircuitBreakerOpenError":
        return "Heliox detected too many consecutive failures in this task and halted further attempts to prevent runaway behavior.";
      case "ActionBudgetExceededError":
        return "A single LLM call would have exceeded the per-action token cap and was blocked before being sent.";
      case "TaskBudgetExceededError":
        return "This task hit its per-task token or USD cap. Subsequent steps were halted.";
      case "BudgetExceededError":
        return "The monthly API spend limit has been reached. Further paid calls are blocked.";
      default:
        return "Heliox halted the task because a budget rule fired.";
    }
  }
</script>

{#if $session.budget?.exceeded}
  <div class="budget-overlay" role="dialog" aria-modal="true" aria-labelledby="budget-dialog-title">
    <div class="budget-dialog">
      <div class="budget-header">
        <span class="warn-icon">&#9888;</span>
        <span id="budget-dialog-title">{titleFor($session.budget.errorType)}</span>
      </div>

      <p class="budget-desc">{descFor($session.budget.errorType)}</p>

      <div class="detail-block">
        <div class="detail-row">
          <span class="detail-label">Reason</span>
          <span class="detail-value">{$session.budget.message}</span>
        </div>
        {#if $session.budget.failureCount}
          <div class="detail-row">
            <span class="detail-label">Failures</span>
            <span class="detail-value">{$session.budget.failureCount} consecutive</span>
          </div>
        {/if}
        <div class="detail-row">
          <span class="detail-label">Task ID</span>
          <code class="detail-value">{$session.budget.taskId}</code>
        </div>
      </div>

      <div class="budget-actions">
        {#if onopensettings}
          <button class="btn-secondary" onclick={handleOpenSettings}>
            Open settings
          </button>
        {/if}
        <button class="btn-primary" onclick={handleAcknowledge}>
          Acknowledge
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .budget-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .budget-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--danger, rgba(255, 60, 60, 0.5));
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 520px;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .budget-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--danger, #ff5050);
  }

  .warn-icon {
    font-size: 22px;
  }

  .budget-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 16px;
  }

  .detail-block {
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .detail-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
    font-size: 12px;
  }

  .detail-label {
    color: var(--text-secondary);
    flex-shrink: 0;
    min-width: 70px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 10px;
  }

  .detail-value {
    color: var(--text-primary);
    word-break: break-word;
    flex: 1;
  }

  code.detail-value {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
  }

  .budget-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .btn-secondary {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-secondary:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-primary {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--danger, #d04040);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .btn-primary:hover {
    background: var(--danger-hover, #b03030);
  }
</style>
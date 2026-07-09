<script lang="ts">
  /**
   * BudgetMeter — displays configured budget caps and surfaces
   * budget-exceeded / circuit-breaker events from the orchestrator.
   *
   * Reads from the settings store (caps) and the session store
   * (most-recent budget event). When session.budget is set, renders
   * a dismissible alert banner; otherwise shows a quiet info display.
   *
   * Live in-flight progress (tokens used vs cap during a running task)
   * will be wired in once the daemon emits periodic budget_update events.
   */

  import { session } from "../stores/session";
  import { settings } from "../stores/settings";

  let acknowledging = $state(false);

  async function handleAcknowledge() {
    acknowledging = true;
    session.acknowledgeBudgetEvent();
    // Brief delay for the dismiss animation
    setTimeout(() => {
      acknowledging = false;
    }, 200);
  }

  // Compact USD formatter — 4 decimals for small amounts
  function fmtUsd(n: number): string {
    return `$${n.toFixed(n < 1 ? 4 : 2)}`;
  }

  // Compact integer formatter with thousands separator
  function fmtInt(n: number): string {
    return n.toLocaleString();
  }
</script>

<div class="budget-meter" class:alerting={$session.budget?.exceeded}>
  <div class="header">
    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
    <span class="title">Budget</span>
  </div>

  {#if $session.budget?.exceeded}
    <div class="alert" class:fading={acknowledging}>
      <div class="alert-header">
        <span class="alert-type">
          {#if $session.budget.errorType === "CircuitBreakerOpenError"}
            Circuit breaker tripped
          {:else if $session.budget.errorType === "ActionBudgetExceededError"}
            Per-action limit hit
          {:else if $session.budget.errorType === "TaskBudgetExceededError"}
            Task budget exceeded
          {:else}
            Monthly budget exceeded
          {/if}
        </span>
        <button class="ack-btn" onclick={handleAcknowledge} title="Dismiss">×</button>
      </div>
      <p class="alert-message">{$session.budget.message}</p>
      {#if $session.budget.failureCount}
        <p class="alert-meta">After {$session.budget.failureCount} consecutive failures</p>
      {/if}
      <p class="alert-meta">Task: <code>{$session.budget.taskId.slice(0, 8)}…</code></p>
    </div>
  {:else}
    <div class="status-ok">
      <span class="ok-dot"></span>
      <span>No budget halts</span>
    </div>
  {/if}

  <div class="live-usage">
    <div class="section-title live-title">Active Session Usage</div>
    <div class="cap-row">
      <span class="cap-label">Tokens consumed</span>
      <span class="cap-value highlight">{fmtInt($session.totalTokens)}</span>
    </div>
    <div class="cap-row">
      <span class="cap-label">Estimated cost</span>
      <span class="cap-value highlight">{fmtUsd($session.estimatedCost)}</span>
    </div>
  </div>

  <div class="caps">
    <div class="section-title">Configured Caps</div>
    <div class="cap-row">
      <span class="cap-label">Monthly</span>
      <span class="cap-value">{fmtUsd($settings.model.budget_monthly_limit_usd)}</span>
    </div>
    <div class="cap-row">
      <span class="cap-label">Per task tokens</span>
      <span class="cap-value">{fmtInt($settings.model.max_tokens_per_task)}</span>
    </div>
    <div class="cap-row">
      <span class="cap-label">Per task USD</span>
      <span class="cap-value">{fmtUsd($settings.model.max_usd_per_task)}</span>
    </div>
    <div class="cap-row">
      <span class="cap-label">Per action tokens</span>
      <span class="cap-value">{fmtInt($settings.model.max_tokens_per_action)}</span>
    </div>
    <div class="cap-row">
      <span class="cap-label">Failure threshold</span>
      <span class="cap-value">{$settings.model.max_consecutive_failures}</span>
    </div>
  </div>

  {#if !$settings.model.budget_enabled}
    <div class="disabled-note">
      Budget enforcement is disabled in settings.
    </div>
  {/if}
</div>

<style>
  .budget-meter {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 16px;
    border: 1px solid rgba(0, 200, 255, 0.18);
    background: rgba(0, 200, 255, 0.04);
    border-radius: 10px;
    backdrop-filter: blur(12px);
    font-family: "Inter", sans-serif;
    min-width: 240px;
    max-width: 320px;
    transition: border-color 0.3s ease, background 0.3s ease;
  }

  .budget-meter.alerting {
    border-color: rgba(255, 60, 60, 0.5);
    background: rgba(255, 60, 60, 0.06);
  }

  .header {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(0, 200, 255, 0.9);
    font-size: 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
  }

  .alerting .header {
    color: rgba(255, 60, 60, 0.9);
  }

  .icon {
    width: 14px;
    height: 14px;
  }

  .title {
    font-weight: 500;
  }

  .alert {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 12px;
    border: 1px solid rgba(255, 60, 60, 0.35);
    background: rgba(255, 60, 60, 0.08);
    border-radius: 6px;
    transition: opacity 0.2s ease;
  }

  .alert.fading {
    opacity: 0;
  }

  .alert-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .alert-type {
    color: rgba(255, 90, 90, 1);
    font-size: 12px;
    font-weight: 600;
  }

  .alert-message {
    color: rgba(255, 220, 220, 0.85);
    font-size: 11px;
    line-height: 1.4;
    margin: 0;
  }

  .alert-meta {
    color: rgba(255, 220, 220, 0.5);
    font-size: 10px;
    margin: 0;
  }

  .alert-meta code {
    background: rgba(255, 255, 255, 0.06);
    padding: 1px 4px;
    border-radius: 3px;
    font-family: monospace;
  }

  .ack-btn {
    background: transparent;
    border: 1px solid rgba(255, 60, 60, 0.4);
    color: rgba(255, 220, 220, 0.9);
    width: 22px;
    height: 22px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 0;
    transition: background 0.15s ease;
  }

  .ack-btn:hover {
    background: rgba(255, 60, 60, 0.2);
  }

  .status-ok {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(200, 200, 220, 0.6);
    font-size: 12px;
    padding: 4px 0;
  }

  .ok-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(0, 255, 136, 0.7);
    box-shadow: 0 0 6px rgba(0, 255, 136, 0.4);
  }

  .live-usage {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 6px 8px;
    background: rgba(0, 255, 136, 0.05);
    border: 1px solid rgba(0, 255, 136, 0.18);
    border-radius: 6px;
    margin-bottom: 2px;
  }

  .section-title {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(0, 200, 255, 0.85);
    margin-bottom: 2px;
  }

  .live-title {
    color: rgba(0, 255, 136, 0.9);
  }

  .highlight {
    color: rgba(0, 255, 136, 0.95);
    font-weight: 700;
  }

  .caps {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 4px;
    border-top: 1px solid rgba(0, 200, 255, 0.1);
  }

  .cap-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    font-size: 11px;
  }

  .cap-label {
    color: rgba(200, 200, 220, 0.55);
  }

  .cap-value {
    color: rgba(220, 230, 240, 0.95);
    font-family: monospace;
  }

  .disabled-note {
    font-size: 10px;
    color: rgba(255, 180, 80, 0.7);
    font-style: italic;
    padding: 4px 0;
    border-top: 1px dashed rgba(255, 180, 80, 0.2);
  }
</style>
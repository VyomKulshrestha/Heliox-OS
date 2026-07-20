<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call, onNotification, offNotification } from "../api/daemon";

  interface HealingAttempt {
    attempt_id: string;
    metric: string;
    goal: string;
    plan_id: string;
    outcome: string;
    max_tier: number;
    irreversible: boolean;
    explanation: string;
  }

  const ALL_METRICS = ["cpu", "memory", "disk"];
  const TIER_LABELS = ["Read Only", "User Write", "System Modify", "Destructive"];
  const HEALING_NOTIFICATIONS = new Set([
    "self_healing_auto_executing",
    "self_healing_confirmation_required",
    "self_healing_complete",
    "self_healing_denied",
    "self_healing_timeout",
  ]);

  let enabled = $state(false);
  let autoExecuteMaxTier = $state(1);
  let watchedMetrics = $state<string[]>([...ALL_METRICS]);
  let attempts = $state<HealingAttempt[]>([]);
  let loading = $state(true);
  let saving = $state(false);
  let saved = $state(false);
  let notificationHandler: ((method: string, params: unknown) => void) | null = null;

  onMount(() => {
    loadStatus();
    notificationHandler = (method) => {
      if (HEALING_NOTIFICATIONS.has(method)) loadStatus();
    };
    onNotification(notificationHandler);
    return () => {
      if (notificationHandler) offNotification(notificationHandler);
    };
  });

  async function loadStatus() {
    try {
      const result = (await call("self_healing_status")) as {
        enabled: boolean;
        auto_execute_max_tier: number;
        watched_metrics: string[];
        attempts: HealingAttempt[];
      };
      enabled = result.enabled ?? false;
      autoExecuteMaxTier = result.auto_execute_max_tier ?? 1;
      watchedMetrics = result.watched_metrics ?? [...ALL_METRICS];
      attempts = result.attempts ?? [];
    } catch {
      /* daemon unreachable -- keep last known state */
    } finally {
      loading = false;
    }
  }

  function toggleMetric(metric: string) {
    watchedMetrics = watchedMetrics.includes(metric)
      ? watchedMetrics.filter((m) => m !== metric)
      : [...watchedMetrics, metric];
  }

  async function save() {
    saving = true;
    saved = false;
    try {
      await call("self_healing_config_update", {
        enabled,
        auto_execute_max_tier: autoExecuteMaxTier,
        watched_metrics: watchedMetrics,
      });
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } finally {
      saving = false;
    }
  }

  async function approve(attempt: HealingAttempt) {
    await call("confirm", { plan_id: attempt.plan_id, confirmed: true });
    await loadStatus();
  }

  async function deny(attempt: HealingAttempt) {
    await call("confirm", { plan_id: attempt.plan_id, confirmed: false });
    await loadStatus();
  }

  function outcomeClass(outcome: string): string {
    if (outcome === "proposed") return "state-paused";
    if (outcome === "auto_executed" || outcome === "confirmed") return "state-running";
    return "state-other";
  }
</script>

<div class="healing-panel">
  <div class="healing-header">
    <h3>{$_('settings.self_healing')}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle Autonomous Healing"
      title="Toggle Autonomous Healing"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="healing-note">{$_('settings.self_healing_desc')}</p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.self_healing_tier')}</span>
        <span class="setting-desc">{$_('settings.self_healing_tier_desc')}</span>
      </div>
      <select class="input-sm" bind:value={autoExecuteMaxTier}>
        {#each TIER_LABELS as label, tier}
          <option value={tier}>{label}</option>
        {/each}
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.self_healing_metrics')}</span>
      </div>
      <div class="metric-checks">
        {#each ALL_METRICS as metric}
          <button
            class="toggle toggle-sm"
            class:active={watchedMetrics.includes(metric)}
            onclick={() => toggleMetric(metric)}
            aria-label={`Toggle ${metric} monitoring`}
          >
            <span class="toggle-knob"></span>
          </button>
          <span class="metric-label">{metric}</span>
        {/each}
      </div>
    </div>

    <div class="healing-actions">
      <button class="btn-save" onclick={save} disabled={saving}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_('settings.save')}
      </button>
    </div>

    <div class="attempts-header">
      <span class="setting-label">{$_('settings.self_healing_attempts')}</span>
      <span class="count">{attempts.length}</span>
    </div>

    {#if attempts.length === 0}
      <div class="empty">{$_('settings.self_healing_attempts_empty')}</div>
    {:else}
      <div class="attempt-list">
        {#each attempts as attempt}
          <div class="attempt-card">
            <div class="attempt-card-header">
              <span class="state-tag {outcomeClass(attempt.outcome)}">{attempt.outcome}</span>
              <span class="source-tag">{attempt.metric}</span>
            </div>
            <div class="attempt-goal">{attempt.goal}</div>
            {#if attempt.explanation}
              <div class="attempt-explanation">{attempt.explanation}</div>
            {/if}
            {#if attempt.outcome === "proposed"}
              <div class="attempt-actions">
                <button class="btn-save" onclick={() => approve(attempt)}>{$_('settings.self_healing_approve')}</button>
                <button class="btn-remove" onclick={() => deny(attempt)}>{$_('settings.self_healing_deny')}</button>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .healing-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .healing-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .healing-note {
    margin: 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }

  .metric-checks {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .toggle-sm {
    transform: scale(0.8);
  }

  .metric-label {
    font-size: 11px;
    color: var(--text-secondary);
    margin-right: 8px;
  }

  .healing-actions {
    display: flex;
    justify-content: flex-end;
  }

  .attempts-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 4px;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .attempt-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .attempt-card {
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 13px;
  }

  .attempt-card-header {
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

  .attempt-goal {
    font-weight: 500;
    margin-bottom: 2px;
  }

  .attempt-explanation {
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .attempt-actions {
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

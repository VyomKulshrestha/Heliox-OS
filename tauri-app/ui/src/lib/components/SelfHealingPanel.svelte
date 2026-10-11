<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call, onNotification, offNotification, requireOkResult, type DaemonStatusResult } from "../api/daemon";
  import { changedFields, onConfigChange } from "../api/configSync";

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

  interface MonitorStatus {
    task_id: string;
    status: string;
    condition: string;
    interval_seconds: number;
    last_run: number;
    run_count: number;
    error_count: number;
    last_result: Record<string, unknown>;
  }

  const ALL_METRICS = ["cpu", "memory", "disk"];
  const TIER_LABELS = ["Read Only", "User Write", "System Modify", "Destructive"];
  const HEALING_NOTIFICATIONS = new Set(["self_healing_confirmation_required", "self_healing_attempt_updated"]);

  let enabled = $state(false);
  let autoExecuteMaxTier = $state(1);
  let watchedMetrics = $state<string[]>([...ALL_METRICS]);
  let attempts = $state<HealingAttempt[]>([]);
  let monitors = $state<Record<string, MonitorStatus>>({});
  let loading = $state(true);
  let statusAvailable = $state(false);
  let saving = $state(false);
  let saved = $state(false);
  let error = $state("");
  let notificationHandler: ((method: string, params: unknown) => void) | null = null;
  // The config the daemon last reported. Healing attempts refresh the status often; those
  // refreshes must not overwrite toggles the person has changed but not saved yet.
  let savedConfig = $state<Record<string, unknown> | null>(null);

  function formConfig(): Record<string, unknown> {
    return { enabled, auto_execute_max_tier: autoExecuteMaxTier, watched_metrics: [...watchedMetrics] };
  }

  const dirty = $derived(savedConfig !== null && Object.keys(changedFields(savedConfig, formConfig())).length > 0);

  onMount(() => {
    loadStatus();
    notificationHandler = (method) => {
      if (HEALING_NOTIFICATIONS.has(method)) loadStatus();
    };
    onNotification(notificationHandler);
    const stopConfigSync = onConfigChange({ methods: ["self_healing_config_update"] }, () => {
      if (!saving) loadStatus();
    });
    return () => {
      if (notificationHandler) offNotification(notificationHandler);
      stopConfigSync();
    };
  });

  async function loadStatus() {
    try {
      const result = (await call("self_healing_status")) as {
        status?: string;
        message?: string;
        error?: string;
        enabled: boolean;
        auto_execute_max_tier: number;
        watched_metrics: string[];
        monitors: Record<string, MonitorStatus>;
        attempts: HealingAttempt[];
      };
      requireOkResult(result, "Autonomous healing status is unavailable.");
      if (!dirty) {
        enabled = result.enabled ?? false;
        autoExecuteMaxTier = result.auto_execute_max_tier ?? 1;
        watchedMetrics = result.watched_metrics ?? [...ALL_METRICS];
        savedConfig = formConfig();
      }
      monitors = result.monitors ?? {};
      attempts = result.attempts ?? [];
      statusAvailable = true;
      error = "";
    } catch (cause) {
      statusAvailable = false;
      error = cause instanceof Error ? cause.message : String(cause);
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
    error = "";
    try {
      const values = formConfig();
      const result = (await call(
        "self_healing_config_update",
        savedConfig ? changedFields(savedConfig, values) : values,
      )) as { status: string; message?: string };
      if (result.status !== "ok") throw new Error(result.message || "Configuration was not saved");
      savedConfig = values;
      await loadStatus();
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      saving = false;
    }
  }

  async function resolveAttempt(attempt: HealingAttempt, confirmed: boolean) {
    error = "";
    try {
      requireOkResult(
        (await call("confirm", { plan_id: attempt.plan_id, confirmed })) as DaemonStatusResult,
        confirmed ? "The healing action was not approved." : "The healing action was not denied.",
      );
      await loadStatus();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not resolve the healing proposal.";
    }
  }

  function approve(attempt: HealingAttempt) {
    return resolveAttempt(attempt, true);
  }

  function deny(attempt: HealingAttempt) {
    return resolveAttempt(attempt, false);
  }

  function outcomeClass(outcome: string): string {
    if (outcome === "proposed") return "state-paused";
    if (outcome === "executing" || outcome === "auto_executed" || outcome === "confirmed") return "state-running";
    return "state-other";
  }

  function monitorValue(metric: string, monitor: MonitorStatus): string {
    const value = monitor.last_result?.[`${metric}_percent`];
    return typeof value === "number" ? `${value}%` : "Waiting for first check";
  }
</script>

<div class="healing-panel">
  <div class="healing-header">
    <h3>{$_("settings.self_healing")}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle Autonomous Healing"
      title="Toggle Autonomous Healing"
      disabled={loading || !statusAvailable}
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="healing-note">{$_("settings.self_healing_desc")}</p>

  {#if error}
    <div class="panel-error" role="alert">{error}</div>
  {/if}

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.self_healing_tier")}</span>
        <span class="setting-desc">{$_("settings.self_healing_tier_desc")}</span>
      </div>
      <select class="input-sm" bind:value={autoExecuteMaxTier} disabled={!statusAvailable}>
        {#each TIER_LABELS as label, tier}
          <option value={tier}>{label}</option>
        {/each}
      </select>
    </div>

    <div class="monitor-grid" aria-label={$_("settings.self_healing_monitor_status")}>
      {#each watchedMetrics as metric}
        {@const monitor = monitors[metric]}
        <div class="monitor-card">
          <div class="monitor-title">
            <span class="monitor-dot" class:running={enabled && monitor?.status === "running"}></span>
            <strong>{metric.toUpperCase()}</strong>
            <span class="monitor-state">{enabled ? (monitor?.status ?? "unavailable") : "disabled"}</span>
          </div>
          <div class="monitor-detail">
            {monitor
              ? `${monitor.condition} · ${monitorValue(metric, monitor)} · ${monitor.run_count} checks`
              : "Monitor unavailable"}
          </div>
        </div>
      {/each}
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.self_healing_metrics")}</span>
      </div>
      <div class="metric-checks">
        {#each ALL_METRICS as metric}
          <button
            class="toggle toggle-sm"
            class:active={watchedMetrics.includes(metric)}
            onclick={() => toggleMetric(metric)}
            aria-label={`Toggle ${metric} monitoring`}
            disabled={!statusAvailable}
          >
            <span class="toggle-knob"></span>
          </button>
          <span class="metric-label">{metric}</span>
        {/each}
      </div>
    </div>

    <div class="healing-actions">
      <button class="btn-save" onclick={save} disabled={saving || !statusAvailable}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_("settings.save")}
      </button>
    </div>

    <div class="attempts-header">
      <span class="setting-label">{$_("settings.self_healing_attempts")}</span>
      <span class="count">{attempts.length}</span>
    </div>

    {#if attempts.length === 0}
      <div class="empty">{$_("settings.self_healing_attempts_empty")}</div>
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
                <button class="btn-save" onclick={() => approve(attempt)}>{$_("settings.self_healing_approve")}</button>
                <button class="btn-remove" onclick={() => deny(attempt)}>{$_("settings.self_healing_deny")}</button>
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
    margin: 0;
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
    padding: 12px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }

  .panel-error {
    padding: 8px 10px;
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 6px;
    background: rgba(248, 113, 113, 0.08);
    color: var(--danger);
    font-size: 11px;
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .setting-info {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 13px;
    font-weight: 500;
  }

  .setting-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .toggle {
    position: relative;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
    cursor: pointer;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 11px;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .input-sm {
    width: 150px;
    padding: 6px 8px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 12px;
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

  .btn-save {
    padding: 6px 14px;
    border: 0;
    border-radius: 6px;
    background: var(--accent);
    color: white;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }

  .btn-save:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .monitor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
  }

  .monitor-card {
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-primary);
  }

  .monitor-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
  }

  .monitor-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
  }

  .monitor-dot.running {
    background: var(--success);
    box-shadow: 0 0 6px rgba(74, 222, 128, 0.45);
  }

  .monitor-state {
    margin-left: auto;
    color: var(--text-muted);
    text-transform: capitalize;
  }

  .monitor-detail {
    margin-top: 4px;
    color: var(--text-muted);
    font-size: 10px;
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

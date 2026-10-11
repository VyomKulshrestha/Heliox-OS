<script lang="ts">
  /**
   * User Manual Supervision settings -- the single biggest privacy-surface
   * toggle in this app. `enabled` (screen/OCR-based cognitive coaching +
   * risk warnings) and `keyboardMouseHookEnabled` (a global keyboard/mouse
   * hook) are deliberately separate toggles, mirroring
   * SupervisionConfig's own enabled/keyboard_mouse_hook_enabled split.
   * The hook toggle stays disabled until the user checks a one-time
   * "I understand" box in this session -- this checkbox is local UI state
   * only, never persisted, so it re-appears next time Settings is opened.
   */
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call, requireOkResult } from "../api/daemon";
  import { changedFields, onConfigChange } from "../api/configSync";

  let enabled = $state(false);
  let keyboardMouseHookEnabled = $state(false);
  let cognitiveCoachingEnabled = $state(true);
  let riskPatternDetectionEnabled = $state(true);
  let hookHealthy = $state(false);
  let loading = $state(true);
  let statusAvailable = $state(false);
  let saving = $state(false);
  let saved = $state(false);
  let error = $state("");
  let hookWarningUnderstood = $state(false);
  let restartNotice = $state("");
  // What the daemon last reported; a save sends only the toggles changed here.
  let savedValues = $state<Record<string, unknown> | null>(null);

  function formValues(): Record<string, unknown> {
    return {
      enabled,
      keyboard_mouse_hook_enabled: keyboardMouseHookEnabled,
      cognitive_coaching_enabled: cognitiveCoachingEnabled,
      risk_pattern_detection_enabled: riskPatternDetectionEnabled,
    };
  }

  const dirty = $derived(savedValues !== null && Object.keys(changedFields(savedValues, formValues())).length > 0);

  onMount(() => {
    void loadStatus();
    return onConfigChange({ methods: ["supervision_config_update"] }, () => {
      if (!dirty && !saving) void loadStatus();
    });
  });

  async function loadStatus() {
    try {
      const result = (await call("supervision_status")) as {
        status?: string;
        message?: string;
        error?: string;
        enabled: boolean;
        keyboard_mouse_hook_enabled: boolean;
        cognitive_coaching_enabled: boolean;
        risk_pattern_detection_enabled: boolean;
        hook_healthy: boolean;
      };
      requireOkResult(result, "Supervision status is unavailable.");
      enabled = result.enabled ?? false;
      keyboardMouseHookEnabled = result.keyboard_mouse_hook_enabled ?? false;
      cognitiveCoachingEnabled = result.cognitive_coaching_enabled ?? true;
      riskPatternDetectionEnabled = result.risk_pattern_detection_enabled ?? true;
      hookHealthy = result.hook_healthy ?? false;
      savedValues = formValues();
      statusAvailable = true;
      error = "";
    } catch (cause) {
      statusAvailable = false;
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      loading = false;
    }
  }

  async function save() {
    saving = true;
    saved = false;
    error = "";
    restartNotice = "";
    try {
      const values = formValues();
      const result = requireOkResult(
        (await call("supervision_config_update", savedValues ? changedFields(savedValues, values) : values)) as {
          status?: string;
          message?: string;
          hook_healthy?: boolean;
          restart_required?: boolean;
        },
        "Supervision settings were not saved",
      );
      hookHealthy = result.hook_healthy ?? false;
      savedValues = values;
      if (result.restart_required) restartNotice = "Saved. Restart Heliox for the new timing to take effect.";
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      saving = false;
    }
  }

  function toggleHook() {
    if (!keyboardMouseHookEnabled && !hookWarningUnderstood) return;
    keyboardMouseHookEnabled = !keyboardMouseHookEnabled;
  }
</script>

<div class="supervision-panel">
  <div class="supervision-header">
    <h3>{$_("settings.supervision")}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle User Manual Supervision"
      title="Toggle User Manual Supervision"
      disabled={loading || !statusAvailable}
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="supervision-warning">{$_("settings.supervision_warning")}</p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    {#if restartNotice}
      <p class="panel-notice" role="status">{restartNotice}</p>
    {/if}
    {#if error}
      <p class="panel-error" role="alert">{error}</p>
    {/if}
    {#if enabled}
      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.supervision_cognitive_coaching")}</span>
          <span class="setting-desc">{$_("settings.supervision_cognitive_coaching_desc")}</span>
        </div>
        <button
          class="toggle toggle-sm"
          class:active={cognitiveCoachingEnabled}
          onclick={() => (cognitiveCoachingEnabled = !cognitiveCoachingEnabled)}
          aria-label="Toggle cognitive coaching"
        >
          <span class="toggle-knob"></span>
        </button>
      </div>

      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.supervision_risk_detection")}</span>
          <span class="setting-desc">{$_("settings.supervision_risk_detection_desc")}</span>
        </div>
        <button
          class="toggle toggle-sm"
          class:active={riskPatternDetectionEnabled}
          onclick={() => (riskPatternDetectionEnabled = !riskPatternDetectionEnabled)}
          aria-label="Toggle risk pattern detection"
        >
          <span class="toggle-knob"></span>
        </button>
      </div>

      <div class="supervision-hook-section">
        <p class="supervision-hook-warning">{$_("settings.supervision_hook_warning")}</p>

        {#if !keyboardMouseHookEnabled}
          <label class="hook-understood">
            <input type="checkbox" bind:checked={hookWarningUnderstood} />
            {$_("settings.supervision_hook_understood")}
          </label>
        {/if}

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-label">{$_("settings.supervision_hook_enabled")}</span>
            <span class="setting-desc">{$_("settings.supervision_hook_enabled_desc")}</span>
          </div>
          <button
            class="toggle toggle-sm"
            class:active={keyboardMouseHookEnabled}
            disabled={!keyboardMouseHookEnabled && !hookWarningUnderstood}
            onclick={toggleHook}
            aria-label="Toggle keyboard/mouse hook"
          >
            <span class="toggle-knob"></span>
          </button>
        </div>

        {#if keyboardMouseHookEnabled}
          <p class="hook-status" class:hook-status-bad={!hookHealthy}>
            {hookHealthy ? $_("settings.supervision_hook_healthy") : $_("settings.supervision_hook_unhealthy")}
          </p>
        {/if}
      </div>
    {/if}

    <div class="supervision-actions">
      <button class="btn-save" onclick={save} disabled={saving || !statusAvailable}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_("settings.save")}
      </button>
    </div>
  {/if}
</div>

<style>
  .supervision-panel {
    display: flex;
    flex-direction: column;
  }

  .supervision-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }

  .supervision-warning {
    margin: 10px 14px;
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

  .panel-notice {
    margin: 8px 0 0;
    color: var(--text-secondary);
    font-size: 12px;
  }

  .panel-error {
    margin: 10px 14px 0;
    padding: 8px 10px;
    color: var(--danger, #ef4444);
    font-size: 11px;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: var(--radius-sm);
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
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
    transition: all 0.2s;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
  }

  .toggle:disabled {
    cursor: not-allowed;
    opacity: 0.45;
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

  .toggle-sm {
    transform: scale(0.8);
  }

  .supervision-hook-section {
    margin: 10px 14px 0;
    padding: 10px 12px;
    border: 1px solid var(--warning, #f59e0b);
    border-radius: var(--radius-sm);
    background: rgba(245, 158, 11, 0.06);
  }

  .supervision-hook-warning {
    margin: 0 0 8px 0;
    font-size: 11px;
    line-height: 1.5;
    color: var(--warning, #f59e0b);
  }

  .hook-understood {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 8px;
    cursor: pointer;
  }

  .hook-status {
    margin: 8px 0 0 0;
    font-size: 11px;
    color: var(--success, #22c55e);
  }

  .hook-status-bad {
    color: var(--danger, #ef4444);
  }

  .supervision-actions {
    display: flex;
    justify-content: flex-end;
    padding: 10px 14px 12px;
  }

  .btn-save {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    white-space: nowrap;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-save:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .btn-save:disabled {
    cursor: not-allowed;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
  }
</style>

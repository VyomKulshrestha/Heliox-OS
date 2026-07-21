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
  import { call } from "../api/daemon";

  let enabled = $state(false);
  let keyboardMouseHookEnabled = $state(false);
  let cognitiveCoachingEnabled = $state(true);
  let riskPatternDetectionEnabled = $state(true);
  let hookHealthy = $state(false);
  let loading = $state(true);
  let saving = $state(false);
  let saved = $state(false);
  let hookWarningUnderstood = $state(false);

  onMount(loadStatus);

  async function loadStatus() {
    try {
      const result = (await call("supervision_status")) as {
        enabled: boolean;
        keyboard_mouse_hook_enabled: boolean;
        cognitive_coaching_enabled: boolean;
        risk_pattern_detection_enabled: boolean;
        hook_healthy: boolean;
      };
      enabled = result.enabled ?? false;
      keyboardMouseHookEnabled = result.keyboard_mouse_hook_enabled ?? false;
      cognitiveCoachingEnabled = result.cognitive_coaching_enabled ?? true;
      riskPatternDetectionEnabled = result.risk_pattern_detection_enabled ?? true;
      hookHealthy = result.hook_healthy ?? false;
    } catch {
      /* daemon unreachable -- keep last known state */
    } finally {
      loading = false;
    }
  }

  async function save() {
    saving = true;
    saved = false;
    try {
      const result = (await call("supervision_config_update", {
        enabled,
        keyboard_mouse_hook_enabled: keyboardMouseHookEnabled,
        cognitive_coaching_enabled: cognitiveCoachingEnabled,
        risk_pattern_detection_enabled: riskPatternDetectionEnabled,
      })) as { hook_healthy: boolean };
      hookHealthy = result.hook_healthy ?? false;
      saved = true;
      setTimeout(() => (saved = false), 2500);
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
    <h3>{$_('settings.supervision')}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle User Manual Supervision"
      title="Toggle User Manual Supervision"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="supervision-warning">{$_('settings.supervision_warning')}</p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if enabled}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.supervision_cognitive_coaching')}</span>
        <span class="setting-desc">{$_('settings.supervision_cognitive_coaching_desc')}</span>
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
        <span class="setting-label">{$_('settings.supervision_risk_detection')}</span>
        <span class="setting-desc">{$_('settings.supervision_risk_detection_desc')}</span>
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
      <p class="supervision-hook-warning">{$_('settings.supervision_hook_warning')}</p>

      {#if !keyboardMouseHookEnabled}
        <label class="hook-understood">
          <input type="checkbox" bind:checked={hookWarningUnderstood} />
          {$_('settings.supervision_hook_understood')}
        </label>
      {/if}

      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_('settings.supervision_hook_enabled')}</span>
          <span class="setting-desc">{$_('settings.supervision_hook_enabled_desc')}</span>
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
          {hookHealthy ? $_('settings.supervision_hook_healthy') : $_('settings.supervision_hook_unhealthy')}
        </p>
      {/if}
    </div>

    <div class="supervision-actions">
      <button class="btn-save" onclick={save} disabled={saving}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_('settings.save')}
      </button>
    </div>
  {/if}
</div>

<style>
  .supervision-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .supervision-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .supervision-warning {
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

  .toggle-sm {
    transform: scale(0.8);
  }

  .supervision-hook-section {
    margin-top: 4px;
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
  }
</style>

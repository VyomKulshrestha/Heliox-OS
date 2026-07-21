<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call } from "../api/daemon";

  let enabled = $state(false);
  let narrateSteps = $state(true);
  let interruptOnRisk = $state(true);
  let confirmTimeoutSeconds = $state(120);
  let loading = $state(true);
  let saving = $state(false);
  let saved = $state(false);

  onMount(loadStatus);

  async function loadStatus() {
    try {
      const result = (await call("narration_status")) as {
        enabled: boolean;
        narrate_steps: boolean;
        interrupt_on_risk: boolean;
        confirm_timeout_seconds: number;
      };
      enabled = result.enabled ?? false;
      narrateSteps = result.narrate_steps ?? true;
      interruptOnRisk = result.interrupt_on_risk ?? true;
      confirmTimeoutSeconds = result.confirm_timeout_seconds ?? 120;
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
      await call("narration_config_update", {
        enabled,
        narrate_steps: narrateSteps,
        interrupt_on_risk: interruptOnRisk,
        confirm_timeout_seconds: confirmTimeoutSeconds,
      });
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } finally {
      saving = false;
    }
  }
</script>

<div class="narration-panel">
  <div class="narration-header">
    <h3>{$_('settings.narration')}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle Live Execution Narrator"
      title="Toggle Live Execution Narrator"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="narration-note">{$_('settings.narration_desc')}</p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.narration_narrate_steps')}</span>
        <span class="setting-desc">{$_('settings.narration_narrate_steps_desc')}</span>
      </div>
      <button
        class="toggle toggle-sm"
        class:active={narrateSteps}
        onclick={() => (narrateSteps = !narrateSteps)}
        aria-label="Toggle step narration"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.narration_interrupt_on_risk')}</span>
        <span class="setting-desc">{$_('settings.narration_interrupt_on_risk_desc')}</span>
      </div>
      <button
        class="toggle toggle-sm"
        class:active={interruptOnRisk}
        onclick={() => (interruptOnRisk = !interruptOnRisk)}
        aria-label="Toggle risk interrupts"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.narration_timeout')}</span>
        <span class="setting-desc">{$_('settings.narration_timeout_desc')}</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={confirmTimeoutSeconds}
        onchange={(e) => (confirmTimeoutSeconds = Number((e.target as HTMLInputElement).value))}
        min="10"
        max="600"
        step="10"
      />
    </div>

    <div class="narration-actions">
      <button class="btn-save" onclick={save} disabled={saving}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_('settings.save')}
      </button>
    </div>
  {/if}
</div>

<style>
  .narration-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .narration-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .narration-note {
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

  .narration-actions {
    display: flex;
    justify-content: flex-end;
  }
</style>

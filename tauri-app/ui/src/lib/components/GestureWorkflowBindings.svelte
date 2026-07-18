<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call } from "../api/daemon";

  interface Binding {
    gesture_name: string;
    goal_template: string;
    enabled: boolean;
  }

  // Mirrors GestureControl.svelte's GESTURE_EMOJIS key set -- any named
  // gesture can be bound to a workflow goal.
  const KNOWN_GESTURES = [
    "palm", "thumbs_up", "thumbs_down", "peace", "fist", "point_up", "rock", "ok",
    "call_me", "finger_gun", "pinch", "middle_finger", "pinky_up", "vulcan",
    "crossed_fingers", "snap_ready", "devil_horns", "palm_down", "palm_up", "three_up", "four_up",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down", "circular_cw", "circular_ccw",
    "palm_push", "palm_pull", "two_finger_swipe_left", "two_finger_swipe_right",
  ];

  let enabled = $state(false);
  let bindings = $state<Binding[]>([]);
  let loading = $state(true);
  let saving = $state(false);
  let saved = $state(false);

  onMount(loadBindings);

  async function loadBindings() {
    loading = true;
    try {
      const result = (await call("gesture_workflow_bindings_get")) as { enabled: boolean; bindings: Binding[] };
      enabled = result.enabled ?? false;
      bindings = result.bindings ?? [];
    } catch {
      enabled = false;
      bindings = [];
    } finally {
      loading = false;
    }
  }

  function addBinding() {
    bindings = [...bindings, { gesture_name: KNOWN_GESTURES[0], goal_template: "", enabled: true }];
  }

  function removeBinding(index: number) {
    bindings = bindings.filter((_, i) => i !== index);
  }

  async function save() {
    saving = true;
    saved = false;
    try {
      await call("gesture_workflow_bindings_update", { enabled, bindings });
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } finally {
      saving = false;
    }
  }
</script>

<div class="bindings-editor">
  <div class="bindings-header">
    <h3>{$_('settings.gesture_workflows')}</h3>
    <button class="toggle" class:active={enabled} onclick={() => (enabled = !enabled)} aria-label="Toggle gesture workflow bindings">
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="bindings-note">{$_('settings.gesture_workflows_desc')}</p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="binding-list">
      {#each bindings as binding, i}
        <div class="binding-row">
          <select class="input-sm" bind:value={binding.gesture_name}>
            {#each KNOWN_GESTURES as g}
              <option value={g}>{g}</option>
            {/each}
          </select>
          <input
            type="text"
            class="input-md"
            placeholder={$_('settings.gesture_workflows_goal_placeholder')}
            bind:value={binding.goal_template}
          />
          <button
            class="toggle toggle-sm"
            class:active={binding.enabled}
            onclick={() => (binding.enabled = !binding.enabled)}
            aria-label="Toggle binding"
          >
            <span class="toggle-knob"></span>
          </button>
          <button class="btn-remove" onclick={() => removeBinding(i)} aria-label="Remove binding">✕</button>
        </div>
      {/each}
    </div>

    <div class="bindings-actions">
      <button class="btn-add" onclick={addBinding}>{$_('settings.gesture_workflows_add')}</button>
      <button class="btn-save" onclick={save} disabled={saving}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_('settings.save')}
      </button>
    </div>
  {/if}
</div>

<style>
  .bindings-editor {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .bindings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
  }

  .bindings-note {
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

  .binding-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .binding-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .binding-row .input-md {
    flex: 1;
  }

  .toggle-sm {
    transform: scale(0.8);
  }

  .btn-remove {
    background: none;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--danger);
    cursor: pointer;
    padding: 4px 8px;
    font-size: 12px;
  }

  .bindings-actions {
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
  }

  .btn-add {
    font-size: 12px;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
  }
</style>

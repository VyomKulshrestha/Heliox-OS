<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { call, requireResultStatus } from "../api/daemon";
  import { changedFields, onConfigChange } from "../api/configSync";

  interface Binding {
    gesture_name: string;
    goal_template: string;
    enabled: boolean;
  }

  // Mirrors GestureControl.svelte's GESTURE_EMOJIS key set -- any named
  // gesture can be bound to a workflow goal.
  const FALLBACK_GESTURES = [
    "palm",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "fist",
    "point_up",
    "rock",
    "ok",
    "call_me",
    "finger_gun",
    "pinch",
    "middle_finger",
    "pinky_up",
    "vulcan",
    "crossed_fingers",
    "snap_ready",
    "devil_horns",
    "palm_down",
    "palm_up",
    "three_up",
    "four_up",
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "circular_cw",
    "circular_ccw",
    "palm_push",
    "palm_pull",
    "two_finger_swipe_left",
    "two_finger_swipe_right",
  ];

  let enabled = $state(false);
  let bindings = $state<Binding[]>([]);
  let loading = $state(true);
  let statusAvailable = $state(false);
  let saving = $state(false);
  let saved = $state(false);
  let error = $state("");
  let supportedGestures = $state<string[]>([...FALLBACK_GESTURES]);
  let validationMessage = $derived(validateBindings());
  // The bindings the daemon last reported; a save sends only what changed here.
  let savedValues = $state<Record<string, unknown> | null>(null);

  function formValues(): Record<string, unknown> {
    return {
      enabled,
      bindings: bindings.map((binding) => ({ ...binding, goal_template: binding.goal_template.trim() })),
    };
  }

  const dirty = $derived(savedValues !== null && Object.keys(changedFields(savedValues, formValues())).length > 0);

  onMount(() => {
    void loadBindings();
    return onConfigChange({ methods: ["gesture_workflow_bindings_update"] }, () => {
      if (!dirty && !saving) void loadBindings();
    });
  });

  async function loadBindings() {
    loading = true;
    try {
      const result = (await call("gesture_workflow_bindings_get")) as {
        status?: string;
        message?: string;
        error?: string;
        enabled: boolean;
        bindings: Binding[];
        supported_gestures?: string[];
      };
      requireResultStatus(result, "ok", "Gesture workflow bindings are unavailable.");
      enabled = result.enabled ?? false;
      bindings = result.bindings ?? [];
      supportedGestures = result.supported_gestures?.length ? result.supported_gestures : [...FALLBACK_GESTURES];
      savedValues = formValues();
      statusAvailable = true;
      error = "";
    } catch (cause) {
      statusAvailable = false;
      enabled = false;
      bindings = [];
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      loading = false;
    }
  }

  function addBinding() {
    const used = new Set(bindings.map((binding) => binding.gesture_name));
    const available = supportedGestures.find((gesture) => !used.has(gesture));
    if (!available) {
      error = $_("settings.gesture_workflows_no_gestures");
      return;
    }
    error = "";
    bindings = [...bindings, { gesture_name: available, goal_template: "", enabled: true }];
  }

  function removeBinding(index: number) {
    bindings = bindings.filter((_, i) => i !== index);
    error = "";
  }

  function validateBindings(): string {
    if (enabled && !bindings.some((binding) => binding.enabled)) {
      return $_("settings.gesture_workflows_enabled_required");
    }
    const seen = new Set<string>();
    for (const binding of bindings) {
      if (!binding.goal_template.trim()) return $_("settings.gesture_workflows_goal_required");
      if (seen.has(binding.gesture_name)) return $_("settings.gesture_workflows_duplicate");
      seen.add(binding.gesture_name);
    }
    return "";
  }

  async function save() {
    if (validationMessage) {
      error = validationMessage;
      return;
    }
    saving = true;
    saved = false;
    error = "";
    try {
      const values = formValues();
      const result = (await call(
        "gesture_workflow_bindings_update",
        savedValues ? changedFields(savedValues, values) : values,
      )) as {
        status: string;
        message?: string;
        enabled?: boolean;
        bindings?: Binding[];
        supported_gestures?: string[];
      };
      requireResultStatus(result, "ok", "Gesture workflow bindings were not saved.");
      enabled = result.enabled ?? enabled;
      bindings = result.bindings ?? bindings;
      supportedGestures = result.supported_gestures?.length ? result.supported_gestures : supportedGestures;
      savedValues = formValues();
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      saving = false;
    }
  }
</script>

<div class="bindings-editor">
  <div class="bindings-header">
    <h3>{$_("settings.gesture_workflows")}</h3>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => (enabled = !enabled)}
      aria-label="Toggle gesture workflow bindings"
      aria-pressed={enabled}
      disabled={loading || !statusAvailable}
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="bindings-note">{$_("settings.gesture_workflows_desc")}</p>
  <p class="runtime-note">{$_("settings.gesture_workflows_live_desc")}</p>

  {#if error || validationMessage}
    <div class="panel-error" role="alert">{error || validationMessage}</div>
  {/if}

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if statusAvailable}
    <div class="binding-list">
      {#each bindings as binding, i}
        <div class="binding-row">
          <select class="input-sm" bind:value={binding.gesture_name} aria-label={`Gesture for binding ${i + 1}`}>
            {#each supportedGestures as g}
              <option value={g}>{g}</option>
            {/each}
          </select>
          <input
            type="text"
            class="input-md"
            placeholder={$_("settings.gesture_workflows_goal_placeholder")}
            bind:value={binding.goal_template}
            aria-label={`Workflow goal for binding ${i + 1}`}
          />
          <button
            class="toggle toggle-sm"
            class:active={binding.enabled}
            onclick={() => (binding.enabled = !binding.enabled)}
            aria-label={`Toggle binding ${i + 1}`}
            aria-pressed={binding.enabled}
          >
            <span class="toggle-knob"></span>
          </button>
          <button class="btn-remove" onclick={() => removeBinding(i)} aria-label={`Remove binding ${i + 1}`}>✕</button>
        </div>
      {/each}
    </div>

    <div class="bindings-actions">
      <button class="btn-add" onclick={addBinding}>{$_("settings.gesture_workflows_add")}</button>
      <button class="btn-save" onclick={save} disabled={saving || !!validationMessage}>
        {saving ? "Saving..." : saved ? "✓ Saved" : $_("settings.save")}
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
    margin: 0;
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

  .runtime-note {
    margin: -4px 0 0;
    color: var(--success);
    font-size: 10px;
  }

  .panel-error {
    padding: 8px 10px;
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 6px;
    background: rgba(248, 113, 113, 0.08);
    color: var(--danger);
    font-size: 11px;
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

  .toggle-sm {
    transform: scale(0.8);
  }

  .input-sm,
  .input-md {
    padding: 6px 8px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 12px;
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
</style>

<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { neural } from "../stores/neural";
  import SSVEPStimulusGrid from "./SSVEPStimulusGrid.svelte";

  interface Props {
    showPreview?: boolean;
  }

  let { showPreview = true }: Props = $props();

  const armed = $derived(["armed_safe_ui", "armed_safe_desktop", "previewed", "cooldown"].includes($neural.state));

  onMount(() => {
    const emergencyKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && ($neural.connected || $neural.preview)) void neural.disarm("keyboard_escape");
    };
    window.addEventListener("keydown", emergencyKey, true);
    return () => window.removeEventListener("keydown", emergencyKey, true);
  });
</script>

{#if armed && $neural.stimulusEnabled}
  <SSVEPStimulusGrid />
{/if}

{#if $neural.connected}
  <div class="neural-dock" class:armed class:with-stimulus={armed && $neural.stimulusEnabled} aria-live="polite">
    <span class="pulse"></span>
    <div>
      <strong>{$_("neural.dock_title")}</strong>
      <small>{$neural.state.replaceAll("_", " ")} · {$neural.quality?.quality ?? "waiting for signal"}</small>
    </div>
    <button onclick={() => neural.disarm()}>{$_("neural.disarm")} · Esc</button>
  </div>
{/if}

{#if showPreview && $neural.preview}
  <div class="preview-overlay" role="dialog" aria-modal="true" aria-labelledby="neural-preview-title">
    <div class="preview-card">
      <div class="eyebrow">{$_("neural.decoded_candidate")}</div>
      <h3 id="neural-preview-title">
        {$neural.preview.staged_task?.label ??
          $neural.preview.resolved_command_id ??
          $neural.preview.canonical_goal.replaceAll("_", " ")}
      </h3>
      {#if $neural.preview.staged_task}
        <p class="task-goal">{$neural.preview.staged_task.goal}</p>
        <p>{$_("neural.task_preview_body")}</p>
      {:else}
        <p>
          {$_("neural.preview_body")}
          {#if !$neural.preview.requires_non_neural_approval}{$_("neural.auto_commit")}{/if}
        </p>
      {/if}
      {#if $neural.preview.world_model}
        <div class="world-model">
          <strong>{$_("neural.world_model")}</strong>
          <span>{JSON.stringify($neural.preview.world_model)}</span>
        </div>
      {/if}
      <div class="preview-actions">
        <button class="cancel" onclick={() => neural.cancelPreview()}>{$_("neural.cancel")}</button>
        {#if $neural.preview.requires_non_neural_approval}
          <button class="approve" onclick={() => neural.approvePreview()}>{$_("neural.approve")}</button>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .neural-dock {
    position: absolute;
    right: 18px;
    bottom: 72px;
    z-index: 80;
    display: flex;
    align-items: center;
    gap: 10px;
    max-width: 420px;
    padding: 9px 10px;
    background: color-mix(in srgb, var(--bg-secondary) 94%, transparent);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
  }
  .neural-dock.armed {
    border-color: color-mix(in srgb, var(--accent) 55%, var(--border));
  }
  .neural-dock.with-stimulus {
    bottom: 176px;
  }
  .pulse {
    width: 8px;
    height: 8px;
    flex: 0 0 auto;
    background: var(--text-muted);
    border-radius: 50%;
  }
  .armed .pulse {
    background: var(--success);
    box-shadow: 0 0 10px var(--success);
  }
  .neural-dock div {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .neural-dock strong {
    font-size: 11px;
  }
  .neural-dock small {
    overflow: hidden;
    color: var(--text-muted);
    font-size: 9px;
    text-overflow: ellipsis;
    text-transform: capitalize;
    white-space: nowrap;
  }
  .neural-dock button,
  .preview-actions button {
    padding: 6px 9px;
    color: var(--danger);
    background: transparent;
    border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
    border-radius: 7px;
    font-size: 10px;
    white-space: nowrap;
  }
  .preview-overlay {
    position: absolute;
    inset: 0;
    z-index: 140;
    display: grid;
    place-items: center;
    padding: 24px;
    background: rgba(0, 0, 0, 0.7);
  }
  .preview-card {
    width: min(500px, 100%);
    padding: 22px;
    background: var(--bg-secondary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-lg);
    box-shadow: 0 22px 70px rgba(0, 0, 0, 0.45);
  }
  .eyebrow {
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  h3 {
    margin: 8px 0;
    font-size: 19px;
  }
  p {
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.5;
  }
  .task-goal {
    padding: 9px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .world-model {
    display: grid;
    gap: 5px;
    max-height: 110px;
    overflow: auto;
    padding: 9px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 10px;
  }
  .world-model span {
    color: var(--text-muted);
    overflow-wrap: anywhere;
  }
  .preview-actions {
    display: flex;
    justify-content: flex-end;
    gap: 9px;
    margin-top: 16px;
  }
  .preview-actions .approve {
    color: var(--on-accent);
    background: var(--accent);
    border-color: var(--accent);
  }
</style>

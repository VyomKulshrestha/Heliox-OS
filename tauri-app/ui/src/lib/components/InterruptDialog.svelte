<script lang="ts">
  /**
   * InterruptDialog — modal shown when the Live Execution Narrator
   * pre-emptively pauses a plan/action on a risk signal (a WARN-level
   * Agent Gateway critic verdict, or a dom_diff.assess_target problem)
   * and is waiting for the user to say whether to continue or stop.
   *
   * Self-subscribes to the narration store and renders conditionally,
   * same self-guarding pattern as BudgetExceededDialog.svelte. The spoken
   * interjection is triggered by narration.ts itself the moment the
   * "execution_interrupt" notification arrives, so this dialog and the
   * voice interruption always appear together.
   *
   * When kind === "action_preview" (the "simulate before executing" gate
   * for autonomous background tasks), this also renders a real screenshot
   * with the target UI element highlighted, plus a real measured DOM diff
   * summary for browser actions — never a generated image, see
   * SECURITY.md's Pre-Execution Target Assessment section for why.
   */

  import { _ } from "svelte-i18n";
  import { narration } from "../stores/narration";

  let naturalWidth = $state(0);
  let naturalHeight = $state(0);

  function onScreenshotLoad(e: Event) {
    const img = e.currentTarget as HTMLImageElement;
    naturalWidth = img.naturalWidth;
    naturalHeight = img.naturalHeight;
  }

  const bboxStyle = $derived.by(() => {
    const bbox = $narration.preview?.bbox;
    if (!bbox || !naturalWidth || !naturalHeight) return "";
    const left = (bbox.x / naturalWidth) * 100;
    const top = (bbox.y / naturalHeight) * 100;
    const width = (bbox.w / naturalWidth) * 100;
    const height = (bbox.h / naturalHeight) * 100;
    return `left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;`;
  });
</script>

{#if $narration.active}
  <div class="interrupt-overlay" role="dialog" aria-modal="true" aria-labelledby="interrupt-dialog-title">
    <div class="interrupt-dialog" class:with-preview={$narration.kind === "action_preview"}>
      <div class="interrupt-header">
        <span class="warn-icon">&#9888;</span>
        <span id="interrupt-dialog-title">
          {$narration.kind === "action_preview" ? $_("interrupt.preview_title") : $_("interrupt.title")}
        </span>
      </div>

      <p class="interrupt-body">{$narration.reason || $_("interrupt.body")}</p>

      {#if $narration.kind === "action_preview" && $narration.preview}
        {@const preview = $narration.preview}
        {#if preview.screenshot_base64}
          <div class="preview-screenshot-wrap">
            <img
              class="preview-screenshot"
              src={`data:image/png;base64,${preview.screenshot_base64}`}
              alt={preview.target_label || "Preview of screen before action"}
              onload={onScreenshotLoad}
            />
            {#if bboxStyle}
              <div class="preview-bbox" style={bboxStyle}></div>
            {/if}
          </div>
        {/if}
        {#if preview.dom_diff?.summary}
          <p class="preview-dom-diff">{$_("interrupt.dom_diff_prefix")}: {preview.dom_diff.summary}</p>
        {/if}
      {/if}

      {#if $narration.responseError}
        <p class="interrupt-error" role="alert">{$narration.responseError}</p>
      {/if}

      <div class="interrupt-actions">
        <button class="btn-deny" disabled={$narration.responsePending} onclick={() => narration.respond(false)}
          >{$_("interrupt.stop")}</button
        >
        <button class="btn-confirm" disabled={$narration.responsePending} onclick={() => narration.respond(true)}
          >{$_("interrupt.continue")}</button
        >
      </div>
    </div>
  </div>
{/if}

<style>
  .interrupt-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .interrupt-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--warning);
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 480px;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .interrupt-dialog.with-preview {
    max-width: 640px;
  }

  .preview-screenshot-wrap {
    position: relative;
    width: 100%;
    margin-bottom: 16px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .preview-screenshot {
    display: block;
    width: 100%;
    height: auto;
  }

  .preview-bbox {
    position: absolute;
    border: 2px solid var(--accent);

    border-radius: 2px;
    pointer-events: none;
  }

  .preview-dom-diff {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 16px;
    font-family: var(--font-mono, monospace);
  }

  .interrupt-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .warn-icon {
    color: var(--warning);
    font-size: 20px;
  }

  .interrupt-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .interrupt-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .interrupt-error {
    margin: 0 0 16px;
    color: var(--danger);
    font-size: 12px;
  }

  .btn-deny {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-deny:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-confirm {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    color: var(--on-accent);
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .btn-confirm:hover {
    background: var(--accent-hover);
    color: var(--on-accent);
  }

  .btn-deny:disabled,
  .btn-confirm:disabled {
    cursor: wait;
    opacity: 0.6;
  }
</style>

<script lang="ts">
  import { _ } from "svelte-i18n";
  import type { UndoPreview } from "../stores/session";

  interface Props {
    preview: UndoPreview;
    submitting?: boolean;
    error?: string;
    onconfirm: () => void;
    oncancel: () => void;
  }

  let { preview, submitting = false, error = "", onconfirm, oncancel }: Props = $props();

  let ready = $derived(preview.items.filter((item) => item.state === "ready"));
  let leftAlone = $derived(preview.items.filter((item) => item.state !== "ready"));
</script>

<div class="confirm-overlay">
  <div class="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="undo-title">
    <div class="confirm-header">
      <span class="undo-icon" aria-hidden="true">&#8630;</span>
      <span id="undo-title">{$_("undo.title")}</span>
    </div>

    <p class="confirm-body">{$_("undo.intro")}</p>

    {#if ready.length}
      <h4>{$_("undo.will_undo")}</h4>
      <ul class="undo-list" data-testid="undo-ready">
        {#each ready as item (item.seq)}
          <li>
            <span class="item-text">{item.description}</span>
            {#if item.target}<code>{item.target}</code>{/if}
          </li>
        {/each}
      </ul>
    {:else}
      <p class="confirm-body">{$_("undo.nothing_ready")}</p>
    {/if}

    {#if leftAlone.length}
      <h4>{$_("undo.left_alone")}</h4>
      <ul class="undo-list muted" data-testid="undo-left-alone">
        {#each leftAlone as item (item.seq)}
          <li>
            <span class="item-text">{item.description}</span>
            {#if item.target}<code>{item.target}</code>{/if}
          </li>
        {/each}
      </ul>
    {/if}

    {#if preview.not_undoable.length}
      <h4>{$_("undo.cannot_undo")}</h4>
      <ul class="undo-list muted" data-testid="undo-not-undoable">
        {#each preview.not_undoable as item, i (i)}
          <li>
            <span class="item-text"><strong>{item.type}</strong> {item.reason}</span>
            {#if item.target}<code>{item.target}</code>{/if}
          </li>
        {/each}
      </ul>
    {/if}

    {#if error}
      <p class="confirm-error" role="alert">{error}</p>
    {/if}

    <div class="confirm-actions">
      <button class="btn-deny" disabled={submitting} onclick={oncancel}>{$_("undo.cancel")}</button>
      <button class="btn-confirm" disabled={submitting || ready.length === 0} onclick={onconfirm}>
        {submitting ? $_("undo.working") : $_("undo.confirm")}
      </button>
    </div>
  </div>
</div>

<style>
  .confirm-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .confirm-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 520px;
    width: 100%;
    max-height: 100%;
    overflow-y: auto;
    box-shadow: var(--shadow);
  }

  .confirm-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .undo-icon {
    color: var(--accent);
    font-size: 20px;
  }

  .confirm-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 14px;
    line-height: 1.5;
  }

  h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 4px 0 6px;
  }

  .undo-list {
    list-style: none;
    margin: 0 0 14px;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .undo-list li {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 12px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    background: var(--bg-tertiary);
  }

  .undo-list.muted li {
    color: var(--text-secondary);
  }

  .undo-list code {
    font-size: 11px;
    color: var(--text-muted, var(--text-secondary));
    overflow-wrap: anywhere;
  }

  .confirm-error {
    font-size: 12px;
    color: var(--danger);
    margin-bottom: 12px;
  }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
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
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: opacity 0.15s;
  }

  .btn-confirm:disabled,
  .btn-deny:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>

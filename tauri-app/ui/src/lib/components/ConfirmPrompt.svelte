<script lang="ts">
  interface Props {
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    onconfirm: () => void;
    oncancel: () => void;
  }

  let { message, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false, onconfirm, oncancel }: Props =
    $props();
</script>

<div class="confirm-overlay">
  <div class="confirm-dialog">
    <div class="confirm-header">
      <span class="warn-icon">&#9888;</span>
    </div>

    <p class="confirm-body">{message}</p>

    <div class="confirm-actions">
      <button class="btn-deny" onclick={oncancel}>{cancelLabel}</button>
      <button class={danger ? "btn-confirm btn-danger" : "btn-confirm"} onclick={onconfirm}>
        {confirmLabel}
      </button>
    </div>
  </div>
</div>

<style>
  .confirm-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 200;
    padding: 24px;
  }

  .confirm-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 480px;
    width: 100%;
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

  .warn-icon {
    color: var(--warning);
    font-size: 20px;
  }

  .confirm-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
    white-space: pre-line;
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
    transition: background 0.15s;
  }

  .btn-confirm:hover {
    background: var(--accent-hover);
  }

  .btn-danger {
    background: var(--danger);
  }

  .btn-danger:hover {
    background: var(--danger);
    opacity: 0.85;
  }
</style>

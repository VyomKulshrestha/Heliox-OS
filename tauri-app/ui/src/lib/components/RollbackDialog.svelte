<script lang="ts">
  interface Props {
    onconfirm: () => void;
    oncancel: () => void;
  }

  let { onconfirm, oncancel }: Props = $props();
</script>

<div class="confirm-overlay">
  <div class="confirm-dialog">
    <div class="confirm-header">
      <span class="warn-icon">&#9888;</span>
      <span>Undo Last Action?</span>
    </div>

    <p class="confirm-body">
      This will revert your entire disk state to the snapshot taken just before the last
      action ran &mdash; including any other changes made since then, not just what the
      action did. This cannot be undone.
    </p>

    <div class="confirm-actions">
      <button class="btn-deny" title="Cancel" onclick={oncancel}>Cancel</button>
      <button class="btn-confirm btn-danger" title="Revert everything" onclick={onconfirm}>
        Yes, revert everything
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
    color: var(--danger);
    font-size: 20px;
  }

  .confirm-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
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

  .btn-danger {
    background: var(--danger);
  }

  .btn-danger:hover {
    background: var(--danger);
    opacity: 0.85;
  }
</style>

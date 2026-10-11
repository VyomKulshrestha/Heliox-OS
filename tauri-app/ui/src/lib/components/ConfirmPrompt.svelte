<script lang="ts">
  import { onMount } from "svelte";

  interface Props {
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    onconfirm: () => void;
    oncancel: () => void;
  }

  let {
    message,
    confirmLabel = "Confirm",
    cancelLabel = "Cancel",
    danger = false,
    onconfirm,
    oncancel,
  }: Props = $props();

  const uid = $props.id();
  const messageId = `${uid}-message`;
  let dialog = $state<HTMLDivElement>();
  let cancelButton = $state<HTMLButtonElement>();
  let confirmButton = $state<HTMLButtonElement>();

  onMount(() => {
    // Remember what had focus so closing the prompt puts the person back where they were.
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    // A risky action starts on Cancel so a stray Enter does not confirm it.
    (danger ? cancelButton : confirmButton)?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  });

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      oncancel();
      return;
    }
    if (event.key !== "Tab" || !dialog) return;
    // Keep Tab and Shift+Tab inside the prompt while it is open.
    const buttons = Array.from(dialog.querySelectorAll<HTMLButtonElement>("button:not([disabled])"));
    if (buttons.length === 0) return;
    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || !dialog.contains(active))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || !dialog.contains(active))) {
      event.preventDefault();
      first.focus();
    }
  }

  function keepFocus(event: MouseEvent) {
    // A click on the dimmed backdrop must not move focus to the page behind it.
    if (event.target === event.currentTarget) event.preventDefault();
  }
</script>

<div class="confirm-overlay" role="presentation" onmousedown={keepFocus}>
  <div
    bind:this={dialog}
    class="confirm-dialog"
    role={danger ? "alertdialog" : "dialog"}
    aria-modal="true"
    aria-label={confirmLabel}
    aria-describedby={messageId}
    tabindex="-1"
    onkeydown={handleKeydown}
  >
    <div class="confirm-header">
      <span class="warn-icon" aria-hidden="true">&#9888;</span>
    </div>

    <p class="confirm-body" id={messageId}>{message}</p>

    <div class="confirm-actions">
      <button bind:this={cancelButton} class="btn-deny" onclick={oncancel}>{cancelLabel}</button>
      <button bind:this={confirmButton} class={danger ? "btn-confirm btn-danger" : "btn-confirm"} onclick={onconfirm}>
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

  .confirm-dialog:focus {
    outline: none;
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

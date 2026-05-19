<script lang="ts">
  interface Props {
    show: boolean;
    onconfirm: () => void;
    oncancel: () => void;
  }

  let { show, onconfirm, oncancel }: Props = $props();

  function onBackdropClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      oncancel();
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Escape" && show) {
      oncancel();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if show}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="modal-backdrop" onclick={onBackdropClick}>
    <div class="modal-content">
      <div class="modal-header">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="alert-icon"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
        <h3>Reset Settings to Defaults?</h3>
      </div>
      
      <div class="modal-body">
        <p>This action will clear all your custom configurations, including:</p>
        <ul>
          <li>Model selections and API keys</li>
          <li>Security and restriction settings</li>
          <li>Saved UI preferences</li>
        </ul>
        <p class="warning-text">This action cannot be undone.</p>
      </div>

      <div class="modal-actions">
        <button class="btn-cancel" onclick={oncancel}>Cancel</button>
        <button class="btn-danger" onclick={onconfirm}>Yes, Reset Everything</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
  }

  .modal-content {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg, 12px);
    width: 100%;
    max-width: 420px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    overflow: hidden;
    animation: slideUp 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes slideUp {
    from {
      opacity: 0;
      transform: translateY(20px) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .modal-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    background: rgba(239, 68, 68, 0.1);
  }

  .alert-icon {
    color: #ef4444;
  }

  h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .modal-body {
    padding: 24px;
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  .modal-body p {
    margin: 0 0 12px 0;
  }

  .modal-body ul {
    margin: 0 0 16px 0;
    padding-left: 20px;
  }

  .modal-body li {
    margin-bottom: 4px;
  }

  .warning-text {
    color: #ef4444;
    font-weight: 500;
    margin: 0 !important;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-tertiary);
    border-top: 1px solid var(--border);
  }

  button {
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    border-radius: var(--radius-md, 6px);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-cancel {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-primary);
  }

  .btn-cancel:hover {
    background: var(--bg-primary);
  }

  .btn-danger {
    background: #ef4444;
    border: 1px solid transparent;
    color: white;
  }

  .btn-danger:hover {
    background: #dc2626;
  }
</style>

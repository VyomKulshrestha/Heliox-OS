<script lang="ts">
  import { call } from "../api/daemon";

  // Props
  let { onReplay }: { onReplay: (command: string) => void } = $props();

  // State
  let isOpen = $state(false);
  type HistoryEntry = {
    raw_input: string;
    created_at: string;
    execution_status: string;
  };

  let history: HistoryEntry[] = $state([]);
  let loading = $state(false);

  // Fetch history when panel opens
  async function fetchHistory() {
    loading = true;
    try {
      const result = await call<{ plans?: HistoryEntry[] }>("get_plan_history", { limit: 30, offset: 0 });
      history = result.plans ?? [];
    } catch (e) {
      history = [];
    } finally {
      loading = false;
    }
  }

  function togglePanel() {
    isOpen = !isOpen;
    if (isOpen) fetchHistory();
  }

  function formatTime(iso: string): string {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function statusClass(status: string): string {
    if (status === "success") return "status-success";
    if (status === "error" || status === "partial_failure") return "status-error";
    return "status-other";
  }

  function statusLabel(status: string): string {
    if (status === "success") return "✓";
    if (status === "error") return "✗";
    if (status === "partial_failure") return "⚠";
    return "•";
  }
</script>

<div class="history-wrapper">
  <button class="history-toggle" onclick={togglePanel} title="Command History">
    🕒 History
  </button>

  {#if isOpen}
    <div class="history-panel">
      <div class="history-header">
        <span class="history-title">Command History</span>
        <button class="close-btn" onclick={() => isOpen = false}>✕</button>
      </div>

      {#if loading}
        <div class="history-loading">Loading...</div>
      {:else if history.length === 0}
        <div class="history-empty">No commands yet.</div>
      {:else}
        <div class="history-list">
          {#each history as entry}
            <div class="history-item">
              <span class="status-dot {statusClass(entry.execution_status)}">
                {statusLabel(entry.execution_status)}
              </span>
              <div class="entry-info">
                <span class="entry-cmd">{entry.raw_input}</span>
                <span class="entry-time">{formatTime(entry.created_at)}</span>
              </div>
              <button
                class="replay-btn"
                title="Replay this command"
                onclick={() => { onReplay(entry.raw_input); isOpen = false; }}
              >
                ↩ Replay
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .history-wrapper {
    position: relative;
  }

  .history-toggle {
    padding: 5px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    background: transparent;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }
  .history-toggle:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .history-panel {
    position: absolute;
    bottom: 110%;
    left: 0;
    width: 340px;
    max-height: 360px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    z-index: 100;
    animation: fadeIn 0.15s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .history-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .close-btn {
    font-size: 12px;
    color: var(--text-muted);
    background: none;
  }
  .close-btn:hover { color: var(--text-primary); }

  .history-loading,
  .history-empty {
    padding: 20px;
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
  }

  .history-list {
    overflow-y: auto;
    flex: 1;
  }

  .history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }
  .history-item:hover { background: var(--bg-hover); }

  .status-dot {
    font-size: 12px;
    font-weight: 700;
    width: 18px;
    text-align: center;
    flex-shrink: 0;
  }
  .status-success { color: var(--success); }
  .status-error   { color: var(--danger); }
  .status-other   { color: var(--text-muted); }

  .entry-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .entry-cmd {
    font-size: 12px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .entry-time {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 1px;
  }

  .replay-btn {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: var(--radius-sm);
    color: var(--accent);
    background: var(--accent-muted);
    flex-shrink: 0;
    transition: all 0.15s;
  }
  .replay-btn:hover {
    background: var(--accent);
    color: white;
  }
</style>

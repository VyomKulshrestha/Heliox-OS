<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireResultStatus } from "../api/daemon";
  import { invoke } from "../api/invoke";
  import { loadLocalCommandHistory, type LocalCommandStatus } from "../utils/chatSessions";
  import BackgroundTasksPanel from "./BackgroundTasksPanel.svelte";

  async function openLogsFolder() {
    await invoke("open_logs_folder");
  }

  interface HistoryEntry {
    id: number | string;
    timestamp: string;
    user_input: string;
    outcome: LocalCommandStatus;
    explanation: string;
  }

  let entries: HistoryEntry[] = $state([]);
  let loading = $state(true);
  let historyNotice = $state("");

  function localEntries(): HistoryEntry[] {
    return loadLocalCommandHistory(typeof localStorage === "undefined" ? null : localStorage).map((entry) => ({
      id: entry.id,
      timestamp: new Date(entry.timestamp).toISOString(),
      user_input: entry.text,
      outcome: entry.status,
      explanation: entry.explanation,
    }));
  }

  onMount(async () => {
    try {
      const result = (await call("get_history", { limit: 100 })) as {
        status?: string;
        message?: string;
        error?: string;
        entries?: Array<Omit<HistoryEntry, "outcome"> & { success: boolean }>;
      };
      requireResultStatus(result, "ok", "Activity history is unavailable.");
      let loaded: HistoryEntry[] = (result.entries ?? []).map(({ success, ...entry }) => ({
        ...entry,
        outcome: success ? "success" : "error",
      }));
      if (loaded.length === 0) {
        loaded = localEntries();
        if (loaded.length > 0) historyNotice = "Showing local chat records because daemon activity history is empty.";
      }
      entries = loaded;
    } catch (cause) {
      entries = localEntries();
      historyNotice = `Daemon activity history is unavailable; showing local records only. ${
        cause instanceof Error ? cause.message : String(cause)
      }`;
    } finally {
      loading = false;
    }
  });

  function formatTime(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleString();
    } catch {
      return iso;
    }
  }
</script>

<div class="activity-log">
  <div class="log-header">
    <h2>Activity Log</h2>
    <div class="header-right">
      <span class="count">{entries.length} entries</span>
      <button class="open-logs-btn" onclick={openLogsFolder} title="Open Logs Folder"> 📂 Open Logs </button>
    </div>
  </div>

  <BackgroundTasksPanel />

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    {#if historyNotice}
      <div class="history-notice" role="status">{historyNotice}</div>
    {/if}
    {#if entries.length === 0}
      <div class="empty">No activity yet. Send a command to get started.</div>
    {:else}
      <div class="log-list">
        {#each entries as entry}
          <div
            class="log-entry"
            class:failed={entry.outcome === "error" || entry.outcome === "partial_failure"}
            class:unverified={entry.outcome === "unverified"}
          >
            <div class="entry-header">
              <span
                class="entry-status"
                class:success={entry.outcome === "success"}
                class:unverified={entry.outcome === "unverified"}
              >
                {entry.outcome === "success"
                  ? "OK"
                  : entry.outcome === "partial_failure"
                    ? "PARTIAL"
                    : entry.outcome === "unverified"
                      ? "LOCAL"
                      : "FAIL"}
              </span>
              <span class="entry-time">{formatTime(entry.timestamp)}</span>
            </div>
            <div class="entry-input">{entry.user_input}</div>
            {#if entry.explanation}
              <div class="entry-explanation">{entry.explanation}</div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .activity-log {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .log-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .empty {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 13px;
  }

  .log-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 16px;
  }

  .history-notice {
    padding: 8px 16px;
    border-bottom: 1px solid var(--warning);
    color: var(--warning);
    font-size: 11px;
  }

  .log-entry {
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
  }

  .log-entry.failed {
    border-color: var(--danger);
  }

  .log-entry.unverified {
    border-color: var(--warning);
  }

  .entry-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .entry-status {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 8px;
    border-radius: 10px;
    background: var(--danger-bg);
    color: var(--danger);
  }

  .entry-status.success {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .entry-status.unverified {
    background: rgba(245, 158, 11, 0.12);
    color: var(--warning);
  }

  .entry-time {
    font-size: 11px;
    color: var(--text-muted);
  }

  .entry-input {
    font-size: 13px;
    color: var(--text-primary);
  }

  .entry-explanation {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .open-logs-btn {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
    transition: background 0.15s;
  }
  .open-logs-btn:hover {
    background: var(--accent, #38bdf8);
    color: #000;
  }
</style>

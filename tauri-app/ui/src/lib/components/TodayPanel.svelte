<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { session } from "../stores/session";
  import { settings } from "../stores/settings";

  type TaskRow = {
    receipt_id: string;
    finished_at: string;
    status: string;
    summary: string;
    request: string;
    action_count: number;
  };
  type AppTime = { app: string; seconds: number; top_titles: string[] };
  type UndoRow = { task_key: string; undoable: number; targets: string[] };
  type Overview = {
    status: string;
    message?: string;
    label: string;
    brief: string;
    tasks: TaskRow[];
    timeline_enabled: boolean;
    usage: { total_seconds: number; apps: AppTime[] } | null;
    undoable: UndoRow[];
  };

  let day = $state<"today" | "yesterday">("today");
  let overview = $state<Overview | null>(null);
  let error = $state("");
  let notice = $state("");
  let saving = $state(false);

  let briefOn = $derived(Boolean($settings.brief?.enabled));
  let topSeconds = $derived(Math.max(1, ...(overview?.usage?.apps ?? []).map((entry) => entry.seconds)));

  function duration(seconds: number): string {
    const minutes = Math.max(1, Math.round(seconds / 60));
    if (minutes < 60) return `${minutes} min`;
    const rest = minutes % 60;
    return rest ? `${Math.floor(minutes / 60)} h ${rest} min` : `${minutes / 60} h`;
  }

  function clock(iso: string): string {
    return new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  async function load() {
    error = "";
    try {
      const result = await call<Overview>("today_overview", { day });
      if (result.status !== "ok") throw new Error(result.message || "The overview is not available.");
      overview = result;
    } catch (err) {
      error = err instanceof Error ? err.message : "The overview is not available.";
    }
  }

  function show(next: "today" | "yesterday") {
    day = next;
    void load();
  }

  async function saveBrief(values: Record<string, unknown>) {
    saving = true;
    const result = await settings.updateSectionDetailed("brief", values);
    saving = false;
    if (!result.ok) error = result.message;
  }

  async function sendNow() {
    try {
      const result = await call<{ status: string; message?: string }>("daily_brief_now");
      notice = result.status === "ok" ? "Sent. It also appears in the Command chat." : result.message || "Not sent.";
    } catch (err) {
      notice = err instanceof Error ? err.message : "Not sent.";
    }
  }

  onMount(load);
</script>

<div class="today" data-testid="today-panel">
  <header class="today-header">
    <div class="switch" role="group" aria-label="Day">
      <button type="button" class:active={day === "today"} onclick={() => show("today")}>Today</button>
      <button type="button" class:active={day === "yesterday"} onclick={() => show("yesterday")}>Yesterday</button>
    </div>
    <button type="button" class="quiet" onclick={load}>Refresh</button>
  </header>

  <div class="today-body">
    {#if error}<p class="message" role="alert">{error}</p>{/if}

    {#if overview}
      <p class="brief">{overview.brief}</p>

      <section aria-labelledby="today-tasks">
        <h3 id="today-tasks">What Heliox did</h3>
        {#if overview.tasks.length === 0}
          <p class="empty">No tasks ran actions {overview.label}.</p>
        {:else}
          <ul class="rows">
            {#each [...overview.tasks].reverse() as task (task.receipt_id)}
              <li>
                <span class="time">{clock(task.finished_at)}</span>
                <span class="row-text">
                  <strong>{task.request || task.summary || "Task"}</strong>
                  <small class:bad={task.status !== "success" && task.status !== "completed"}
                    >{task.status} · {task.action_count} action{task.action_count === 1 ? "" : "s"}</small
                  >
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <section aria-labelledby="today-time">
        <h3 id="today-time">Where your time went</h3>
        {#if !overview.timeline_enabled}
          <p class="empty">Turn on the activity timeline in Settings to see which apps you used.</p>
        {:else if !overview.usage || overview.usage.apps.length === 0}
          <p class="empty">Nothing recorded {overview.label}.</p>
        {:else}
          <ul class="bars">
            {#each overview.usage.apps as entry (entry.app)}
              <li>
                <span class="bar-label"><strong>{entry.app}</strong> {duration(entry.seconds)}</span>
                <span class="bar" style:width={`${Math.round((entry.seconds / topSeconds) * 100)}%`}></span>
                {#if entry.top_titles[0]}<small>{entry.top_titles[0]}</small>{/if}
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <section aria-labelledby="today-undo">
        <h3 id="today-undo">Can still be undone</h3>
        {#if overview.undoable.length === 0}
          <p class="empty">Nothing to undo {overview.label}.</p>
        {:else}
          <ul class="rows">
            {#each overview.undoable as row (row.task_key)}
              <li>
                <span class="row-text">
                  <strong>{row.targets.join(", ") || "Changes"}</strong>
                  <small>{row.undoable} change{row.undoable === 1 ? "" : "s"}</small>
                </span>
                <button type="button" onclick={() => session.openUndoPreview(row.task_key)}>Undo</button>
              </li>
            {/each}
          </ul>
        {/if}
      </section>
    {/if}

    <section aria-labelledby="today-brief">
      <h3 id="today-brief">Daily brief</h3>
      <p class="empty">One short summary of the day at a time you choose, as a notification and in chat.</p>
      <div class="brief-controls">
        <label
          ><input
            type="checkbox"
            checked={briefOn}
            disabled={saving}
            onchange={(event) => saveBrief({ enabled: event.currentTarget.checked })}
          /> Send a daily brief at</label
        >
        <input
          type="time"
          value={$settings.brief?.time ?? "18:00"}
          disabled={saving}
          aria-label="Brief time"
          onchange={(event) => saveBrief({ time: event.currentTarget.value })}
        />
        <label
          ><input
            type="checkbox"
            checked={Boolean($settings.brief?.speak)}
            disabled={saving}
            onchange={(event) => saveBrief({ speak: event.currentTarget.checked })}
          /> Read it aloud</label
        >
        <button type="button" onclick={sendNow}>Send now</button>
      </div>
      {#if notice}<p class="empty" role="status">{notice}</p>{/if}
    </section>
  </div>
</div>

<style>
  .today {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .today-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }
  .today-body {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .switch {
    display: flex;
    gap: 4px;
  }
  button {
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }
  .switch button.active {
    border-color: var(--accent);
    color: var(--accent);
  }
  button.quiet {
    color: var(--text-secondary);
  }
  .brief {
    margin: 0;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-secondary);
    font-size: 14px;
    line-height: 1.55;
  }
  h3 {
    margin: 0 0 8px;
    font-size: 13px;
  }
  .empty,
  small {
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  small.bad {
    color: var(--danger);
  }
  .message {
    margin: 0;
    color: var(--danger);
    font-size: 12px;
  }
  .rows,
  .bars {
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .rows li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .time {
    flex-shrink: 0;
    width: 64px;
    color: var(--text-secondary);
    font: 11px var(--font-mono);
  }
  .row-text {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .row-text strong {
    overflow: hidden;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bars li {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .bar-label {
    font-size: 12px;
    color: var(--text-secondary);
  }
  .bar-label strong {
    color: var(--text-primary);
  }
  .bar {
    display: block;
    height: 6px;
    min-width: 4px;
    border-radius: 3px;
    background: var(--accent);
  }
  .brief-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin: 8px 0 0;
    font-size: 12px;
  }
  .brief-controls label {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .brief-controls input[type="time"] {
    padding: 5px 8px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font: inherit;
  }
</style>

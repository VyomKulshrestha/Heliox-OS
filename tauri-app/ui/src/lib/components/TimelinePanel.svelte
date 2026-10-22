<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  type Span = { start: number; end: number; app: string; title: string };
  type AppTime = { app: string; seconds: number; top_titles: string[] };
  type Status = { status: string; message?: string; paused_until?: number | null; span_count?: number };

  let status = $state<Status | null>(null);
  let today = $state<AppTime[]>([]);
  let query = $state("");
  let results = $state<Span[] | null>(null);
  let error = $state("");
  let saving = $state(false);
  let clearArmed = $state(false);

  let enabled = $derived(Boolean($settings.timeline?.enabled));

  function duration(seconds: number): string {
    const minutes = Math.max(1, Math.round(seconds / 60));
    if (minutes < 60) return `${minutes} min`;
    const rest = minutes % 60;
    return rest ? `${Math.floor(minutes / 60)} h ${rest} min` : `${minutes / 60} h`;
  }

  function clock(epoch: number): string {
    return new Date(epoch * 1000).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
  }

  async function request<T extends { status: string; message?: string }>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T | null> {
    error = "";
    try {
      const result = await call<T>(method, params);
      if (result.status !== "ok") throw new Error(result.message || "The daemon refused the request.");
      return result;
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
      return null;
    }
  }

  async function load() {
    status = await request<Status>("timeline_status");
    const summary = await request<{ status: string; summary?: { apps?: AppTime[] } }>("timeline_summary", {
      when: "today",
    });
    today = summary?.summary?.apps ?? [];
  }

  async function toggle() {
    saving = true;
    const result = await settings.updateSectionDetailed("timeline", { enabled: !enabled });
    saving = false;
    if (!result.ok) error = result.message;
    await load();
  }

  async function pause(minutes: number) {
    await request("timeline_pause", { minutes });
    await load();
  }

  async function clearAll() {
    if (!clearArmed) {
      clearArmed = true;
      setTimeout(() => (clearArmed = false), 5000);
      return;
    }
    clearArmed = false;
    await request("timeline_clear", { confirmed: true });
    results = null;
    await load();
  }

  async function search(event: SubmitEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    const result = await request<{ status: string; spans: Span[] }>("timeline_search", { query: query.trim() });
    results = result?.spans ?? [];
  }

  onMount(load);
</script>

<section class="timeline-card" aria-labelledby="timeline-heading" data-testid="timeline-panel">
  <div class="section-title">
    <div>
      <span class="eyebrow">Memory</span>
      <h3 id="timeline-heading">Activity timeline</h3>
      <p>
        Records which app and window title you had in front, and when, so you can ask "what was I working on yesterday
        afternoon?". Never pixels or screen text. Password managers and private windows are skipped, and it stays on
        this computer for {$settings.timeline?.retention_days ?? 7} days.
      </p>
      {#if enabled && status}
        <small
          >{status.span_count ?? 0} entries{status.paused_until
            ? ` · paused until ${new Date(status.paused_until * 1000).toLocaleTimeString()}`
            : ""}</small
        >
      {/if}
    </div>
    <button
      class="toggle"
      class:active={enabled}
      onclick={toggle}
      disabled={saving}
      aria-label="Record activity timeline"
      title="Record activity timeline"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}

  {#if enabled}
    <div class="controls">
      {#if status?.paused_until}
        <button type="button" onclick={() => pause(0)}>Resume</button>
      {:else}
        <button type="button" onclick={() => pause(60)}>Pause 1 hour</button>
      {/if}
      <button type="button" onclick={clearAll}>{clearArmed ? "Click again to delete" : "Clear timeline"}</button>
    </div>

    {#if today.length}
      <ul class="rows" aria-label="Today">
        {#each today as entry (entry.app)}
          <li>
            <strong>{entry.app}</strong>
            <small>{duration(entry.seconds)}{entry.top_titles[0] ? ` · ${entry.top_titles[0]}` : ""}</small>
          </li>
        {/each}
      </ul>
    {/if}

    <form class="search" onsubmit={search}>
      <input
        bind:value={query}
        maxlength="200"
        placeholder="Find a window, file or page"
        aria-label="Search timeline"
      />
      <button type="submit">Search</button>
    </form>
    {#if results}
      {#if results.length === 0}
        <p class="empty">Nothing found.</p>
      {:else}
        <ul class="rows" aria-label="Search results">
          {#each results as span (span.start)}
            <li>
              <strong>{span.title || span.app}</strong>
              <small>{span.app} · {clock(span.start)} · {duration(span.end - span.start)}</small>
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  {/if}
</section>

<style>
  .timeline-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p,
  small {
    display: block;
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .eyebrow {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
    font-size: 11px;
  }
  .controls,
  .search {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .search input {
    flex: 1;
    min-width: 0;
    padding: 6px 9px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 12px;
  }
  .empty {
    padding: 0 20px 14px;
  }
  .rows {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .rows li {
    padding: 9px 20px;
    border-top: 1px solid var(--border);
    min-width: 0;
  }
  .rows strong {
    display: block;
    overflow: hidden;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  button:not(.toggle) {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  .toggle {
    position: relative;
    flex-shrink: 0;
    width: 40px;
    height: 22px;
    border-radius: 11px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    cursor: pointer;
  }
  .toggle.active {
    background: var(--accent);
  }
  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    transition: transform 0.15s;
  }
  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>

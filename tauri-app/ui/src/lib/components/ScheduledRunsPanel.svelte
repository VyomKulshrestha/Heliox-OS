<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type ScheduledRun = {
    run_id: string;
    name: string;
    goal: string;
    allowed_actions: string[];
    every_minutes: number;
    daily_at: string;
    weekdays: number[];
    enabled: boolean;
    next_run_at: number;
    last_status: string;
    last_job_status: string;
  };

  type Result = { status: string; message?: string; run?: ScheduledRun; runs?: ScheduledRun[] };

  const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const READ_AND_NOTIFY = "file_read, file_list, file_search, system_info, disk_usage, memory_usage, notify";

  let runs = $state<ScheduledRun[]>([]);
  let error = $state("");
  let busy = $state(false);
  let showForm = $state(false);

  let name = $state("");
  let goal = $state("");
  let timing = $state<"interval" | "daily">("daily");
  let everyMinutes = $state(60);
  let dailyAt = $state("09:00");
  let weekdays = $state<number[]>([]);
  let allowedActions = $state(READ_AND_NOTIFY);

  async function request(method: string, params: Record<string, unknown> = {}): Promise<Result | null> {
    error = "";
    try {
      const result = await call<Result>(method, params);
      if (result.status !== "ok") throw new Error(result.message || "The daemon refused the request.");
      return result;
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
      return null;
    }
  }

  async function load() {
    const result = await request("scheduled_run_list");
    if (result?.runs) runs = result.runs;
  }

  async function create(event: SubmitEvent) {
    event.preventDefault();
    busy = true;
    const actions = allowedActions
      .split(/[\s,]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    const result = await request("scheduled_run_create", {
      name,
      goal,
      allowed_actions: actions,
      ...(timing === "interval" ? { every_minutes: everyMinutes } : { daily_at: dailyAt, weekdays }),
    });
    busy = false;
    if (result?.run) {
      runs = [...runs, result.run];
      showForm = false;
      name = "";
      goal = "";
    }
  }

  async function update(method: string, run: ScheduledRun, extra: Record<string, unknown> = {}) {
    const result = await request(method, { run_id: run.run_id, ...extra });
    if (result?.run) runs = runs.map((item) => (item.run_id === run.run_id ? result.run! : item));
  }

  async function remove(run: ScheduledRun) {
    if (await request("scheduled_run_delete", { run_id: run.run_id })) {
      runs = runs.filter((item) => item.run_id !== run.run_id);
    }
  }

  function toggleDay(day: number) {
    weekdays = weekdays.includes(day) ? weekdays.filter((item) => item !== day) : [...weekdays, day].sort();
  }

  function timetable(run: ScheduledRun): string {
    if (run.every_minutes) return `Every ${run.every_minutes} min`;
    const days = run.weekdays.length ? run.weekdays.map((day) => DAYS[day]).join(", ") : "every day";
    return `Daily at ${run.daily_at}, ${days}`;
  }

  function lastRun(run: ScheduledRun): string {
    if (run.last_status === "started") return `Last run: ${run.last_job_status || "started"}`;
    if (run.last_status === "missed") return "Last run: missed while Heliox was off";
    if (run.last_status === "skipped_still_running") return "Last run: skipped, previous run still going";
    if (run.last_status === "failed_to_start") return "Last run: could not start";
    return "Not run yet";
  }

  onMount(() => void load());
</script>

<section class="schedule-card" aria-labelledby="scheduled-runs-heading">
  <div class="section-title">
    <div>
      <span class="eyebrow">Automation</span>
      <h3 id="scheduled-runs-heading">Scheduled runs</h3>
      <p>
        Heliox works on a goal by itself on a timetable, using only the actions you approve. Actions that need approval
        still wait for you.
      </p>
    </div>
    <button class="secondary" type="button" onclick={() => (showForm = !showForm)}>
      {showForm ? "Close" : "New schedule"}
    </button>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}

  {#if showForm}
    <form class="form" onsubmit={create}>
      <label>Name <input bind:value={name} maxlength="80" required /></label>
      <label>Goal <textarea bind:value={goal} maxlength="2000" rows="2" required></textarea></label>
      <div class="timing">
        <select bind:value={timing} aria-label="Timing">
          <option value="daily">Daily at</option>
          <option value="interval">Every N minutes</option>
        </select>
        {#if timing === "daily"}
          <input type="time" bind:value={dailyAt} aria-label="Time of day" />
          <div class="days" role="group" aria-label="Weekdays">
            {#each DAYS as day, index (day)}
              <label class="day"
                ><input type="checkbox" checked={weekdays.includes(index)} onchange={() => toggleDay(index)} />
                {day}</label
              >
            {/each}
          </div>
        {:else}
          <input type="number" min="5" bind:value={everyMinutes} aria-label="Minutes between runs" />
        {/if}
      </div>
      <label
        >Approved actions <input bind:value={allowedActions} />
        <small>Comma-separated action names. The run refuses any other action.</small></label
      >
      <button class="primary" type="submit" disabled={busy}>Save schedule</button>
    </form>
  {/if}

  {#if runs.length === 0 && !showForm}
    <p class="empty">No schedules yet.</p>
  {:else}
    <ul class="runs">
      {#each runs as run (run.run_id)}
        <li>
          <div class="run-text">
            <strong>{run.name}</strong>
            <small>{timetable(run)} · {lastRun(run)}</small>
            {#if run.enabled}<small>Next: {new Date(run.next_run_at * 1000).toLocaleString()}</small>{/if}
          </div>
          <div class="run-buttons">
            <label class="toggle"
              ><input
                type="checkbox"
                checked={run.enabled}
                aria-label={`Enable ${run.name}`}
                onchange={() => update("scheduled_run_set_enabled", run, { enabled: !run.enabled })}
              /> On</label
            >
            <button type="button" onclick={() => update("scheduled_run_now", run)}>Run now</button>
            <button type="button" onclick={() => remove(run)}>Delete</button>
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .schedule-card {
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
  .form {
    display: grid;
    gap: 10px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }
  .form label {
    display: grid;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .form input,
  .form textarea,
  .form select {
    padding: 7px 9px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font: inherit;
    font-size: 12px;
  }
  .timing {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .days {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .form .day,
  .toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
  }
  .empty {
    padding: 12px 20px 16px;
    border-top: 1px solid var(--border);
  }
  .runs {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .runs li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .run-text {
    min-width: 0;
  }
  .run-text strong {
    font-size: 12px;
  }
  .run-buttons {
    display: flex;
    flex-shrink: 0;
    align-items: center;
    gap: 6px;
  }
  button {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  button.primary {
    justify-self: start;
    border-color: var(--accent);
    color: var(--accent);
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>

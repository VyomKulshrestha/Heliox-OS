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

  let day = $state<"today" | "yesterday" | "week">("today");
  let focusMinutes = $state(25);
  type QuietWindow = { id: string; days: number[]; start: number; end: number };
  let quietWindows = $state<QuietWindow[]>([]);
  let quietDays = $state("weekdays");
  let quietFrom = $state("09:00");
  let quietTo = $state("12:00");
  const DAY_SETS: Record<string, number[]> = {
    everyday: [0, 1, 2, 3, 4, 5, 6],
    weekdays: [0, 1, 2, 3, 4],
    weekends: [5, 6],
  };
  let overview = $state<Overview | null>(null);
  let error = $state("");
  let notice = $state("");
  let saving = $state(false);

  type Reminder = { id: string; due: number; text: string };
  type Workspace = { name: string; apps: string[] };
  type Note = { id: string; text: string; created_at: number };
  type Nudge = { id: string; match: string; minutes: number };
  type Todo = { id: string; text: string; created_at: number; done: boolean; done_at: number | null };
  let notes = $state<Note[]>([]);
  let noteText = $state("");
  let todos = $state<Todo[]>([]);
  let todoText = $state("");
  let openTodos = $derived(todos.filter((item) => !item.done));
  let doneTodos = $derived(todos.filter((item) => item.done));
  type Watcher = { id: string; kind: string; label: string; created_at: number };
  type Copied = { id: string; text: string; at: number; app: string };
  let copied = $state<Copied[]>([]);
  let clipboardOn = $derived(Boolean($settings.clipboard?.history));
  type HealthApp = { app: string; cpu: number; memory: number; processes: number };
  type Health = { summary: string; cpu: number; memory: { percent: number }; apps: HealthApp[] };
  let health = $state<Health | null>(null);
  let checking = $state(false);
  let watchers = $state<Watcher[]>([]);
  let watchText = $state("");
  let nudges = $state<Nudge[]>([]);
  let nudgeMatch = $state("");
  let nudgeMinutes = $state(60);
  let workspaces = $state<Workspace[]>([]);
  let workspaceName = $state("");
  let historyQuery = $state("");
  let history = $state<TaskRow[] | null>(null);
  let reminders = $state<Reminder[]>([]);
  let reminderText = $state("");
  let reminderMinutes = $state(20);

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

  async function startFocus() {
    try {
      const result = await call<{ status: string; message?: string }>("focus_start", {
        minutes: Number(focusMinutes),
      });
      notice = result.status === "ok" ? "Focus started. Reminders and the brief will wait." : result.message || "";
    } catch (err) {
      notice = err instanceof Error ? err.message : "Focus did not start.";
    }
  }

  async function startPomodoro() {
    try {
      const result = await call<{ status: string; message?: string }>("focus_start", {
        minutes: 25,
        break_minutes: 5,
      });
      notice = result.status === "ok" ? "Pomodoro started: 25 minutes, then a 5-minute break." : result.message || "";
    } catch (err) {
      notice = err instanceof Error ? err.message : "The pomodoro did not start.";
    }
  }

  function minutesOf(clock: string): number {
    const [hours, minutes] = clock.split(":").map(Number);
    return hours * 60 + minutes;
  }

  function clockOf(minutes: number): string {
    return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
  }

  function daysLabel(days: number[]): string {
    const key = days.join(",");
    if (key === "0,1,2,3,4,5,6") return "Every day";
    if (key === "0,1,2,3,4") return "Weekdays";
    if (key === "5,6") return "Weekends";
    return days.map((day) => ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day]).join(", ");
  }

  async function addQuietHours(event: SubmitEvent) {
    event.preventDefault();
    const next = await listCall<QuietWindow>("focus_schedule_add", "windows", {
      days: DAY_SETS[quietDays],
      start: minutesOf(quietFrom),
      end: minutesOf(quietTo),
    });
    if (next) quietWindows = next;
  }

  async function removeQuietHours(id: string) {
    quietWindows = (await listCall<QuietWindow>("focus_schedule_delete", "windows", { id })) ?? quietWindows;
  }

  async function exportReport() {
    try {
      const result = await call<{ status: string; message?: string; path?: string }>("report_export", { day });
      notice = result.status === "ok" ? `Report saved to ${result.path}` : result.message || "Not saved.";
    } catch (err) {
      notice = err instanceof Error ? err.message : "Not saved.";
    }
  }

  function show(next: "today" | "yesterday" | "week") {
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

  async function loadReminders() {
    try {
      const result = await call<{ status: string; reminders?: Reminder[] }>("reminder_list");
      reminders = result.reminders ?? [];
    } catch {
      reminders = [];
    }
  }

  async function addReminder(event: SubmitEvent) {
    event.preventDefault();
    const result = await call<{ status: string; message?: string; reminders?: Reminder[] }>("reminder_add", {
      text: reminderText,
      in_minutes: Number(reminderMinutes),
    });
    if (result.status !== "ok") error = result.message || "The reminder was not saved.";
    else {
      reminderText = "";
      reminders = result.reminders ?? reminders;
    }
  }

  async function cancelReminder(id: string) {
    const result = await call<{ status: string; reminders?: Reminder[] }>("reminder_cancel", { id });
    reminders = result.reminders ?? reminders.filter((item) => item.id !== id);
  }

  function dueText(epoch: number): string {
    return new Date(epoch * 1000).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
  }

  async function loadWorkspaces() {
    try {
      const result = await call<{ status: string; workspaces?: Workspace[] }>("workspace_list");
      workspaces = result.workspaces ?? [];
    } catch {
      workspaces = [];
    }
  }

  async function saveWorkspace(event: SubmitEvent) {
    event.preventDefault();
    const result = await call<{ status: string; message?: string; workspaces?: Workspace[] }>("workspace_save", {
      name: workspaceName,
    });
    if (result.status !== "ok") error = result.message || "The setup was not saved.";
    else {
      workspaceName = "";
      workspaces = result.workspaces ?? workspaces;
    }
  }

  async function deleteWorkspace(name: string) {
    const result = await call<{ status: string; workspaces?: Workspace[] }>("workspace_delete", { name });
    workspaces = result.workspaces ?? workspaces.filter((item) => item.name !== name);
  }

  async function searchHistory(event: SubmitEvent) {
    event.preventDefault();
    if (!historyQuery.trim()) return;
    const result = await call<{ status: string; message?: string; receipts?: TaskRow[] }>("receipts_search", {
      query: historyQuery.trim(),
    });
    history = result.receipts ?? [];
  }

  async function listCall<T>(method: string, key: string, params: Record<string, unknown> = {}): Promise<T[] | null> {
    try {
      const result = await call<Record<string, unknown>>(method, params);
      if (result.status !== "ok") {
        error = String(result.message || "The request failed.");
        return null;
      }
      return (result[key] as T[] | undefined) ?? [];
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
      return null;
    }
  }

  async function addNote(event: SubmitEvent) {
    event.preventDefault();
    const next = await listCall<Note>("note_add", "notes", { text: noteText });
    if (next) {
      notes = next;
      noteText = "";
    }
  }

  async function deleteNote(id: string) {
    notes = (await listCall<Note>("note_delete", "notes", { id })) ?? notes;
  }

  async function addTodo(event: SubmitEvent) {
    event.preventDefault();
    const next = await listCall<Todo>("todo_add", "todos", { text: todoText });
    if (next) {
      todos = next;
      todoText = "";
    }
  }

  async function setTodoDone(id: string, done: boolean) {
    todos = (await listCall<Todo>("todo_set_done", "todos", { id, done })) ?? todos;
  }

  async function deleteTodo(id: string) {
    todos = (await listCall<Todo>("todo_delete", "todos", { id })) ?? todos;
  }

  async function addWatcher(event: SubmitEvent) {
    event.preventDefault();
    const next = await listCall<Watcher>("watch_add", "watchers", { text: watchText });
    if (next) {
      watchers = next;
      watchText = "";
    }
  }

  async function cancelWatcher(id: string) {
    watchers = (await listCall<Watcher>("watch_cancel", "watchers", { id })) ?? watchers;
  }

  async function checkComputer() {
    checking = true;
    try {
      const result = await call<Health & { status: string; message?: string }>("system_health");
      if (result.status === "ok") health = result;
      else error = String(result.message || "The check failed.");
    } catch (err) {
      error = err instanceof Error ? err.message : "The check failed.";
    } finally {
      checking = false;
    }
  }

  async function loadCopied() {
    copied = (await listCall<Copied>("clipboard_history_list", "items")) ?? [];
  }

  async function copyAgain(id: string) {
    copied = (await listCall<Copied>("clipboard_history_copy", "items", { id })) ?? copied;
  }

  async function clearDoneTodos() {
    todos = (await listCall<Todo>("todo_clear_done", "todos")) ?? todos;
  }

  async function addNudge(event: SubmitEvent) {
    event.preventDefault();
    const next = await listCall<Nudge>("nudge_add", "nudges", { match: nudgeMatch, minutes: Number(nudgeMinutes) });
    if (next) {
      nudges = next;
      nudgeMatch = "";
    }
  }

  async function deleteNudge(id: string) {
    nudges = (await listCall<Nudge>("nudge_delete", "nudges", { id })) ?? nudges;
  }

  onMount(() => {
    void load();
    void loadReminders();
    void loadWorkspaces();
    void listCall<Note>("notes_list", "notes").then((items) => (notes = items ?? []));
    void listCall<Todo>("todo_list", "todos").then((items) => (todos = items ?? []));
    void listCall<Watcher>("watch_list", "watchers").then((items) => (watchers = items ?? []));
    void loadCopied();
    void listCall<Nudge>("nudge_list", "nudges").then((items) => (nudges = items ?? []));
    void listCall<QuietWindow>("focus_schedule_list", "windows").then((items) => (quietWindows = items ?? []));
  });
</script>

<div class="today" data-testid="today-panel">
  <header class="today-header">
    <div class="switch" role="group" aria-label="Day">
      <button type="button" class:active={day === "today"} onclick={() => show("today")}>Today</button>
      <button type="button" class:active={day === "yesterday"} onclick={() => show("yesterday")}>Yesterday</button>
      <button type="button" class:active={day === "week"} onclick={() => show("week")}>This week</button>
    </div>
    <span class="header-actions">
      <button type="button" class="quiet" onclick={exportReport}>Export report</button>
      <button type="button" class="quiet" onclick={load}>Refresh</button>
    </span>
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

    <section aria-labelledby="today-todos">
      <h3 id="today-todos">To-do</h3>
      {#if openTodos.length === 0}
        <p class="empty">Nothing to do. Say "add call the bank to my list", or add one here.</p>
      {:else}
        <ul class="rows">
          {#each openTodos as item (item.id)}
            <li>
              <span class="row-text"><strong>{item.text}</strong></span>
              <button type="button" aria-label={`Mark ${item.text} done`} onclick={() => setTodoDone(item.id, true)}
                >Done</button
              >
              <button type="button" aria-label={`Delete to-do ${item.text}`} onclick={() => deleteTodo(item.id)}
                >Delete</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      {#if doneTodos.length > 0}
        <ul class="rows">
          {#each doneTodos.slice(-5) as item (item.id)}
            <li>
              <span class="row-text"><s>{item.text}</s></span>
              <button
                type="button"
                aria-label={`Mark ${item.text} not done`}
                onclick={() => setTodoDone(item.id, false)}>Undo</button
              >
            </li>
          {/each}
        </ul>
        <button type="button" class="quiet" onclick={clearDoneTodos}>Clear finished ({doneTodos.length})</button>
      {/if}
      <form class="brief-controls" onsubmit={addTodo}>
        <input class="reminder-text" bind:value={todoText} maxlength="200" aria-label="New to-do" required />
        <button type="submit">Add to-do</button>
      </form>
    </section>

    <section aria-labelledby="today-watching">
      <h3 id="today-watching">Watching</h3>
      {#if watchers.length === 0}
        <p class="empty">Not watching anything. Say "tell me when Chrome closes", or type what to wait for here.</p>
      {:else}
        <ul class="rows">
          {#each watchers as watcher (watcher.id)}
            <li>
              <span class="row-text"><strong>Until {watcher.label}</strong></span>
              <button
                type="button"
                aria-label={`Stop watching until ${watcher.label}`}
                onclick={() => cancelWatcher(watcher.id)}>Stop</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={addWatcher}>
        <input
          class="reminder-text"
          bind:value={watchText}
          maxlength="200"
          aria-label="Tell me when"
          placeholder="Chrome closes"
          required
        />
        <button type="submit">Watch</button>
      </form>
    </section>

    {#if clipboardOn}
      <section aria-labelledby="today-copied">
        <h3 id="today-copied">Copied earlier</h3>
        {#if copied.length === 0}
          <p class="empty">Nothing yet. What you copy shows here for a few hours, in memory only.</p>
        {:else}
          <ul class="rows">
            {#each copied.slice(0, 8) as item (item.id)}
              <li>
                <span class="row-text"
                  ><strong>{item.text.length > 80 ? `${item.text.slice(0, 79)}…` : item.text}</strong></span
                >
                <button
                  type="button"
                  aria-label={`Copy again: ${item.text.slice(0, 40)}`}
                  onclick={() => copyAgain(item.id)}>Copy</button
                >
              </li>
            {/each}
          </ul>
        {/if}
        <button type="button" class="quiet" onclick={loadCopied}>Refresh copied</button>
      </section>
    {/if}

    <section aria-labelledby="today-computer">
      <h3 id="today-computer">This computer</h3>
      {#if health}
        <p class="brief">{health.summary}</p>
        <ul class="rows">
          {#each health.apps.slice(0, 5) as app (app.app)}
            <li>
              <span class="row-text"
                ><strong>{app.app}</strong>
                <small
                  >{(app.memory / 1024 ** 3).toFixed(1)} GB · {Math.round(app.cpu)}% processor{app.processes > 1
                    ? ` · ${app.processes} processes`
                    : ""}</small
                ></span
              >
            </li>
          {/each}
        </ul>
      {:else}
        <p class="empty">See what is using your processor, memory and disks. Or ask "why is my computer slow?"</p>
      {/if}
      <button type="button" class="quiet" onclick={checkComputer} disabled={checking}
        >{checking ? "Checking…" : "Check now"}</button
      >
    </section>

    <section aria-labelledby="today-notes">
      <h3 id="today-notes">Notes</h3>
      {#if notes.length === 0}
        <p class="empty">No notes. Say "note that the plumber comes Tuesday", or add one here.</p>
      {:else}
        <ul class="rows">
          {#each notes.slice(0, 8) as note (note.id)}
            <li>
              <span class="row-text"><strong>{note.text}</strong></span>
              <button type="button" aria-label={`Delete note ${note.text}`} onclick={() => deleteNote(note.id)}
                >Delete</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={addNote}>
        <input class="reminder-text" bind:value={noteText} maxlength="500" aria-label="Note" required />
        <button type="submit">Add note</button>
      </form>
    </section>

    <section aria-labelledby="today-nudges">
      <h3 id="today-nudges">Usage nudges</h3>
      <p class="empty">
        Heliox tells you once a day when an app or site passes your limit. Uses the activity timeline; nothing is
        blocked.
      </p>
      {#if nudges.length}
        <ul class="rows">
          {#each nudges as nudge (nudge.id)}
            <li>
              <span class="row-text"><strong>{nudge.match}</strong> <small>after {nudge.minutes} min a day</small></span
              >
              <button type="button" aria-label={`Remove nudge ${nudge.match}`} onclick={() => deleteNudge(nudge.id)}
                >Remove</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={addNudge}>
        <label
          >Tell me after <input
            class="minutes"
            type="number"
            min="1"
            max="1440"
            bind:value={nudgeMinutes}
            aria-label="Nudge minutes"
          /> min on</label
        >
        <input
          class="reminder-text"
          bind:value={nudgeMatch}
          maxlength="60"
          placeholder="YouTube"
          aria-label="App or site"
          required
        />
        <button type="submit">Add nudge</button>
      </form>
    </section>

    <section aria-labelledby="today-setups">
      <h3 id="today-setups">Setups</h3>
      <p class="empty">
        Save the apps you have open, then say "open my coding setup" to bring them back. Opening goes through the normal
        approvals.
      </p>
      {#if workspaces.length}
        <ul class="rows">
          {#each workspaces as workspace (workspace.name)}
            <li>
              <span class="row-text">
                <strong>{workspace.name}</strong>
                <small>{workspace.apps.join(", ")}</small>
              </span>
              <button type="button" onclick={() => session.sendCommand(`open my ${workspace.name} setup`)}>Open</button>
              <button
                type="button"
                aria-label={`Delete ${workspace.name}`}
                onclick={() => deleteWorkspace(workspace.name)}>Delete</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={saveWorkspace}>
        <input
          class="reminder-text"
          bind:value={workspaceName}
          maxlength="40"
          placeholder="coding"
          aria-label="Setup name"
          required
        />
        <button type="submit">Save open apps</button>
      </form>
    </section>

    <section aria-labelledby="today-history">
      <h3 id="today-history">Search what you asked</h3>
      <form class="brief-controls" onsubmit={searchHistory}>
        <input
          class="reminder-text"
          bind:value={historyQuery}
          maxlength="200"
          placeholder="invoice"
          aria-label="Search tasks"
        />
        <button type="submit">Search</button>
      </form>
      {#if history}
        {#if history.length === 0}
          <p class="empty">No task matches.</p>
        {:else}
          <ul class="rows">
            {#each history as task (task.receipt_id)}
              <li>
                <span class="time"
                  >{new Date(task.finished_at).toLocaleDateString([], { month: "short", day: "numeric" })}</span
                >
                <span class="row-text">
                  <strong>{task.request || task.summary}</strong>
                  <small>{task.status} · {task.action_count} action{task.action_count === 1 ? "" : "s"}</small>
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      {/if}
    </section>

    <section aria-labelledby="today-focus">
      <h3 id="today-focus">Focus</h3>
      <p class="empty">
        Holds reminders and the daily brief and keeps Heliox quiet; approvals and answers still come through. You can
        also say "focus for 45 minutes".
      </p>
      <div class="brief-controls">
        <label
          >Focus for <input
            class="minutes"
            type="number"
            min="1"
            max="480"
            bind:value={focusMinutes}
            aria-label="Focus minutes"
          /> min</label
        >
        <button type="button" onclick={startFocus}>Start focus</button>
        <button type="button" onclick={startPomodoro}>Start pomodoro</button>
      </div>
      {#if quietWindows.length}
        <ul class="rows">
          {#each quietWindows as window (window.id)}
            <li>
              <span class="row-text"
                ><strong>{daysLabel(window.days)}</strong>
                <small>{clockOf(window.start)} to {clockOf(window.end)}</small></span
              >
              <button
                type="button"
                aria-label={`Remove quiet hours ${clockOf(window.start)}`}
                onclick={() => removeQuietHours(window.id)}>Remove</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={addQuietHours}>
        <label
          >Quiet hours <select bind:value={quietDays} aria-label="Quiet days">
            <option value="weekdays">Weekdays</option>
            <option value="everyday">Every day</option>
            <option value="weekends">Weekends</option>
          </select></label
        >
        <input type="time" bind:value={quietFrom} aria-label="Quiet from" />
        <input type="time" bind:value={quietTo} aria-label="Quiet until" />
        <button type="submit">Add quiet hours</button>
      </form>
    </section>

    <section aria-labelledby="today-reminders">
      <h3 id="today-reminders">Reminders</h3>
      {#if reminders.length === 0}
        <p class="empty">No reminders. Say "remind me in 20 minutes to stretch", or add one here.</p>
      {:else}
        <ul class="rows">
          {#each reminders as reminder (reminder.id)}
            <li>
              <span class="time">{dueText(reminder.due)}</span>
              <span class="row-text"><strong>{reminder.text}</strong></span>
              <button type="button" aria-label={`Cancel ${reminder.text}`} onclick={() => cancelReminder(reminder.id)}
                >Cancel</button
              >
            </li>
          {/each}
        </ul>
      {/if}
      <form class="brief-controls" onsubmit={addReminder}>
        <label
          >Remind me in <input
            class="minutes"
            type="number"
            min="1"
            max="10080"
            bind:value={reminderMinutes}
            aria-label="Minutes"
          /> min to</label
        >
        <input
          class="reminder-text"
          bind:value={reminderText}
          maxlength="300"
          placeholder="stretch"
          aria-label="Reminder"
          required
        />
        <button type="submit">Add</button>
      </form>
    </section>

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
  .switch,
  .header-actions {
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
  .brief-controls .minutes {
    width: 64px;
  }
  .brief-controls .reminder-text {
    flex: 1;
    min-width: 140px;
  }
  .brief-controls input[type="number"],
  .brief-controls .reminder-text,
  .brief-controls input[type="time"] {
    padding: 5px 8px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font: inherit;
  }
</style>

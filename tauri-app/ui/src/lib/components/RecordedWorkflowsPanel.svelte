<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireOkResult, requireResultStatus, type DaemonStatusResult } from "../api/daemon";

  interface Step {
    action_type: string;
    parameters: Record<string, unknown>;
  }
  interface Workflow {
    id: string;
    name: string;
    goal: string;
    steps: Step[];
    step_count: number;
    sample_count: number;
    omitted_actions: number;
    incomplete: boolean;
    needs_review: boolean;
    schedule: { next_run_at: string; repeat_seconds: number } | null;
  }
  interface RecordingStatus {
    active: boolean;
    name: string;
    capture_visual: boolean;
    step_count: number;
    sample_count: number;
    omitted_actions: number;
  }

  let workflows = $state<Workflow[]>([]);
  let recording = $state<RecordingStatus>({
    active: false, name: "", capture_visual: false, step_count: 0, sample_count: 0, omitted_actions: 0,
  });
  let name = $state("");
  let goal = $state("");
  let consent = $state(false);
  let captureVisual = $state(false);
  let busy = $state(false);
  let error = $state("");
  let notice = $state("");
  let runAt = $state<Record<string, string>>({});
  let repeat = $state<Record<string, string>>({});
  let reviewed = $state<Record<string, boolean>>({});

  onMount(() => {
    void refresh();
    const timer = window.setInterval(() => { if (recording.active) void refreshStatus(); }, 4000);
    return () => window.clearInterval(timer);
  });

  function message(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  async function refreshStatus() {
    try {
      const result = requireOkResult(
        (await call("recorded_workflow_status", {})) as DaemonStatusResult & RecordingStatus,
        "Recording status unavailable",
      ) as DaemonStatusResult & RecordingStatus;
      recording = result;
    } catch (cause) { error = message(cause); }
  }

  async function refresh() {
    await refreshStatus();
    try {
      const result = requireOkResult(
        (await call("recorded_workflow_list", {})) as DaemonStatusResult & { workflows: Workflow[] },
        "Saved workflows unavailable",
      ) as DaemonStatusResult & { workflows: Workflow[] };
      workflows = result.workflows ?? [];
    } catch (cause) { error = message(cause); }
  }

  async function start() {
    if (busy || !consent || !name.trim()) return;
    busy = true; error = ""; notice = "";
    try {
      requireOkResult((await call("recorded_workflow_start", {
        name: name.trim(), goal: goal.trim(), consent: true, capture_visual: captureVisual,
      })) as DaemonStatusResult, "Could not start recording");
      consent = false;
      captureVisual = false;
      name = "";
      goal = "";
      notice = "Recording started. Only Heliox-executed actions are observed.";
      await refreshStatus();
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }

  async function stop(save: boolean) {
    if (busy) return;
    busy = true; error = ""; notice = "";
    try {
      requireOkResult((await call("recorded_workflow_stop", { save })) as DaemonStatusResult,
        "Could not stop recording");
      notice = save ? "Recording saved as a draft. Review its steps before running it." : "Recording discarded.";
      await refresh();
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }

  async function run(workflow: Workflow) {
    if (busy || workflow.incomplete) return;
    busy = true; error = ""; notice = "";
    try {
      requireResultStatus((await call("recorded_workflow_run", {
        workflow_id: workflow.id, reviewed: reviewed[workflow.id] === true,
      })) as DaemonStatusResult, "submitted", "Could not run workflow");
      notice = `${workflow.name} submitted through Heliox's normal approvals and verification.`;
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }

  async function schedule(workflow: Workflow) {
    const local = runAt[workflow.id];
    if (busy || !local || workflow.incomplete) return;
    const timestamp = new Date(local);
    if (Number.isNaN(timestamp.getTime()) || timestamp.getTime() <= Date.now()) {
      error = "Choose a future date and time.";
      return;
    }
    busy = true; error = ""; notice = "";
    try {
      requireOkResult((await call("recorded_workflow_schedule", {
        workflow_id: workflow.id,
        run_at: timestamp.toISOString(),
        repeat_seconds: Number(repeat[workflow.id] ?? "0"),
        reviewed: reviewed[workflow.id] === true,
      })) as DaemonStatusResult, "Could not schedule workflow");
      notice = `${workflow.name} scheduled. Heliox will still ask for required approvals.`;
      await refresh();
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }

  async function unschedule(workflow: Workflow) {
    if (busy) return;
    busy = true; error = ""; notice = "";
    try {
      requireOkResult((await call("recorded_workflow_unschedule", { workflow_id: workflow.id })) as DaemonStatusResult,
        "Could not remove schedule");
      notice = `${workflow.name} is no longer scheduled.`;
      await refresh();
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }

  async function remove(workflow: Workflow) {
    if (busy || !window.confirm(`Delete saved workflow “${workflow.name}” and its local visual samples?`)) return;
    busy = true; error = ""; notice = "";
    try {
      requireOkResult((await call("recorded_workflow_delete", { workflow_id: workflow.id })) as DaemonStatusResult,
        "Could not delete workflow");
      notice = "Saved workflow deleted.";
      await refresh();
    } catch (cause) { error = message(cause); }
    finally { busy = false; }
  }
</script>

<section class="recorded-panel" aria-labelledby="recorded-title">
  <div class="heading">
    <div>
      <h3 id="recorded-title">Recorded workflows</h3>
      <p>Turn actions Heliox executes into a reviewable recipe. Recording is always off after restart.</p>
    </div>
    <span class:live={recording.active} class="state">{recording.active ? "Recording" : "Off"}</span>
  </div>

  {#if recording.active}
    <div class="active-box" role="status">
      <strong>{recording.name}</strong>
      <span>{recording.step_count} replayable steps · {recording.omitted_actions} omitted · {recording.sample_count} visual pairs</span>
      <span>{recording.capture_visual ? "16×16 screen features are being recorded locally." : "Visual feature capture is off."}</span>
      <div class="actions">
        <button disabled={busy} onclick={() => void stop(true)}>Stop and save draft</button>
        <button class="secondary" disabled={busy} onclick={() => void stop(false)}>Discard</button>
      </div>
    </div>
  {:else}
    <div class="setup">
      <label>Workflow name <input bind:value={name} maxlength="100" placeholder="e.g. Morning status review" /></label>
      <label>Task goal <input bind:value={goal} maxlength="2000" placeholder="What should Heliox achieve when rerun?" /></label>
      <label class="check"><input type="checkbox" bind:checked={consent} /> I want Heliox to record actions it executes until I stop it.</label>
      <label class="check"><input type="checkbox" bind:checked={captureVisual} /> Also capture 16×16 before/after screen features for local world-model research.</label>
      {#if captureVisual}
        <p class="warning">These small features can still reveal screen content. They are encrypted locally, never sent to the model, and excluded from workflow previews.</p>
      {/if}
      <button disabled={busy || !consent || !name.trim()} onclick={() => void start()}>Start recording</button>
    </div>
  {/if}

  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if notice}<p class="notice" role="status">{notice}</p>{/if}

  <div class="saved-heading"><h4>Saved drafts</h4><button class="secondary" onclick={() => void refresh()}>Refresh</button></div>
  {#if workflows.length === 0}
    <p class="empty">No saved recordings yet.</p>
  {:else}
    <div class="saved-list">
      {#each workflows as workflow (workflow.id)}
        <details class="workflow">
          <summary><strong>{workflow.name}</strong><span>{workflow.step_count} steps · {workflow.sample_count} private samples</span></summary>
          <p class="goal">Goal: {workflow.goal}</p>
          {#if workflow.incomplete}<p class="warning">{workflow.omitted_actions} actions were excluded (for example typed text, scripts, or parallel actions). Re-record a complete task before replaying or scheduling.</p>{/if}
          {#if workflow.needs_review && !workflow.incomplete}
            <label class="check"><input type="checkbox" checked={reviewed[workflow.id] === true} onchange={(event) => reviewed[workflow.id] = event.currentTarget.checked} /> I reviewed these coordinate-based steps and understand the current screen may differ.</label>
          {/if}
          <ol>{#each workflow.steps as step}<li><code>{step.action_type}</code> <span>{JSON.stringify(step.parameters)}</span></li>{/each}</ol>
          {#if workflow.schedule}
            <p class="schedule-note">Next run: {new Date(workflow.schedule.next_run_at).toLocaleString()} {workflow.schedule.repeat_seconds ? `(every ${workflow.schedule.repeat_seconds / 3600} hours)` : "(once)"}</p>
          {/if}
          <div class="actions">
            <button disabled={busy || workflow.incomplete || (workflow.needs_review && !reviewed[workflow.id])} onclick={() => void run(workflow)}>Run with approvals</button>
            {#if workflow.schedule}<button class="secondary" disabled={busy} onclick={() => void unschedule(workflow)}>Remove schedule</button>{/if}
            <button class="danger" disabled={busy} onclick={() => void remove(workflow)}>Delete</button>
          </div>
          {#if !workflow.schedule}
            <div class="schedule-form">
              <label>First run <input type="datetime-local" value={runAt[workflow.id] ?? ""} onchange={(event) => runAt[workflow.id] = event.currentTarget.value} /></label>
              <label>Repeat <select value={repeat[workflow.id] ?? "0"} onchange={(event) => repeat[workflow.id] = event.currentTarget.value}><option value="0">Once</option><option value="86400">Daily</option><option value="604800">Weekly</option></select></label>
              <button class="secondary" disabled={busy || workflow.incomplete || !runAt[workflow.id] || (workflow.needs_review && !reviewed[workflow.id])} onclick={() => void schedule(workflow)}>Schedule with approvals</button>
            </div>
          {/if}
        </details>
      {/each}
    </div>
  {/if}
</section>

<style>
  .recorded-panel { display: grid; gap: 16px; color: var(--text-primary); }
  .heading, .saved-heading { display: flex; align-items: start; justify-content: space-between; gap: 16px; }
  h3, h4 { margin: 0; }
  h3 { font-size: 16px; } h4 { font-size: 14px; }
  p { margin: 6px 0 0; line-height: 1.5; color: var(--text-secondary); }
  .state { border: 1px solid var(--border); border-radius: 999px; padding: 5px 10px; white-space: nowrap; }
  .state.live { color: var(--accent); border-color: currentColor; }
  .setup, .active-box, .workflow { display: grid; gap: 12px; border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: var(--bg-secondary); }
  label { display: grid; gap: 6px; font-size: 13px; }
  input:not([type="checkbox"]), select { min-width: 0; width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-primary); color: var(--text-primary); font: inherit; }
  .check { display: flex; align-items: start; gap: 8px; line-height: 1.5; }
  .check input { margin: 3px 0 0; }
  .warning { color: var(--warning); }
  .error { color: var(--danger); }
  .notice { color: var(--success); }
  .active-box span { font-size: 13px; color: var(--text-secondary); }
  .actions, .schedule-form { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
  .schedule-form label { flex: 1 1 150px; }
  button { min-height: 36px; padding: 7px 12px; border: 1px solid var(--accent); border-radius: 8px; background: var(--accent); color: var(--bg-primary); cursor: pointer; font: inherit; }
  button.secondary { background: transparent; color: var(--text-primary); border-color: var(--border); }
  button.danger { background: transparent; color: var(--danger); border-color: currentColor; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .saved-list { display: grid; gap: 10px; }
  summary { cursor: pointer; display: flex; gap: 12px; justify-content: space-between; }
  summary span { font-size: 12px; color: var(--text-secondary); }
  .workflow ol { margin: 0; padding-left: 22px; display: grid; gap: 6px; }
  .workflow li { overflow-wrap: anywhere; font-size: 12px; }
  .workflow li span { color: var(--text-secondary); }
  .schedule-note { font-size: 13px; }
  @media (max-width: 640px) { .heading, summary { flex-direction: column; } }
</style>

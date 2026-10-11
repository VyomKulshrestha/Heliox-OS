<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireOkResult, requireResultStatus, type DaemonStatusResult } from "../api/daemon";
  import ConfirmPrompt from "./ConfirmPrompt.svelte";

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
    visual_gaps: number;
    omitted_actions: number;
    incomplete: boolean;
    needs_review: boolean;
    overlap_warning: boolean;
    placeholders: string[];
    feedback_count: number;
    feedback_visual_sample_count: number;
    schedule: { next_run_at: string; repeat_seconds: number } | null;
  }
  interface RecordingStatus {
    active: boolean;
    name: string;
    capture_visual: boolean;
    capture_manual: boolean;
    manual_hook_healthy: boolean;
    step_count: number;
    sample_count: number;
    visual_gaps: number;
    omitted_actions: number;
  }

  let workflows = $state<Workflow[]>([]);
  let recording = $state<RecordingStatus>({
    active: false,
    name: "",
    capture_visual: false,
    capture_manual: false,
    manual_hook_healthy: false,
    step_count: 0,
    sample_count: 0,
    visual_gaps: 0,
    omitted_actions: 0,
  });
  let name = $state("");
  let goal = $state("");
  let consent = $state(false);
  let captureVisual = $state(false);
  let captureManual = $state(false);
  let manualConsent = $state(false);
  let busy = $state(false);
  let error = $state("");
  let notice = $state("");
  let runAt = $state<Record<string, string>>({});
  let repeat = $state<Record<string, string>>({});
  let reviewed = $state<Record<string, boolean>>({});
  let textValues = $state<Record<string, Record<string, string>>>({});
  let captureReplayVisual = $state<Record<string, boolean>>({});
  let pendingDelete = $state<Workflow | null>(null);
  let lastRun = $state<{
    workflowId: string;
    jobId: string;
    status: string;
    feedbackSaved: boolean;
    captureVisual: boolean;
  } | null>(null);

  function runFinished(status: string): boolean {
    return ["success", "partial", "failed", "cancelled"].includes(status);
  }

  function textReady(workflow: Workflow): boolean {
    return (workflow.placeholders ?? []).every((key) => (textValues[workflow.id]?.[key] ?? "").trim().length > 0);
  }

  onMount(() => {
    void refresh();
    const timer = window.setInterval(() => {
      if (recording.active) void refreshStatus();
      if (lastRun && !runFinished(lastRun.status)) void refreshRunStatus();
    }, 4000);
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
    } catch (cause) {
      error = message(cause);
    }
  }

  async function refresh() {
    await refreshStatus();
    try {
      const result = requireOkResult(
        (await call("recorded_workflow_list", {})) as DaemonStatusResult & { workflows: Workflow[] },
        "Saved workflows unavailable",
      ) as DaemonStatusResult & { workflows: Workflow[] };
      workflows = result.workflows ?? [];
    } catch (cause) {
      error = message(cause);
    }
  }

  async function refreshRunStatus() {
    if (!lastRun) return;
    try {
      const result = (await call("autonomous_job", { job_id: lastRun.jobId })) as {
        job_id?: string;
        status?: string;
        error?: string;
      };
      if (result.error) throw new Error(result.error);
      if (result.job_id === lastRun?.jobId && result.status) {
        lastRun = { ...lastRun, status: result.status };
      }
    } catch (cause) {
      error = message(cause);
    }
  }

  async function start() {
    if (busy || !consent || !name.trim() || (captureManual && !manualConsent)) return;
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_start", {
          name: name.trim(),
          goal: goal.trim(),
          consent: true,
          capture_visual: captureVisual,
          capture_manual: captureManual,
          manual_consent: captureManual && manualConsent,
        })) as DaemonStatusResult,
        "Could not start recording",
      );
      consent = false;
      captureVisual = false;
      captureManual = false;
      manualConsent = false;
      name = "";
      goal = "";
      notice = "Recording started. Stop it when your demonstration is complete.";
      await refreshStatus();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function stop(save: boolean) {
    if (busy) return;
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_stop", { save })) as DaemonStatusResult,
        "Could not stop recording",
      );
      notice = save ? "Recording saved as a draft. Review its steps before running it." : "Recording discarded.";
      await refresh();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function run(workflow: Workflow) {
    if (busy || workflow.incomplete || !textReady(workflow)) return;
    busy = true;
    error = "";
    notice = "";
    try {
      const result = requireResultStatus(
        (await call("recorded_workflow_run", {
          workflow_id: workflow.id,
          reviewed: reviewed[workflow.id] === true,
          text_values: textValues[workflow.id] ?? {},
          capture_visual: captureReplayVisual[workflow.id] === true,
        })) as DaemonStatusResult & { job?: { job_id: string; status: string } },
        "submitted",
        "Could not run workflow",
      );
      if (!result.job?.job_id) throw new Error("The workflow run did not return a job ID");
      const capturedVisual = captureReplayVisual[workflow.id] === true;
      lastRun = {
        workflowId: workflow.id,
        jobId: result.job.job_id,
        status: result.job.status,
        feedbackSaved: false,
        captureVisual: capturedVisual,
      };
      textValues[workflow.id] = {};
      captureReplayVisual[workflow.id] = false;
      notice = `${workflow.name} submitted through Heliox's normal approvals and verification.`;
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function saveFeedback(workflow: Workflow, outcome: "achieved" | "partial" | "not_achieved") {
    if (busy || !lastRun || lastRun.workflowId !== workflow.id || !runFinished(lastRun.status)) return;
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_feedback", {
          workflow_id: workflow.id,
          job_id: lastRun.jobId,
          outcome,
        })) as DaemonStatusResult,
        "Could not save outcome feedback",
      );
      lastRun = { ...lastRun, feedbackSaved: true };
      notice = lastRun.captureVisual
        ? "Your goal-outcome rating was saved locally; any retained replay visual transitions are encrypted at rest and available only to offline world-model research."
        : "Your goal-outcome rating was saved locally. It is separate from the executor status.";
      await refresh();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function schedule(workflow: Workflow) {
    const local = runAt[workflow.id];
    if (busy || !local || workflow.incomplete || (workflow.placeholders ?? []).length) return;
    const timestamp = new Date(local);
    if (Number.isNaN(timestamp.getTime()) || timestamp.getTime() <= Date.now()) {
      error = "Choose a future date and time.";
      return;
    }
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_schedule", {
          workflow_id: workflow.id,
          run_at: timestamp.toISOString(),
          repeat_seconds: Number(repeat[workflow.id] ?? "0"),
          reviewed: reviewed[workflow.id] === true,
        })) as DaemonStatusResult,
        "Could not schedule workflow",
      );
      notice = `${workflow.name} scheduled. Heliox will still ask for required approvals.`;
      await refresh();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function unschedule(workflow: Workflow) {
    if (busy) return;
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_unschedule", { workflow_id: workflow.id })) as DaemonStatusResult,
        "Could not remove schedule",
      );
      notice = `${workflow.name} is no longer scheduled.`;
      await refresh();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  function askToRemove(workflow: Workflow) {
    if (!busy) pendingDelete = workflow;
  }

  async function confirmRemove() {
    const workflow = pendingDelete;
    pendingDelete = null;
    if (!workflow || busy) return;
    busy = true;
    error = "";
    notice = "";
    try {
      requireOkResult(
        (await call("recorded_workflow_delete", { workflow_id: workflow.id })) as DaemonStatusResult,
        "Could not delete workflow",
      );
      notice = "Saved workflow deleted.";
      await refresh();
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }
</script>

{#if pendingDelete}
  <ConfirmPrompt
    message={`Delete saved workflow “${pendingDelete.name}” and its local visual samples?`}
    confirmLabel="Delete workflow"
    danger={true}
    onconfirm={() => void confirmRemove()}
    oncancel={() => (pendingDelete = null)}
  />
{/if}

<section class="recorded-panel" aria-labelledby="recorded-title">
  <div class="heading">
    <div>
      <h3 id="recorded-title">Recorded workflows</h3>
      <p>
        Turn Heliox actions, and optionally your manual desktop actions, into a reviewable recipe. Recording is always
        off after restart.
      </p>
    </div>
    <span class:live={recording.active} class="state">{recording.active ? "Recording" : "Off"}</span>
  </div>

  {#if recording.active}
    <div class="active-box" role="status">
      <strong>{recording.name}</strong>
      <span
        >{recording.step_count} replayable steps · {recording.omitted_actions} omitted · {recording.sample_count} visual pairs
        · {recording.visual_gaps ?? 0} visual gaps</span
      >
      <span
        >{recording.capture_visual
          ? recording.capture_manual
            ? "A rolling in-memory 16×16 feature supports manual clicks; unused frames are discarded. Eligible Heliox mouse actions also produce visual pairs."
            : "16×16 screen features are captured around eligible Heliox mouse actions only."
          : "Visual feature capture is off."}</span
      >
      {#if recording.capture_manual}<span
          >{recording.manual_hook_healthy
            ? "Manual desktop input is being observed."
            : "Manual input hook is not healthy; stop and review this draft."}</span
        >{/if}
      <div class="actions">
        <button disabled={busy} onclick={() => void stop(true)}>Stop and save draft</button>
        <button class="secondary" disabled={busy} onclick={() => void stop(false)}>Discard</button>
      </div>
    </div>
  {:else}
    <div class="setup">
      <label>Workflow name <input bind:value={name} maxlength="100" placeholder="e.g. Morning status review" /></label>
      <label
        >Task goal <input
          bind:value={goal}
          maxlength="2000"
          placeholder="What should Heliox achieve when rerun?"
        /></label
      >
      <label class="check"
        ><input type="checkbox" bind:checked={consent} /> I want Heliox to record actions it executes until I stop it.</label
      >
      <label class="check"
        ><input type="checkbox" bind:checked={captureManual} /> Also record my manual desktop actions.</label
      >
      {#if captureManual}
        <label class="check"
          ><input type="checkbox" bind:checked={manualConsent} /> I consent to a temporary desktop-wide mouse and keyboard
          hook until I stop recording.</label
        >
        <p class="warning">
          Mouse coordinates, drag endpoints, and scrolls are saved locally. Typed characters are never saved; text entry
          becomes an editable placeholder. Shortcuts and unknown actions make the draft incomplete. Pause before opening
          sensitive screens.
        </p>
      {/if}
      <label class="check"
        ><input type="checkbox" bind:checked={captureVisual} /> Also capture local 16×16 before/after screen features for
        world-model research.</label
      >
      {#if captureVisual}
        <p class="warning">
          Eligible Heliox mouse actions create pairs. With manual recording also enabled, a rolling in-memory feature
          supports manual clicks; unused frames are discarded. Saved pairs are encrypted locally and excluded from
          workflow previews. Even small features can reveal screen content. Pause before viewing sensitive screens.
        </p>
      {/if}
      <button
        disabled={busy || !consent || !name.trim() || (captureManual && !manualConsent)}
        onclick={() => void start()}>Start recording</button
      >
    </div>
  {/if}

  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if notice}<p class="notice" role="status">{notice}</p>{/if}

  <div class="saved-heading">
    <h4>Saved drafts</h4>
    <button class="secondary" onclick={() => void refresh()}>Refresh</button>
  </div>
  {#if workflows.length === 0}
    <p class="empty">No saved recordings yet.</p>
  {:else}
    <div class="saved-list">
      {#each workflows as workflow (workflow.id)}
        <details class="workflow">
          <summary
            ><strong>{workflow.name}</strong><span
              >{workflow.step_count} steps · {workflow.sample_count} private samples{workflow.feedback_count
                ? ` · ${workflow.feedback_count} goal ratings`
                : ""}{workflow.feedback_visual_sample_count
                ? ` · ${workflow.feedback_visual_sample_count} replay pairs`
                : ""}</span
            ></summary
          >
          <p class="goal">Goal: {workflow.goal}</p>
          {#if workflow.visual_gaps}<p class="warning">
              {workflow.visual_gaps} manual clicks have no reliable visual pair. Their recipe steps remain, but they were
              excluded from local visual training.
            </p>{/if}
          {#if workflow.incomplete}<p class="warning">
              {workflow.omitted_actions} actions were excluded (for example Heliox-entered text, shortcuts, scripts, or parallel
              actions). This draft cannot run or schedule; those steps need a parameterized workflow.
            </p>{/if}
          {#if workflow.overlap_warning}<p class="warning">
              Heliox also acted during this recording. Manual input at the same moment may have been missed; review the
              steps carefully.
            </p>{/if}
          {#if workflow.needs_review && !workflow.incomplete}
            <label class="check"
              ><input
                type="checkbox"
                checked={reviewed[workflow.id] === true}
                onchange={(event) => (reviewed[workflow.id] = event.currentTarget.checked)}
              /> I reviewed these manual or coordinate-based steps and understand the current screen may differ.</label
            >
          {/if}
          {#each workflow.placeholders ?? [] as key}
            <label
              >Text for step {Number(key) + 1}
              <input
                type="text"
                maxlength="2000"
                autocomplete="off"
                value={textValues[workflow.id]?.[key] ?? ""}
                oninput={(event) => {
                  textValues[workflow.id] ??= {};
                  textValues[workflow.id][key] = event.currentTarget.value;
                }}
                placeholder="Enter at run time; not saved"
              /></label
            >
          {/each}
          {#if (workflow.placeholders ?? []).length}<p class="warning">
              Text values are sent to the configured model and may appear in normal task history. Do not enter passwords
              or other secrets. Workflows with text cannot be scheduled unattended.
            </p>{/if}
          <ol>
            {#each workflow.steps as step}<li>
                <code>{step.action_type}</code> <span>{JSON.stringify(step.parameters)}</span>
              </li>{/each}
          </ol>
          {#if workflow.schedule}
            <p class="schedule-note">
              Next run: {new Date(workflow.schedule.next_run_at).toLocaleString()}
              {workflow.schedule.repeat_seconds ? `(every ${workflow.schedule.repeat_seconds / 3600} hours)` : "(once)"}
            </p>
          {/if}
          <label class="check"
            ><input
              type="checkbox"
              checked={captureReplayVisual[workflow.id] === true}
              onchange={(event) => (captureReplayVisual[workflow.id] = event.currentTarget.checked)}
            /> Capture local 16×16 before/after features for this replay’s world-model research.</label
          >
          {#if captureReplayVisual[workflow.id]}
            <p class="warning">
              Only eligible Heliox mouse actions in this run are captured. Features can reveal screen content; they stay
              in a bounded in-memory buffer until you rate this run, then are encrypted at rest locally. If you skip
              feedback, they are discarded after 30 minutes or on restart. No raw screenshot or typed text is stored.
              This consent applies to one run only.
            </p>
          {/if}
          <div class="actions">
            <button
              disabled={busy ||
                workflow.incomplete ||
                !textReady(workflow) ||
                (workflow.needs_review && !reviewed[workflow.id])}
              onclick={() => void run(workflow)}>Run with approvals</button
            >
            {#if workflow.schedule}<button class="secondary" disabled={busy} onclick={() => void unschedule(workflow)}
                >Remove schedule</button
              >{/if}
            <button class="danger" disabled={busy} onclick={() => askToRemove(workflow)}>Delete</button>
          </div>
          {#if lastRun?.workflowId === workflow.id}
            <div class="run-feedback">
              <p>Executor status: {lastRun.status}. This does not prove your goal was achieved.</p>
              {#if !runFinished(lastRun.status)}
                <button class="secondary" disabled={busy} onclick={() => void refreshRunStatus()}
                  >Check run status</button
                >
              {:else if lastRun.feedbackSaved}
                <p>
                  {lastRun.captureVisual
                    ? "Goal-outcome rating and any replay visual pairs saved locally."
                    : "Goal-outcome rating saved locally."}
                </p>
              {:else}
                <p>Did this run achieve your goal? Your answer is optional and stored locally with this workflow.</p>
                <div class="actions" role="group" aria-label="Rate this run's goal outcome">
                  <button class="secondary" disabled={busy} onclick={() => void saveFeedback(workflow, "achieved")}
                    >Achieved</button
                  >
                  <button class="secondary" disabled={busy} onclick={() => void saveFeedback(workflow, "partial")}
                    >Partly achieved</button
                  >
                  <button class="secondary" disabled={busy} onclick={() => void saveFeedback(workflow, "not_achieved")}
                    >Not achieved</button
                  >
                </div>
              {/if}
            </div>
          {/if}
          {#if !workflow.schedule && !(workflow.placeholders ?? []).length}
            <div class="schedule-form">
              <p>Times use your local timezone. The daemon must be running to dispatch a scheduled task.</p>
              <label
                >First run <input
                  type="datetime-local"
                  value={runAt[workflow.id] ?? ""}
                  onchange={(event) => (runAt[workflow.id] = event.currentTarget.value)}
                /></label
              >
              <label
                >Repeat <select
                  value={repeat[workflow.id] ?? "0"}
                  onchange={(event) => (repeat[workflow.id] = event.currentTarget.value)}
                  ><option value="0">Once</option><option value="86400">Daily</option><option value="604800"
                    >Weekly</option
                  ></select
                ></label
              >
              <button
                class="secondary"
                disabled={busy ||
                  workflow.incomplete ||
                  !runAt[workflow.id] ||
                  (workflow.needs_review && !reviewed[workflow.id])}
                onclick={() => void schedule(workflow)}>Schedule with approvals</button
              >
            </div>
          {/if}
        </details>
      {/each}
    </div>
  {/if}
</section>

<style>
  .recorded-panel {
    display: grid;
    gap: 16px;
    color: var(--text-primary);
  }
  .heading,
  .saved-heading {
    display: flex;
    align-items: start;
    justify-content: space-between;
    gap: 16px;
  }
  h3,
  h4 {
    margin: 0;
  }
  h3 {
    font-size: 16px;
  }
  h4 {
    font-size: 14px;
  }
  p {
    margin: 6px 0 0;
    line-height: 1.5;
    color: var(--text-secondary);
  }
  .state {
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 10px;
    white-space: nowrap;
  }
  .state.live {
    color: var(--accent);
    border-color: currentColor;
  }
  .setup,
  .active-box,
  .workflow {
    display: grid;
    gap: 12px;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    background: var(--bg-secondary);
  }
  .run-feedback {
    display: grid;
    gap: 8px;
    border-top: 1px solid var(--border);
    padding-top: 12px;
  }
  label {
    display: grid;
    gap: 6px;
    font-size: 13px;
  }
  input:not([type="checkbox"]),
  select {
    min-width: 0;
    width: 100%;
    box-sizing: border-box;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font: inherit;
  }
  .check {
    display: flex;
    align-items: start;
    gap: 8px;
    line-height: 1.5;
  }
  .check input {
    margin: 3px 0 0;
  }
  .warning {
    color: var(--warning);
  }
  .error {
    color: var(--danger);
  }
  .notice {
    color: var(--success);
  }
  .active-box span {
    font-size: 13px;
    color: var(--text-secondary);
  }
  .actions,
  .schedule-form {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: end;
  }
  .schedule-form label {
    flex: 1 1 150px;
  }
  .schedule-form p {
    flex-basis: 100%;
    font-size: 12px;
  }
  button {
    min-height: 36px;
    padding: 7px 12px;
    border: 1px solid var(--accent);
    border-radius: 8px;
    background: var(--accent);
    color: var(--bg-primary);
    cursor: pointer;
    font: inherit;
  }
  button.secondary {
    background: transparent;
    color: var(--text-primary);
    border-color: var(--border);
  }
  button.danger {
    background: transparent;
    color: var(--danger);
    border-color: currentColor;
  }
  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  button:focus-visible,
  input:focus-visible,
  select:focus-visible,
  summary:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .saved-list {
    display: grid;
    gap: 10px;
  }
  summary {
    cursor: pointer;
    display: flex;
    gap: 12px;
    justify-content: space-between;
  }
  summary span {
    font-size: 12px;
    color: var(--text-secondary);
  }
  .workflow ol {
    margin: 0;
    padding-left: 22px;
    display: grid;
    gap: 6px;
  }
  .workflow li {
    overflow-wrap: anywhere;
    font-size: 12px;
  }
  .workflow li span {
    color: var(--text-secondary);
  }
  .schedule-note {
    font-size: 13px;
  }
  @media (max-width: 640px) {
    .heading,
    summary {
      flex-direction: column;
    }
  }
</style>

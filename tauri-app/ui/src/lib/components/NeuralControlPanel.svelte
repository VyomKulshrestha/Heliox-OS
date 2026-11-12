<script lang="ts">
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { invoke } from "../api/invoke";
  import { neural } from "../stores/neural";
  import { session } from "../stores/session";

  let source = $state<"synthetic" | "playback" | "brainflow" | "lsl">("synthetic");
  let artifactPath = $state("");
  let playbackPath = $state("");
  let boardId = $state(0);
  let serialPort = $state("");
  let lslName = $state("HelioxEEG");
  let syntheticFrequency = $state(12);
  let warningAccepted = $state(false);
  let recordRaw = $state(false);
  let recordingFile = $state("");
  let recordingPurpose = $state("local accessibility calibration");
  let retentionDays = $state(7);
  let allowBidsExport = $state(false);
  let exportRecording = $state("");
  let exportDestination = $state("");
  let exportMessage = $state("");
  let benchmarkBusy = $state(false);
  let benchmarkError = $state("");
  let benchmarkResult = $state<Record<string, unknown> | null>(null);
  let eegbciSubject = $state(1);
  let stagedTaskLabel = $state("");
  let stagedTaskGoal = $state("");

  onMount(() => {
    void neural.refresh();
    const timer = setInterval(() => void neural.refresh(), 1500);
    return () => clearInterval(timer);
  });

  function start() {
    void neural.startSidecar({
      source,
      artifactPath: artifactPath || null,
      playbackPath: playbackPath || null,
      boardId,
      serialPort,
      lslName,
      syntheticFrequency,
      recordRaw,
      recordingFile: recordingFile || null,
      recordingPurpose,
      retentionDays,
      allowBidsExport,
    });
  }

  async function exportBids() {
    exportMessage = "";
    try {
      await invoke("export_neural_recording", { recording: exportRecording, destination: exportDestination });
      exportMessage = $_("neural.export_success");
    } catch (cause) {
      exportMessage = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function percent(value: number | undefined): string {
    return `${Math.round((value ?? 0) * 100)}%`;
  }

  function resultText(key: string): string {
    const result = $neural.lastResult;
    if (!result) return "";
    const job = (result.job ?? {}) as Record<string, unknown>;
    const value = key === "job_id" ? job.job_id : (result[key] ?? job[key]);
    return value === null || value === undefined ? "" : String(value);
  }

  async function runBenchmark(benchmark: "brainflow-synthetic" | "eegbci") {
    benchmarkBusy = true;
    benchmarkError = "";
    benchmarkResult = null;
    try {
      benchmarkResult = await invoke<Record<string, unknown>>("run_neural_benchmark", {
        benchmark,
        subject: benchmark === "eegbci" ? eegbciSubject : null,
        runs: benchmark === "eegbci" ? [6, 10, 14] : null,
      });
    } catch (cause) {
      benchmarkError = cause instanceof Error ? cause.message : String(cause);
    } finally {
      benchmarkBusy = false;
    }
  }

  async function stageAutonomousTask() {
    const staged = await neural.stageTask(stagedTaskLabel, stagedTaskGoal, $session.activeSessionId);
    if (staged) {
      stagedTaskLabel = "";
      stagedTaskGoal = "";
    }
  }

  function evidenceLabel(kind: string): string {
    if (kind === "live_eeg") return $_("neural.live_eeg");
    if (kind === "recorded_eeg") return $_("neural.recorded_eeg");
    if (kind === "synthetic") return $_("neural.synthetic_evidence");
    return kind || $_("neural.waiting_evidence");
  }
</script>

<div class="neural-panel">
  <div class="panel-header">
    <div>
      <h3>{$_("neural.title")}</h3>
      <span class="subtitle">{$_("neural.subtitle")}</span>
    </div>
    <span class:online={$neural.connected} class="status-pill">
      {$neural.connected ? $_("neural.connected") : $_("neural.disconnected")}
    </span>
  </div>

  <div class="boundary-note">
    <strong>{$_("neural.research_only")}</strong>
    <span>{$_("neural.boundary")}</span>
  </div>

  <label class="consent-row">
    <input type="checkbox" bind:checked={warningAccepted} onchange={() => neural.setStimulusEnabled(warningAccepted)} />
    <span>{$_("neural.photosensitivity")}</span>
  </label>

  <div class="configuration-grid">
    <label>
      <span>{$_("neural.source")}</span>
      <select bind:value={source} disabled={$neural.sidecarRunning}>
        <option value="synthetic">{$_("neural.synthetic")}</option>
        <option value="playback">{$_("neural.playback")}</option>
        <option value="brainflow">BrainFlow</option>
        <option value="lsl">Lab Streaming Layer</option>
      </select>
    </label>
    {#if source !== "synthetic"}
      <label class="wide">
        <span>{$_("neural.artifact_path")}</span>
        <input bind:value={artifactPath} placeholder="C:\path\calibration.json" disabled={$neural.sidecarRunning} />
      </label>
    {/if}
    {#if source === "playback"}
      <label class="wide">
        <span>{$_("neural.playback_path")}</span>
        <input bind:value={playbackPath} placeholder="C:\path\recording.npz" disabled={$neural.sidecarRunning} />
      </label>
    {:else if source === "brainflow"}
      <label>
        <span>{$_("neural.board_id")}</span>
        <input type="number" bind:value={boardId} disabled={$neural.sidecarRunning} />
      </label>
      <label>
        <span>{$_("neural.serial_port")}</span>
        <input bind:value={serialPort} placeholder="COM3" disabled={$neural.sidecarRunning} />
      </label>
    {:else if source === "lsl"}
      <label>
        <span>{$_("neural.lsl_name")}</span>
        <input bind:value={lslName} disabled={$neural.sidecarRunning} />
      </label>
    {:else}
      <label>
        <span>{$_("neural.simulated_target")}</span>
        <select bind:value={syntheticFrequency} disabled={$neural.sidecarRunning}>
          <option value={8}>8 Hz · focus left</option>
          <option value={10}>10 Hz · focus right</option>
          <option value={12}>12 Hz · select</option>
          <option value={15}>15 Hz · cancel</option>
        </select>
      </label>
    {/if}
  </div>

  <div class="recording-controls">
    <label class="consent-row compact">
      <input type="checkbox" bind:checked={recordRaw} disabled={$neural.sidecarRunning} />
      <span>{$_("neural.record_consent")}</span>
    </label>
    {#if recordRaw}
      <div class="configuration-grid">
        <label class="wide">
          <span>{$_("neural.recording_purpose")}</span>
          <input bind:value={recordingPurpose} maxlength="256" disabled={$neural.sidecarRunning} />
        </label>
        <label>
          <span>{$_("neural.recording_file")}</span>
          <input bind:value={recordingFile} placeholder="Optional new .neeg path" disabled={$neural.sidecarRunning} />
        </label>
        <label>
          <span>{$_("neural.retention_days")}</span>
          <input type="number" min="1" max="365" bind:value={retentionDays} disabled={$neural.sidecarRunning} />
        </label>
        <label class="consent-row compact wide">
          <input type="checkbox" bind:checked={allowBidsExport} disabled={$neural.sidecarRunning} />
          <span>{$_("neural.export_consent")}</span>
        </label>
      </div>
    {/if}
    <details>
      <summary>{$_("neural.export_title")}</summary>
      <div class="configuration-grid">
        <label>
          <span>{$_("neural.encrypted_recording")}</span>
          <input bind:value={exportRecording} placeholder="C:\path\session.neeg" />
        </label>
        <label>
          <span>{$_("neural.export_destination")}</span>
          <input bind:value={exportDestination} placeholder="C:\path\new-bids-dataset" />
        </label>
      </div>
      <div class="actions">
        <button onclick={exportBids} disabled={!exportRecording.trim() || !exportDestination.trim()}>
          {$_("neural.export_button")}
        </button>
        {#if exportMessage}<span class="export-message">{exportMessage}</span>{/if}
      </div>
    </details>
  </div>

  <div class="benchmark-controls">
    <div class="goal-header">
      <div>
        <strong>{$_("neural.no_hardware_title")}</strong>
        <span>{$_("neural.no_hardware_body")}</span>
      </div>
      <span class="evidence-badge">{$_("neural.recorded_not_live")}</span>
    </div>
    <div class="benchmark-actions">
      <button onclick={() => runBenchmark("brainflow-synthetic")} disabled={benchmarkBusy}>
        {$_("neural.run_brainflow_synthetic")}
      </button>
      <label>
        <span>{$_("neural.eegbci_subject")}</span>
        <input type="number" min="1" max="109" bind:value={eegbciSubject} disabled={benchmarkBusy} />
      </label>
      <button
        onclick={() => runBenchmark("eegbci")}
        disabled={benchmarkBusy || eegbciSubject < 1 || eegbciSubject > 109}
      >
        {benchmarkBusy ? $_("neural.benchmark_running") : $_("neural.run_eegbci")}
      </button>
    </div>
    {#if benchmarkResult}
      <div class="benchmark-result" aria-live="polite">
        <strong>{evidenceLabel(String(benchmarkResult.evidence_kind ?? ""))}</strong>
        {#if benchmarkResult.dataset}<span>{String(benchmarkResult.dataset)}</span>{/if}
        {#if benchmarkResult.balanced_accuracy !== undefined}
          <span>
            {$_("neural.benchmark_accuracy")}: {percent(Number(benchmarkResult.balanced_accuracy))} ·
            {$_("neural.benchmark_chance")}: {percent(Number(benchmarkResult.chance_level))} ·
            {String(benchmarkResult.fold_count)}
            {$_("neural.held_runs")}
          </span>
        {:else}
          <span>
            {String(benchmarkResult.sample_count)} samples · {String(benchmarkResult.sample_rate_hz)} Hz ·
            {String(benchmarkResult.signal_quality)}
          </span>
        {/if}
      </div>
    {/if}
    {#if benchmarkError}<div class="error" role="alert">{benchmarkError}</div>{/if}
  </div>

  <div class="actions">
    {#if !$neural.sidecarRunning}
      <button class="primary" onclick={start} disabled={!warningAccepted || $neural.busy}>
        {$_("neural.start")}
      </button>
    {:else}
      <button onclick={() => neural.stopSidecar()} disabled={$neural.busy}>{$_("neural.stop")}</button>
    {/if}
    <button
      onclick={() => neural.beginCalibration()}
      disabled={!$neural.connected || $neural.state !== "connected_uncalibrated" || $neural.busy}
      >{$_("neural.calibrate")}</button
    >
    <button
      onclick={() => neural.arm("navigate")}
      disabled={!$neural.calibrated || $neural.state !== "observe_only" || $neural.busy}>{$_("neural.arm_ui")}</button
    >
    <button
      onclick={() => neural.arm("safe_desktop")}
      disabled={!$neural.calibrated || $neural.state !== "observe_only" || $neural.busy}
      >{$_("neural.arm_desktop")}</button
    >
    <button class="danger" onclick={() => neural.disarm()} disabled={!$neural.connected}>{$_("neural.disarm")}</button>
  </div>

  <div class="status-grid">
    <div><span>{$_("neural.state")}</span><strong>{$neural.state.replaceAll("_", " ")}</strong></div>
    <div>
      <span>{$_("neural.evidence")}</span>
      <strong class="evidence-badge" class:live={$neural.evidenceKind === "live_eeg"}>
        {evidenceLabel($neural.evidenceKind)}
      </strong>
    </div>
    <div><span>{$_("neural.transport")}</span><strong>{$neural.transport || "—"}</strong></div>
    <div>
      <span>{$_("neural.signal")}</span>
      <strong class:good={$neural.quality?.quality === "good"}>{$neural.quality?.quality ?? "waiting"}</strong>
    </div>
    <div>
      <span>{$_("neural.buffer")}</span><strong>{$neural.bufferedSamples} / {$neural.droppedSamples} dropped</strong>
    </div>
    <div><span>{$_("neural.calibration")}</span><strong>{$neural.calibrated ? "verified" : "required"}</strong></div>
    <div>
      <span>{$_("neural.held_accuracy")}</span>
      <strong>{percent($neural.calibrationMetrics?.balanced_accuracy)}</strong>
    </div>
  </div>

  {#if $neural.quality?.artifact_flags.length}
    <div class="artifact-warning" role="alert">
      {$_("neural.artifacts")}: {$neural.quality.artifact_flags.join(", ")} · {$neural.quality.reasons.join(", ")}
    </div>
  {/if}

  <div class="task-launcher">
    <div class="goal-header">
      <div>
        <strong>{$_("neural.task_launcher")}</strong>
        <span>{$_("neural.task_launcher_body")}</span>
      </div>
      <span class="evidence-badge">{$_("neural.task_authority")}</span>
    </div>
    <div class="configuration-grid task-form">
      <label>
        <span>{$_("neural.task_label")}</span>
        <input bind:value={stagedTaskLabel} maxlength="80" placeholder="Research and summarize" />
      </label>
      <label class="wide">
        <span>{$_("neural.task_goal")}</span>
        <textarea
          bind:value={stagedTaskGoal}
          maxlength="2000"
          rows="3"
          placeholder="Research the topic, compare the evidence, and save a verified summary."></textarea>
      </label>
    </div>
    <div class="actions task-actions">
      <button class="primary" onclick={stageAutonomousTask} disabled={$neural.busy || stagedTaskGoal.trim().length < 3}>
        {$_("neural.stage_task")}
      </button>
    </div>
    <div class="staged-task-list">
      {#each $neural.stagedTasks as task (task.task_id)}
        <div class:focused={$neural.focusedCommandId === task.command_id} class="staged-task">
          <div>
            <strong>{task.label}</strong>
            <small>{task.goal}</small>
          </div>
          <button onclick={() => neural.removeStagedTask(task.task_id)} disabled={$neural.busy}>
            {$_("neural.remove_task")}
          </button>
        </div>
      {:else}
        <p class="empty">{$_("neural.no_staged_tasks")}</p>
      {/each}
    </div>
    {#if $neural.lastResult?.staged_task}
      <div class:failed={resultText("status") === "failed"} class="task-result" aria-live="polite">
        <strong>{$_("neural.task_result")}: {resultText("status") || $_("neural.task_result_unknown")}</strong>
        {#if resultText("job_id")}
          <span>{$_("neural.task_job_id")}: <code>{resultText("job_id")}</code></span>
        {/if}
        {#if resultText("error")}<span>{resultText("error")}</span>{/if}
        <small>{$_("neural.task_result_body")}</small>
      </div>
    {/if}
  </div>

  <div class="goal-list" aria-label={$_("neural.safe_goals")}>
    <div class="goal-header">
      <strong>{$_("neural.safe_goals")}</strong>
      <span>{$_("neural.physical_disabled")}</span>
    </div>
    {#each $neural.safeGoals as goal (goal.command_id)}
      <div class:focused={$neural.focusedCommandId === goal.command_id} class="goal">
        <span>{goal.label}</span>
        <small>{goal.description}</small>
        <code>Tier {goal.permission_tier} · {goal.action_type}</code>
      </div>
    {:else}
      <p class="empty">{$_("neural.no_goals")}</p>
    {/each}
  </div>

  {#if $neural.waiting}<div class="waiting" role="status">{$neural.waiting}</div>{/if}
  {#if $neural.error}<div class="error" role="alert">{$neural.error}</div>{/if}
</div>

<style>
  .neural-panel {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
  }
  .panel-header,
  .actions,
  .goal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .panel-header {
    padding: 14px;
    background: var(--bg-tertiary);
  }
  h3 {
    margin: 0;
    font-size: 14px;
  }
  .subtitle,
  label span,
  .status-grid span,
  small,
  .goal-header span {
    color: var(--text-muted);
    font-size: 11px;
  }
  .status-pill {
    padding: 4px 9px;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 10px;
    text-transform: uppercase;
  }
  .status-pill.online,
  .good {
    color: var(--success);
    border-color: color-mix(in srgb, var(--success) 45%, transparent);
  }
  .boundary-note {
    display: grid;
    gap: 4px;
    padding: 12px 14px;
    color: var(--warning);
    background: color-mix(in srgb, var(--warning) 8%, transparent);
    font-size: 11px;
  }
  .boundary-note span {
    color: var(--text-secondary);
    line-height: 1.45;
  }
  .consent-row {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    padding: 12px 14px 0;
  }
  .consent-row input {
    margin-top: 2px;
  }
  .consent-row.compact {
    padding: 0;
  }
  .recording-controls {
    display: grid;
    gap: 10px;
    margin: 0 14px 12px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .benchmark-controls {
    display: grid;
    gap: 10px;
    margin: 0 14px 12px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--accent) 4%, transparent);
  }
  .benchmark-controls .goal-header > div,
  .benchmark-result {
    display: grid;
    gap: 4px;
  }
  .benchmark-actions {
    display: flex;
    align-items: end;
    flex-wrap: wrap;
    gap: 8px;
  }
  .benchmark-actions input {
    width: 76px;
  }
  .benchmark-result {
    padding: 9px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 11px;
  }
  .evidence-badge {
    color: var(--warning);
  }
  .evidence-badge.live {
    color: var(--success);
  }
  details summary {
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 11px;
  }
  .recording-controls .configuration-grid {
    padding: 10px 0 0;
  }
  .recording-controls .actions {
    padding: 8px 0 0;
  }
  .export-message {
    color: var(--text-muted);
    font-size: 10px;
  }
  .configuration-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    padding: 12px 14px;
  }
  label {
    display: grid;
    gap: 5px;
  }
  label.wide {
    grid-column: 1 / -1;
  }
  input,
  select,
  textarea {
    min-width: 0;
    padding: 7px 9px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  textarea {
    resize: vertical;
  }
  .actions {
    justify-content: flex-start;
    flex-wrap: wrap;
    padding: 0 14px 12px;
  }
  button {
    padding: 7px 11px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  button.primary {
    color: var(--on-accent);
    background: var(--accent);
    border-color: var(--accent);
  }
  button.danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 50%, transparent);
  }
  button:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }
  .status-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    padding: 12px 14px;
    border-top: 1px solid var(--border);
  }
  .status-grid div {
    display: grid;
    gap: 4px;
    padding: 9px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .status-grid strong {
    overflow: hidden;
    font-size: 11px;
    text-overflow: ellipsis;
    text-transform: capitalize;
  }
  .waiting {
    padding: 10px 14px;
    color: var(--text-secondary);
    background: var(--warning-bg);
    font-size: 11px;
  }
  .artifact-warning,
  .error {
    padding: 10px 14px;
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 9%, transparent);
    font-size: 11px;
  }
  .goal-list {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    padding: 12px 14px 14px;
    border-top: 1px solid var(--border);
  }
  .task-launcher {
    display: grid;
    gap: 10px;
    margin: 0 14px 12px;
    padding: 10px;
    background: color-mix(in srgb, var(--accent) 5%, var(--bg-primary));
    border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
    border-radius: var(--radius-sm);
  }
  .task-form {
    padding: 0;
  }
  .task-actions {
    padding: 0;
  }
  .staged-task-list {
    display: grid;
    gap: 7px;
  }
  .task-result {
    display: grid;
    gap: 4px;
    padding: 9px;
    color: var(--success);
    background: color-mix(in srgb, var(--success) 7%, var(--bg-primary));
    border: 1px solid color-mix(in srgb, var(--success) 35%, var(--border));
    border-radius: var(--radius-sm);
    font-size: 11px;
  }
  .task-result.failed {
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 7%, var(--bg-primary));
    border-color: color-mix(in srgb, var(--danger) 35%, var(--border));
  }
  .task-result span,
  .task-result small {
    color: var(--text-secondary);
  }
  .staged-task {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 9px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .staged-task.focused {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 35%, transparent);
  }
  .staged-task div {
    display: grid;
    gap: 4px;
    min-width: 0;
  }
  .staged-task small {
    overflow: hidden;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
  }
  .goal-header,
  .empty {
    grid-column: 1 / -1;
  }
  .goal {
    display: grid;
    gap: 4px;
    padding: 9px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }
  .goal.focused {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 35%, transparent);
  }
  .goal code {
    color: var(--text-muted);
    font-size: 9px;
  }
  .empty {
    color: var(--text-muted);
    font-size: 11px;
  }
  @media (max-width: 900px) {
    .status-grid,
    .goal-list {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>

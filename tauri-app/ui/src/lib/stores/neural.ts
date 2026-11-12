import { get, writable } from "svelte/store";
import { call, offNotification, onNotification, requireResultStatus } from "../api/daemon";
import { invoke } from "../api/invoke";
import { lateRetry, startingMessage } from "../api/lateParts";

export type NeuralSessionState =
  | "disconnected"
  | "connected_uncalibrated"
  | "calibrating"
  | "observe_only"
  | "armed_safe_ui"
  | "armed_safe_desktop"
  | "previewed"
  | "cooldown";

export interface NeuralQuality {
  quality: "good" | "degraded" | "reject";
  artifact_flags: string[];
  channel_std_uv: number[];
  line_noise_ratio: number;
  muscle_ratio: number;
  estimated_missing_samples: number;
  timestamp_jitter_ratio: number;
  reasons: string[];
}

export interface NeuralPreview {
  status: "previewed";
  preview_id: string;
  intent_id: string;
  intent_class: string;
  command_id: string | null;
  resolved_command_id: string | null;
  canonical_goal: string;
  requested_scope: string;
  state_revision: number;
  created_at_ns: number;
  eligible_at_ns: number;
  expires_at_ns: number;
  requires_non_neural_approval: boolean;
  world_model: Record<string, unknown> | null;
  staged_task: NeuralStagedTask | null;
}

export interface NeuralSafeGoal {
  command_id: string;
  label: string;
  description: string;
  action_type: string;
  permission_tier: number;
}

export interface NeuralStagedTask {
  task_id: string;
  command_id: string;
  label: string;
  goal: string;
  session_id: string;
  created_at_ns: number;
  authority: "explicit_non_neural_staging";
}

export interface NeuralState {
  state: NeuralSessionState;
  connected: boolean;
  sidecarRunning: boolean;
  sidecarPid: number | null;
  stimulusEnabled: boolean;
  sessionId: string;
  sourceId: string;
  boardKind: string;
  transport: string;
  evidenceKind: string;
  calibrated: boolean;
  calibrationId: string;
  decoderVersion: string;
  calibrationMetrics: {
    epoch_count: number;
    block_count: number;
    balanced_accuracy: number;
    expected_calibration_error: number;
    per_class_recall: Record<string, number>;
  } | null;
  armedScope: string;
  stateRevision: number;
  focusedCommandId: string;
  safeGoals: NeuralSafeGoal[];
  stagedTasks: NeuralStagedTask[];
  quality: NeuralQuality | null;
  bufferedSamples: number;
  droppedSamples: number;
  preview: NeuralPreview | null;
  lastResult: Record<string, unknown> | null;
  busy: boolean;
  /** Neural input starts after Heliox is ready: until it has, the daemon's sentence saying so (not an error). */
  waiting: string;
  error: string;
}

const DEFAULT_STATE: NeuralState = {
  state: "disconnected",
  connected: false,
  sidecarRunning: false,
  sidecarPid: null,
  stimulusEnabled: false,
  sessionId: "",
  sourceId: "",
  boardKind: "",
  transport: "",
  evidenceKind: "",
  calibrated: false,
  calibrationId: "",
  decoderVersion: "",
  calibrationMetrics: null,
  armedScope: "observe",
  stateRevision: 0,
  focusedCommandId: "",
  safeGoals: [],
  stagedTasks: [],
  quality: null,
  bufferedSamples: 0,
  droppedSamples: 0,
  preview: null,
  lastResult: null,
  busy: false,
  waiting: "",
  error: "",
};

function readableError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause || "Unknown neural control error");
}

function createNeuralControl() {
  const store = writable<NeuralState>({ ...DEFAULT_STATE });
  let autoCommit: ReturnType<typeof setTimeout> | null = null;

  function clearAutoCommit() {
    if (autoCommit) clearTimeout(autoCommit);
    autoCommit = null;
  }

  function applyStatus(payload: Record<string, unknown>) {
    store.update((current) => ({
      ...current,
      state: String(payload.state ?? current.state) as NeuralSessionState,
      connected: Boolean(payload.connected ?? current.connected),
      sessionId: String(payload.session_id ?? current.sessionId),
      sourceId: String(payload.source_id ?? current.sourceId),
      boardKind: String(payload.board_kind ?? current.boardKind),
      transport: String(payload.transport ?? current.transport),
      evidenceKind: String(payload.evidence_kind ?? current.evidenceKind),
      calibrated: Boolean(payload.calibrated ?? current.calibrated),
      calibrationId: String(payload.calibration_id ?? current.calibrationId),
      decoderVersion: String(payload.decoder_version ?? current.decoderVersion),
      calibrationMetrics:
        (payload.calibration_metrics as NeuralState["calibrationMetrics"]) ?? current.calibrationMetrics,
      armedScope: String(payload.armed_scope ?? current.armedScope),
      stateRevision: Number(payload.state_revision ?? current.stateRevision),
      focusedCommandId: String(payload.focused_command_id ?? current.focusedCommandId),
      safeGoals: (payload.safe_goals as NeuralSafeGoal[] | undefined) ?? current.safeGoals,
      stagedTasks: (payload.staged_tasks as NeuralStagedTask[] | undefined) ?? current.stagedTasks,
      waiting: "",
      error: "",
    }));
  }

  const retry = lateRetry();

  /** A reply that says neural input is still starting: show its sentence as a waiting state (not in the error
   * field) and ask again when the daemon says the part is ready, or after a short interval. True when it was. */
  function stillStarting(result: unknown): boolean {
    const message = startingMessage(result);
    if (!message) return false;
    store.update((current) => ({ ...current, waiting: message, error: "" }));
    retry.schedule(() => void refresh());
    return true;
  }

  async function commitPreview(worldModelApproved: boolean) {
    const current = get(store);
    if (!current.preview) return;
    clearAutoCommit();
    store.update((value) => ({ ...value, busy: true, error: "" }));
    try {
      const result = (await call("neural_commit", {
        preview_id: current.preview.preview_id,
        expected_revision: current.preview.state_revision,
        world_model_approved: worldModelApproved,
      })) as Record<string, unknown>;
      requireResultStatus(
        result,
        ["committed", "submitted", "completed", "failed"],
        "The daemon did not complete the neural intent.",
      );
      const terminalError =
        result.status === "failed" ? String(result.error ?? "The neural intent failed without a verified result.") : "";
      store.update((value) => ({
        ...value,
        preview: null,
        lastResult: result,
        busy: false,
        error: terminalError,
      }));
      await refresh();
      if (terminalError) store.update((value) => ({ ...value, error: terminalError }));
    } catch (cause) {
      store.update((value) => ({ ...value, busy: false, error: readableError(cause) }));
    }
  }

  function acceptPreview(payload: NeuralPreview) {
    clearAutoCommit();
    store.update((current) => ({ ...current, preview: payload, lastResult: null, error: "" }));
    if (!payload.requires_non_neural_approval) {
      const cancellationMs = Math.max(0, (payload.eligible_at_ns - payload.created_at_ns) / 1_000_000);
      autoCommit = setTimeout(() => void commitPreview(false), cancellationMs);
    }
  }

  const notificationHandler = (method: string, params: unknown) => {
    const payload = (params ?? {}) as Record<string, unknown>;
    if (method === "neural_status") applyStatus(payload);
    else if (method === "neural_observation") {
      store.update((current) => ({
        ...current,
        quality: (payload.quality as NeuralQuality | undefined) ?? null,
        bufferedSamples: Number(payload.buffered_samples ?? 0),
        droppedSamples: Number(payload.dropped_samples ?? 0),
      }));
    } else if (method === "neural_preview") acceptPreview(payload as unknown as NeuralPreview);
    else if (method === "neural_navigation" || method === "neural_result") {
      clearAutoCommit();
      store.update((current) => ({
        ...current,
        preview: null,
        lastResult: payload,
        focusedCommandId: String(payload.focused_command_id ?? current.focusedCommandId),
      }));
      void refresh();
    } else if (method === "neural_disarmed") {
      clearAutoCommit();
      applyStatus(payload);
      store.update((current) => ({ ...current, preview: null }));
    }
  };

  onNotification(notificationHandler);
  const hot = (import.meta as ImportMeta & { hot?: { dispose(callback: () => void): void } }).hot;
  if (hot) hot.dispose(() => offNotification(notificationHandler));

  async function refresh() {
    try {
      const payload = (await call("neural_status")) as Record<string, unknown>;
      if (stillStarting(payload)) {
        store.update((current) => ({ ...current, connected: false })); // asked again once the part is ready
      } else if (payload.status === "unavailable") {
        applyStatus(payload);
        store.update((current) => ({
          ...current,
          connected: false,
          error: String(payload.error ?? "Neural controller is unavailable in the Heliox daemon."),
        }));
      } else {
        requireResultStatus(payload, "ok", "The daemon returned an invalid neural status response.");
        applyStatus(payload);
      }
    } catch (cause) {
      store.update((current) => ({ ...current, error: readableError(cause) }));
    }
    try {
      const process = await invoke<{ running?: boolean; pid?: number | null }>("get_neural_sidecar_status");
      store.update((current) => ({
        ...current,
        sidecarRunning: Boolean(process?.running),
        sidecarPid: process?.pid ?? null,
      }));
    } catch {
      // Browser development can still observe a manually started sidecar.
    }
  }

  async function startSidecar(options: Record<string, unknown>) {
    store.update((current) => ({ ...current, busy: true, error: "" }));
    try {
      const process = await invoke<{ running: boolean; pid: number | null }>("start_neural_sidecar", { options });
      store.update((current) => ({
        ...current,
        busy: false,
        sidecarRunning: process.running,
        sidecarPid: process.pid,
      }));
      for (let attempt = 0; attempt < 20; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 250));
        await refresh();
        if (get(store).connected) return;
      }
      throw new Error("The sidecar started but did not pair with the daemon in time");
    } catch (cause) {
      store.update((current) => ({ ...current, busy: false, error: readableError(cause) }));
    }
  }

  async function stopSidecar() {
    await disarm("sidecar_stop");
    try {
      await invoke("stop_neural_sidecar");
    } catch (cause) {
      store.update((current) => ({ ...current, error: readableError(cause) }));
    }
    store.update((current) => ({ ...current, sidecarRunning: false, sidecarPid: null }));
    await refresh();
  }

  async function beginCalibration() {
    const current = get(store);
    if (!current.sessionId) return;
    store.update((value) => ({ ...value, busy: true, error: "" }));
    try {
      const result = (await call("neural_begin_calibration", { session_id: current.sessionId })) as Record<
        string,
        unknown
      >;
      if (stillStarting(result)) return;
      requireResultStatus(result, "ok", "The daemon did not begin neural calibration.");
      applyStatus(result);
    } catch (cause) {
      store.update((value) => ({ ...value, error: readableError(cause) }));
    } finally {
      store.update((value) => ({ ...value, busy: false }));
    }
  }

  async function arm(scope: "navigate" | "safe_desktop") {
    const current = get(store);
    if (!current.sessionId) return;
    store.update((value) => ({ ...value, busy: true, error: "" }));
    try {
      const result = (await call("neural_arm", {
        session_id: current.sessionId,
        scope,
        user_authorized: true,
      })) as Record<string, unknown>;
      if (stillStarting(result)) return;
      requireResultStatus(result, "ok", "The daemon did not arm neural control.");
      applyStatus(result);
    } catch (cause) {
      store.update((value) => ({ ...value, error: readableError(cause) }));
    } finally {
      store.update((value) => ({ ...value, busy: false }));
    }
  }

  async function disarm(reason = "user_emergency_disarm") {
    clearAutoCommit();
    try {
      const result = (await call("neural_disarm", { reason })) as Record<string, unknown>;
      if (stillStarting(result)) return;
      requireResultStatus(result, "ok", "The daemon did not confirm neural disarm.");
      applyStatus(result);
      store.update((current) => ({ ...current, preview: null }));
    } catch (cause) {
      store.update((current) => ({ ...current, error: readableError(cause), preview: null }));
    }
  }

  async function deferPreviewForSafetyDecision() {
    const current = get(store);
    if (!current.preview) return;
    clearAutoCommit();
    store.update((value) => ({
      ...value,
      preview: null,
      error: "Neural preview cancelled because another Heliox safety decision requires your attention.",
    }));
    try {
      const result = (await call("neural_disarm", { reason: "higher_priority_safety_decision" })) as Record<
        string,
        unknown
      >;
      requireResultStatus(result, "ok", "The daemon did not confirm neural safety disarm.");
      applyStatus(result);
      store.update((value) => ({
        ...value,
        preview: null,
        error: "Neural preview cancelled because another Heliox safety decision requires your attention.",
      }));
    } catch (cause) {
      store.update((value) => ({ ...value, error: readableError(cause), preview: null }));
    }
  }

  async function stageTask(label: string, goal: string, sessionId = "neural"): Promise<boolean> {
    store.update((current) => ({ ...current, busy: true, error: "" }));
    try {
      const result = (await call("neural_stage_task", {
        label,
        goal,
        session_id: sessionId,
      })) as Record<string, unknown>;
      if (stillStarting(result)) return false;
      requireResultStatus(result, "ok", "The daemon did not stage the neural task.");
      applyStatus(result);
      return true;
    } catch (cause) {
      store.update((current) => ({ ...current, error: readableError(cause) }));
      return false;
    } finally {
      store.update((current) => ({ ...current, busy: false }));
    }
  }

  async function removeStagedTask(taskId: string): Promise<boolean> {
    store.update((current) => ({ ...current, busy: true, error: "" }));
    try {
      const result = (await call("neural_remove_staged_task", { task_id: taskId })) as Record<string, unknown>;
      if (stillStarting(result)) return false;
      requireResultStatus(result, "ok", "The daemon did not remove the staged neural task.");
      applyStatus(result);
      return true;
    } catch (cause) {
      store.update((current) => ({ ...current, error: readableError(cause) }));
      return false;
    } finally {
      store.update((current) => ({ ...current, busy: false }));
    }
  }

  return {
    subscribe: store.subscribe,
    refresh,
    startSidecar,
    stopSidecar,
    beginCalibration,
    arm,
    disarm,
    deferPreviewForSafetyDecision,
    stageTask,
    removeStagedTask,
    approvePreview: () => commitPreview(true),
    cancelPreview: () => disarm("preview_cancelled_by_user"),
    setStimulusEnabled(enabled: boolean) {
      store.update((current) => ({ ...current, stimulusEnabled: enabled }));
    },
  };
}

export const neural = createNeuralControl();

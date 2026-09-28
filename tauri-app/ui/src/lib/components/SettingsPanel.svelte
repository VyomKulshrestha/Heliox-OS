<script lang="ts">
  import { settings } from "../stores/settings";
  import { _, locale } from "svelte-i18n";
  import { session } from "../stores/session";
  import { call, requireOkResult, requireResultStatus, type DaemonStatusResult } from "../api/daemon";
  import { invoke } from "../api/invoke";
  import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
  import ConfirmPrompt from "./ConfirmPrompt.svelte";
  import PermissionAuditLog from "./PermissionAuditLog.svelte";
  import GatewayPolicyEditor from "./GatewayPolicyEditor.svelte";
  import GatewayAuditLog from "./GatewayAuditLog.svelte";
  import RiskWorldModelPanel from "./RiskWorldModelPanel.svelte";
  import TemporalMemoryPanel from "./TemporalMemoryPanel.svelte";
  import OnlineLearningPanel from "./OnlineLearningPanel.svelte";
  import StrategyEvolutionPanel from "./StrategyEvolutionPanel.svelte";
  import EvolutionHarnessPanel from "./EvolutionHarnessPanel.svelte";
  import AgentMeshPanel from "./AgentMeshPanel.svelte";
  import PeerMeshPanel from "./PeerMeshPanel.svelte";
  import AirHandoffPanel from "./AirHandoffPanel.svelte";
  import CalendarIntegrationPanel from "./CalendarIntegrationPanel.svelte";
  import EmailIntegrationPanel from "./EmailIntegrationPanel.svelte";
  import SshIntegrationPanel from "./SshIntegrationPanel.svelte";
  import GestureWorkflowBindings from "./GestureWorkflowBindings.svelte";
  import VoiceGestureWorkflowStatus from "./VoiceGestureWorkflowStatus.svelte";
  import RecordedWorkflowsPanel from "./RecordedWorkflowsPanel.svelte";
  import SelfHealingPanel from "./SelfHealingPanel.svelte";
  import NarrationPanel from "./NarrationPanel.svelte";
  import SupervisionPanel from "./SupervisionPanel.svelte";
  import NeuralControlPanel from "./NeuralControlPanel.svelte";
  import { getSharedGestureCalibrationStore } from "../gesture/calibration";
  import { gazeRuntime, resetGazeRuntime } from "../stores/gazeRuntime";
  import { defaultHotkey, isNativeTauriRuntime, normalizeHotkeyValue } from "../hotkey";
  import { companion } from "../stores/companion";

  let {
    onOpenCommand = () => {},
  }: {
    onOpenCommand?: () => void;
  } = $props();

  let pendingConfirm = $state<{ message: string; danger: boolean; onConfirm: () => void } | null>(null);

  function askConfirm(message: string, onConfirm: () => void, danger = false) {
    pendingConfirm = { message, danger, onConfirm };
  }

  function resolveConfirm(accepted: boolean) {
    const action = pendingConfirm?.onConfirm;
    pendingConfirm = null;
    if (accepted && action) action();
  }
  let apiKeyInput = $state("");
  let apiKeySaved = $state(false);
  let apiKeySaving = $state(false);
  let apiKeyError = $state("");
  let rootToast = $state("");
  let rootToastType = $state<"success" | "warning">("success");
  let rootSaving = $state(false);
  let elevationRequesting = $state(false);
  let elevationMessage = $state("");
  let rootRuntime = $state<{
    status: "ok";
    root_policy_enabled: boolean;
    process_elevated: boolean;
    platform: string;
    detail: string;
  } | null>(null);
  let snapshotSaving = $state(false);
  let snapshotToast = $state("");
  let snapshotRuntime = $state<{
    status: "ok";
    enabled: boolean;
    backend: string;
    available: boolean;
    ready: boolean;
    detail: string;
    retention_supported: boolean;
    retention_count: number;
    retention_detail: string;
  } | null>(null);
  let dryRunSaving = $state(false);
  let dryRunToast = $state("");
  let previewSaving = $state(false);
  let previewToast = $state("");
  let typedDecisionSaving = $state(false);
  let typedDecisionToast = $state("");
  let typedDecisionPaths = $state({ artifact_dir: "", report_path: "" });
  let gestureCursorToast = $state("");
  let visionToast = $state("");
  let gestureCalibrationToast = $state("");
  let voiceCalibrationToast = $state("");
  let speechToast = $state("");
  let speechSaving = $state(false);
  let speechTesting = $state(false);
  let recognitionTesting = $state(false);
  let audioInputDevices = $state<
    Array<{
      id: string;
      name: string;
      hostapi: string;
      is_default: boolean;
    }>
  >([]);
  let audioInputMessage = $state("");

  $effect(() => {
    if ($locale) {
      localStorage.setItem("locale", $locale);
    }
  });

  $effect(() => {
    const config = $settings.typed_decision;
    if (config) {
      typedDecisionPaths = {
        artifact_dir: config.artifact_dir || "",
        report_path: config.report_path || "",
      };
    }
  });

  let hotkeyInput = $state("");
  let hotkeySaved = $state(false);
  let hotkeyError = $state("");
  const hotkeySupported = isNativeTauriRuntime();

  const cloudModels: Record<string, string[]> = {
    gemini: ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    openai: ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
    openrouter: [
      "openrouter/auto",
      "deepseek/deepseek-v4-pro",
      "~deepseek/deepseek-v4-flash-latest",
      "~anthropic/claude-sonnet-latest",
      "~google/gemini-pro-latest",
      "~openai/gpt-latest",
    ],
    claude: ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
    meta: ["muse-spark-1.1"],
  };

  let availableOllamaModels = $state<string[]>([]);
  type SubscriptionStatus = {
    status?: string;
    provider: string;
    installed: boolean;
    authenticated: boolean;
    subscription: boolean;
    version?: string;
    message: string;
    last_usage?: Record<string, number>;
    session_usage?: Record<string, number>;
  };
  let subscriptionStatus = $state<SubscriptionStatus | null>(null);
  let subscriptionStatusLoading = $state(false);
  let subscriptionActionMessage = $state("");

  $effect(() => {
    call("list_ollama_models")
      .then((raw) => {
        const result = requireOkResult(
          raw as DaemonStatusResult & { models?: string[] },
          "Ollama model discovery is unavailable.",
        );
        if (!Array.isArray(result.models)) throw new Error("Ollama model discovery is incomplete.");
        availableOllamaModels = result.models;
      })
      .catch((error) => {
        availableOllamaModels = [];
        console.error(error);
      });
  });

  $effect(() => {
    const provider = $settings.model.subscription_provider;
    void refreshSubscriptionStatus(provider, false);
  });

  // Load the current hotkey when component mounts
  invoke("get_hotkey")
    .then((val) => {
      hotkeyInput = normalizeHotkeyValue(val);
    })
    .catch(() => {
      hotkeyInput = defaultHotkey();
      hotkeyError = "Could not read the active system shortcut.";
    });

  async function saveHotkey() {
    hotkeyError = "";
    if (!hotkeySupported) {
      hotkeyError = "Open Heliox as the desktop app to register a system-wide shortcut.";
      return;
    }
    const trimmed = hotkeyInput.trim();
    if (!trimmed) {
      hotkeyError = "Enter a shortcut such as Ctrl+Space or Alt+H.";
      return;
    }
    try {
      await invoke("set_hotkey", { shortcut: trimmed });
      await settings.updateSection("", { hotkey: trimmed });
      hotkeyInput = trimmed;
      hotkeySaved = true;
      setTimeout(() => (hotkeySaved = false), 3000);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      hotkeyError = detail ? `Shortcut was not changed: ${detail}` : "Invalid shortcut. Try Ctrl+Space or Alt+H.";
    }
  }

  async function refreshRootRuntime() {
    try {
      rootRuntime = requireOkResult(
        (await call("get_security_status")) as DaemonStatusResult,
        "Security runtime status is unavailable.",
      ) as typeof rootRuntime;
    } catch {
      rootRuntime = null;
    }
  }

  $effect(() => {
    refreshRootRuntime();
    const retry = setInterval(() => {
      if (!rootRuntime?.process_elevated) refreshRootRuntime();
    }, 2000);
    return () => clearInterval(retry);
  });

  async function refreshSnapshotRuntime() {
    try {
      snapshotRuntime = requireOkResult(
        (await call("get_snapshot_status")) as DaemonStatusResult,
        "Snapshot runtime status is unavailable.",
      ) as typeof snapshotRuntime;
    } catch {
      snapshotRuntime = null;
    }
  }

  $effect(() => {
    refreshSnapshotRuntime();
    const retry = setInterval(() => {
      if (!snapshotRuntime?.ready) refreshSnapshotRuntime();
    }, 2000);
    return () => clearInterval(retry);
  });

  $effect(() => {
    if (elevationRequesting && rootRuntime?.process_elevated) {
      elevationRequesting = false;
      elevationMessage = "Administrator mode is active. Snapshot protection is refreshing.";
      refreshSnapshotRuntime();
    }
  });

  async function requestAdministratorRestart() {
    if (elevationRequesting) return;
    elevationRequesting = true;
    elevationMessage = "Waiting for the Windows UAC prompt. Choose Yes to continue.";

    try {
      const result = await call<{ status: string; message: string }>("restart_elevated");
      elevationMessage = result.message;
      if (result.status !== "prompted") {
        elevationRequesting = false;
      }
    } catch (err) {
      elevationRequesting = false;
      elevationMessage =
        err instanceof Error
          ? `Administrator restart failed: ${err.message}`
          : "Administrator restart failed. The existing daemon is still running.";
    }
  }

  async function applyRootToggle(turningOn: boolean) {
    rootSaving = true;
    const synced = await settings.updateSection("security", { root_enabled: turningOn }, { requireDaemon: true });
    rootSaving = false;

    if (!synced) {
      rootToast = "Root policy was not changed because the Heliox daemon could not confirm it.";
      rootToastType = "warning";
      setTimeout(() => (rootToast = ""), 5000);
      return;
    }

    await refreshRootRuntime();
    if (turningOn) {
      rootToast = rootRuntime?.process_elevated
        ? "Root policy enabled. Administrator/root privileges are detected."
        : "Root policy enabled. OS elevation is not detected; protected operations may still be denied.";
      rootToastType = "warning";
    } else {
      rootToast = "Root policy disabled. Root-tier actions are blocked by Heliox.";
      rootToastType = "success";
    }
    setTimeout(() => (rootToast = ""), 5000);
  }

  function toggleRoot() {
    if (rootSaving) return;
    const turningOn = !$settings.security.root_enabled;
    if (turningOn) {
      askConfirm(
        "⚠️ ENABLE ROOT ACCESS?\n\n" +
          "This unlocks Heliox's full power:\n" +
          "• Admin/sudo shell commands\n" +
          "• System service management\n" +
          "• Protected file modifications\n" +
          "• Registry & disk-level operations\n\n" +
          "Actions requiring elevated privileges will no longer be blocked by Heliox policy.\n" +
          "The operating system may still deny them unless the daemon is elevated.\n" +
          "Only enable this if you trust the AI agent with system-level access.",
        () => applyRootToggle(true),
        true,
      );
      return;
    }
    applyRootToggle(false);
  }

  async function applyAutoSnapshotToggle(turningOn: boolean) {
    snapshotSaving = true;
    const synced = await settings.updateSection(
      "security",
      { snapshot_on_destructive: turningOn },
      { requireDaemon: true },
    );
    snapshotSaving = false;

    if (!synced) {
      snapshotToast = "Auto-Snapshot was not changed because the daemon could not confirm it.";
      setTimeout(() => (snapshotToast = ""), 5000);
      return;
    }

    await refreshSnapshotRuntime();
    snapshotToast = turningOn
      ? snapshotRuntime?.ready
        ? "Auto-Snapshot enabled. The snapshot backend is ready."
        : "Auto-Snapshot enabled, but its backend is not ready. Destructive actions will be blocked."
      : "Auto-Snapshot disabled. Confirmed destructive actions can run without a rollback point.";
    setTimeout(() => (snapshotToast = ""), 5000);
  }

  function toggleAutoSnapshot() {
    if (snapshotSaving) return;
    const turningOn = !$settings.security.snapshot_on_destructive;
    if (!turningOn) {
      askConfirm(
        "DISABLE AUTO-SNAPSHOT?\n\n" +
          "Destructive actions will be able to run after confirmation without first creating a rollback point.",
        () => applyAutoSnapshotToggle(false),
        true,
      );
      return;
    }
    applyAutoSnapshotToggle(true);
  }

  function snapshotBackendLabel(backend: string): string {
    if (backend === "windows_restore_point") return "Windows Restore Point";
    if (backend === "btrfs") return "Btrfs";
    if (backend === "timeshift") return "Timeshift";
    return "No backend";
  }

  async function applyDryRunToggle(turningOn: boolean) {
    dryRunSaving = true;
    const synced = await settings.updateSection("security", { dry_run: turningOn }, { requireDaemon: true });
    dryRunSaving = false;

    if (!synced) {
      dryRunToast = "Dry Run was not changed because the daemon could not confirm it.";
      setTimeout(() => (dryRunToast = ""), 5000);
      return;
    }

    dryRunToast = turningOn
      ? "Dry Run enabled. Plans will be simulated and audited without executing actions."
      : "Dry Run disabled. Future approved plans can change the OS, files, and processes.";
    setTimeout(() => (dryRunToast = ""), 5000);
  }

  function toggleDryRun() {
    if (dryRunSaving) return;
    const turningOn = !$settings.security.dry_run;
    if (!turningOn) {
      askConfirm(
        "RETURN TO LIVE EXECUTION?\n\n" +
          "Future approved plans can change the operating system, files, processes, and browser state.",
        () => applyDryRunToggle(false),
        true,
      );
      return;
    }
    applyDryRunToggle(true);
  }

  function setMode(mode: string) {
    settings.updateSection("model", { mode });
  }

  function setProvider(provider: string) {
    settings.updateSection("model", { provider });
  }

  function setCloudProvider(cloud_provider: string) {
    settings.updateSection("model", { cloud_provider, cloud_model: "", provider: "cloud" });
  }

  function updateCloudModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { cloud_model: val });
  }

  async function setSubscriptionProvider(subscription_provider: "codex" | "claude") {
    subscriptionStatus = null;
    subscriptionActionMessage = "";
    await settings.updateSection("model", {
      provider: "subscription",
      subscription_provider,
      subscription_model: "",
    });
    await refreshSubscriptionStatus(subscription_provider, true);
  }

  function updateSubscriptionModel(e: Event) {
    settings.updateSection("model", { subscription_model: (e.target as HTMLInputElement).value.trim() });
  }

  function updateSubscriptionTimeout(e: Event) {
    const value = Number.parseInt((e.target as HTMLInputElement).value, 10);
    if (Number.isFinite(value)) settings.updateSection("model", { subscription_timeout_seconds: value });
  }

  function updateSubscriptionPromptCap(e: Event) {
    const value = Number.parseInt((e.target as HTMLInputElement).value, 10);
    if (Number.isFinite(value)) settings.updateSection("model", { subscription_max_prompt_chars: value });
  }

  async function refreshSubscriptionStatus(provider = $settings.model.subscription_provider, refresh = true) {
    subscriptionStatusLoading = true;
    subscriptionActionMessage = "";
    try {
      subscriptionStatus = requireOkResult(
        await call<SubscriptionStatus>("subscription_status", { provider, refresh }),
        "Subscription status is unavailable.",
      );
    } catch (error) {
      subscriptionStatus = {
        provider,
        installed: false,
        authenticated: false,
        subscription: false,
        message: error instanceof Error ? error.message : "Cannot connect to the Heliox daemon.",
      };
    } finally {
      subscriptionStatusLoading = false;
    }
  }

  async function startSubscriptionLogin() {
    subscriptionActionMessage = "";
    try {
      const result = requireResultStatus(
        await call<{ status?: string; message?: string; error?: string }>("subscription_login", {
          provider: $settings.model.subscription_provider,
        }),
        "started",
        "The official subscription sign-in flow did not start.",
      );
      subscriptionActionMessage = result.message ?? "";
    } catch (error) {
      subscriptionActionMessage = error instanceof Error ? error.message : "Could not start the official sign-in flow.";
    }
  }

  const pocketTtsVoices = ["alba", "giovanni", "lola"];
  const kokoroTtsVoices = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
  ];

  async function refreshAudioInputDevices() {
    try {
      const result = await call<{
        status?: string;
        error?: string;
        devices: Array<{
          id: string;
          name: string;
          hostapi: string;
          is_default: boolean;
        }>;
        message?: string;
      }>("list_audio_input_devices");
      requireOkResult(result, "Microphone inputs are unavailable.");
      if (!Array.isArray(result.devices)) throw new Error("Microphone input status is incomplete.");
      audioInputDevices = result.devices ?? [];
      audioInputMessage = result.message ?? "";
    } catch (err) {
      audioInputDevices = [];
      audioInputMessage = err instanceof Error ? err.message : "Could not enumerate microphone inputs.";
    }
  }

  $effect(() => {
    refreshAudioInputDevices();
  });

  async function updateAudioInput(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    speechSaving = true;
    const synced = await settings.updateSection("voice", { input_device: val }, { requireDaemon: true });
    speechSaving = false;
    speechToast = synced
      ? "Microphone input saved. An active listener was refreshed automatically."
      : "Microphone input was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function updateSpeechRecognition(e: Event) {
    const target = e.target as HTMLSelectElement;
    speechSaving = true;
    const synced = await settings.updateSection("voice", { [target.name]: target.value }, { requireDaemon: true });
    speechSaving = false;
    speechToast = synced
      ? "Speech recognition setting saved."
      : "Speech recognition was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function updateTtsEngine(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    const ttsVoice =
      val === "kokoro_tts" ? "af_heart" : val === "pocket_tts" ? "alba" : ($settings.voice?.tts_voice ?? "af_heart");
    speechSaving = true;
    const synced = await settings.updateSection(
      "voice",
      { tts_engine: val, tts_voice: ttsVoice },
      { requireDaemon: true },
    );
    speechSaving = false;
    const engineName = val === "kokoro_tts" ? "Kokoro TTS" : val === "pocket_tts" ? "Pocket TTS" : "OS Voice";
    speechToast = synced
      ? `Speech engine changed to ${engineName}.`
      : "Speech engine was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function updateTtsVoice(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    speechSaving = true;
    const synced = await settings.updateSection("voice", { tts_voice: val }, { requireDaemon: true });
    speechSaving = false;
    speechToast = synced
      ? `TTS voice changed to ${val}.`
      : "Speech voice was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  function testConfiguredVoice() {
    speechTesting = true;
    speechToast = "Testing the configured daemon voice…";
    companion.speak({
      channel: "user_speech",
      text: "Heliox voice is ready.",
      dedupeKey: `voice-test:${Date.now()}`,
      onEnd: () => {
        speechTesting = false;
        speechToast = "Voice test completed.";
        setTimeout(() => (speechToast = ""), 5000);
      },
      onError: () => {
        speechTesting = false;
        speechToast = "Voice test failed in both the daemon and browser fallback.";
        setTimeout(() => (speechToast = ""), 5000);
      },
    });
  }

  function stopConfiguredVoice() {
    companion.stopAll();
    speechTesting = false;
    speechToast = "Voice test stopped.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function testSpeechRecognition() {
    recognitionTesting = true;
    speechToast = "Listening now—say one short phrase. It will not be executed.";
    try {
      const result = requireOkResult(
        await call<{
          status: string;
          message?: string;
          transcript?: string;
          language?: string;
          latency_ms?: number;
        }>("voice_recognition_test", { timeout_seconds: 12 }),
        "The microphone recognition test failed.",
      );
      const language = result.language ? ` · ${result.language}` : "";
      const latency = typeof result.latency_ms === "number" ? ` · ${result.latency_ms} ms` : "";
      speechToast = `Heard: “${result.transcript ?? ""}”${language}${latency}`;
    } catch (error) {
      speechToast = error instanceof Error ? `Recognition test failed: ${error.message}` : "Recognition test failed.";
    } finally {
      recognitionTesting = false;
      setTimeout(() => (speechToast = ""), 12000);
    }
  }

  function updateGpuLimit(e: Event) {
    const val = parseInt((e.target as HTMLInputElement).value) || 0;
    settings.updateSection("model", { gpu_memory_limit_mb: val });
  }

  async function updateRetention(e: Event) {
    const input = e.target as HTMLInputElement;
    const parsed = Number.parseInt(input.value, 10);
    const val = Number.isFinite(parsed) ? Math.min(100, Math.max(1, parsed)) : 10;
    input.value = String(val);
    const synced = await settings.updateSection("security", { snapshot_retention_count: val }, { requireDaemon: true });
    snapshotToast = synced
      ? `Snapshot retention saved at ${val}.`
      : "Snapshot retention was not changed because the daemon could not confirm it.";
    setTimeout(() => (snapshotToast = ""), 5000);
  }

  function updateScreenVisionInterval(e: Event) {
    const rawValue = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(rawValue)) return;
    const capture_interval_seconds = Math.min(60, Math.max(0.5, rawValue));
    settings.updateSection("screen_vision", { capture_interval_seconds });
  }

  // Gesture cursor control drives the real OS mouse cursor - default off,
  // must be an explicit opt-in via this toggle (never silently enabled).
  async function toggleGestureCursor() {
    const turningOn = !$settings.gesture_cursor?.enabled;
    const synced = await settings.updateSection("gesture_cursor", {
      enabled: turningOn,
    });
    gestureCursorToast = synced
      ? turningOn
        ? "Gesture Cursor enabled. Start the camera and explicitly enter Cursor Mode to control the pointer."
        : "Gesture Cursor disabled. Any active Cursor Mode was stopped."
      : "Gesture Cursor was not changed because the daemon could not confirm it.";
    setTimeout(() => (gestureCursorToast = ""), 5000);
  }

  async function toggleGazeTracking() {
    const turningOn = !$settings.vision?.gaze_tracking_enabled;
    const synced = await settings.updateSection("vision", {
      gaze_tracking_enabled: turningOn,
    });
    if (synced && !turningOn) resetGazeRuntime();
    visionToast = synced
      ? turningOn
        ? "Gaze tracking enabled. Active camera controls switch to the compatible 3D backend automatically."
        : "Gaze tracking disabled. Active camera controls reconcile automatically."
      : "Gaze tracking was not changed because the daemon could not confirm it.";
    setTimeout(() => (visionToast = ""), 5000);
  }

  async function toggleEnhanced3DHandTracking() {
    const turningOn = $settings.vision?.mediapipe_backend !== "tasks";
    const synced = await settings.updateSection("vision", {
      mediapipe_backend: turningOn ? "tasks" : "legacy",
    });
    visionToast = synced
      ? turningOn
        ? "Enhanced 3D hand tracking enabled. Active camera controls restart automatically."
        : $settings.vision?.gaze_tracking_enabled
          ? "Compatibility mode saved, but gaze still requires 3D tracking while both features are active."
          : "2D compatibility mode enabled. Active camera controls restart automatically."
      : "Camera tracking was not changed because the daemon could not confirm it.";
    setTimeout(() => (visionToast = ""), 5000);
  }

  // "Simulate before executing" preview adds real latency (a screenshot +
  // VLM call, plus a real dry-run browser tab for browser actions) before
  // every autonomous action - default off, explicit opt-in like gesture_cursor.
  async function togglePreview() {
    if (previewSaving) return;
    const turningOn = !$settings.preview?.enabled;
    previewSaving = true;
    const synced = await settings.updateSection("preview", { enabled: turningOn }, { requireDaemon: true });
    previewSaving = false;
    previewToast = synced
      ? turningOn
        ? "Preview gate enabled. Autonomous actions now require a real preview and approval."
        : "Preview gate disabled."
      : "Preview was not changed because the daemon could not confirm it.";
    setTimeout(() => (previewToast = ""), 5000);
  }

  async function saveTypedDecisionPaths() {
    if (typedDecisionSaving) return;
    typedDecisionSaving = true;
    const saved = await settings.updateSection(
      "typed_decision",
      {
        artifact_dir: typedDecisionPaths.artifact_dir.trim(),
        report_path: typedDecisionPaths.report_path.trim(),
      },
      { requireDaemon: true },
    );
    typedDecisionSaving = false;
    typedDecisionToast = saved
      ? "Local model paths saved. Scoring still requires explicit enablement."
      : "Paths were not saved. Use absolute local paths and check that Heliox is connected.";
    setTimeout(() => (typedDecisionToast = ""), 5000);
  }

  async function toggleTypedDecision() {
    if (typedDecisionSaving) return;
    const turningOn = !$settings.typed_decision?.enabled;
    if (turningOn && (!typedDecisionPaths.artifact_dir.trim() || !typedDecisionPaths.report_path.trim())) {
      typedDecisionToast = "Set and save the local artifact directory and report file before enabling.";
      setTimeout(() => (typedDecisionToast = ""), 5000);
      return;
    }
    typedDecisionSaving = true;
    const saved = await settings.updateSection(
      "typed_decision",
      {
        enabled: turningOn,
        ...(turningOn
          ? {
              artifact_dir: typedDecisionPaths.artifact_dir.trim(),
              report_path: typedDecisionPaths.report_path.trim(),
            }
          : {}),
      },
      { requireDaemon: true },
    );
    typedDecisionSaving = false;
    typedDecisionToast = saved
      ? turningOn
        ? "Research-only scoring enabled. The model loads on its first MCP request."
        : "Scoring disabled and its model released from GPU memory."
      : "The setting was not changed. Check the paths and daemon connection.";
    setTimeout(() => (typedDecisionToast = ""), 5000);
  }

  async function updateGestureCursorSensitivity(e: Event) {
    const rawValue = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(rawValue)) return;
    const sensitivity = Math.min(3, Math.max(0.1, rawValue));
    (e.target as HTMLInputElement).value = String(sensitivity);
    const synced = await settings.updateSection("gesture_cursor", { sensitivity });
    gestureCursorToast = synced
      ? `Cursor sensitivity saved at ${sensitivity}.`
      : "Sensitivity was not changed because the daemon could not confirm it.";
    setTimeout(() => (gestureCursorToast = ""), 5000);
  }

  async function updateGestureCursorBlend(e: Event) {
    const rawValue = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(rawValue)) return;
    const blend = Math.min(1, Math.max(0, rawValue));
    (e.target as HTMLInputElement).value = String(blend);
    const synced = await settings.updateSection("gesture_cursor", { blend });
    gestureCursorToast = synced
      ? `Prediction blend saved at ${blend}.`
      : "Prediction blend was not changed because the daemon could not confirm it.";
    setTimeout(() => (gestureCursorToast = ""), 5000);
  }

  // Shared with GestureControl so Reset changes the thresholds used by an
  // active camera immediately, and learned sample counts update live.
  const gestureCalibration = getSharedGestureCalibrationStore();
  let gestureCalibrationSnapshot = $state(gestureCalibration.getSnapshot());

  $effect(() => {
    return gestureCalibration.subscribe((snapshot) => {
      gestureCalibrationSnapshot = snapshot;
    });
  });

  async function toggleGestureCalibration() {
    const turningOn = !$settings.adaptive_calibration?.gesture_enabled;
    const synced = await settings.updateSection("adaptive_calibration", {
      gesture_enabled: turningOn,
    });
    gestureCalibrationToast = synced
      ? turningOn
        ? "Adaptive gesture calibration enabled."
        : "Adaptive gesture calibration paused."
      : "Calibration was not changed because the daemon could not confirm it.";
    setTimeout(() => (gestureCalibrationToast = ""), 5000);
  }

  function resetGestureCalibration() {
    askConfirm(
      "RESET LEARNED GESTURE CALIBRATION?\n\nThis permanently removes the personalized pinch and thumb thresholds learned in this browser.",
      () => {
        gestureCalibration.reset();
        gestureCalibrationToast = "Learned gesture calibration cleared. Shipped thresholds are active.";
        setTimeout(() => (gestureCalibrationToast = ""), 5000);
      },
      true,
    );
  }

  let voiceVariants = $state<{ text: string; confirmed_count: number }[]>([]);
  let voicePromotionThreshold = $state(5);
  let voiceVariantsAvailable = $state(false);

  async function loadVoiceVariants() {
    try {
      const res = requireOkResult(
        (await call("list_wake_variants")) as DaemonStatusResult & {
          variants?: { text: string; confirmed_count: number }[];
          promotion_threshold?: number;
        },
        "Wake-word calibration status is unavailable.",
      );
      if (!Array.isArray(res.variants)) throw new Error("Wake-word calibration status is incomplete.");
      voiceVariants = res.variants;
      voicePromotionThreshold = res.promotion_threshold ?? 5;
      voiceVariantsAvailable = true;
    } catch {
      voiceVariants = [];
      voiceVariantsAvailable = false;
    }
  }

  $effect(() => {
    loadVoiceVariants();
    const refresh = setInterval(loadVoiceVariants, 5000);
    return () => clearInterval(refresh);
  });

  async function toggleVoiceCalibration() {
    const turningOn = !$settings.adaptive_calibration?.voice_wake_word_enabled;
    const synced = await settings.updateSection(
      "adaptive_calibration",
      { voice_wake_word_enabled: turningOn },
      { requireDaemon: true },
    );
    voiceCalibrationToast = synced
      ? turningOn
        ? "Adaptive wake-word matching enabled."
        : "Adaptive wake-word matching disabled."
      : "Wake-word calibration was not changed because the daemon could not confirm it.";
    setTimeout(() => (voiceCalibrationToast = ""), 5000);
  }

  async function applyVoiceCalibrationReset() {
    try {
      requireOkResult(
        await call<DaemonStatusResult>("reset_wake_calibration"),
        "The daemon did not reset learned wake-word variants.",
      );
      voiceVariants = [];
      voiceCalibrationToast = "Learned wake-word variants cleared from the live listener and disk.";
    } catch {
      voiceCalibrationToast = "Wake-word variants were not cleared because the daemon could not confirm it.";
    }
    setTimeout(() => (voiceCalibrationToast = ""), 5000);
  }

  function resetVoiceCalibration() {
    askConfirm(
      "RESET LEARNED WAKE WORDS?\n\nThis permanently removes every personalized wake-word variant from the live listener and disk.",
      () => void applyVoiceCalibrationReset(),
      true,
    );
  }

  function updateOllamaModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { ollama_model: val });
  }

  async function saveApiKey() {
    if (!apiKeyInput.trim()) return;
    apiKeySaving = true;
    apiKeyError = "";
    try {
      const provider = $settings.model.cloud_provider || "gemini";
      const result = await call<{ status: string; message?: string }>("store_api_key", {
        provider,
        api_key: apiKeyInput.trim(),
      });
      if (result.status !== "ok") {
        throw new Error(result.message || "The daemon did not confirm secure API-key storage.");
      }
      apiKeySaved = true;
      apiKeyInput = "";
      setTimeout(() => {
        apiKeySaved = false;
      }, 3000);
    } catch (err) {
      console.error("Failed to save API key:", err);
      apiKeyError = err instanceof Error ? err.message : "The API key could not be stored securely.";
    } finally {
      apiKeySaving = false;
    }
  }

  // Function to reset all settings to defaults
  function handleReset() {
    askConfirm(
      $_("settings.reset_confirm"),
      () => {
        void settings.reset();
      },
      true,
    );
  }
  function toggleTheme() {
    const currentTheme = $settings.theme || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";

    // Update the central store root section directly
    // The store's internal side-effects will automatically manage document classes and localStorage synchronization
    settings.updateSection("", { theme: nextTheme });
  }

  async function testNotification() {
    try {
      // Try Tauri native notification first
      let granted = await isPermissionGranted();
      if (!granted) {
        const permission = await requestPermission();
        granted = permission === "granted";
      }
      if (granted) {
        sendNotification({
          title: "Heliox OS",
          body: "Test notification — desktop notifications are working! 🚀",
        });
        return;
      }
    } catch {
      // Tauri plugin not available (running in browser) — fall through
    }

    // Fallback: browser Notification API
    try {
      if (!("Notification" in window)) {
        alert("Heliox OS — Notifications are not supported in this browser.");
        return;
      }
      let perm = Notification.permission;
      if (perm === "default") {
        perm = await Notification.requestPermission();
      }
      if (perm === "granted") {
        new Notification("Heliox OS", {
          body: "Test notification — desktop notifications are working! 🚀",
          icon: "/favicon.png",
        });
      } else {
        alert("Notification permission was denied. Enable it in your browser settings.");
      }
    } catch (err) {
      console.error("Browser notification failed:", err);
      alert("Heliox OS — Test notification triggered! (popups blocked by browser)");
    }
  }

  type RestrictionKey = "protected_folders" | "protected_packages" | "blocked_commands";

  let newRestrictionEntry = $state<Record<RestrictionKey, string>>({
    protected_folders: "",
    protected_packages: "",
    blocked_commands: "",
  });

  function addRestrictionEntry(key: RestrictionKey) {
    const value = newRestrictionEntry[key].trim();
    if (!value) return;
    const current = $settings.restrictions?.[key] ?? [];
    if (current.includes(value)) {
      newRestrictionEntry[key] = "";
      return;
    }
    settings.updateSection("restrictions", { [key]: [...current, value] });
    newRestrictionEntry[key] = "";
  }

  function removeRestrictionEntry(key: RestrictionKey, value: string) {
    const current = $settings.restrictions?.[key] ?? [];
    settings.updateSection("restrictions", { [key]: current.filter((v) => v !== value) });
  }

  const restrictionFields: { key: RestrictionKey; labelKey: string; placeholder: string }[] = [
    { key: "protected_folders", labelKey: "settings.protected_folders", placeholder: "/path/to/folder" },
    { key: "protected_packages", labelKey: "settings.protected_packages", placeholder: "package-name" },
    { key: "blocked_commands", labelKey: "settings.blocked_commands", placeholder: "command-name" },
  ];
</script>

<div class="settings-panel">
  {#if pendingConfirm}
    <ConfirmPrompt
      message={pendingConfirm.message}
      danger={pendingConfirm.danger}
      onconfirm={() => resolveConfirm(true)}
      oncancel={() => resolveConfirm(false)}
    />
  {/if}

  <h2>{$_("settings.title")}</h2>

  <section class="settings-group">
    <h3>{$_("settings.appearance")}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.light_mode")}</span>
        <span class="setting-desc">{$_("settings.light_mode_desc")}</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.theme === "light"}
        onclick={toggleTheme}
        aria-label="Toggle Light Mode"
        title="Toggle Light Mode"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.language")}</span>
        <span class="setting-desc">{$_("settings.language_desc")}</span>
      </div>
      <select class="input-md" bind:value={$locale}>
        <option value="en">English</option>
        <option value="hi">Hindi</option>
      </select>
    </div>
  </section>

  <section class="settings-group">
    <h3>Keyboard Shortcut</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Global Hotkey</span>
        <span class="setting-desc">Summon Heliox from anywhere (default: Ctrl+Space)</span>
        <span class="hotkey-status" class:unsupported={!hotkeySupported}>
          {hotkeySupported
            ? `Active system shortcut: ${hotkeyInput || defaultHotkey()}`
            : "Desktop app required — browsers cannot register system-wide shortcuts."}
        </span>
      </div>
      <div class="api-key-row">
        <input
          type="text"
          class="input-md"
          bind:value={hotkeyInput}
          placeholder="Ctrl+Space"
          aria-label="Global hotkey"
          disabled={!hotkeySupported}
        />
        <button class="btn-save" onclick={saveHotkey} disabled={!hotkeySupported}>
          {hotkeySaved ? "✓ Saved!" : "Save"}
        </button>
      </div>
    </div>
    {#if hotkeyError}
      <div style="padding: 6px 14px; font-size: 11px; color: var(--accent);">
        {hotkeyError}
      </div>
    {/if}
  </section>

  <section class="settings-group">
    <h3>{$_("settings.security")}</h3>

    {#if rootToast}
      <div
        class="root-toast"
        class:root-toast-warning={rootToastType === "warning"}
        class:root-toast-success={rootToastType === "success"}
      >
        {rootToast}
      </div>
    {/if}

    {#if snapshotToast}
      <div class="root-toast root-toast-warning">
        {snapshotToast}
      </div>
    {/if}

    {#if dryRunToast}
      <div
        class="root-toast"
        class:root-toast-success={$settings.security.dry_run}
        class:root-toast-warning={!$settings.security.dry_run}
      >
        {dryRunToast}
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.root_access")}</span>
        <span class="setting-desc">{$_("settings.root_access_desc")}</span>
        <span class="security-setting-status">
          {$settings.security.root_enabled
            ? rootRuntime
              ? rootRuntime.process_elevated
                ? "Policy enabled · Administrator/root detected"
                : "Policy enabled · OS elevation not detected"
              : "Policy enabled · privilege status unavailable"
            : "Policy disabled · root-tier actions blocked"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.root_enabled}
        onclick={toggleRoot}
        aria-label="Toggle Root Access"
        title="Toggle Root Access"
        aria-pressed={$settings.security.root_enabled}
        disabled={rootSaving || !rootRuntime}
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    {#if $settings.security.root_enabled}
      <div class="root-status-banner">
        <span class="root-icon">⚡</span>
        <div class="root-status-info">
          <span class="root-status-title">
            {rootRuntime
              ? rootRuntime.process_elevated
                ? "Root policy active · OS elevated"
                : "Root policy active · OS elevation not detected"
              : "Root policy active · privilege status unavailable"}
          </span>
          <span class="root-status-desc">
            {rootRuntime?.detail ||
              "Root-tier actions are allowed by Heliox policy; checking the daemon's OS privileges."}
          </span>
          {#if rootRuntime?.platform === "win32" && !rootRuntime.process_elevated}
            <span class="root-status-desc"> Windows will show a UAC prompt. Heliox stays running if you cancel. </span>
          {/if}
        </div>
        {#if rootRuntime?.platform === "win32" && !rootRuntime.process_elevated}
          <button class="elevation-button" onclick={requestAdministratorRestart} disabled={elevationRequesting}>
            {elevationRequesting ? "Waiting for UAC..." : "Restart as Administrator"}
          </button>
        {/if}
      </div>
    {/if}

    {#if elevationMessage}
      <div class="elevation-message">{elevationMessage}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.auto_snapshot")}</span>
        <span class="setting-desc">{$_("settings.auto_snapshot_desc")}</span>
        <span class="security-setting-status">
          {$settings.security.snapshot_on_destructive
            ? snapshotRuntime
              ? snapshotRuntime.ready
                ? `Protection ready · ${snapshotBackendLabel(snapshotRuntime.backend)}`
                : `Enabled but unavailable · ${snapshotBackendLabel(snapshotRuntime.backend)}`
              : "Enabled · backend status unavailable"
            : "Disabled · no automatic rollback point"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.snapshot_on_destructive}
        onclick={toggleAutoSnapshot}
        aria-label="Toggle Auto Snapshot"
        title="Toggle Auto Snapshot"
        aria-pressed={$settings.security.snapshot_on_destructive}
        disabled={snapshotSaving || !snapshotRuntime}
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    {#if $settings.security.snapshot_on_destructive && snapshotRuntime && !snapshotRuntime.ready}
      <div class="snapshot-status-banner">
        <span class="root-icon">🛡</span>
        <div class="root-status-info">
          <span class="snapshot-status-title">Snapshot protection is fail-closed</span>
          <span class="root-status-desc">{snapshotRuntime.detail}</span>
        </div>
        {#if rootRuntime?.platform === "win32" && !rootRuntime.process_elevated}
          <button
            class="elevation-button elevation-button-danger"
            onclick={requestAdministratorRestart}
            disabled={elevationRequesting}
          >
            {elevationRequesting ? "Waiting for UAC..." : "Fix: Restart as Administrator"}
          </button>
        {/if}
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.dry_run")}</span>
        <span class="setting-desc">{$_("settings.dry_run_desc")}</span>
        <span class="security-setting-status">
          {$settings.security.dry_run
            ? "Active · every planned action is simulation-only"
            : "Inactive · approved actions execute normally"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.dry_run}
        onclick={toggleDryRun}
        aria-label="Toggle Dry Run Mode"
        title="Toggle Dry Run Mode"
        aria-pressed={$settings.security.dry_run}
        disabled={dryRunSaving}
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    {#if $settings.security.dry_run}
      <div class="dry-run-status-banner">
        <span class="root-icon">🧪</span>
        <div class="root-status-info">
          <span class="dry-run-status-title">Dry Run is active</span>
          <span class="root-status-desc">
            Plans are validated, simulated, and written to the audit trail. Action handlers are not called.
          </span>
        </div>
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.snapshot_retention")}</span>
        <span class="setting-desc">{$_("settings.snapshot_retention_desc")}</span>
        {#if snapshotRuntime}
          <span class="security-setting-status">{snapshotRuntime.retention_detail}</span>
        {/if}
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.security.snapshot_retention_count}
        onchange={updateRetention}
        min="1"
        max="100"
        disabled={!snapshotRuntime?.retention_supported}
        title={snapshotRuntime?.retention_detail || "Checking snapshot backend"}
      />
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.usage")}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.total_tokens")}</span>
        <span class="setting-desc">{$_("settings.total_tokens_desc")}</span>
      </div>
      <span>{$session.totalTokens}</span>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.estimated_cost")}</span>
        <span class="setting-desc">{$_("settings.estimated_cost_desc")}</span>
      </div>
      <span>
        {$settings.model.provider === "ollama"
          ? $_("settings.free_local")
          : $settings.model.provider === "subscription"
            ? "Included plan quota"
            : `$${$session.estimatedCost.toFixed(4)}`}
      </span>
    </div>

    <div class="setting-row">
      <button class="btn-save" onclick={() => session.resetUsage()}>{$_("settings.reset_usage")}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.screen_vision")}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.capture_interval")}</span>
        <span class="setting-desc">{$_("settings.capture_interval_desc")}</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.screen_vision?.capture_interval_seconds ?? 3}
        onchange={updateScreenVisionInterval}
        min="0.5"
        max="60"
        step="0.5"
      />
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.camera_intelligence")}</h3>
    <p class="gesture-cursor-warning">{$_("settings.camera_intelligence_desc")}</p>

    {#if visionToast}
      <div class="root-toast root-toast-warning">{visionToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.enhanced_3d_hand_tracking")}</span>
        <span class="setting-desc">{$_("settings.enhanced_3d_hand_tracking_desc")}</span>
        <span class="security-setting-status">
          {$settings.vision?.mediapipe_backend === "tasks"
            ? $_("settings.enhanced_3d_status_on")
            : $settings.vision?.gaze_tracking_enabled
              ? $_("settings.enhanced_3d_status_forced")
              : $_("settings.enhanced_3d_status_off")}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.vision?.mediapipe_backend === "tasks"}
        onclick={toggleEnhanced3DHandTracking}
        aria-label="Toggle Enhanced 3D Hand Tracking"
        title="Toggle Enhanced 3D Hand Tracking"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <h4 class="settings-subheading">{$_("settings.gaze_tracking")}</h4>
    <p class="setting-section-note">{$_("settings.gaze_tracking_desc")}</p>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.gaze_tracking_enabled")}</span>
        <span class="setting-desc">{$_("settings.gaze_tracking_enabled_desc")}</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.vision?.gaze_tracking_enabled}
        onclick={toggleGazeTracking}
        aria-label="Toggle Gaze Tracking"
        title="Toggle Gaze Tracking"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div
      class="gaze-status-row"
      class:active={$settings.vision?.gaze_tracking_enabled && $gazeRuntime.phase === "active"}
      class:error={$settings.vision?.gaze_tracking_enabled &&
        ($gazeRuntime.phase === "error" || $gazeRuntime.daemonStatus === "error")}
    >
      <span class="gaze-status-dot"></span>
      <div class="gaze-status-copy">
        <strong>
          {#if !$settings.vision?.gaze_tracking_enabled}
            {$_("settings.gaze_status_off")}
          {:else if $gazeRuntime.phase === "loading"}
            {$_("settings.gaze_status_loading")}
          {:else if $gazeRuntime.phase === "scanning"}
            {$_("settings.gaze_status_scanning")}
          {:else if $gazeRuntime.phase === "stabilizing"}
            Gaze stabilizing
          {:else if $gazeRuntime.phase === "active" && $gazeRuntime.region}
            {$_("settings.gaze_status_active")}: {$gazeRuntime.region}
            ({Math.round(($gazeRuntime.confidence ?? 0) * 100)}%)
          {:else if $gazeRuntime.phase === "error"}
            {$_("settings.gaze_status_error")}
          {:else}
            {$_("settings.gaze_status_ready")}
          {/if}
        </strong>
        <span>
          {#if $settings.vision?.gaze_tracking_enabled && $gazeRuntime.message}
            {$gazeRuntime.message}
          {:else if $settings.vision?.gaze_tracking_enabled}
            {$_("settings.gaze_status_ready_desc")}
          {:else}
            {$_("settings.gaze_status_off_desc")}
          {/if}
        </span>
      </div>
      {#if $settings.vision?.gaze_tracking_enabled && !$gazeRuntime.cameraActive}
        <button class="gaze-open-command" onclick={onOpenCommand}>
          {$_("settings.gaze_open_command")}
        </button>
      {/if}
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.preview")}</h3>
    <p class="gesture-cursor-warning">{$_("settings.preview_desc")}</p>

    {#if previewToast}
      <div
        class="root-toast"
        class:root-toast-success={$settings.preview?.enabled}
        class:root-toast-warning={!$settings.preview?.enabled}
      >
        {previewToast}
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.preview_enabled")}</span>
        <span class="setting-desc">{$_("settings.preview_enabled_desc")}</span>
        <span class="security-setting-status">
          {$settings.preview?.enabled
            ? "Active · autonomous actions fail closed if a real preview is unavailable"
            : "Inactive · no pre-action preview gate"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.preview?.enabled}
        onclick={togglePreview}
        aria-label="Toggle simulate-before-executing preview"
        title="Toggle simulate-before-executing preview"
        disabled={previewSaving}
      >
        <span class="toggle-knob"></span>
      </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>Typed decision model · research</h3>
    <p class="setting-section-note">
      Optional local MCP scoring for dynamic yes/no, choice, or ordered options. Disabled by default;
      scores never approve actions or enter the Heliox planner. Inputs are not saved or uploaded.
    </p>
    {#if typedDecisionToast}
      <div class="root-toast" role="status">{typedDecisionToast}</div>
    {/if}
    <label class="typed-decision-field">
      <span class="setting-label">Local artifact directory</span>
      <input
        class="input-md"
        type="text"
        bind:value={typedDecisionPaths.artifact_dir}
        autocomplete="off"
        spellcheck="false"
        placeholder="Absolute path containing adapter/ and head.npz"
        aria-label="Local typed decision model artifact directory"
        disabled={typedDecisionSaving}
      />
    </label>
    <label class="typed-decision-field">
      <span class="setting-label">Research report file</span>
      <input
        class="input-md"
        type="text"
        bind:value={typedDecisionPaths.report_path}
        autocomplete="off"
        spellcheck="false"
        placeholder="Absolute path to the model report JSON"
        aria-label="Local typed decision model report path"
        disabled={typedDecisionSaving}
      />
    </label>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Research-only MCP scoring</span>
        <span class="setting-desc">
          {$settings.typed_decision?.enabled
            ? "Enabled · weights load on demand and are released when disabled"
            : "Disabled · no model files are loaded or downloaded"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.typed_decision?.enabled}
        onclick={toggleTypedDecision}
        aria-label="Toggle research-only typed decision scoring"
        title="Toggle research-only typed decision scoring"
        disabled={typedDecisionSaving}
      >
        <span class="toggle-knob"></span>
      </button>
    </div>
    <div class="setting-row">
      <span class="setting-desc">The MCP tool supports at most 64 options per request.</span>
      <button class="typed-decision-save" onclick={saveTypedDecisionPaths} disabled={typedDecisionSaving}>
        Save paths
      </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.gesture_cursor")}</h3>

    <p class="gesture-cursor-warning">{$_("settings.gesture_cursor_warning")}</p>

    {#if gestureCursorToast}
      <div class="root-toast root-toast-warning">{gestureCursorToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.gesture_cursor_enabled")}</span>
        <span class="setting-desc">{$_("settings.gesture_cursor_enabled_desc")}</span>
        <span class="security-setting-status">
          {$settings.gesture_cursor?.enabled
            ? "Enabled · cursor movement still requires the visible Cursor Mode button"
            : "Disabled · camera gestures cannot move or click the OS cursor"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.gesture_cursor?.enabled}
        onclick={toggleGestureCursor}
        aria-label="Toggle Gesture Cursor Control"
        title="Toggle Gesture Cursor Control"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    {#if $settings.gesture_cursor?.enabled}
      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.gesture_cursor_sensitivity")}</span>
          <span class="setting-desc">{$_("settings.gesture_cursor_sensitivity_desc")}</span>
        </div>
        <input
          type="number"
          class="input-sm"
          value={$settings.gesture_cursor?.sensitivity ?? 1.0}
          onchange={updateGestureCursorSensitivity}
          min="0.1"
          max="3"
          step="0.1"
        />
      </div>

      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.gesture_cursor_blend")}</span>
          <span class="setting-desc">{$_("settings.gesture_cursor_blend_desc")}</span>
        </div>
        <input
          type="number"
          class="input-sm"
          value={$settings.gesture_cursor?.blend ?? 0.3}
          onchange={updateGestureCursorBlend}
          min="0"
          max="1"
          step="0.05"
        />
      </div>
    {/if}
  </section>

  <section class="settings-group">
    <h3>{$_("settings.gesture_calibration")}</h3>
    <p class="gesture-cursor-warning">{$_("settings.gesture_calibration_desc")}</p>

    {#if gestureCalibrationToast}
      <div class="root-toast root-toast-success">{gestureCalibrationToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.gesture_calibration_enabled")}</span>
        <span class="setting-desc">{$_("settings.gesture_calibration_enabled_desc")}</span>
        <span class="security-setting-status">
          {$settings.adaptive_calibration?.gesture_enabled
            ? "Learning active · bounded thresholds update after confirmed samples"
            : "Paused · shipped thresholds are used without learning"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.adaptive_calibration?.gesture_enabled}
        onclick={toggleGestureCalibration}
        aria-label="Toggle Adaptive Gesture Calibration"
        title="Toggle Adaptive Gesture Calibration"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.gesture_calibration_learned")}</span>
        <span class="setting-desc">
          {#if gestureCalibrationSnapshot.pinchSampleCount === 0 && gestureCalibrationSnapshot.thumbSampleCount === 0}
            {$_("settings.gesture_calibration_no_data")}
          {:else}
            {$_("settings.gesture_calibration_pinch_samples")}: {gestureCalibrationSnapshot.pinchSampleCount} · {$_(
              "settings.gesture_calibration_thumb_samples",
            )}: {gestureCalibrationSnapshot.thumbSampleCount}
          {/if}
        </span>
      </div>
      <button class="btn-save" onclick={resetGestureCalibration}>{$_("settings.gesture_calibration_reset")}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.voice_calibration")}</h3>
    <p class="gesture-cursor-warning">{$_("settings.voice_calibration_desc")}</p>

    {#if voiceCalibrationToast}
      <div class="root-toast root-toast-success">{voiceCalibrationToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.voice_calibration_enabled")}</span>
        <span class="setting-desc">{$_("settings.voice_calibration_enabled_desc")}</span>
        <span class="security-setting-status">
          {$settings.adaptive_calibration?.voice_wake_word_enabled
            ? "Active · the daemon's Hey Heliox listener can use promoted variants"
            : "Inactive · only exact built-in wake words are accepted"}
        </span>
      </div>
      <button
        class="toggle"
        class:active={$settings.adaptive_calibration?.voice_wake_word_enabled}
        onclick={toggleVoiceCalibration}
        aria-label="Toggle Adaptive Wake-Word Matching"
        title="Toggle Adaptive Wake-Word Matching"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.voice_calibration_learned")}</span>
        <span class="setting-desc">
          {#if !voiceVariantsAvailable}
            Calibration status unavailable
          {:else if voiceVariants.length === 0}
            {$_("settings.voice_calibration_no_data")}
          {:else}
            {#each voiceVariants as variant}
              "{variant.text}" ({variant.confirmed_count}/{voicePromotionThreshold}){#if variant !== voiceVariants[voiceVariants.length - 1]},
              {/if}
            {/each}
          {/if}
        </span>
      </div>
      <button class="btn-save" onclick={resetVoiceCalibration} disabled={!voiceVariantsAvailable}
        >{$_("settings.voice_calibration_reset")}</button
      >
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.voice_speech")}</h3>
    <p class="gesture-cursor-warning">{$_("settings.voice_speech_desc")}</p>

    {#if speechToast}
      <div
        class="root-toast"
        class:root-toast-success={!speechToast.includes("not changed") && !speechToast.includes("failed")}
        class:root-toast-warning={speechToast.includes("not changed") || speechToast.includes("failed")}
      >
        {speechToast}
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Microphone Input</span>
        <span class="setting-desc">Audio device used by Heliox Active and local wake-word recognition.</span>
        {#if audioInputMessage}
          <span class="security-setting-status">{audioInputMessage}</span>
        {/if}
      </div>
      <select
        class="input-md"
        value={$settings.voice?.input_device ?? "auto"}
        onchange={updateAudioInput}
        disabled={speechSaving || audioInputDevices.length === 0}
      >
        <option value="auto">Automatic (system default)</option>
        {#each audioInputDevices as device}
          <option value={device.id}>
            {device.name} — {device.hostapi}{device.is_default ? " (system default)" : ""}
          </option>
        {/each}
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Speech recognition engine</span>
        <span class="setting-desc"
          >Automatic prefers faster local recognition and falls back safely when unavailable.</span
        >
      </div>
      <select
        class="input-md"
        name="transcription_engine"
        value={$settings.voice?.transcription_engine ?? "auto"}
        onchange={updateSpeechRecognition}
        disabled={speechSaving}
      >
        <option value="auto">Automatic (recommended)</option>
        <option value="faster_whisper">Faster-Whisper only</option>
        <option value="openai_whisper">OpenAI Whisper only</option>
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Recognition quality</span>
        <span class="setting-desc"
          >Small is the recommended accuracy/speed balance; larger models need more memory.</span
        >
      </div>
      <select
        class="input-md"
        name="whisper_model"
        value={$settings.voice?.whisper_model ?? "small"}
        onchange={updateSpeechRecognition}
        disabled={speechSaving}
      >
        <option value="tiny">Fastest (tiny)</option>
        <option value="base">Fast (base)</option>
        <option value="small">Balanced (small, recommended)</option>
        <option value="medium">More accurate (medium)</option>
        <option value="turbo">Most accurate fast model (turbo)</option>
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Recognition language</span>
        <span class="setting-desc"
          >Automatic supports multilingual speech; selecting a language can improve speed and accuracy.</span
        >
      </div>
      <select
        class="input-md"
        name="language"
        value={$settings.voice?.language ?? "auto"}
        onchange={updateSpeechRecognition}
        disabled={speechSaving}
      >
        <option value="auto">Automatic language detection</option>
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="es">Spanish</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="it">Italian</option>
        <option value="ja">Japanese</option>
        <option value="ko">Korean</option>
        <option value="pt">Portuguese</option>
        <option value="zh">Chinese</option>
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Test speech recognition</span>
        <span class="setting-desc">Shows exactly what Heliox hears without executing the phrase.</span>
      </div>
      <button class="btn-save" onclick={testSpeechRecognition} disabled={speechSaving || recognitionTesting}>
        {recognitionTesting ? "Listening…" : "Test microphone"}
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.tts_engine")}</span>
        <span class="setting-desc">{$_("settings.tts_engine_desc")}</span>
        <span class="security-setting-status">Used by voice replies, live narration, supervision, and this test.</span>
      </div>
      <select
        class="input-md"
        value={$settings.voice?.tts_engine ?? "kokoro_tts"}
        onchange={updateTtsEngine}
        disabled={speechSaving || speechTesting}
      >
        <option value="kokoro_tts">Kokoro TTS (Natural)</option>
        <option value="pocket_tts">Pocket TTS</option>
        <option value="os_native">OS Voice</option>
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.tts_voice")}</span>
        <span class="setting-desc">{$_("settings.tts_voice_desc")}</span>
      </div>
      <select
        class="input-md"
        value={$settings.voice?.tts_voice ?? "af_heart"}
        onchange={updateTtsVoice}
        disabled={$settings.voice?.tts_engine === "os_native" || speechSaving || speechTesting}
      >
        {#each $settings.voice?.tts_engine === "kokoro_tts" ? kokoroTtsVoices : pocketTtsVoices as voiceOption}
          <option value={voiceOption}>{voiceOption}</option>
        {/each}
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Test configured voice</span>
        <span class="setting-desc">Speaks one short sentence through the selected engine and voice.</span>
      </div>
      {#if speechTesting}
        <button class="btn-save" onclick={stopConfiguredVoice}>Stop test</button>
      {:else}
        <button class="btn-save" onclick={testConfiguredVoice} disabled={speechSaving}>Test voice</button>
      {/if}
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.model")}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.provider")}</span>
        <span class="setting-desc">{$_("settings.provider_desc")}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.provider === "ollama"} onclick={() => setProvider("ollama")}
          >Ollama</button
        >
        <button class:active={$settings.model.provider === "cloud"} onclick={() => setProvider("cloud")}>Cloud</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.reasoning_mode")}</span>
        <span class="setting-desc">{$_("settings.reasoning_mode_desc")}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.mode === "lightweight"} onclick={() => setMode("lightweight")}
          >{$_("settings.light")}</button
        >
        <button class:active={$settings.model.mode === "full"} onclick={() => setMode("full")}
          >{$_("settings.full")}</button
        >
      </div>
    </div>

    {#if $settings.model.provider === "ollama"}
      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.ollama_model")}</span>
          <span class="setting-desc">{$_("settings.ollama_model_desc")}</span>
        </div>
        {#if availableOllamaModels.length > 0}
          <select class="input-md" value={$settings.model.ollama_model} onchange={updateOllamaModel}>
            {#each availableOllamaModels as modelOption}
              <option value={modelOption}>{modelOption}</option>
            {/each}
          </select>
        {:else}
          <input
            type="text"
            class="input-md"
            value={$settings.model.ollama_model}
            onchange={updateOllamaModel}
            placeholder="llama3.1:8b"
          />
        {/if}
      </div>

      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{$_("settings.gpu_memory")}</span>
          <span class="setting-desc">{$_("settings.gpu_memory_desc")}</span>
        </div>
        <input
          type="number"
          class="input-sm"
          value={$settings.model.gpu_memory_limit_mb}
          onchange={updateGpuLimit}
          min="0"
          step="512"
        />
      </div>

      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">Idle model unload</span>
          <span class="setting-desc"
            >Seconds to keep a local model in memory after its last request; 0 unloads immediately.</span
          >
        </div>
        <input
          type="number"
          class="input-sm"
          value={$settings.model.idle_unload_seconds}
          onchange={(event) =>
            settings.updateSection("model", {
              idle_unload_seconds: Number.parseInt((event.target as HTMLInputElement).value, 10) || 0,
            })}
          min="0"
          max="86400"
          step="30"
        />
      </div>
    {/if}
  </section>

  <section class="settings-group">
    <h3>{$_("settings.cloud_api")}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.cloud_provider")}</span>
        <span class="setting-desc">{$_("settings.cloud_provider_desc")}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.cloud_provider === "gemini"} onclick={() => setCloudProvider("gemini")}
          >Gemini</button
        >
        <button class:active={$settings.model.cloud_provider === "openai"} onclick={() => setCloudProvider("openai")}
          >OpenAI</button
        >
        <button
          class:active={$settings.model.cloud_provider === "openrouter"}
          onclick={() => setCloudProvider("openrouter")}>OpenRouter</button
        >
        <button class:active={$settings.model.cloud_provider === "claude"} onclick={() => setCloudProvider("claude")}
          >Claude</button
        >
        <button class:active={$settings.model.cloud_provider === "meta"} onclick={() => setCloudProvider("meta")}
          >Meta</button
        >
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.cloud_model")}</span>
        <span class="setting-desc">{$_("settings.cloud_model_desc")}</span>
      </div>
      {#if $settings.model.cloud_provider === "openrouter"}
        <input
          class="input-md"
          list="openrouter-model-options"
          value={$settings.model.cloud_model}
          oninput={updateCloudModel}
          placeholder="openrouter/auto or any OpenRouter model slug"
        />
        <datalist id="openrouter-model-options">
          {#each cloudModels.openrouter as modelOption}
            <option value={modelOption}></option>
          {/each}
        </datalist>
      {:else}
        <select class="input-md" value={$settings.model.cloud_model} onchange={updateCloudModel}>
          <option value="">Default for provider</option>
          {#each cloudModels[$settings.model.cloud_provider || "gemini"] || [] as modelOption}
            <option value={modelOption}>{modelOption}</option>
          {/each}
        </select>
      {/if}
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.api_key")}</span>
        <span class="setting-desc">{$_("settings.api_key_desc")}</span>
      </div>
      <div class="api-key-row">
        <input
          type="password"
          class="input-md"
          bind:value={apiKeyInput}
          placeholder={$settings.model.cloud_provider === "openrouter"
            ? "sk-or-v1-..."
            : $_("settings.api_key_placeholder")}
        />
        <button class="btn-save" onclick={saveApiKey} disabled={apiKeySaving}>
          {apiKeySaved ? $_("settings.saved") : apiKeySaving ? $_("settings.saving") : $_("settings.save")}
        </button>
      </div>
      {#if apiKeyError}
        <p class="api-key-error" role="alert">{apiKeyError}</p>
      {/if}
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.restrictions")}</h3>

    {#each restrictionFields as field}
      <div class="restriction-editor">
        <span class="setting-label">{$_(field.labelKey)}</span>
        <div class="restriction-chips">
          {#each $settings.restrictions?.[field.key] ?? [] as entry}
            <span class="restriction-chip">
              <code>{entry}</code>
              <button class="chip-remove" title="Remove" onclick={() => removeRestrictionEntry(field.key, entry)}
                >&times;</button
              >
            </span>
          {:else}
            <span class="restriction-empty">{$_("settings.configured")}: 0</span>
          {/each}
        </div>
        <div class="restriction-add-row">
          <input
            type="text"
            class="input-md"
            placeholder={field.placeholder}
            bind:value={newRestrictionEntry[field.key]}
            onkeydown={(e) => {
              if (e.key === "Enter") addRestrictionEntry(field.key);
            }}
          />
          <button class="btn-save" onclick={() => addRestrictionEntry(field.key)}>Add</button>
        </div>
      </div>
    {/each}
  </section>

  <section class="settings-group subscription-panel">
    <h3>Existing AI subscription</h3>
    <div class="subscription-notice">
      Heliox invokes the official local CLI in a tool-free, read-only planning sandbox. Codex or Claude owns the login;
      Heliox never reads browser cookies, OAuth tokens, or API keys.
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Subscription provider</span>
        <span class="setting-desc">Use a ChatGPT Codex or Claude Code subscription already signed in on this PC.</span>
      </div>
      <div class="btn-group">
        <button
          class:active={$settings.model.subscription_provider === "codex"}
          onclick={() => setSubscriptionProvider("codex")}>Codex</button
        >
        <button
          class:active={$settings.model.subscription_provider === "claude"}
          onclick={() => setSubscriptionProvider("claude")}>Claude Code</button
        >
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info subscription-status-copy">
        <span class="setting-label">Connection</span>
        {#if subscriptionStatusLoading}
          <span class="setting-desc">Checking the official CLI...</span>
        {:else if subscriptionStatus}
          <span
            class:status-ok={subscriptionStatus.subscription}
            class:status-warning={!subscriptionStatus.subscription}
          >
            {subscriptionStatus.subscription ? "Subscription connected" : "Not connected"}
            {subscriptionStatus.version ? ` · ${subscriptionStatus.version}` : ""}
          </span>
          <span class="setting-desc">{subscriptionStatus.message}</span>
        {:else}
          <span class="setting-desc">Status has not been checked.</span>
        {/if}
      </div>
      <div class="subscription-actions">
        <button class="btn-save" onclick={startSubscriptionLogin} disabled={subscriptionStatusLoading}>
          {subscriptionStatus?.authenticated ? "Reconnect" : "Connect"}
        </button>
        <button
          class="btn-secondary"
          onclick={() => refreshSubscriptionStatus($settings.model.subscription_provider, true)}
          disabled={subscriptionStatusLoading}>Refresh</button
        >
      </div>
    </div>
    {#if subscriptionActionMessage}
      <p class="subscription-message" role="status">{subscriptionActionMessage}</p>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Model</span>
        <span class="setting-desc">Leave blank for the CLI default, or enter any model supported by your plan.</span>
      </div>
      <input
        type="text"
        class="input-md subscription-model-input"
        value={$settings.model.subscription_model}
        onchange={updateSubscriptionModel}
        placeholder={$settings.model.subscription_provider === "codex" ? "Default Codex model" : "Default Claude model"}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Planning timeout</span>
        <span class="setting-desc">Stops a stalled CLI request; allowed range is 15–600 seconds.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.model.subscription_timeout_seconds}
        onchange={updateSubscriptionTimeout}
        min="15"
        max="600"
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Prompt context cap</span>
        <span class="setting-desc"
          >Limits Heliox context before each request. Lower values reduce usage; 48,000 characters is the balanced
          default.</span
        >
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.model.subscription_max_prompt_chars}
        onchange={updateSubscriptionPromptCap}
        min="16000"
        max="200000"
        step="1000"
      />
    </div>

    {#if subscriptionStatus?.last_usage && Object.keys(subscriptionStatus.last_usage).length > 0}
      <div class="usage-strip">
        Last call: {subscriptionStatus.last_usage.input_tokens ?? 0} provider input · {subscriptionStatus.last_usage
          .uncached_input_tokens ??
          subscriptionStatus.last_usage.input_tokens ??
          0}
        uncached · {subscriptionStatus.last_usage.cached_input_tokens ?? 0} cached · {subscriptionStatus.last_usage
          .output_tokens ?? 0} output. Heliox supplied about {subscriptionStatus.last_usage
          .heliox_estimated_prompt_tokens ?? 0} prompt tokens.
      </div>
    {/if}
    {#if subscriptionStatus?.session_usage && Object.keys(subscriptionStatus.session_usage).length > 0}
      <div class="usage-strip">
        This Heliox session: {subscriptionStatus.session_usage.input_tokens ?? 0} provider input · {subscriptionStatus
          .session_usage.cached_input_tokens ?? 0} cached · {subscriptionStatus.session_usage.output_tokens ?? 0} output tokens.
      </div>
    {/if}
  </section>

  <section class="settings-group audit-log-section">
    <PermissionAuditLog />
  </section>

  <section class="settings-group audit-log-section">
    <GatewayPolicyEditor />
  </section>

  <section class="settings-group audit-log-section">
    <GatewayAuditLog />
  </section>

  <section class="settings-group feature-panel-section">
    <RiskWorldModelPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <NeuralControlPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <TemporalMemoryPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <OnlineLearningPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <StrategyEvolutionPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <EvolutionHarnessPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <AgentMeshPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <GestureWorkflowBindings />
  </section>

  <section class="settings-group feature-panel-section">
    <VoiceGestureWorkflowStatus />
  </section>

  <section class="settings-group feature-panel-section">
    <RecordedWorkflowsPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <SelfHealingPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <NarrationPanel />
  </section>

  <section class="settings-group feature-panel-section">
    <SupervisionPanel />
  </section>

  <section class="settings-group">
    <h3>Data & Export</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Export Chat History</span>
        <span class="setting-desc">Save the current session history and tool logs to your Downloads folder.</span>
      </div>
      <div class="btn-group">
        <button onclick={() => session.exportChat("markdown")}>Markdown</button>
        <button onclick={() => session.exportChat("json")}>JSON</button>
      </div>
    </div>
  </section>

  <section class="settings-group">
    <h3>Budget &amp; Limits</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Request rate limit</span>
        <span class="setting-desc"
          >Throttle LLM calls across providers to prevent bursts and accidental quota exhaustion.</span
        >
      </div>
      <div class="btn-group">
        <button
          class:active={$settings.model.rate_limit_enabled}
          onclick={() => settings.updateSection("model", { rate_limit_enabled: true })}>On</button
        >
        <button
          class:active={!$settings.model.rate_limit_enabled}
          onclick={() => settings.updateSection("model", { rate_limit_enabled: false })}>Off</button
        >
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Sustained requests per minute</span>
        <span class="setting-desc">Continuous request rate after the initial burst is consumed.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="1"
        max="10000"
        step="1"
        value={$settings.model.rate_limit_rpm}
        onchange={(event) =>
          settings.updateSection("model", {
            rate_limit_rpm: Number.parseInt((event.target as HTMLInputElement).value, 10) || 1,
          })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Request burst capacity</span>
        <span class="setting-desc">Maximum immediate calls before sustained throttling begins.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="1"
        max="1000"
        step="1"
        value={$settings.model.rate_limit_burst}
        onchange={(event) =>
          settings.updateSection("model", {
            rate_limit_burst: Number.parseInt((event.target as HTMLInputElement).value, 10) || 1,
          })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Budget enforcement</span>
        <span class="setting-desc"
          >Master switch for all budget checks. When off, none of the limits below are enforced.</span
        >
      </div>
      <div class="btn-group">
        <button
          class:active={$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: true })}>On</button
        >
        <button
          class:active={!$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: false })}>Off</button
        >
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Monthly USD limit</span>
        <span class="setting-desc">Cumulative cap across all cloud calls for the current calendar month.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="0.5"
        value={$settings.model.budget_monthly_limit_usd}
        onchange={(e) =>
          settings.updateSection("model", {
            budget_monthly_limit_usd: parseFloat((e.target as HTMLInputElement).value) || 0,
          })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-action token cap</span>
        <span class="setting-desc"
          >Maximum estimated input tokens for a single LLM call. Oversized prompts are blocked before being sent.</span
        >
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="500"
        value={$settings.model.max_tokens_per_action}
        onchange={(e) =>
          settings.updateSection("model", {
            max_tokens_per_action: parseInt((e.target as HTMLInputElement).value, 10) || 0,
          })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task token cap</span>
        <span class="setting-desc"
          >Cumulative token budget for a single user task. Halts the task cleanly when exceeded.</span
        >
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="5000"
        value={$settings.model.max_tokens_per_task}
        onchange={(e) =>
          settings.updateSection("model", {
            max_tokens_per_task: parseInt((e.target as HTMLInputElement).value, 10) || 0,
          })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task USD cap</span>
        <span class="setting-desc"
          >Cumulative USD cap per task. Useful for bounding cloud spend on autonomous loops.</span
        >
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="0.01"
        value={$settings.model.max_usd_per_task}
        onchange={(e) =>
          settings.updateSection("model", { max_usd_per_task: parseFloat((e.target as HTMLInputElement).value) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Failure threshold</span>
        <span class="setting-desc">Circuit breaker trips after this many consecutive action failures in a task.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="1"
        max="20"
        step="1"
        value={$settings.model.max_consecutive_failures}
        onchange={(e) =>
          settings.updateSection("model", {
            max_consecutive_failures: parseInt((e.target as HTMLInputElement).value, 10) || 1,
          })}
      />
    </div>
  </section>

  <!--Adding a reset button to clear all settings and return to defaults, with a confirmation prompt to prevent accidental resets -->
  <section class="settings-group">
    <h3>{$_("settings.reset_header")}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.reset_label")}</span>
        <span class="setting-desc">
          {$_("settings.reset_desc")}
        </span>
      </div>

      <button class="btn-save" onclick={handleReset}>
        {$_("settings.reset_button")}
      </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_("settings.debug")}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_("settings.notifications")}</span>
        <span class="setting-desc">{$_("settings.notifications_desc")}</span>
      </div>
      <button class="btn-save" onclick={testNotification}>{$_("settings.test_popup")}</button>
    </div>
  </section>

  <CalendarIntegrationPanel />
  <EmailIntegrationPanel />
  <SshIntegrationPanel />
  <PeerMeshPanel />
  <AirHandoffPanel />
</div>

<style>
  .settings-panel {
    height: 100%;
    overflow-y: auto;
    padding: 16px;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .settings-group {
    margin-bottom: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .audit-log-section {
    height: 420px;
  }

  h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  .settings-subheading {
    margin: 0;
    padding: 10px 14px 4px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    border-top: 1px solid var(--border);
  }

  .setting-section-note {
    margin: 0;
    padding: 0 14px 10px;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-muted);
  }

  .typed-decision-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .typed-decision-field .input-md {
    width: 100%;
  }

  .typed-decision-save {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-tertiary);
    color: var(--text-primary);
    cursor: pointer;
    flex-shrink: 0;
  }

  .typed-decision-save:disabled {
    cursor: wait;
    opacity: 0.6;
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .setting-row:last-child {
    border-bottom: none;
  }

  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 13px;
    font-weight: 500;
  }

  .setting-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .toggle {
    width: 40px;
    height: 22px;
    border-radius: 11px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    position: relative;
    transition: all 0.2s;
    cursor: pointer;
    flex-shrink: 0;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    transition: transform 0.2s;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .btn-group {
    display: flex;
    gap: 2px;
    background: var(--bg-primary);
    border-radius: var(--radius-sm);
    padding: 2px;
  }

  .btn-group button {
    padding: 4px 12px;
    font-size: 11px;
    color: var(--text-secondary);
    background: transparent;
    border-radius: 4px;
    transition: all 0.15s;
  }

  .btn-group button:hover {
    color: var(--text-primary);
  }

  .btn-group button.active {
    background: var(--accent);
    color: white;
  }

  .input-sm {
    width: 80px;
    padding: 5px 8px;
    font-size: 13px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    text-align: right;
  }

  .input-md {
    width: 160px;
    padding: 5px 8px;
    font-size: 13px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .restriction-editor {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .restriction-editor:last-child {
    border-bottom: none;
  }

  .restriction-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }

  .restriction-chip {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 4px 3px 10px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    font-size: 12px;
  }

  .restriction-chip code {
    font-family: var(--font-mono);
    color: var(--text-primary);
  }

  .chip-remove {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    color: var(--text-secondary);
    font-size: 13px;
    line-height: 1;
  }

  .chip-remove:hover {
    background: var(--danger-bg);
    color: var(--danger);
  }

  .restriction-empty {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .restriction-add-row {
    display: flex;
    gap: 6px;
  }

  .restriction-add-row .input-md {
    flex: 1;
    width: auto;
  }

  .api-key-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .btn-save {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
    white-space: nowrap;
  }

  .btn-save:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .btn-save:disabled {
    cursor: not-allowed;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }

  .subscription-notice,
  .usage-strip {
    padding: 10px 14px;
    color: var(--text-secondary);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    line-height: 1.5;
  }

  .subscription-status-copy {
    max-width: min(620px, 70%);
  }

  .status-ok,
  .status-warning {
    font-size: 12px;
    font-weight: 700;
  }

  .status-ok {
    color: var(--success);
  }

  .status-warning {
    color: var(--warning);
  }

  .subscription-actions {
    display: flex;
    gap: 6px;
  }

  .btn-secondary {
    padding: 5px 12px;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
  }

  .btn-secondary:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .subscription-message {
    margin: 0;
    padding: 8px 14px;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
  }

  .subscription-model-input {
    width: 240px;
  }

  .btn-group button:disabled {
    cursor: not-allowed;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
  }

  /* Root Access Toast & Banner */
  .root-toast {
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 0;
    animation: toastSlide 0.3s ease-out;
  }

  .root-toast-warning {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border-bottom: 1px solid rgba(245, 158, 11, 0.3);
  }

  .root-toast-success {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border-bottom: 1px solid rgba(16, 185, 129, 0.3);
  }

  @keyframes toastSlide {
    from {
      opacity: 0;
      transform: translateY(-6px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .gesture-cursor-warning {
    margin: 0;
    padding: 10px 14px;
    font-size: 11px;
    line-height: 1.4;
    color: var(--warning, #f59e0b);
    background: rgba(245, 158, 11, 0.08);
    border-bottom: 1px solid var(--border);
  }

  .hotkey-status {
    margin-top: 3px;
    font-size: 11px;
    color: var(--success, #10b981);
  }

  .hotkey-status.unsupported {
    color: var(--warning, #f59e0b);
  }

  .gaze-status-row {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 14px;
    background: rgba(245, 158, 11, 0.06);
  }

  .gaze-status-row.active {
    background: rgba(16, 185, 129, 0.08);
  }

  .gaze-status-row.error {
    background: rgba(239, 68, 68, 0.08);
  }

  .gaze-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    background: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.55);
  }

  .gaze-status-row.active .gaze-status-dot {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
  }

  .gaze-status-row.error .gaze-status-dot {
    background: #ef4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
  }

  .gaze-status-copy {
    min-width: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .gaze-status-copy strong {
    font-size: 11px;
    text-transform: capitalize;
  }

  .gaze-status-copy span {
    color: var(--text-muted);
    font-size: 10px;
    line-height: 1.35;
  }

  .gaze-open-command {
    flex-shrink: 0;
    padding: 6px 10px;
    border: 1px solid rgba(180, 120, 255, 0.45);
    border-radius: 6px;
    background: rgba(180, 120, 255, 0.1);
    color: var(--text-primary);
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
  }

  .gaze-open-command:hover {
    background: rgba(180, 120, 255, 0.18);
  }

  .root-status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(245, 158, 11, 0.08);
    border-bottom: 1px solid var(--border);
  }

  .root-icon {
    font-size: 18px;
    flex-shrink: 0;
  }

  .root-status-info {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
  }

  .root-status-title {
    font-size: 12px;
    font-weight: 600;
    color: #f59e0b;
  }

  .root-status-desc {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.3;
  }

  .security-setting-status {
    margin-top: 3px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .snapshot-status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(239, 68, 68, 0.08);
    border-bottom: 1px solid rgba(239, 68, 68, 0.22);
  }

  .snapshot-status-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--danger, #ef4444);
  }

  .elevation-button {
    flex-shrink: 0;
    padding: 7px 11px;
    border: 1px solid rgba(245, 158, 11, 0.48);
    border-radius: 6px;
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }

  .elevation-button:hover:not(:disabled) {
    background: rgba(245, 158, 11, 0.2);
  }

  .elevation-button-danger {
    border-color: rgba(239, 68, 68, 0.5);
    background: rgba(239, 68, 68, 0.12);
    color: var(--danger, #ef4444);
  }

  .elevation-button:disabled {
    cursor: wait;
    opacity: 0.65;
  }

  .elevation-message {
    padding: 8px 14px;
    border-bottom: 1px solid rgba(245, 158, 11, 0.24);
    background: rgba(245, 158, 11, 0.07);
    color: var(--text-secondary);
    font-size: 11px;
  }

  .dry-run-status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(16, 185, 129, 0.08);
    border-bottom: 1px solid rgba(16, 185, 129, 0.22);
  }

  .dry-run-status-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--success, #10b981);
  }

  .api-key-error {
    margin: 6px 0 0;
    max-width: 520px;
    color: var(--danger, #ef4444);
    font-size: 11px;
    line-height: 1.4;
  }
</style>

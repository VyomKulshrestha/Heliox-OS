<script lang="ts">
  import { settings } from "../stores/settings";
  import { _, locale } from 'svelte-i18n';
  import { session } from "../stores/session";
  import { call } from "../api/daemon";
  import { invoke } from "../api/invoke";
  import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
  import ConfirmPrompt from "./ConfirmPrompt.svelte";
  import PermissionAuditLog from "./PermissionAuditLog.svelte";
  import GatewayPolicyEditor from "./GatewayPolicyEditor.svelte";
  import GatewayAuditLog from "./GatewayAuditLog.svelte";
  import GestureWorkflowBindings from "./GestureWorkflowBindings.svelte";
  import VoiceGestureWorkflowStatus from "./VoiceGestureWorkflowStatus.svelte";
  import SelfHealingPanel from "./SelfHealingPanel.svelte";
  import NarrationPanel from "./NarrationPanel.svelte";
  import SupervisionPanel from "./SupervisionPanel.svelte";
  import { getSharedGestureCalibrationStore } from "../gesture/calibration";
  import {
    gazeRuntime,
    resetGazeRuntime,
  } from "../stores/gazeRuntime";
  import {
    defaultHotkey,
    isNativeTauriRuntime,
    normalizeHotkeyValue,
  } from "../hotkey";
  import { speakText, stopSpeech } from "../utils/tts";

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
  let rootToast = $state("");
  let rootToastType = $state<"success" | "warning">("success");
  let rootSaving = $state(false);
  let elevationRequesting = $state(false);
  let elevationMessage = $state("");
  let rootRuntime = $state<{
    root_policy_enabled: boolean;
    process_elevated: boolean;
    platform: string;
    detail: string;
  } | null>(null);
  let snapshotSaving = $state(false);
  let snapshotToast = $state("");
  let snapshotRuntime = $state<{
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
  let gestureCursorToast = $state("");
  let gestureCalibrationToast = $state("");
  let voiceCalibrationToast = $state("");
  let speechToast = $state("");
  let speechSaving = $state(false);
  let speechTesting = $state(false);
  let audioInputDevices = $state<Array<{
    id: string;
    name: string;
    hostapi: string;
    is_default: boolean;
  }>>([]);
  let audioInputMessage = $state("");

  $effect(() => {
    if ($locale) {
      localStorage.setItem("locale", $locale);
    }
  });

  let hotkeyInput = $state("");
  let hotkeySaved = $state(false);
  let hotkeyError = $state("");
  const hotkeySupported = isNativeTauriRuntime();

  const cloudModels: Record<string, string[]> = {
    gemini: ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    openai: ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
    claude: ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
    meta: ["muse-spark-1.1"]
  };

  let availableOllamaModels = $state<string[]>([]);

  $effect(() => {
    call("list_ollama_models").then((res: any) => {
      if (res && res.models) {
        availableOllamaModels = res.models;
      }
    }).catch(console.error);
  });

  // Load the current hotkey when component mounts
  invoke("get_hotkey").then((val) => {
    hotkeyInput = normalizeHotkeyValue(val);
  }).catch(() => {
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
      hotkeyError = detail
        ? `Shortcut was not changed: ${detail}`
        : "Invalid shortcut. Try Ctrl+Space or Alt+H.";
    }
  }

  async function refreshRootRuntime() {
    try {
      rootRuntime = await call("get_security_status");
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
      snapshotRuntime = await call("get_snapshot_status");
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
      elevationMessage = err instanceof Error
        ? `Administrator restart failed: ${err.message}`
        : "Administrator restart failed. The existing daemon is still running.";
    }
  }

  async function applyRootToggle(turningOn: boolean) {
    rootSaving = true;
    const synced = await settings.updateSection(
      "security",
      { root_enabled: turningOn },
      { requireDaemon: true },
    );
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
        true
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
      ? (snapshotRuntime?.ready
        ? "Auto-Snapshot enabled. The snapshot backend is ready."
        : "Auto-Snapshot enabled, but its backend is not ready. Destructive actions will be blocked.")
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
    const synced = await settings.updateSection(
      "security",
      { dry_run: turningOn },
      { requireDaemon: true },
    );
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
    settings.updateSection("model", { cloud_provider, provider: "cloud" });
  }

  function updateCloudModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { cloud_model: val });
  }

  const pocketTtsVoices = ["alba", "giovanni", "lola"];

  async function refreshAudioInputDevices() {
    try {
      const result = await call<{
        devices: Array<{
          id: string;
          name: string;
          hostapi: string;
          is_default: boolean;
        }>;
        message?: string;
      }>("list_audio_input_devices");
      audioInputDevices = result.devices ?? [];
      audioInputMessage = result.message ?? "";
    } catch (err) {
      audioInputDevices = [];
      audioInputMessage = err instanceof Error
        ? err.message
        : "Could not enumerate microphone inputs.";
    }
  }

  $effect(() => {
    refreshAudioInputDevices();
  });

  async function updateAudioInput(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    speechSaving = true;
    const synced = await settings.updateSection(
      "voice",
      { input_device: val },
      { requireDaemon: true },
    );
    speechSaving = false;
    speechToast = synced
      ? "Microphone input saved. Restart Heliox Active to use it."
      : "Microphone input was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function updateTtsEngine(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    speechSaving = true;
    const synced = await settings.updateSection(
      "voice",
      { tts_engine: val },
      { requireDaemon: true },
    );
    speechSaving = false;
    speechToast = synced
      ? `Speech engine changed to ${val === "pocket_tts" ? "Pocket TTS" : "OS Voice"}.`
      : "Speech engine was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  async function updateTtsVoice(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    speechSaving = true;
    const synced = await settings.updateSection(
      "voice",
      { tts_voice: val },
      { requireDaemon: true },
    );
    speechSaving = false;
    speechToast = synced
      ? `Pocket TTS voice changed to ${val}.`
      : "Speech voice was not changed because the daemon could not confirm it.";
    setTimeout(() => (speechToast = ""), 5000);
  }

  function testConfiguredVoice() {
    speechTesting = true;
    speechToast = "Testing the configured daemon voice…";
    speakText("Heliox voice is ready.", {
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
    stopSpeech();
    speechTesting = false;
    speechToast = "Voice test stopped.";
    setTimeout(() => (speechToast = ""), 5000);
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
    const synced = await settings.updateSection(
      "security",
      { snapshot_retention_count: val },
      { requireDaemon: true },
    );
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
      ? (turningOn
        ? "Gesture Cursor enabled. Start the camera and explicitly enter Cursor Mode to control the pointer."
        : "Gesture Cursor disabled. Any active Cursor Mode was stopped.")
      : (turningOn
        ? "Enabled for this UI session, but daemon persistence could not be confirmed."
        : "Disabled locally; daemon persistence could not be confirmed.");
    setTimeout(() => (gestureCursorToast = ""), 5000);
  }

  async function toggleGazeTracking() {
    const turningOn = !$settings.vision?.gaze_tracking_enabled;
    await settings.updateSection("vision", {
      gaze_tracking_enabled: turningOn,
    });
    if (!turningOn) resetGazeRuntime();
  }

  // "Simulate before executing" preview adds real latency (a screenshot +
  // VLM call, plus a real dry-run browser tab for browser actions) before
  // every autonomous action - default off, explicit opt-in like gesture_cursor.
  async function togglePreview() {
    if (previewSaving) return;
    const turningOn = !$settings.preview?.enabled;
    previewSaving = true;
    const synced = await settings.updateSection(
      "preview",
      { enabled: turningOn },
      { requireDaemon: true },
    );
    previewSaving = false;
    previewToast = synced
      ? (turningOn
        ? "Preview gate enabled. Autonomous actions now require a real preview and approval."
        : "Preview gate disabled.")
      : "Preview was not changed because the daemon could not confirm it.";
    setTimeout(() => (previewToast = ""), 5000);
  }

  async function updateGestureCursorSensitivity(e: Event) {
    const rawValue = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(rawValue)) return;
    const sensitivity = Math.min(3, Math.max(0.1, rawValue));
    (e.target as HTMLInputElement).value = String(sensitivity);
    const synced = await settings.updateSection("gesture_cursor", { sensitivity });
    gestureCursorToast = synced
      ? `Cursor sensitivity saved at ${sensitivity}.`
      : "Sensitivity changed locally, but daemon persistence could not be confirmed.";
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
      : "Prediction blend changed locally, but daemon persistence could not be confirmed.";
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
      ? (turningOn ? "Adaptive gesture calibration enabled." : "Adaptive gesture calibration paused.")
      : "Calibration changed locally, but daemon persistence could not be confirmed.";
    setTimeout(() => (gestureCalibrationToast = ""), 5000);
  }

  function resetGestureCalibration() {
    gestureCalibration.reset();
    gestureCalibrationToast = "Learned gesture calibration cleared. Shipped thresholds are active.";
    setTimeout(() => (gestureCalibrationToast = ""), 5000);
  }

  let voiceVariants = $state<{ text: string; confirmed_count: number }[]>([]);
  let voicePromotionThreshold = $state(5);
  let voiceVariantsAvailable = $state(true);

  async function loadVoiceVariants() {
    try {
      const res: any = await call("list_wake_variants");
      if (res && res.variants) {
        voiceVariants = res.variants;
        voicePromotionThreshold = res.promotion_threshold ?? 5;
        voiceVariantsAvailable = true;
      }
    } catch {
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
      ? (turningOn ? "Adaptive wake-word matching enabled." : "Adaptive wake-word matching disabled.")
      : "Wake-word calibration was not changed because the daemon could not confirm it.";
    setTimeout(() => (voiceCalibrationToast = ""), 5000);
  }

  async function resetVoiceCalibration() {
    try {
      const result = await call<{ status: string }>("reset_wake_calibration");
      if (result.status !== "ok") throw new Error("Daemon rejected reset");
      voiceVariants = [];
      voiceCalibrationToast = "Learned wake-word variants cleared from the live listener and disk.";
    } catch {
      voiceCalibrationToast = "Wake-word variants were not cleared because the daemon could not confirm it.";
    }
    setTimeout(() => (voiceCalibrationToast = ""), 5000);
  }

  function updateOllamaModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { ollama_model: val });
  }

  async function saveApiKey() {
    if (!apiKeyInput.trim()) return;
    apiKeySaving = true;
    try {
      const provider = $settings.model.cloud_provider || "gemini";
      await call("store_api_key", { provider, api_key: apiKeyInput.trim() });
      apiKeySaved = true;
      apiKeyInput = "";
      setTimeout(() => {
        apiKeySaved = false;
      }, 3000);
    } catch (err) {
      console.error("Failed to save API key:", err);
    } finally {
      apiKeySaving = false;
    }
  }
  
  // Function to reset all settings to defaults
  function handleReset() {
    askConfirm($_('settings.reset_confirm'), () => { void settings.reset(); }, true);
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

  <h2>{$_('settings.title')}</h2>

  <section class="settings-group">
    <h3>{$_('settings.appearance')}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.light_mode')}</span>
        <span class="setting-desc">{$_('settings.light_mode_desc')}</span>
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
        <span class="setting-label">{$_('settings.language')}</span>
        <span class="setting-desc">{$_('settings.language_desc')}</span>
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
    <h3>{$_('settings.security')}</h3>

    {#if rootToast}
      <div class="root-toast" class:root-toast-warning={rootToastType === 'warning'} class:root-toast-success={rootToastType === 'success'}>
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
        <span class="setting-label">{$_('settings.root_access')}</span>
        <span class="setting-desc">{$_('settings.root_access_desc')}</span>
        <span class="security-setting-status">
          {$settings.security.root_enabled
            ? (rootRuntime
              ? (rootRuntime.process_elevated
                ? "Policy enabled · Administrator/root detected"
                : "Policy enabled · OS elevation not detected")
              : "Policy enabled · privilege status unavailable")
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
        disabled={rootSaving}
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
              ? (rootRuntime.process_elevated
                ? "Root policy active · OS elevated"
                : "Root policy active · OS elevation not detected")
              : "Root policy active · privilege status unavailable"}
          </span>
          <span class="root-status-desc">
            {rootRuntime?.detail ||
              "Root-tier actions are allowed by Heliox policy; checking the daemon's OS privileges."}
          </span>
          {#if rootRuntime?.platform === "win32" && !rootRuntime.process_elevated}
            <span class="root-status-desc">
              Windows will show a UAC prompt. Heliox stays running if you cancel.
            </span>
          {/if}
        </div>
        {#if rootRuntime?.platform === "win32" && !rootRuntime.process_elevated}
          <button
            class="elevation-button"
            onclick={requestAdministratorRestart}
            disabled={elevationRequesting}
          >
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
        <span class="setting-label">{$_('settings.auto_snapshot')}</span>
        <span class="setting-desc">{$_('settings.auto_snapshot_desc')}</span>
        <span class="security-setting-status">
          {$settings.security.snapshot_on_destructive
            ? (snapshotRuntime
              ? (snapshotRuntime.ready
                ? `Protection ready · ${snapshotBackendLabel(snapshotRuntime.backend)}`
                : `Enabled but unavailable · ${snapshotBackendLabel(snapshotRuntime.backend)}`)
              : "Enabled · backend status unavailable")
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
        disabled={snapshotSaving}
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
        <span class="setting-label">{$_('settings.dry_run')}</span>
        <span class="setting-desc">{$_('settings.dry_run_desc')}</span>
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
        <span class="setting-label">{$_('settings.snapshot_retention')}</span>
        <span class="setting-desc">{$_('settings.snapshot_retention_desc')}</span>
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
    <h3>{$_('settings.usage')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.total_tokens')}</span>
        <span class="setting-desc">{$_('settings.total_tokens_desc')}</span>
      </div>
      <span>{$session.totalTokens}</span>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.estimated_cost')}</span>
        <span class="setting-desc">{$_('settings.estimated_cost_desc')}</span>
      </div>
      <span>
        {$settings.model.provider === "ollama" ? $_('settings.free_local') : `$${$session.estimatedCost.toFixed(4)}`}
      </span>
    </div>

    <div class="setting-row">
      <button class="btn-save" onclick={() => session.resetUsage()}>{$_('settings.reset_usage')}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.screen_vision')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.capture_interval')}</span>
        <span class="setting-desc">{$_('settings.capture_interval_desc')}</span>
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
    <h3>{$_('settings.gaze_tracking')}</h3>
    <p class="gesture-cursor-warning">{$_('settings.gaze_tracking_desc')}</p>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.gaze_tracking_enabled')}</span>
        <span class="setting-desc">{$_('settings.gaze_tracking_enabled_desc')}</span>
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
      class:error={$settings.vision?.gaze_tracking_enabled && ($gazeRuntime.phase === "error" || $gazeRuntime.daemonStatus === "error")}
    >
      <span class="gaze-status-dot"></span>
      <div class="gaze-status-copy">
        <strong>
          {#if !$settings.vision?.gaze_tracking_enabled}
            {$_('settings.gaze_status_off')}
          {:else if $gazeRuntime.phase === "loading"}
            {$_('settings.gaze_status_loading')}
          {:else if $gazeRuntime.phase === "scanning"}
            {$_('settings.gaze_status_scanning')}
          {:else if $gazeRuntime.phase === "active" && $gazeRuntime.region}
            {$_('settings.gaze_status_active')}: {$gazeRuntime.region}
            ({Math.round(($gazeRuntime.confidence ?? 0) * 100)}%)
          {:else if $gazeRuntime.phase === "error"}
            {$_('settings.gaze_status_error')}
          {:else}
            {$_('settings.gaze_status_ready')}
          {/if}
        </strong>
        <span>
          {#if $settings.vision?.gaze_tracking_enabled && $gazeRuntime.message}
            {$gazeRuntime.message}
          {:else if $settings.vision?.gaze_tracking_enabled}
            {$_('settings.gaze_status_ready_desc')}
          {:else}
            {$_('settings.gaze_status_off_desc')}
          {/if}
        </span>
      </div>
      {#if $settings.vision?.gaze_tracking_enabled && !$gazeRuntime.cameraActive}
        <button class="gaze-open-command" onclick={onOpenCommand}>
          {$_('settings.gaze_open_command')}
        </button>
      {/if}
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.preview')}</h3>
    <p class="gesture-cursor-warning">{$_('settings.preview_desc')}</p>

    {#if previewToast}
      <div class="root-toast" class:root-toast-success={$settings.preview?.enabled} class:root-toast-warning={!$settings.preview?.enabled}>
        {previewToast}
      </div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.preview_enabled')}</span>
        <span class="setting-desc">{$_('settings.preview_enabled_desc')}</span>
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
    <h3>{$_('settings.gesture_cursor')}</h3>

    <p class="gesture-cursor-warning">{$_('settings.gesture_cursor_warning')}</p>

    {#if gestureCursorToast}
      <div class="root-toast root-toast-warning">{gestureCursorToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.gesture_cursor_enabled')}</span>
        <span class="setting-desc">{$_('settings.gesture_cursor_enabled_desc')}</span>
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
          <span class="setting-label">{$_('settings.gesture_cursor_sensitivity')}</span>
          <span class="setting-desc">{$_('settings.gesture_cursor_sensitivity_desc')}</span>
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
          <span class="setting-label">{$_('settings.gesture_cursor_blend')}</span>
          <span class="setting-desc">{$_('settings.gesture_cursor_blend_desc')}</span>
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
    <h3>{$_('settings.gesture_calibration')}</h3>
    <p class="gesture-cursor-warning">{$_('settings.gesture_calibration_desc')}</p>

    {#if gestureCalibrationToast}
      <div class="root-toast root-toast-success">{gestureCalibrationToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.gesture_calibration_enabled')}</span>
        <span class="setting-desc">{$_('settings.gesture_calibration_enabled_desc')}</span>
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
        <span class="setting-label">{$_('settings.gesture_calibration_learned')}</span>
        <span class="setting-desc">
          {#if gestureCalibrationSnapshot.pinchSampleCount === 0 && gestureCalibrationSnapshot.thumbSampleCount === 0}
            {$_('settings.gesture_calibration_no_data')}
          {:else}
            {$_('settings.gesture_calibration_pinch_samples')}: {gestureCalibrationSnapshot.pinchSampleCount} · {$_('settings.gesture_calibration_thumb_samples')}: {gestureCalibrationSnapshot.thumbSampleCount}
          {/if}
        </span>
      </div>
      <button class="btn-save" onclick={resetGestureCalibration}>{$_('settings.gesture_calibration_reset')}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.voice_calibration')}</h3>
    <p class="gesture-cursor-warning">{$_('settings.voice_calibration_desc')}</p>

    {#if voiceCalibrationToast}
      <div class="root-toast root-toast-success">{voiceCalibrationToast}</div>
    {/if}

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.voice_calibration_enabled')}</span>
        <span class="setting-desc">{$_('settings.voice_calibration_enabled_desc')}</span>
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
        <span class="setting-label">{$_('settings.voice_calibration_learned')}</span>
        <span class="setting-desc">
          {#if !voiceVariantsAvailable}
            Calibration status unavailable
          {:else if voiceVariants.length === 0}
            {$_('settings.voice_calibration_no_data')}
          {:else}
            {#each voiceVariants as variant}
              "{variant.text}" ({variant.confirmed_count}/{voicePromotionThreshold}){#if variant !== voiceVariants[voiceVariants.length - 1]}, {/if}
            {/each}
          {/if}
        </span>
      </div>
      <button class="btn-save" onclick={resetVoiceCalibration}>{$_('settings.voice_calibration_reset')}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.voice_speech')}</h3>
    <p class="gesture-cursor-warning">{$_('settings.voice_speech_desc')}</p>

    {#if speechToast}
      <div class="root-toast" class:root-toast-success={!speechToast.includes("not changed") && !speechToast.includes("failed")} class:root-toast-warning={speechToast.includes("not changed") || speechToast.includes("failed")}>
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
        disabled={speechSaving}
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
        <span class="setting-label">{$_('settings.tts_engine')}</span>
        <span class="setting-desc">{$_('settings.tts_engine_desc')}</span>
        <span class="security-setting-status">Used by voice replies, live narration, supervision, and this test.</span>
      </div>
      <select class="input-md" value={$settings.voice?.tts_engine ?? "pocket_tts"} onchange={updateTtsEngine} disabled={speechSaving || speechTesting}>
        <option value="pocket_tts">Pocket TTS</option>
        <option value="os_native">OS Voice</option>
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.tts_voice')}</span>
        <span class="setting-desc">{$_('settings.tts_voice_desc')}</span>
      </div>
      <select
        class="input-md"
        value={$settings.voice?.tts_voice ?? "alba"}
        onchange={updateTtsVoice}
        disabled={$settings.voice?.tts_engine === "os_native" || speechSaving || speechTesting}
      >
        {#each pocketTtsVoices as voiceOption}
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
    <h3>{$_('settings.model')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.provider')}</span>
        <span class="setting-desc">{$_('settings.provider_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.provider === "ollama"} onclick={() => setProvider("ollama")}>Ollama</button>
        <button class:active={$settings.model.provider === "cloud"} onclick={() => setProvider("cloud")}>Cloud</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.reasoning_mode')}</span>
        <span class="setting-desc">{$_('settings.reasoning_mode_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.mode === "lightweight"} onclick={() => setMode("lightweight")}>{$_('settings.light')}</button>
        <button class:active={$settings.model.mode === "full"} onclick={() => setMode("full")}>{$_('settings.full')}</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.ollama_model')}</span>
        <span class="setting-desc">{$_('settings.ollama_model_desc')}</span>
      </div>
      {#if availableOllamaModels.length > 0}
        <select
          class="input-md"
          value={$settings.model.ollama_model}
          onchange={updateOllamaModel}
        >
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
        <span class="setting-label">{$_('settings.gpu_memory')}</span>
        <span class="setting-desc">{$_('settings.gpu_memory_desc')}</span>
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
  </section>

  <section class="settings-group">
    <h3>{$_('settings.cloud_api')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.cloud_provider')}</span>
        <span class="setting-desc">{$_('settings.cloud_provider_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.cloud_provider === "gemini"} onclick={() => setCloudProvider("gemini")}>Gemini</button>
        <button class:active={$settings.model.cloud_provider === "openai"} onclick={() => setCloudProvider("openai")}>OpenAI</button>
        <button class:active={$settings.model.cloud_provider === "claude"} onclick={() => setCloudProvider("claude")}>Claude</button>
        <button class:active={$settings.model.cloud_provider === "meta"} onclick={() => setCloudProvider("meta")}>Meta</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.cloud_model')}</span>
        <span class="setting-desc">{$_('settings.cloud_model_desc')}</span>
      </div>
      <select
        class="input-md"
        value={$settings.model.cloud_model}
        onchange={updateCloudModel}
      >
        <option value="">Default for provider</option>
        {#each (cloudModels[$settings.model.cloud_provider || "gemini"] || []) as modelOption}
          <option value={modelOption}>{modelOption}</option>
        {/each}
      </select>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.api_key')}</span>
        <span class="setting-desc">{$_('settings.api_key_desc')}</span>
      </div>
      <div class="api-key-row">
        <input type="password" class="input-md" bind:value={apiKeyInput} placeholder={$_('settings.api_key_placeholder')} />
        <button class="btn-save" onclick={saveApiKey} disabled={apiKeySaving}>
          {apiKeySaved ? $_('settings.saved') : apiKeySaving ? $_('settings.saving') : $_('settings.save')}
        </button>
      </div>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.restrictions')}</h3>

    {#each restrictionFields as field}
      <div class="restriction-editor">
        <span class="setting-label">{$_(field.labelKey)}</span>
        <div class="restriction-chips">
          {#each ($settings.restrictions?.[field.key] ?? []) as entry}
            <span class="restriction-chip">
              <code>{entry}</code>
              <button
                class="chip-remove"
                title="Remove"
                onclick={() => removeRestrictionEntry(field.key, entry)}
              >&times;</button>
            </span>
          {:else}
            <span class="restriction-empty">{$_('settings.configured')}: 0</span>
          {/each}
        </div>
        <div class="restriction-add-row">
          <input
            type="text"
            class="input-md"
            placeholder={field.placeholder}
            bind:value={newRestrictionEntry[field.key]}
            onkeydown={(e) => { if (e.key === "Enter") addRestrictionEntry(field.key); }}
          />
          <button class="btn-save" onclick={() => addRestrictionEntry(field.key)}>Add</button>
        </div>
      </div>
    {/each}
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

  <section class="settings-group audit-log-section">
    <GestureWorkflowBindings />
  </section>

  <section class="settings-group audit-log-section">
    <VoiceGestureWorkflowStatus />
  </section>

  <section class="settings-group audit-log-section">
    <SelfHealingPanel />
  </section>

  <section class="settings-group audit-log-section">
    <NarrationPanel />
  </section>

  <section class="settings-group audit-log-section">
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
        <span class="setting-label">Budget enforcement</span>
        <span class="setting-desc">Master switch for all budget checks. When off, none of the limits below are enforced.</span>
      </div>
      <div class="btn-group">
        <button
          class:active={$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: true })}
        >On</button>
        <button
          class:active={!$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: false })}
        >Off</button>
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
        onchange={(e) => settings.updateSection("model", { budget_monthly_limit_usd: parseFloat((e.target as HTMLInputElement).value) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-action token cap</span>
        <span class="setting-desc">Maximum estimated input tokens for a single LLM call. Oversized prompts are blocked before being sent.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="500"
        value={$settings.model.max_tokens_per_action}
        onchange={(e) => settings.updateSection("model", { max_tokens_per_action: parseInt((e.target as HTMLInputElement).value, 10) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task token cap</span>
        <span class="setting-desc">Cumulative token budget for a single user task. Halts the task cleanly when exceeded.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="5000"
        value={$settings.model.max_tokens_per_task}
        onchange={(e) => settings.updateSection("model", { max_tokens_per_task: parseInt((e.target as HTMLInputElement).value, 10) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task USD cap</span>
        <span class="setting-desc">Cumulative USD cap per task. Useful for bounding cloud spend on autonomous loops.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="0.01"
        value={$settings.model.max_usd_per_task}
        onchange={(e) => settings.updateSection("model", { max_usd_per_task: parseFloat((e.target as HTMLInputElement).value) || 0 })}
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
        onchange={(e) => settings.updateSection("model", { max_consecutive_failures: parseInt((e.target as HTMLInputElement).value, 10) || 1 })}
      />
    </div>
  </section>

  <!--Adding a reset button to clear all settings and return to defaults, with a confirmation prompt to prevent accidental resets -->
  <section class="settings-group">
    <h3>{$_('settings.reset_header')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.reset_label')}</span>
        <span class="setting-desc">
          {$_('settings.reset_desc')}
        </span>
      </div>

      <button class="btn-save" onclick={handleReset}>
        {$_('settings.reset_button')}
      </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.debug')}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.notifications')}</span>
        <span class="setting-desc">{$_('settings.notifications_desc')}</span>
      </div>
      <button class="btn-save" onclick={testNotification}>{$_('settings.test_popup')}</button>
    </div>
  </section>
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
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
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
</style>

<script lang="ts">
  /**
   * GestureControl v3 — 30+ hand gesture recognition engine.
   *
   * STATIC POSE GESTURES:
   *  ✋ Open Palm       → Cancel / Stop
   *  👍 Thumbs Up      → Confirm plan
   *  👎 Thumbs Down    → Deny / Reject
   *  ✌️ Peace Sign      → Toggle voice mode
   *  👊 Fist           → Execute last command
   *  👆 Point Up       → Scroll up
   *  🤟 Rock           → System info
   *  👌 OK Sign        → Accept / Acknowledge
   *  🤙 Call Me        → Open settings
   *  🔫 Finger Gun     → Screenshot
   *  🤏 Pinch          → Grab / Select
   *  🖕 Middle Finger  → Emergency stop
   *  🌸 Pinky Up       → Fancy mode
   *  🖖 Vulcan         → Diagnostics
   *  🤞 Crossed Fingers → Luck / Random action
   *  ☝️ Index Only      → Focus mode
   *  🫰 Snap Ready     → Quick launch
   *  🤘 Devil Horns    → Play music
   *  🫳 Palm Down      → Mute / Silence
   *  🫴 Palm Up        → Unmute / Restore
   *  ✌️+👆 Three Up     → Brightness up
   *  🖖+✋ Four Up      → Brightness down
   *
   * MOTION-BASED GESTURES:
   *  👈 Swipe Left     → Previous tab
   *  👉 Swipe Right    → Next tab
   *  ↕️ Swipe Up        → Scroll up fast
   *  ↕️ Swipe Down      → Scroll down fast
   *  🔄 Circular CW    → Volume up
   *  🔄 Circular CCW   → Volume down
   *  🫸 Palm Push      → Confirm AI action
   *  🫷 Palm Pull      → Cancel AI action
   *  ✌️ Two-Finger Swipe Left → Switch workspace left
   *  ✌️ Two-Finger Swipe Right → Switch workspace right
   */

  import { session } from "../stores/session";
  import { settings } from "../stores/settings";
  import { tick } from "svelte";
  import { Hands, type Results } from "@mediapipe/hands";
  import { FilesetResolver, HandLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision";
  import {
    LandmarkFilterBank,
    computeHandQuality,
    isThumbExtended,
    thumbExtensionRatio,
    handSize,
    predictCursorTarget,
    trajectoryAgreement,
    THUMB_EXTENDED_RATIO,
    type Landmark,
  } from "../gesture/spatialModel";
  import {
    toWristRelative3D,
    handSize3D,
    pinchDistance3D,
    detectPushPull3D,
    WorldModelFilterBank,
  } from "../gesture/worldModel";
  import {
    estimateGazeRegion,
    shouldSendGazeUpdate,
    type GazeRegion,
  } from "../gesture/gazeTracking";
  import {
    gazeRuntime,
    resetGazeRuntime,
    updateGazeRuntime,
  } from "../stores/gazeRuntime";
  import { isTauriRuntime } from "../utils/runtime";
  import { GestureCalibrationStore, classifyOutcome, REVERSAL_WINDOW_MS, type GestureEvent } from "../gesture/calibration";
  import { classifyControlGesture } from "../gesture/workflowControl";

  // ── Props ──
  let { onGesture = (name: string) => {} }: { onGesture?: (name: string) => void } = $props();

  // ── State ──
  let isActive = $state(false);
  let currentGesture = $state("");
  let confidence = $state(0);
  let cameraError = $state("");
  let showCamera = $state(false);
  let gestureHistory: string[] = $state([]);
  
  let videoEl: HTMLVideoElement | undefined = $state();
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let trailCanvas: HTMLCanvasElement | undefined = $state();
  let stream: MediaStream | null = null;
  let hands: Hands | null = null;
  let handLandmarker: HandLandmarker | null = null;
  // Frozen at startGestures() time from $settings.vision.mediapipe_backend —
  // changing the setting mid-session requires stopping/restarting the
  // engine, not a hot-swap (see VisionConfig.mediapipe_backend in config.py).
  let activeBackend: "legacy" | "tasks" = "legacy";
  // Most recent worldLandmarks from the "tasks" backend — the "legacy"
  // backend never populates this (no metric-scale 3D available). Consumed
  // by handleFrameResult()'s 3D push/pull + metric pinch confirmation
  // checks below (see worldModel.ts).
  let lastWorldLandmarks: Landmark[] | null = null;

  // ── Gaze tracking (third input modality, see gazeTracking.ts) ──
  //
  // A SEPARATE model from the hand backend — runs (or doesn't) independent
  // of activeBackend's legacy/tasks choice, gated only on
  // $settings.vision.gaze_tracking_enabled. Only ever sends a coarse
  // region label + confidence to the backend, never raw face landmarks or
  // video — see gazeTracking.ts's module docstring.
  let faceLandmarker: FaceLandmarker | null = null;
  let gazeTrackingActive = false;
  let lastGazeRegion: GazeRegion | null = null;
  let lastGazeSentAt = 0;
  let gazeFrameCounter = 0;
  // Face detection runs far less often than hand detection -- a coarse
  // "which rough direction" signal doesn't need 30fps, and running two
  // ML inference passes every single frame on a CPU delegate is a real
  // cost this keeps in check.
  const GAZE_FRAME_INTERVAL = 6;

  let animFrameId: number = 0;
  let lastGestureTime = 0;
  let candidateGesture = "";
  let candidateCount = 0;
  const REQUIRED_FRAMES = 5;

  // ── Spatial/world-model layer: temporal landmark filtering + quality gating ──
  const landmarkFilter = new LandmarkFilterBank();
  const QUALITY_CONFIDENCE_FLOOR = 0.35;

  // ── 3D world-model layer ("tasks" backend only — see worldModel.ts) ──
  //
  // worldModelFilter smooths worldLandmarks the same way landmarkFilter
  // smooths the 2D landmarks; feeds the metric pinch-distance confirmation
  // check below. worldWristHistory stays RAW (unfiltered), same rationale
  // as the existing 2D wristHistory: detectPushPull3D's threshold is tuned
  // against real motion, which filtering would damp.
  const worldModelFilter = new WorldModelFilterBank();
  let worldWristHistory: Landmark[] = [];
  const WORLD_MOTION_BUFFER_SIZE = 8; // mirrors detectPushPull()'s 8-frame gate

  // Shared with classifyGesture()'s OK/pinch checks — single source of truth
  // so the cursor-mode click logic can't silently drift from the discrete
  // pinch gesture's own threshold.
  const PINCH_DISTANCE_THRESHOLD = 0.05;

  // How far ahead classifyGesture()'s swipe-direction agreement check looks —
  // deliberately short and not user-configurable (unlike the cursor bridge's
  // prediction_ms setting): this only nudges confidence for an
  // already-classified swipe, it doesn't drive anything continuous.
  const MOTION_PREDICTION_MS = 50;

  // ── On-device gesture calibration (continual-learning loop) ──
  //
  // Personalizes PINCH_DISTANCE_THRESHOLD/THUMB_EXTENDED_RATIO from implicit
  // confirm/reversal signals — see calibration.ts. Gated on
  // $settings.adaptive_calibration.gesture_enabled (default on); the store
  // itself is always constructed (cheap, a no-op localStorage read) so
  // toggling the setting mid-session doesn't require re-mounting anything.
  const gestureCalibration = new GestureCalibrationStore();
  let pendingCalibrationEvent: GestureEvent | null = null;
  let pendingCalibrationTimer: ReturnType<typeof setTimeout> | null = null;

  function resolvePendingCalibration(next: GestureEvent | null) {
    if (pendingCalibrationTimer) {
      clearTimeout(pendingCalibrationTimer);
      pendingCalibrationTimer = null;
    }
    if (!pendingCalibrationEvent) return;
    const outcome = classifyOutcome(pendingCalibrationEvent, next);
    gestureCalibration.recordOutcome(pendingCalibrationEvent, outcome);
    pendingCalibrationEvent = null;
  }

  /** Drops any pending calibration window WITHOUT recording an outcome —
   * used when the engine itself stops mid-window, since we genuinely don't
   * know whether that gesture would have been confirmed or reversed. */
  function cancelPendingCalibration() {
    if (pendingCalibrationTimer) {
      clearTimeout(pendingCalibrationTimer);
      pendingCalibrationTimer = null;
    }
    pendingCalibrationEvent = null;
  }

  /** Called right after a gesture fires (executeGestureAction/onGesture) —
   * resolves whatever calibration-relevant gesture was pending (this new
   * fire is its "next"), then starts a new pending window if the gesture
   * that just fired is itself calibration-relevant. */
  function trackGestureForCalibration(name: string, metricValue: number) {
    if (!$settings.adaptive_calibration?.gesture_enabled) return;
    const now = Date.now();
    resolvePendingCalibration({ name, timestamp: now, metricValue });

    if (name === "pinch" || name === "ok" || name === "thumbs_up" || name === "thumbs_down") {
      pendingCalibrationEvent = { name, timestamp: now, metricValue };
      pendingCalibrationTimer = setTimeout(() => resolvePendingCalibration(null), REVERSAL_WINDOW_MS);
    }
  }

  // ── Gesture Cursor Control (continuous gesture-to-cursor bridge) ──
  //
  // Off by default (gated on $settings.gesture_cursor.enabled) and only
  // toggled by an explicit UI button, never by a gesture — this drives the
  // real OS mouse cursor. While active: index-fingertip position (blended
  // with its predicted near-future position from spatialModel.ts) drives the
  // cursor, pinch fires a click, and every other discrete gesture is
  // suppressed so reaching for e.g. a swipe doesn't misfire while pointing.
  // Open palm and stopGestures() both force-exit cursor mode immediately as
  // safety escape hatches.
  let cursorModeActive = $state(false);
  let lastCursorX = 0;
  let lastCursorY = 0;
  let pinchClickFired = false; // debounce: one click per pinch-close, not per frame

  async function moveGestureCursor(x: number, y: number): Promise<void> {
    try {
      if (isTauriRuntime()) {
        const { invoke } = await import("../api/invoke");
        await invoke("move_gesture_cursor", { x, y });
      } else {
        const { call } = await import("../api/daemon");
        await call("cursor_move", { x, y });
      }
    } catch {
      // Best-effort — a single dropped cursor-move frame isn't worth surfacing.
    }
  }

  async function clickGestureCursor(x: number, y: number): Promise<void> {
    try {
      if (isTauriRuntime()) {
        const { invoke } = await import("../api/invoke");
        await invoke("click_gesture_cursor");
      } else {
        // The daemon fallback has no "click at current position" concept —
        // reuse the same coordinates the last cursor_move call already sent.
        const { call } = await import("../api/daemon");
        await call("cursor_click", { x, y });
      }
    } catch {
      // ignore
    }
  }

  function exitCursorMode() {
    cursorModeActive = false;
    pinchClickFired = false;
  }

  function toggleCursorMode() {
    if (!$settings.gesture_cursor?.enabled) return;
    if (cursorModeActive) {
      exitCursorMode();
    } else {
      cursorModeActive = true;
    }
  }

  /** Per-frame cursor tracking + pinch-to-click while cursor mode is active.
   * `landmarks` must be the temporally-filtered set (same space as raw). */
  function updateGestureCursor(landmarks: Landmark[]) {
    const indexTip = landmarks[8];
    const thumbTip = landmarks[4];
    const predictionMs = $settings.gesture_cursor?.prediction_ms ?? 80;
    const blend = $settings.gesture_cursor?.blend ?? 0.3;
    const predicted = landmarkFilter.predictAhead(predictionMs);

    const target = predicted ? predictCursorTarget(indexTip, predicted[8], blend) : indexTip;

    // The video element is mirrored (`transform: scaleX(-1)`) for natural
    // selfie-view display, but MediaPipe processes the raw, unmirrored
    // frame — flip x so cursor motion matches what the user sees (moving
    // their hand right visually moves the cursor right).
    const screenX = Math.round(
      Math.max(0, Math.min(window.screen.width - 1, (1 - target.x) * window.screen.width))
    );
    const screenY = Math.round(Math.max(0, Math.min(window.screen.height - 1, target.y * window.screen.height)));
    lastCursorX = screenX;
    lastCursorY = screenY;
    void moveGestureCursor(screenX, screenY);

    // Pinch-to-click on the PREDICTED distance, so the click fires before
    // the pinch pose has fully, stably closed — the literal "fire before a
    // gesture completes" case this predictive layer targets.
    const predictedThumb = predicted ? predicted[4] : thumbTip;
    const predictedIndex = predicted ? predicted[8] : indexTip;
    const predictedPinchDist = Math.hypot(predictedThumb.x - predictedIndex.x, predictedThumb.y - predictedIndex.y);
    const effectivePinchThreshold = $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectivePinchThreshold(PINCH_DISTANCE_THRESHOLD)
      : PINCH_DISTANCE_THRESHOLD;

    if (predictedPinchDist < effectivePinchThreshold) {
      if (!pinchClickFired) {
        pinchClickFired = true;
        void clickGestureCursor(screenX, screenY);
      }
    } else {
      pinchClickFired = false;
    }
  }

  // Finger trail tracking for air drawing
  let fingerTrail: { x: number; y: number; t: number }[] = [];
  let prevIndexPos: { x: number; y: number } | null = null;

  // Motion tracking buffers for dynamic gestures
  let wristHistory: { x: number; y: number; z: number; t: number }[] = [];
  let indexHistory: { x: number; y: number; t: number }[] = [];
  const MOTION_BUFFER_SIZE = 20;

  const GESTURE_COOLDOWN_MS = 1200;
  const MAX_TRAIL_LENGTH = 60;

  // Gesture emoji map — 30+ gestures
  const GESTURE_EMOJIS: Record<string, string> = {
    // Static poses
    palm: "✋", thumbs_up: "👍", thumbs_down: "👎", peace: "✌️",
    fist: "👊", point_up: "👆", rock: "🤟", ok: "👌",
    call_me: "🤙", finger_gun: "🔫", pinch: "🤏",
    middle_finger: "🖕", pinky_up: "🌸", vulcan: "🖖",
    crossed_fingers: "🤞", snap_ready: "🫰", devil_horns: "🤘",
    palm_down: "🫳", palm_up: "🫴", three_up: "🔆", four_up: "🔅",
    // Motion-based
    swipe_left: "👈", swipe_right: "👉", swipe_up: "⬆️", swipe_down: "⬇️",
    circular_cw: "🔄", circular_ccw: "🔃",
    palm_push: "🫸", palm_pull: "🫷",
    two_finger_swipe_left: "⏪", two_finger_swipe_right: "⏩",
  };

  // ── MediaPipe Hands Loading (legacy backend) ──
  let mpLoaded = $state(false);
  let mpLoading = $state(false);
  const MEDIAPIPE_HANDS_ASSET_BASE = "/mediapipe/hands";
  const MEDIAPIPE_TASKS_VISION_ASSET_BASE = "/mediapipe/tasks-vision";

  async function loadMediaPipe() {
    if (mpLoaded && hands) return true;
    mpLoading = true;

    try {
      hands = new Hands({
        locateFile: (file: string) =>
          `${MEDIAPIPE_HANDS_ASSET_BASE}/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(onHandResults);
      await hands.initialize();
      mpLoaded = true;
      return true;
    } catch (e) {
      cameraError = "Failed to load gesture detection assets.";
      console.error("MediaPipe load error:", e);
      return false;
    } finally {
      mpLoading = false;
    }
  }

  // ── MediaPipe Tasks-Vision HandLandmarker Loading ("tasks" backend) ──
  //
  // Real-metric-scale worldLandmarks (see GESTURES.md's "3D World-Model
  // Layer" section) require this newer Tasks API — the legacy `Hands`
  // callback API above never exposes metric 3D. CPU delegate is used
  // unconditionally: GPU delegate support inside Tauri's embedded webview
  // (WebView2/WebKitGTK/WKWebView) hasn't been verified cross-platform.
  async function loadHandLandmarker() {
    if (mpLoaded && handLandmarker) return true;
    mpLoading = true;

    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_TASKS_VISION_ASSET_BASE);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `${MEDIAPIPE_TASKS_VISION_ASSET_BASE}/hand_landmarker.task`,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      mpLoaded = true;
      return true;
    } catch (e) {
      cameraError = "Failed to load gesture detection assets.";
      console.error("MediaPipe Tasks-Vision load error:", e);
      return false;
    } finally {
      mpLoading = false;
    }
  }

  // ── MediaPipe Tasks-Vision FaceLandmarker Loading (gaze tracking) ──
  //
  // A separate model from the hand backend, loaded independently and only
  // when $settings.vision.gaze_tracking_enabled is on (see gazeTracking.ts).
  // Same CPU-delegate rationale as loadHandLandmarker() above.
  async function loadFaceLandmarker() {
    if (faceLandmarker) return true;

    updateGazeRuntime({
      phase: "loading",
      cameraActive: isActive,
      region: null,
      confidence: null,
      daemonStatus: "idle",
      message: "Loading the on-device FaceLandmarker model…",
    });
    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_TASKS_VISION_ASSET_BASE);
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `${MEDIAPIPE_TASKS_VISION_ASSET_BASE}/face_landmarker.task`,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
      });
      return true;
    } catch (e) {
      console.error("MediaPipe FaceLandmarker load error (gaze tracking disabled for this session):", e);
      faceLandmarker = null;
      updateGazeRuntime({
        phase: "error",
        cameraActive: isActive,
        message: "Gaze model failed to load. Gesture control can still run.",
      });
      return false;
    }
  }

  async function activateGazeTracking(): Promise<void> {
    if (!isActive || gazeTrackingActive || !$settings.vision?.gaze_tracking_enabled) return;
    const loaded = await loadFaceLandmarker();
    if (!isActive || !$settings.vision?.gaze_tracking_enabled) {
      if (faceLandmarker) {
        try { faceLandmarker.close(); } catch { /* ignore */ }
        faceLandmarker = null;
      }
      return;
    }
    gazeTrackingActive = loaded;
    if (loaded) {
      updateGazeRuntime({
        phase: "scanning",
        cameraActive: true,
        region: null,
        confidence: null,
        daemonStatus: "idle",
        message: "Camera and gaze model are on. Looking for your face…",
      });
    }
  }

  function deactivateGazeTracking(): void {
    if (faceLandmarker) {
      try { faceLandmarker.close(); } catch { /* ignore */ }
      faceLandmarker = null;
    }
    gazeTrackingActive = false;
    lastGazeRegion = null;
    lastGazeSentAt = 0;
    gazeFrameCounter = 0;
    resetGazeRuntime();
  }

  async function toggleGestures() {
    if (isActive) stopGestures();
    else await startGestures();
  }

  // Tracks a PAUSED/WAITING_FOR_TRIGGER VoiceGestureWorkflow sourced from
  // "gesture" (see daemon/pilot/agents/voice_gesture_workflow.py) so a
  // recognized control gesture (classifyControlGesture) can resume/cancel it
  // instead of firing its normal action. null when no such workflow exists.
  let pendingWorkflowId: string | null = null;
  let workflowNotificationHandler: ((method: string, params: unknown) => void) | null = null;
  // gesture_name -> goal_template, enabled bindings only (see
  // GestureWorkflowConfig in config.py) -- refreshed once per engine start,
  // not re-polled every frame.
  let gestureWorkflowBindings: Record<string, string> = {};

  async function subscribeToWorkflowState() {
    const { onNotification, call } = await import("../api/daemon");
    workflowNotificationHandler = (method, params) => {
      if (method !== "voice_gesture_workflow_state") return;
      const wf = params as { workflow_id: string; invocation_source: string; state: string };
      if (wf.invocation_source !== "gesture") return;
      if (wf.state === "paused" || wf.state === "waiting_for_trigger") {
        pendingWorkflowId = wf.workflow_id;
      } else if (wf.workflow_id === pendingWorkflowId) {
        pendingWorkflowId = null;
      }
    };
    onNotification(workflowNotificationHandler);

    try {
      const result = (await call("voice_gesture_workflow_list")) as {
        workflows: Array<{ workflow_id: string; invocation_source: string; state: string }>;
      };
      const existing = result.workflows?.find(
        (w) => w.invocation_source === "gesture" && (w.state === "paused" || w.state === "waiting_for_trigger")
      );
      if (existing) pendingWorkflowId = existing.workflow_id;
    } catch {
      // Daemon not ready yet -- fine, future voice_gesture_workflow_state
      // notifications will still populate pendingWorkflowId.
    }

    try {
      const policy = (await call("gesture_workflow_bindings_get")) as {
        enabled: boolean;
        bindings: Array<{ gesture_name: string; goal_template: string; enabled: boolean }>;
      };
      gestureWorkflowBindings =
        policy.enabled && policy.bindings
          ? Object.fromEntries(policy.bindings.filter((b) => b.enabled && b.goal_template).map((b) => [b.gesture_name, b.goal_template]))
          : {};
    } catch {
      gestureWorkflowBindings = {};
    }
  }

  async function unsubscribeFromWorkflowState() {
    if (workflowNotificationHandler) {
      const { offNotification } = await import("../api/daemon");
      offNotification(workflowNotificationHandler);
      workflowNotificationHandler = null;
    }
    pendingWorkflowId = null;
    gestureWorkflowBindings = {};
  }

  async function dispatchWorkflowControl(intent: "continue" | "cancel", workflowId: string) {
    const { call } = await import("../api/daemon");
    const method = intent === "continue" ? "voice_gesture_workflow_resume" : "voice_gesture_workflow_cancel";
    try {
      await call(method, { workflow_id: workflowId });
    } catch {
      // best-effort -- if the RPC fails the workflow simply stays paused,
      // no worse than before the gesture fired
    }
  }

  async function startBoundWorkflow(goalTemplate: string) {
    const { call } = await import("../api/daemon");
    try {
      await call("voice_gesture_workflow_submit", { goal: goalTemplate, invocation_source: "gesture" });
    } catch {
      // best-effort -- if submission fails, nothing was started; the user
      // can retry the gesture
    }
  }

  async function startGestures() {
    cameraError = "";
    resetGazeRuntime();
    if ($settings.vision?.gaze_tracking_enabled) {
      updateGazeRuntime({
        phase: "loading",
        message: "Preparing camera controls and on-device models…",
      });
    }
    activeBackend = $settings.vision?.mediapipe_backend === "tasks" ? "tasks" : "legacy";
    const loaded = activeBackend === "tasks" ? await loadHandLandmarker() : await loadMediaPipe();
    if (!loaded) {
      if ($settings.vision?.gaze_tracking_enabled) {
        updateGazeRuntime({
          phase: "error",
          message: cameraError || "Camera controls failed to load.",
        });
      }
      return;
    }

    await subscribeToWorkflowState();

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
    } catch (e: any) {
      cameraError = `Camera error: ${e.name || e.message || 'Access denied or no device found'}`;
      console.error("Camera error:", e);
      if ($settings.vision?.gaze_tracking_enabled) {
        updateGazeRuntime({
          phase: "error",
          cameraActive: false,
          message: cameraError,
        });
      }
      void unsubscribeFromWorkflowState();
      return;
    }

    isActive = true;
    showCamera = true;
    fingerTrail = [];
    
    // Wait for Svelte to render the `<video>` element before assigning the stream
    await tick();

    if (videoEl) {
      videoEl.srcObject = stream;
      try {
        await videoEl.play();
      } catch (e) {
        console.error("Video play failed", e);
      }
    }

    detectFrame();
  }

  let stopping = false;

  function stopGestures() {
    if (stopping) return; // Guard against double-fire
    stopping = true;

    // 1. Stop the animation frame loop FIRST (prevents new MediaPipe sends)
    isActive = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = 0; }

    // 2. Close MediaPipe (whichever backend was active) to release the
    // video element reference
    if (hands) {
      try { hands.close(); } catch { /* ignore */ }
      hands = null;
    }
    if (handLandmarker) {
      try { handLandmarker.close(); } catch { /* ignore */ }
      handLandmarker = null;
    }
    deactivateGazeTracking();
    lastWorldLandmarks = null;

    // 3. Stop camera tracks AFTER MediaPipe is closed
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    // 4. Clear video element source
    if (videoEl) {
      videoEl.srcObject = null;
    }

    // 5. Reset UI state
    showCamera = false;
    currentGesture = "";
    confidence = 0;
    fingerTrail = [];
    prevIndexPos = null;
    candidateGesture = "";
    candidateCount = 0;
    wristHistory = [];
    indexHistory = [];
    landmarkFilter.reset();
    exitCursorMode(); // safety hatch: never leave cursor mode active with the engine stopped
    cancelPendingCalibration();
    void unsubscribeFromWorkflowState();

    stopping = false;
  }

  async function detectFrame() {
    if (!isActive || !videoEl || stopping) return;

    if (activeBackend === "tasks") {
      if (handLandmarker) {
        try {
          const result = handLandmarker.detectForVideo(videoEl, performance.now());
          const landmarks = (result.landmarks?.[0] as Landmark[] | undefined) ?? null;
          const worldLandmarks = (result.worldLandmarks?.[0] as Landmark[] | undefined) ?? null;
          const handednessScore = result.handednesses?.[0]?.[0]?.score;
          handleFrameResult(landmarks, worldLandmarks, handednessScore);
        } catch { /* ignore */ }
      }
    } else if (hands) {
      try { await hands.send({ image: videoEl }); } catch { /* ignore */ }
    } else {
      return;
    }

    if (gazeTrackingActive && faceLandmarker) {
      gazeFrameCounter++;
      if (gazeFrameCounter >= GAZE_FRAME_INTERVAL) {
        gazeFrameCounter = 0;
        try {
          const faceResult = faceLandmarker.detectForVideo(videoEl, performance.now());
          const faceLandmarks = faceResult.faceLandmarks?.[0] as { x: number; y: number; z?: number }[] | undefined;
          const estimate = estimateGazeRegion(faceLandmarks ?? null);
          if (estimate) {
            const now = performance.now();
            updateGazeRuntime({
              phase: "active",
              cameraActive: true,
              region: estimate.region,
              confidence: estimate.confidence,
              message: "",
            });
            // Refresh a steady reading before the backend's short fusion
            // window expires, while still avoiding per-frame RPC traffic.
            if (shouldSendGazeUpdate(estimate.region, lastGazeRegion, now, lastGazeSentAt)) {
              lastGazeRegion = estimate.region;
              lastGazeSentAt = now;
              void sendGazeEvent(estimate.region, estimate.confidence);
            }
          } else {
            updateGazeRuntime({
              phase: "scanning",
              cameraActive: true,
              region: null,
              confidence: null,
              message: "Gaze is on, but no face is visible to the camera.",
            });
          }
        } catch { /* ignore */ }
      }
    }

    if (isActive && !stopping) {
      animFrameId = requestAnimationFrame(detectFrame);
    }
  }

  /** Sends only the coarse region label + confidence to the backend --
   * never raw face landmarks or video frames (see gazeTracking.ts's
   * module docstring on the privacy rationale). Best-effort: a failed
   * send just means this one gaze update didn't reach the fusion engine,
   * not a reason to disrupt the gesture/camera pipeline. */
  async function sendGazeEvent(region: GazeRegion, confidence: number): Promise<void> {
    updateGazeRuntime({ daemonStatus: "sending" });
    try {
      const { call } = await import("../api/daemon");
      const response = (await call("gaze_event", { region, confidence })) as {
        status?: "ingested" | "ignored" | "error";
        reason?: string;
        message?: string;
      };
      if (response.status === "ingested") {
        updateGazeRuntime({ daemonStatus: "ingested", message: "" });
      } else {
        updateGazeRuntime({
          daemonStatus: response.status === "ignored" ? "ignored" : "error",
          message:
            response.reason === "confidence_below_threshold"
              ? "Gaze detected locally; confidence is too low for fusion."
              : response.message || "Gaze detected locally, but the daemon did not ingest it.",
        });
      }
    } catch {
      updateGazeRuntime({
        daemonStatus: "error",
        message: "Gaze detected locally, but the daemon connection is unavailable.",
      });
    }
  }

  function onHandResults(results: Results) {
    const landmarks = (results.multiHandLandmarks?.[0] as Landmark[] | undefined) ?? null;
    handleFrameResult(landmarks, null, results.multiHandedness?.[0]?.score);
  }

  /** Backend-agnostic per-frame entry point — both the legacy `Hands`
   * callback path and the "tasks" `HandLandmarker` polling path funnel into
   * this. `worldLandmarks` is only ever non-null from the "tasks" backend.
   * Static-pose classification still runs entirely off the normalized
   * `landmarks` array exactly as before (see spatialModel.ts's docstring on
   * why the ~20 empirically-tuned thresholds aren't being re-expressed in
   * 3D here) — the "tasks" backend only adds a real-metric-depth push/pull
   * check and a metric pinch-distance confirmation signal (see
   * worldModel.ts and GESTURES.md's "3D World-Model Layer" section). */
  function handleFrameResult(
    landmarks: Landmark[] | null,
    worldLandmarks: Landmark[] | null,
    handednessScore: number | undefined,
  ) {
    lastWorldLandmarks = worldLandmarks;

    if (!landmarks || landmarks.length === 0) {
      currentGesture = "";
      confidence = 0;
      prevIndexPos = null;
      candidateGesture = "";
      candidateCount = 0;
      landmarkFilter.reset(); // avoid smearing stale filter state into the next detected hand
      worldModelFilter.reset();
      worldWristHistory = [];
      return;
    }

    // Update motion buffers — deliberately built from RAW (unfiltered) landmarks.
    // Swipe/circular/push-pull thresholds are already tuned against raw jitter;
    // filtering the buffers themselves risks damping the fast motion they detect.
    const now = Date.now();
    const wrist = landmarks[0];
    wristHistory.push({ x: wrist.x, y: wrist.y, z: wrist.z || 0, t: now });
    if (wristHistory.length > MOTION_BUFFER_SIZE) wristHistory.shift();
    const idx = landmarks[8];
    indexHistory.push({ x: idx.x, y: idx.y, t: now });
    if (indexHistory.length > MOTION_BUFFER_SIZE) indexHistory.shift();

    // Temporally-filtered landmarks feed static-pose classification, where
    // single-frame jitter causes flicker between adjacent gesture readings.
    const filteredLandmarks = landmarkFilter.filter(landmarks, now);

    // ── 3D world-model signals ("tasks" backend only) ──
    //
    // worldWristHistory stays RAW, same rationale as wristHistory above.
    // filteredWorldLandmarks (temporally smoothed) feeds the metric pinch
    // confirmation check below, paralleling filteredLandmarks's role in
    // static-pose classification.
    let use3DPushPull = false;
    let pushPull3D: "push" | "pull" | null = null;
    let filteredWorldLandmarks: Landmark[] | null = null;
    if (worldLandmarks) {
      use3DPushPull = true;
      worldWristHistory.push(worldLandmarks[0]);
      if (worldWristHistory.length > WORLD_MOTION_BUFFER_SIZE) worldWristHistory.shift();
      if (worldWristHistory.length >= WORLD_MOTION_BUFFER_SIZE) {
        pushPull3D = detectPushPull3D(worldWristHistory);
        if (pushPull3D) worldWristHistory = []; // mirrors wristHistory's reset-on-fire in detectPushPull()
      }
      filteredWorldLandmarks = worldModelFilter.filter(worldLandmarks, now);
    } else {
      worldModelFilter.reset();
      worldWristHistory = [];
    }

    const gesture = classifyGesture(filteredLandmarks, use3DPushPull, pushPull3D);

    // Scale confidence by detection/geometric quality instead of letting a
    // degenerate (occluded/edge-on) hand pose misfire at full confidence.
    const quality = computeHandQuality(landmarks, handednessScore);
    gesture.confidence *= quality;
    if (gesture.name && gesture.confidence < QUALITY_CONFIDENCE_FLOOR) {
      gesture.name = "";
      gesture.confidence = 0;
    }

    // Metric pinch-distance confirmation ("tasks" backend only) — a
    // camera-distance-invariant depth reading that catches 2D-projection
    // false positives where the thumb and index tip merely overlap in the
    // camera's view without truly being close in real depth. Only ever
    // REDUCES confidence on top of the already-classified 2D result; never
    // replaces or raises it, and the existing 2D PINCH_DISTANCE_THRESHOLD
    // check above is untouched.
    if (filteredWorldLandmarks && (gesture.name === "ok" || gesture.name === "pinch")) {
      const wristRelative = toWristRelative3D(filteredWorldLandmarks);
      const metricSize = handSize3D(wristRelative);
      const metricPinchRatio = pinchDistance3D(wristRelative) / metricSize;
      // Same ratio-based threshold the 2D check conceptually uses, just
      // evaluated in metric wrist-relative space; doubled as a wide
      // tolerance band since this is a confirmation signal, not a
      // replacement for the tuned 2D threshold.
      if (metricPinchRatio > (PINCH_DISTANCE_THRESHOLD / handSize(filteredLandmarks)) * 2) {
        gesture.confidence *= 0.5;
        if (gesture.confidence < QUALITY_CONFIDENCE_FLOOR) {
          gesture.name = "";
          gesture.confidence = 0;
        }
      }
    }

    if (cursorModeActive) {
      // Open palm is the hands-only escape hatch — checked before anything
      // else so it always wins over cursor tracking/pinch-click.
      if (gesture.name === "palm") {
        exitCursorMode();
        drawLandmarks(landmarks);
        return;
      }
      updateGestureCursor(filteredLandmarks);
      // Every other discrete gesture is suppressed while pointing/clicking —
      // reaching for a swipe/peace/thumbs-up mid-point would otherwise
      // misfire constantly.
      drawLandmarks(landmarks);
      return;
    }

    // Track index finger for air drawing
    trackFingerTrail(landmarks);

    if (gesture.name) {
      if (gesture.name === candidateGesture) {
        candidateCount++;
        if (candidateCount >= REQUIRED_FRAMES && gesture.name !== currentGesture) {
          currentGesture = gesture.name;
          confidence = gesture.confidence;
          const now = Date.now();
          if (now - lastGestureTime > GESTURE_COOLDOWN_MS) {
            lastGestureTime = now;
            // A gesture-sourced workflow currently paused/waiting claims
            // continue/cancel gestures instead of them firing their normal
            // action — see subscribeToWorkflowState()/workflowControl.ts.
            const controlIntent = pendingWorkflowId ? classifyControlGesture(gesture.name) : "unknown";
            const boundGoal = !pendingWorkflowId ? gestureWorkflowBindings[gesture.name] : undefined;
            if (controlIntent !== "unknown" && pendingWorkflowId) {
              void dispatchWorkflowControl(controlIntent, pendingWorkflowId);
              pendingWorkflowId = null;
            } else if (boundGoal) {
              // A user-bound gesture starts a workflow instead of its
              // normal default action — see GestureWorkflowConfig in
              // config.py and the Settings gesture-workflow bindings editor.
              void startBoundWorkflow(boundGoal);
              gestureHistory = [...gestureHistory.slice(-4), gesture.name];
            } else {
              executeGestureAction(gesture.name);
              gestureHistory = [...gestureHistory.slice(-4), gesture.name];
              onGesture(gesture.name);
              trackGestureForCalibration(gesture.name, gesture.metricValue ?? 0);
            }
          }
        }
      } else {
        candidateGesture = gesture.name;
        candidateCount = 1;
      }
    } else {
      if (candidateGesture !== "") {
        candidateGesture = "";
        candidateCount = 1;
      } else {
        candidateCount++;
        if (candidateCount >= 3) {
          currentGesture = "";
          confidence = 0;
        }
      }
    }

    drawLandmarks(landmarks);
  }

  // ── Finger Trail Tracking ──
  function trackFingerTrail(landmarks: any[]) {
    const indexTip = landmarks[8];
    const now = Date.now();

    // Only track when only index finger is extended (pointing)
    const isPointing = landmarks[8].y < landmarks[6].y &&
      landmarks[12].y > landmarks[10].y; // Index up, middle down

    if (isPointing && trailCanvas) {
      const x = indexTip.x * trailCanvas.width;
      const y = indexTip.y * trailCanvas.height;
      fingerTrail.push({ x, y, t: now });
      if (fingerTrail.length > MAX_TRAIL_LENGTH) fingerTrail.shift();
      drawTrail();
    } else {
      // Decay trail
      if (fingerTrail.length > 0) {
        fingerTrail = fingerTrail.filter(p => now - p.t < 2000);
        drawTrail();
      }
    }

    prevIndexPos = { x: indexTip.x, y: indexTip.y };
  }

  function drawTrail() {
    if (!trailCanvas) return;
    const ctx = trailCanvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);

    if (fingerTrail.length < 2) return;

    const now = Date.now();
    for (let i = 1; i < fingerTrail.length; i++) {
      const prev = fingerTrail[i - 1];
      const curr = fingerTrail[i];
      const age = (now - curr.t) / 2000;
      const alpha = Math.max(0, 1 - age);

      ctx.strokeStyle = `hsla(${190 + i * 2}, 100%, 65%, ${alpha * 0.7})`;
      ctx.lineWidth = 2 * alpha;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(curr.x, curr.y);
      ctx.stroke();

      // Glowing dot at current position
      if (i === fingerTrail.length - 1 && alpha > 0.5) {
        ctx.fillStyle = `hsla(190, 100%, 70%, ${alpha})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(0, 200, 255, 0.5)";
        ctx.beginPath();
        ctx.arc(curr.x, curr.y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  }

  // ── Enhanced Gesture Classification ──
  interface Gesture {
    name: string;
    confidence: number;
    /** The raw measured value behind a calibration-relevant classification
     * (pinch/OK's thumb-index distance, or thumbs_up/down's thumb-extension
     * ratio) — populated only for gestures calibration.ts tracks. */
    metricValue?: number;
  }

  function classifyGesture(
    landmarks: any[],
    use3DPushPull: boolean = false,
    pushPull3D: "push" | "pull" | null = null,
  ): Gesture {
    const THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
    const INDEX_PIP = 6, MIDDLE_PIP = 10, RING_PIP = 14, PINKY_PIP = 18;
    const THUMB_MCP = 2, INDEX_MCP = 5;
    const WRIST = 0;

    const isExtended = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y;
    // Orientation/handedness-invariant — see spatialModel.ts for why the old
    // `landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x` check broke for left
    // hands and rotated wrists. Threshold may be personalized by the
    // on-device calibration loop (calibration.ts) — falls back to the
    // shipped THUMB_EXTENDED_RATIO default until enough confirmed samples
    // exist.
    const effectiveThumbRatio = $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectiveThumbRatio(THUMB_EXTENDED_RATIO)
      : THUMB_EXTENDED_RATIO;
    const thumbExtended = isThumbExtended(landmarks, handSize(landmarks), effectiveThumbRatio);

    // Same calibration treatment for the pinch/OK-sign distance threshold.
    const effectivePinchThreshold = $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectivePinchThreshold(PINCH_DISTANCE_THRESHOLD)
      : PINCH_DISTANCE_THRESHOLD;

    const indexUp = isExtended(INDEX_TIP, INDEX_PIP);
    const middleUp = isExtended(MIDDLE_TIP, MIDDLE_PIP);
    const ringUp = isExtended(RING_TIP, RING_PIP);
    const pinkyUp = isExtended(PINKY_TIP, PINKY_PIP);

    // Distance helper
    const dist = (a: number, b: number) => {
      const dx = landmarks[a].x - landmarks[b].x;
      const dy = landmarks[a].y - landmarks[b].y;
      return Math.sqrt(dx * dx + dy * dy);
    };

    // 3D distance for push/pull
    const dist3d = (a: number, b: number) => {
      const dx = landmarks[a].x - landmarks[b].x;
      const dy = landmarks[a].y - landmarks[b].y;
      const dz = (landmarks[a].z || 0) - (landmarks[b].z || 0);
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    };

    // ═══════════════════════════════════════════
    // MOTION-BASED GESTURES (check first — they are time-sensitive)
    // ═══════════════════════════════════════════

    // Circular motion detection (volume control)
    const circularResult = detectCircularMotion();
    if (circularResult) return circularResult;

    // Palm push/pull (Z-axis depth change). Under the "tasks" backend,
    // detectPushPull3D's real-metric-depth reading replaces the ad hoc
    // normalized-z check below entirely (see worldModel.ts and GESTURES.md's
    // "3D World-Model Layer" section) — same open-palm pose gate either way.
    const allFingersUpForPushPull = indexUp && middleUp && ringUp && pinkyUp;
    if (use3DPushPull) {
      if (pushPull3D && allFingersUpForPushPull) {
        return { name: pushPull3D === "push" ? "palm_push" : "palm_pull", confidence: 0.72 };
      }
    } else {
      const pushPull = detectPushPull(landmarks);
      if (pushPull) return pushPull;
    }

    // Predicted near-future landmarks — used below only to scale swipe
    // confidence by whether the trajectory agrees with the classified
    // direction (reduces misfires from a single noisy frame), not to
    // re-decide the classification itself. Not applied to
    // detectCircularMotion()/detectPushPull(): "agreement" isn't a simple
    // dx/dy sign check for a tangential/depth motion.
    const predictedMotion = landmarkFilter.predictAhead(MOTION_PREDICTION_MS);

    // Two-finger swipe (peace sign + horizontal motion)
    if (prevIndexPos && indexUp && middleUp && !ringUp && !pinkyUp) {
      const dx = landmarks[WRIST].x - prevIndexPos.x;
      const predictedDx = predictedMotion ? predictedMotion[WRIST].x - prevIndexPos.x : dx;
      if (dx < -0.09) {
        return { name: "two_finger_swipe_left", confidence: 0.75 * trajectoryAgreement(dx, predictedDx) };
      }
      if (dx > 0.09) {
        return { name: "two_finger_swipe_right", confidence: 0.75 * trajectoryAgreement(dx, predictedDx) };
      }
    }

    // Full-hand swipe (all fingers up + horizontal motion)
    if (prevIndexPos && indexUp && middleUp && ringUp && pinkyUp) {
      const dx = landmarks[WRIST].x - prevIndexPos.x;
      const dy = landmarks[WRIST].y - prevIndexPos.y;
      const predictedDx = predictedMotion ? predictedMotion[WRIST].x - prevIndexPos.x : dx;
      const predictedDy = predictedMotion ? predictedMotion[WRIST].y - prevIndexPos.y : dy;
      if (Math.abs(dx) > 0.08) {
        if (dx < -0.08) return { name: "swipe_left", confidence: 0.7 * trajectoryAgreement(dx, predictedDx) };
        if (dx > 0.08) return { name: "swipe_right", confidence: 0.7 * trajectoryAgreement(dx, predictedDx) };
      }
      if (Math.abs(dy) > 0.08) {
        if (dy < -0.08) return { name: "swipe_up", confidence: 0.7 * trajectoryAgreement(dy, predictedDy) };
        if (dy > 0.08) return { name: "swipe_down", confidence: 0.7 * trajectoryAgreement(dy, predictedDy) };
      }
    }

    // ═══════════════════════════════════════════
    // STATIC POSE GESTURES (most specific first)
    // ═══════════════════════════════════════════

    // 🫳 Palm Down — all fingers extended, wrist higher than fingertips
    if (indexUp && middleUp && ringUp && pinkyUp && thumbExtended) {
      const avgTipY = (landmarks[INDEX_TIP].y + landmarks[MIDDLE_TIP].y +
        landmarks[RING_TIP].y + landmarks[PINKY_TIP].y) / 4;
      if (avgTipY > landmarks[WRIST].y + 0.15) {
        return { name: "palm_down", confidence: 0.8 };
      }
      // 🫴 Palm Up — fingertips above wrist significantly
      if (avgTipY < landmarks[WRIST].y - 0.15) {
        return { name: "palm_up", confidence: 0.8 };
      }
    }

    // 🖖 Vulcan Salute — all 4 fingers up, gap between middle+ring
    if (indexUp && middleUp && ringUp && pinkyUp && !thumbExtended) {
      if (dist(MIDDLE_TIP, RING_TIP) > 0.08) {
        return { name: "vulcan", confidence: 0.85 };
      }
    }

    // 👌 OK Sign — thumb tip touching index tip, others up
    if (dist(THUMB_TIP, INDEX_TIP) < effectivePinchThreshold && middleUp && ringUp && pinkyUp) {
      return { name: "ok", confidence: 0.85, metricValue: dist(THUMB_TIP, INDEX_TIP) };
    }

    // 🤏 Pinch — thumb tip close to index tip, others curled
    if (dist(THUMB_TIP, INDEX_TIP) < effectivePinchThreshold && !middleUp && !ringUp && !pinkyUp) {
      return { name: "pinch", confidence: 0.85, metricValue: dist(THUMB_TIP, INDEX_TIP) };
    }

    // 🫰 Snap Ready — thumb touching middle finger, index curled
    if (dist(THUMB_TIP, MIDDLE_TIP) < 0.05 && !indexUp && !ringUp && !pinkyUp) {
      return { name: "snap_ready", confidence: 0.82 };
    }

    // 🤞 Crossed Fingers — index + middle up, close together
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
      if (dist(INDEX_TIP, MIDDLE_TIP) < 0.03) {
        return { name: "crossed_fingers", confidence: 0.8 };
      }
    }

    // 🖕 Middle Finger — only middle extended
    if (!indexUp && middleUp && !ringUp && !pinkyUp && !thumbExtended) {
      return { name: "middle_finger", confidence: 0.9 };
    }

    // 🌸 Pinky Up — only pinky extended
    if (!indexUp && !middleUp && !ringUp && pinkyUp && !thumbExtended) {
      return { name: "pinky_up", confidence: 0.85 };
    }

    // 🤘 Devil Horns — index + pinky up, middle + ring down, thumb tucked
    if (indexUp && !middleUp && !ringUp && pinkyUp && !thumbExtended) {
      // Extra check: index and pinky spread
      if (dist(INDEX_TIP, PINKY_TIP) > 0.1) {
        return { name: "devil_horns", confidence: 0.82 };
      }
    }

    // 🔫 Finger Gun — index + thumb extended, others down, thumb horizontal
    if (thumbExtended && indexUp && !middleUp && !ringUp && !pinkyUp) {
      if (Math.abs(landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y) < 0.08) {
        return { name: "finger_gun", confidence: 0.78 };
      }
    }

    // 🤙 Call Me — thumb + pinky extended, others curled
    if (thumbExtended && !indexUp && !middleUp && !ringUp && pinkyUp) {
      return { name: "call_me", confidence: 0.82 };
    }

    // 👎 Thumbs Down / 👍 Thumbs Up
    if (thumbExtended && !indexUp && !middleUp && !ringUp && !pinkyUp) {
      if (landmarks[THUMB_TIP].y > landmarks[WRIST].y) {
        return { name: "thumbs_down", confidence: 0.8, metricValue: thumbExtensionRatio(landmarks) };
      }
      if (landmarks[THUMB_TIP].y < landmarks[WRIST].y) {
        return { name: "thumbs_up", confidence: 0.8, metricValue: thumbExtensionRatio(landmarks) };
      }
    }

    // 👊 Fist — everything curled
    if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbExtended) {
      return { name: "fist", confidence: 0.85 };
    }

    // ✋ Open Palm — everything extended (default orientation)
    if (indexUp && middleUp && ringUp && pinkyUp && thumbExtended) {
      return { name: "palm", confidence: 0.9 };
    }

    // 🔆 Three Up — index + middle + ring, no pinky
    if (indexUp && middleUp && ringUp && !pinkyUp && !thumbExtended) {
      return { name: "three_up", confidence: 0.78 };
    }

    // 🔅 Four Up — all 4 fingers, no thumb
    if (indexUp && middleUp && ringUp && pinkyUp && !thumbExtended) {
      return { name: "four_up", confidence: 0.78 };
    }

    // ✌️ Peace — index + middle
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
      return { name: "peace", confidence: 0.85 };
    }

    // 👆 Point Up — only index
    if (indexUp && !middleUp && !ringUp && !pinkyUp) {
      return { name: "point_up", confidence: 0.8 };
    }

    // 🤟 Rock — index + pinky (with thumb)
    if (indexUp && !middleUp && !ringUp && pinkyUp) {
      return { name: "rock", confidence: 0.75 };
    }

    return { name: "", confidence: 0 };
  }

  // ── Motion: Circular gesture detection ──
  function detectCircularMotion(): Gesture | null {
    if (indexHistory.length < 12) return null;
    const recent = indexHistory.slice(-12);
    const cx = recent.reduce((s, p) => s + p.x, 0) / recent.length;
    const cy = recent.reduce((s, p) => s + p.y, 0) / recent.length;

    // Check if points form a rough circle around the centroid
    const radii = recent.map(p => Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2));
    const avgRadius = radii.reduce((s, r) => s + r, 0) / radii.length;
    if (avgRadius < 0.03 || avgRadius > 0.2) return null;

    // Check circularity: stddev of radii should be small
    const variance = radii.reduce((s, r) => s + (r - avgRadius) ** 2, 0) / radii.length;
    if (Math.sqrt(variance) > avgRadius * 0.5) return null;

    // Determine direction using cross product sum
    let crossSum = 0;
    for (let i = 1; i < recent.length; i++) {
      const prev = recent[i - 1];
      const curr = recent[i];
      crossSum += (prev.x - cx) * (curr.y - cy) - (prev.y - cy) * (curr.x - cx);
    }

    if (Math.abs(crossSum) < 0.001) return null;

    // Clear buffer to avoid re-triggering
    indexHistory.length = 0;

    if (crossSum > 0) return { name: "circular_cw", confidence: 0.75 };
    return { name: "circular_ccw", confidence: 0.75 };
  }

  // ── Motion: Palm push/pull (Z-axis depth) ──
  function detectPushPull(landmarks: any[]): Gesture | null {
    if (wristHistory.length < 8) return null;
    const old = wristHistory[0];
    const now = wristHistory[wristHistory.length - 1];
    const dz = now.z - old.z;
    const elapsed = now.t - old.t;

    // Only detect if movement happened in < 600ms
    if (elapsed > 600 || elapsed < 100) return null;

    // All fingers must be extended (palm pose)
    const isExtended = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y;
    const allUp = isExtended(8, 6) && isExtended(12, 10) && isExtended(16, 14) && isExtended(20, 18);
    if (!allUp) return null;

    if (dz < -0.06) {
      wristHistory.length = 0;
      return { name: "palm_push", confidence: 0.72 };
    }
    if (dz > 0.06) {
      wristHistory.length = 0;
      return { name: "palm_pull", confidence: 0.72 };
    }
    return null;
  }

  function executeGestureAction(gesture: string) {
    const emoji = GESTURE_EMOJIS[gesture] || "🖐️";
    switch (gesture) {
      // ── Static Pose Actions ──
      case "palm":
        session.addSystemMessage(`${emoji} Stop / Cancel`);
        break;
      case "thumbs_up":
        session.confirm(true);
        session.addSystemMessage(`${emoji} Confirmed!`);
        break;
      case "thumbs_down":
        session.confirm(false);
        session.addSystemMessage(`${emoji} Denied!`);
        break;
      case "peace":
        session.addSystemMessage(`${emoji} Peace! Toggling voice...`);
        break;
      case "fist":
        session.addSystemMessage(`${emoji} Ready to execute!`);
        break;
      case "point_up":
        session.addSystemMessage(`${emoji} Scroll up`);
        break;
      case "rock":
        session.sendCommand("Show me my system info");
        break;
      case "ok":
        session.addSystemMessage(`${emoji} OK! Acknowledged.`);
        break;
      case "finger_gun":
        session.sendCommand("Take a screenshot and save it to the Desktop");
        session.addSystemMessage(`${emoji} Screenshot!`);
        break;
      case "call_me":
        session.addSystemMessage(`${emoji} Opening settings...`);
        break;
      case "pinch":
        session.addSystemMessage(`${emoji} Pinch / Grab`);
        break;
      case "middle_finger":
        session.sendCommand("Cancel all tasks absolutely immediately");
        break;
      case "pinky_up":
        session.addSystemMessage(`${emoji} Fancy!`);
        break;
      case "vulcan":
        session.sendCommand("Show detailed system diagnostics and status");
        session.addSystemMessage(`${emoji} Live long and prosper.`);
        break;
      case "crossed_fingers":
        session.sendCommand("Surprise me with something cool");
        session.addSystemMessage(`${emoji} Feeling lucky...`);
        break;
      case "snap_ready":
        session.sendCommand("Open my most used application");
        session.addSystemMessage(`${emoji} Quick Launch!`);
        break;
      case "devil_horns":
        session.sendCommand("Open the default music player and play music");
        session.addSystemMessage(`${emoji} Rock on! Playing music...`);
        break;
      case "palm_down":
        session.sendCommand("Set volume to 0");
        session.addSystemMessage(`${emoji} Muted!`);
        break;
      case "palm_up":
        session.sendCommand("Set volume to 50");
        session.addSystemMessage(`${emoji} Unmuted! Volume at 50%`);
        break;
      case "three_up":
        session.sendCommand("Increase screen brightness by 20 percent");
        session.addSystemMessage(`${emoji} Brightness up!`);
        break;
      case "four_up":
        session.sendCommand("Decrease screen brightness by 20 percent");
        session.addSystemMessage(`${emoji} Brightness down!`);
        break;

      // ── Motion-Based Actions ──
      case "swipe_left":
        session.addSystemMessage(`${emoji} Previous tab`);
        break;
      case "swipe_right":
        session.addSystemMessage(`${emoji} Next tab`);
        break;
      case "swipe_up":
        session.addSystemMessage(`${emoji} Scroll up!`);
        break;
      case "swipe_down":
        session.addSystemMessage(`${emoji} Scroll down!`);
        break;
      case "circular_cw":
        session.sendCommand("Increase the system volume by 15 percent");
        session.addSystemMessage(`${emoji} Volume Up!`);
        break;
      case "circular_ccw":
        session.sendCommand("Decrease the system volume by 15 percent");
        session.addSystemMessage(`${emoji} Volume Down!`);
        break;
      case "palm_push":
        session.confirm(true);
        session.addSystemMessage(`${emoji} AI Action Confirmed!`);
        break;
      case "palm_pull":
        session.confirm(false);
        session.addSystemMessage(`${emoji} AI Action Cancelled!`);
        break;
      case "two_finger_swipe_left":
        session.sendCommand("Switch to the previous virtual desktop or workspace");
        session.addSystemMessage(`${emoji} Workspace Left`);
        break;
      case "two_finger_swipe_right":
        session.sendCommand("Switch to the next virtual desktop or workspace");
        session.addSystemMessage(`${emoji} Workspace Right`);
        break;
    }
  }

  // ── Canvas Drawing ──
  function drawLandmarks(landmarks: any[]) {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const connections = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],
    ];

    // Neon connections
    ctx.lineWidth = 1.5;
    connections.forEach(([a, b]) => {
      const grad = ctx.createLinearGradient(
        landmarks[a].x * canvasEl!.width, landmarks[a].y * canvasEl!.height,
        landmarks[b].x * canvasEl!.width, landmarks[b].y * canvasEl!.height
      );
      grad.addColorStop(0, "rgba(0, 200, 255, 0.5)");
      grad.addColorStop(1, "rgba(120, 80, 255, 0.5)");
      ctx.strokeStyle = grad;
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * canvasEl!.width, landmarks[a].y * canvasEl!.height);
      ctx.lineTo(landmarks[b].x * canvasEl!.width, landmarks[b].y * canvasEl!.height);
      ctx.stroke();
    });

    // Glow nodes
    landmarks.forEach((lm, i) => {
      const isTip = [4, 8, 12, 16, 20].includes(i);
      const x = lm.x * canvasEl!.width;
      const y = lm.y * canvasEl!.height;

      if (isTip) {
        // Glow effect for tips
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 8);
        glow.addColorStop(0, "rgba(0, 255, 136, 0.4)");
        glow.addColorStop(1, "rgba(0, 255, 136, 0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = isTip ? "rgba(0, 255, 136, 0.9)" : "rgba(0, 200, 255, 0.7)";
      ctx.beginPath();
      ctx.arc(x, y, isTip ? 4 : 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  $effect(() => {
    const enabled = $settings.vision?.gaze_tracking_enabled ?? false;
    if (!enabled) {
      if (gazeTrackingActive || faceLandmarker) deactivateGazeTracking();
    } else if (isActive && !gazeTrackingActive) {
      // This also makes the preference reactive if it is changed while the
      // camera session is already running.
      void activateGazeTracking();
    }
  });

  $effect(() => {
    return () => stopGestures();
  });
</script>

<div class="gesture-control">
  <button
    class="gesture-btn"
    class:active={isActive}
    class:loading={mpLoading}
    onclick={toggleGestures}
    aria-label={isActive ? "Stop camera controls" : "Start camera controls"}
    title={isActive
      ? "Stop camera, gesture, and gaze control"
      : $settings.vision?.gaze_tracking_enabled
        ? "Start camera, hand gestures, and gaze tracking"
        : "Start gesture control (30+ gestures!)"}
  >
    <svg class="hand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M18 11V6a2 2 0 0 0-4 0v1" />
      <path d="M14 10V4a2 2 0 0 0-4 0v6" />
      <path d="M10 10.5V6a2 2 0 0 0-4 0v8" />
      <path d="M18 8a2 2 0 0 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
    </svg>
    {#if mpLoading}
      <span class="loading-dot"></span>
    {/if}
  </button>

  {#if $settings.vision?.gaze_tracking_enabled}
    <button
      class="gaze-runtime-chip"
      class:active={$gazeRuntime.phase === "active"}
      class:error={$gazeRuntime.phase === "error" || $gazeRuntime.daemonStatus === "error"}
      class:scanning={$gazeRuntime.phase === "loading" || $gazeRuntime.phase === "scanning"}
      onclick={() => { if (!isActive) void startGestures(); }}
      title={$gazeRuntime.message || (isActive ? "Live coarse gaze region" : "Start the camera to activate gaze tracking")}
    >
      <span class="gaze-runtime-dot"></span>
      {#if $gazeRuntime.phase === "active" && $gazeRuntime.region}
        Gaze: {$gazeRuntime.region} {Math.round(($gazeRuntime.confidence ?? 0) * 100)}%
      {:else if $gazeRuntime.phase === "loading"}
        Loading gaze…
      {:else if $gazeRuntime.phase === "scanning"}
        Gaze on · find face
      {:else if $gazeRuntime.phase === "error"}
        Gaze error
      {:else}
        Gaze ready · start camera
      {/if}
    </button>
  {/if}

  {#if isActive && $settings.gesture_cursor?.enabled}
    <button
      class="cursor-mode-btn"
      class:active={cursorModeActive}
      onclick={toggleCursorMode}
      title={cursorModeActive
        ? "Exit cursor mode (or show an open palm)"
        : "Enable gesture cursor control — point to move, pinch to click"}
    >
      {cursorModeActive ? "🖱️ Cursor Mode: ON" : "🖱️ Cursor Mode"}
    </button>
  {/if}

  {#if currentGesture}
    <div class="gesture-label" class:high-conf={confidence > 0.8}>
      <span class="gesture-emoji">{GESTURE_EMOJIS[currentGesture] || "🖐️"}</span>
      <span class="gesture-name">{currentGesture.replace(/_/g, " ")}</span>
    </div>
  {/if}

  <!-- Gesture History -->
  {#if gestureHistory.length > 0 && isActive}
    <div class="gesture-history">
      {#each gestureHistory as g}
        <span class="history-emoji">{GESTURE_EMOJIS[g] || "?"}</span>
      {/each}
    </div>
  {/if}

  {#if showCamera}
    <div class="camera-pip" class:gesture-detected={!!currentGesture}>
      <video bind:this={videoEl} class="cam-video" playsinline muted autoplay></video>
      <canvas bind:this={canvasEl} class="cam-overlay" width="320" height="240"></canvas>
      <canvas bind:this={trailCanvas} class="cam-trail" width="320" height="240"></canvas>
      <button class="pip-close" title="Close Camera" onclick={stopGestures}>×</button>
      {#if currentGesture}
        <div class="pip-gesture-tag">
          <span>{GESTURE_EMOJIS[currentGesture] || ""}</span>
          {currentGesture.replace(/_/g, " ")}
        </div>
      {/if}
      <!-- Gesture count badge -->
      <div class="pip-badge">30+ gestures</div>
      {#if $settings.vision?.gaze_tracking_enabled}
        <div
          class="pip-gaze-badge"
          class:active={$gazeRuntime.phase === "active"}
          class:error={$gazeRuntime.phase === "error" || $gazeRuntime.daemonStatus === "error"}
          title={$gazeRuntime.message}
        >
          {#if $gazeRuntime.phase === "active" && $gazeRuntime.region}
            Gaze {$gazeRuntime.region} · {Math.round(($gazeRuntime.confidence ?? 0) * 100)}%
          {:else if $gazeRuntime.phase === "loading"}
            Gaze loading
          {:else if $gazeRuntime.phase === "scanning"}
            Gaze scanning
          {:else}
            Gaze unavailable
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  {#if cameraError}
    <div class="gesture-error">{cameraError}</div>
  {/if}
</div>

<style>
  .gesture-control {
    display: flex;
    align-items: center;
    gap: 6px;
    position: relative;
  }

  .gesture-btn {
    position: relative;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid rgba(180, 120, 255, 0.3);
    background: rgba(180, 120, 255, 0.06);
    color: rgba(180, 120, 255, 0.7);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    flex-shrink: 0;
  }

  .gesture-btn:hover {
    border-color: rgba(180, 120, 255, 0.6);
    background: rgba(180, 120, 255, 0.12);
    color: rgba(180, 120, 255, 1);
    box-shadow: 0 0 15px rgba(180, 120, 255, 0.2);
  }

  .gesture-btn.active {
    border-color: rgba(0, 255, 136, 0.6);
    background: rgba(0, 255, 136, 0.1);
    color: rgba(0, 255, 136, 0.9);
    animation: gesture-pulse 2s ease-in-out infinite;
  }

  @keyframes gesture-pulse {
    0%, 100% { box-shadow: 0 0 8px rgba(0, 255, 136, 0.15); }
    50% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
  }

  .cursor-mode-btn {
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid rgba(255, 120, 60, 0.35);
    background: rgba(255, 120, 60, 0.08);
    color: rgba(255, 150, 90, 0.9);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: all 0.2s ease;
  }

  .cursor-mode-btn:hover {
    border-color: rgba(255, 120, 60, 0.6);
    background: rgba(255, 120, 60, 0.15);
  }

  .cursor-mode-btn.active {
    border-color: rgba(255, 60, 60, 0.7);
    background: rgba(255, 60, 60, 0.18);
    color: rgba(255, 200, 200, 1);
    animation: gesture-pulse 2s ease-in-out infinite;
  }

  .hand-icon { width: 18px; height: 18px; z-index: 1; }

  .loading-dot {
    position: absolute; top: 2px; right: 2px;
    width: 6px; height: 6px; border-radius: 50%;
    background: rgba(255, 200, 0, 0.8);
    animation: blink 0.8s infinite;
  }

  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

  .gaze-runtime-chip {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 12px;
    border: 1px solid rgba(180, 120, 255, 0.35);
    background: rgba(180, 120, 255, 0.08);
    color: rgba(220, 200, 255, 0.9);
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
    cursor: pointer;
  }

  .gaze-runtime-chip.active {
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(0, 255, 136, 0.1);
    color: rgba(130, 255, 195, 0.95);
  }

  .gaze-runtime-chip.scanning {
    border-color: rgba(245, 158, 11, 0.5);
    color: rgba(255, 205, 110, 0.95);
  }

  .gaze-runtime-chip.error {
    border-color: rgba(255, 80, 80, 0.55);
    color: rgba(255, 135, 135, 0.95);
  }

  .gaze-runtime-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 7px currentColor;
  }

  .gesture-label {
    display: flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 12px;
    background: rgba(180, 120, 255, 0.1);
    border: 1px solid rgba(180, 120, 255, 0.3);
    font-size: 11px; color: rgba(180, 120, 255, 0.9);
    white-space: nowrap; animation: fadeIn 0.2s ease;
  }

  .gesture-label.high-conf {
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(0, 255, 136, 0.08);
    color: rgba(0, 255, 136, 0.9);
  }

  .gesture-emoji { font-size: 14px; }
  .gesture-name { text-transform: capitalize; letter-spacing: 0.3px; }

  @keyframes fadeIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }

  /* Gesture History */
  .gesture-history {
    display: flex; gap: 2px;
    padding: 2px 6px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
  }

  .history-emoji {
    font-size: 12px;
    opacity: 0.5;
    transition: opacity 0.3s;
  }
  .history-emoji:last-child { opacity: 1; }

  /* Camera PiP */
  .camera-pip {
    position: fixed; bottom: 80px; right: 16px;
    width: 220px; height: 165px;
    border-radius: 12px; overflow: hidden;
    border: 2px solid rgba(180, 120, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(180, 120, 255, 0.1);
    z-index: 1000; transition: border-color 0.3s;
  }

  .camera-pip.gesture-detected {
    border-color: rgba(0, 255, 136, 0.6);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 255, 136, 0.15);
  }

  .cam-video {
    width: 100%; height: 100%;
    object-fit: cover; transform: scaleX(-1);
  }

  .cam-overlay, .cam-trail {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    transform: scaleX(-1);
    pointer-events: none;
  }

  .pip-close {
    position: absolute; top: 4px; right: 4px;
    width: 20px; height: 20px; border-radius: 50%;
    border: none; background: rgba(0,0,0,0.6);
    color: white; font-size: 12px; cursor: pointer;
    display: flex; align-items: center; justify-content: center; z-index: 2;
  }

  .pip-gesture-tag {
    position: absolute; bottom: 6px; left: 50%;
    transform: translateX(-50%);
    padding: 2px 10px; border-radius: 8px;
    background: rgba(0,0,0,0.7); color: rgba(0, 255, 136, 0.9);
    font-size: 11px; font-family: "Inter", sans-serif;
    text-transform: capitalize; z-index: 2;
    display: flex; align-items: center; gap: 4px;
  }

  .pip-badge {
    position: absolute; top: 4px; left: 4px;
    padding: 1px 6px; border-radius: 6px;
    background: rgba(180, 120, 255, 0.3);
    color: rgba(255,255,255,0.7); font-size: 8px;
    font-family: "Inter", sans-serif; letter-spacing: 0.5px;
    z-index: 2;
  }

  .pip-gaze-badge {
    position: absolute;
    top: 25px;
    left: 4px;
    padding: 2px 6px;
    border-radius: 6px;
    background: rgba(245, 158, 11, 0.75);
    color: white;
    font-size: 9px;
    font-family: "Inter", sans-serif;
    text-transform: capitalize;
    z-index: 2;
  }

  .pip-gaze-badge.active {
    background: rgba(0, 130, 80, 0.82);
  }

  .pip-gaze-badge.error {
    background: rgba(180, 35, 45, 0.85);
  }

  .gesture-error {
    font-size: 10px; color: rgba(255, 80, 80, 0.8); max-width: 160px;
  }
</style>

<script lang="ts">
  /**
   * GestureControl v3 — 30+ hand gesture recognition engine.
   * The mappings below are suggested explicit bindings. Unbound gestures
   * never create OS tasks; only stop and pending-approval controls are built in.
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
  import { multimodal } from "../stores/multimodal";
  import { invoke } from "../api/invoke";
  import { call, offNotification, onNotification, requireResultStatus } from "../api/daemon";
  import { tick, untrack } from "svelte";
  import { Hands, type Results } from "@mediapipe/hands";
  import { FilesetResolver, HandLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision";
  import {
    LandmarkFilterBank,
    assessHandFrame,
    computeHandQuality,
    handFrameRejectionMessage,
    isThumbExtended,
    thumbExtensionRatio,
    handSize,
    mapCursorTargetToScreen,
    predictCursorTarget,
    trajectoryAgreement,
    measureRecentMotion,
    MIN_RELIABLE_HAND_CONFIDENCE,
    THUMB_EXTENDED_RATIO,
    type Landmark,
  } from "../gesture/spatialModel";
  import {
    toWristRelative3D,
    handSize3D,
    pinchDistance3D,
    detectPushPull3D,
    fingerExtensionStates3D,
    classifyThumbState3D,
    WorldModelFilterBank,
  } from "../gesture/worldModel";
  import { TemporalGestureVerifier } from "../gesture/temporalGestureVerifier";
  import {
    GazeTemporalStabilizer,
    assessGazeFrame,
    cameraBackendNeedsRestart,
    gazeRejectionMessage,
    resolveHandBackend,
    shouldRunGazeInference,
    shouldSendGazeUpdate,
    shouldRunVideoInference,
    type GazeRegion,
  } from "../gesture/gazeTracking";
  import { gazeRuntime, resetGazeRuntime, updateGazeRuntime } from "../stores/gazeRuntime";
  import { isTauriRuntime } from "../utils/runtime";
  import {
    classifyOutcome,
    getSharedGestureCalibrationStore,
    REVERSAL_WINDOW_MS,
    type GestureEvent,
  } from "../gesture/calibration";
  import { classifyControlGesture } from "../gesture/workflowControl";
  import { defaultGestureAction } from "../gesture/actionPolicy";
  import { airHandoffGestureCommand } from "../gesture/airHandoffGesture";
  import {
    dispatchGestureToOwner,
    latchedGestureAfterCursorOverride,
    ownerOverridesCursorMode,
    selectGestureDispatchOwner,
  } from "../gesture/gestureDispatchPolicy";
  import {
    activeGestureWorkflowBindings,
    controlGestureWorkflow,
    submitGestureWorkflow,
  } from "../gesture/workflowBindingRuntime";
  import { LatestAsyncDispatcher } from "../gesture/latestAsyncDispatcher";
  import { airHandoff } from "../stores/airHandoff";
  import {
    createGestureArm,
    createPinchHysteresis,
    gazeEventAllowed,
    normalizedPointer,
  } from "../gesture/studyPointerTracker";
  import {
    GESTURE_APPROVAL_MESSAGE_KEYS,
    GESTURE_APPROVAL_MESSAGES,
    gestureApprovalDecision,
    sendGestureAnswer,
  } from "../gesture/approvalGesture";
  import { studyActive } from "../stores/coadaptation";
  import {
    publishStudyPointerFrame,
    reportStudyPointerHandLost,
    reportStudyPointerPalmExit,
    setStudyPointerAvailable,
    studyPointer,
  } from "../stores/studyPointer";
  import { _ } from "svelte-i18n";

  // ── Props ──
  let {
    onGesture = (name: string) => {},
    onActiveChange = (active: boolean) => {},
  }: {
    onGesture?: (name: string) => void;
    onActiveChange?: (active: boolean) => void;
  } = $props();

  // ── State ──
  let isActive = $state(false);
  let currentGesture = $state("");
  let confidence = $state(0);
  let cameraError = $state("");
  let workflowStatus = $state("");
  let workflowError = $state("");
  let showCamera = $state(false);
  let isStarting = $state(false);
  let gestureHistory: string[] = $state([]);
  let worldTracking = $state(false);
  let handAcquiring = $state(false);
  let handDiagnostic = $state("Show one complete hand");
  let handSignalQuality = $state(0);
  let handAcquireFrames = 0;

  let videoEl: HTMLVideoElement | undefined = $state();
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let trailCanvas: HTMLCanvasElement | undefined = $state();
  let stream: MediaStream | null = null;
  let hands: Hands | null = null;
  let handLandmarker: HandLandmarker | null = null;
  // Frozen at startGestures() time. Gaze always uses the Tasks-Vision hand
  // backend too: loading legacy @mediapipe/hands and Tasks-Vision in the
  // same page makes their Emscripten `Module` globals collide and aborts
  // FaceLandmarker initialization.
  let activeBackend: "legacy" | "tasks" = $state("legacy");
  // Most recent worldLandmarks from the "tasks" backend — the "legacy"
  // backend never populates this (no metric-scale 3D available). Consumed
  // by handleFrameResult()'s 3D push/pull + metric pinch confirmation
  // checks below (see worldModel.ts).
  let lastWorldLandmarks: Landmark[] | null = null;

  // ── Gaze tracking (third input modality, see gazeTracking.ts) ──
  //
  // A separate Tasks-Vision model gated on
  // $settings.vision.gaze_tracking_enabled. The hand model is also forced
  // to Tasks-Vision while gaze is enabled to avoid mixing incompatible
  // MediaPipe WASM runtimes. Only ever sends a coarse region label +
  // confidence to the backend, never raw face landmarks or video.
  let faceLandmarker: FaceLandmarker | null = null;
  let faceLandmarkerLoad: Promise<boolean> | null = null;
  let gazeTrackingActive = false;
  let lastGazeRegion: GazeRegion | null = null;
  let lastGazeSentAt = 0;
  let lastGazeInferenceAt = 0;
  let lastHandVideoTime = -1;
  let lastGazeVideoTime = -1;
  const gazeStabilizer = new GazeTemporalStabilizer();

  let animFrameId: number = 0;
  let lastGestureTime = 0;
  let candidateGesture = "";
  let candidateCount = 0;
  const REQUIRED_FRAMES = 5;
  const HAND_ACQUIRE_FRAMES = 8;

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
  const temporalGestureVerifier = new TemporalGestureVerifier();
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
  const gestureCalibration = getSharedGestureCalibrationStore();
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

  $effect(() => {
    if (!$settings.adaptive_calibration?.gesture_enabled) {
      cancelPendingCalibration();
    }
  });

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
  let handDetected = $state(false);
  let cursorRuntimePhase: "idle" | "waiting" | "tracking" | "error" = $state("idle");
  let cursorRuntimeMessage = $state("");
  let lastCursorX = 0;
  let lastCursorY = 0;
  let pinchClickFired = false; // debounce: one click per pinch-close, not per frame
  let cursorRetryAfter = 0;

  async function dispatchGestureCursor({ x, y }: { x: number; y: number }): Promise<void> {
    try {
      if (isTauriRuntime()) {
        await invoke("move_gesture_cursor", { x, y });
      } else {
        requireResultStatus(
          await call<{ status?: string; message?: string }>("cursor_move", { x, y }),
          "ok",
          "Cursor move was rejected",
        );
      }
      cursorRetryAfter = 0;
      if (cursorModeActive) {
        cursorRuntimePhase = "tracking";
        cursorRuntimeMessage = "Hand tracking active — point to move, pinch to click";
      }
    } catch (error) {
      cursorRetryAfter = Date.now() + 1000;
      cursorRuntimePhase = "error";
      cursorRuntimeMessage =
        error instanceof Error ? `Cursor control unavailable: ${error.message}` : "Cursor control unavailable";
      throw error;
    }
  }

  const cursorMoveDispatcher = new LatestAsyncDispatcher(dispatchGestureCursor);

  function moveGestureCursor(x: number, y: number): void {
    // The selection study uses its own in-window pointer; the OS cursor never moves while it is requested.
    if ($studyPointer.requested) return;
    if (Date.now() < cursorRetryAfter) return;
    cursorMoveDispatcher.enqueue({ x, y });
  }

  async function clickGestureCursor(x: number, y: number): Promise<void> {
    if ($studyPointer.requested) return;
    try {
      if (isTauriRuntime()) {
        await invoke("click_gesture_cursor");
      } else {
        // The daemon fallback has no "click at current position" concept —
        // reuse the same coordinates the last cursor_move call already sent.
        requireResultStatus(
          await call<{ status?: string; message?: string }>("cursor_click", { x, y }),
          "ok",
          "Cursor click was rejected",
        );
      }
    } catch (error) {
      cursorRuntimePhase = "error";
      cursorRuntimeMessage =
        error instanceof Error ? `Cursor click unavailable: ${error.message}` : "Cursor click unavailable";
    }
  }

  function exitCursorMode() {
    cursorMoveDispatcher.reset();
    cursorModeActive = false;
    cursorRuntimePhase = "idle";
    cursorRuntimeMessage = "";
    cursorRetryAfter = 0;
    pinchClickFired = false;
    candidateGesture = "";
    candidateCount = 0;
    currentGesture = "";
    confidence = 0;
  }

  $effect(() => {
    if (!$settings.gesture_cursor?.enabled && cursorModeActive) {
      exitCursorMode();
    }
  });

  function toggleCursorMode() {
    if (!$settings.gesture_cursor?.enabled) return;
    if (!cursorModeActive && $studyPointer.requested) return; // Cursor Mode and the study pointer exclude each other
    if (cursorModeActive) {
      exitCursorMode();
    } else {
      candidateGesture = "";
      candidateCount = 0;
      currentGesture = "";
      confidence = 0;
      cursorModeActive = true;
      cursorRuntimePhase = handDetected ? "tracking" : "waiting";
      cursorRuntimeMessage = handDetected
        ? "Hand tracking active — point to move, pinch to click"
        : "Show one hand to the camera";
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
    const sensitivity = $settings.gesture_cursor?.sensitivity ?? 1;
    const { x: screenX, y: screenY } = mapCursorTargetToScreen(
      target,
      window.screen.width,
      window.screen.height,
      sensitivity,
    );
    lastCursorX = screenX;
    lastCursorY = screenY;
    moveGestureCursor(screenX, screenY);

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

  // ── Selection-study pointer and discrete-gesture re-arm (spec 17.2 F2, 17.4) ──
  //
  // While the selection study requests the hand pointer, every verified frame
  // publishes the filtered index tip and the measured pinch into the study's
  // own store (stores/studyPointer.ts), which alone decides when a press may
  // start. The OS cursor is never moved or clicked, Cursor Mode is off, and the
  // open palm (which pauses the study) is the only discrete gesture.
  //
  // When the study pointer is released (any pause, exit, modal or approval)
  // every discrete gesture stays disarmed until the hand has been out of view
  // for 200 ms and is verified again for HAND_ACQUIRE_FRAMES frames, or no hand
  // was seen for 500 ms, and a gesture must begin after that re-arm. The pose
  // held in the study therefore never starts a workflow, navigates a tab or
  // answers an approval. Approve gestures are also disarmed whenever a
  // confirmation dialog becomes visible, so approving by gesture always needs a
  // hand that came back after the dialog appeared.
  const STUDY_HAND_LOST_MS = 200;
  const discreteArm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: HAND_ACQUIRE_FRAMES });
  const approveArm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: HAND_ACQUIRE_FRAMES });
  const studyPinch = createPinchHysteresis(PINCH_DISTANCE_THRESHOLD);
  let studyPointerWasRequested = false;
  let studyHandSeen = false;
  let studyLastHandAt = 0;
  let studyPalmLatched = false;
  // performance.now() of the frame on which the current candidate gesture was first seen.
  let candidateStartedAt = 0;
  let lastSeenConfirmVisibleSince: number | null = null;

  const studyPointerRequested = $derived($studyPointer.requested);
  const confirmVisibleSince = $derived($session.confirmVisibleSince);

  function effectivePinchThreshold(): number {
    return $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectivePinchThreshold(PINCH_DISTANCE_THRESHOLD)
      : PINCH_DISTANCE_THRESHOLD;
  }

  /** Handles `$studyPointer.requested` changing. Runs from an effect and again at the top of every camera frame,
   * so the edge is applied before that frame can dispatch anything. */
  function syncStudyPointerRequested(requested: boolean) {
    if (requested === studyPointerWasRequested) return;
    studyPointerWasRequested = requested;
    if (requested) {
      // The study starts or resumes: no pinch, palm or recognised pose from before carries into it.
      if (cursorModeActive) exitCursorMode();
      studyPinch.reset();
      studyHandSeen = false;
      studyPalmLatched = false;
      candidateGesture = "";
      candidateCount = 0;
      currentGesture = "";
      confidence = 0;
    } else {
      discreteArm.disarm(performance.now());
      studyPinch.reset();
      studyHandSeen = false;
    }
  }

  $effect(() => {
    const requested = studyPointerRequested;
    untrack(() => syncStudyPointerRequested(requested));
  });

  $effect(() => {
    const since = confirmVisibleSince;
    untrack(() => {
      if (since !== null && since !== lastSeenConfirmVisibleSince) approveArm.disarm(performance.now());
      lastSeenConfirmVisibleSince = since;
    });
  });

  function studyPointerModeActive(): boolean {
    return isActive && $studyPointer.requested;
  }

  /** A frame without a verified hand: after more than 200 ms the loss is reported once and the pinch reopens. */
  function noteStudyHandAbsent(frameAt: number) {
    if (studyHandSeen && frameAt - studyLastHandAt > STUDY_HAND_LOST_MS) {
      studyHandSeen = false;
      studyPinch.reset();
      reportStudyPointerHandLost();
    }
  }

  function publishStudyPointer(filteredLandmarks: Landmark[], frameAt: number) {
    studyHandSeen = true;
    studyLastHandAt = frameAt;
    const indexTip = filteredLandmarks[8];
    const thumbTip = filteredLandmarks[4];
    const pinchDistance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
    const pinchClosed = studyPinch.update(pinchDistance, effectivePinchThreshold());
    const { nx, ny } = normalizedPointer(indexTip, $settings.gesture_cursor?.sensitivity ?? 1);
    publishStudyPointerFrame({ nx, ny, pinchClosed, quality: handSignalQuality, t: frameAt });
  }

  // Finger trail tracking for air drawing
  let fingerTrail: { x: number; y: number; t: number }[] = [];

  // Motion tracking buffers for dynamic gestures
  let wristHistory: { x: number; y: number; z: number; t: number }[] = [];
  let indexHistory: { x: number; y: number; t: number }[] = [];
  const MOTION_BUFFER_SIZE = 20;

  const GESTURE_COOLDOWN_MS = 1200;
  const MAX_TRAIL_LENGTH = 60;

  // Gesture emoji map — 30+ gestures
  const GESTURE_EMOJIS: Record<string, string> = {
    // Static poses
    palm: "✋",
    thumbs_up: "👍",
    thumbs_down: "👎",
    peace: "✌️",
    fist: "👊",
    point_up: "👆",
    rock: "🤟",
    ok: "👌",
    call_me: "🤙",
    finger_gun: "🔫",
    pinch: "🤏",
    middle_finger: "🖕",
    pinky_up: "🌸",
    vulcan: "🖖",
    crossed_fingers: "🤞",
    snap_ready: "🫰",
    devil_horns: "🤘",
    palm_down: "🫳",
    palm_up: "🫴",
    three_up: "🔆",
    four_up: "🔅",
    // Motion-based
    swipe_left: "👈",
    swipe_right: "👉",
    swipe_up: "⬆️",
    swipe_down: "⬇️",
    circular_cw: "🔄",
    circular_ccw: "🔃",
    palm_push: "🫸",
    palm_pull: "🫷",
    two_finger_swipe_left: "⏪",
    two_finger_swipe_right: "⏩",
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
        locateFile: (file: string) => `${MEDIAPIPE_HANDS_ASSET_BASE}/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: MIN_RELIABLE_HAND_CONFIDENCE,
        minTrackingConfidence: 0.65,
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
        minHandDetectionConfidence: MIN_RELIABLE_HAND_CONFIDENCE,
        minHandPresenceConfidence: MIN_RELIABLE_HAND_CONFIDENCE,
        minTrackingConfidence: 0.65,
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
    if (faceLandmarkerLoad) return faceLandmarkerLoad;

    updateGazeRuntime({
      phase: "loading",
      cameraActive: isActive,
      region: null,
      confidence: null,
      daemonStatus: "idle",
      message: "Loading the on-device FaceLandmarker model…",
    });
    faceLandmarkerLoad = (async () => {
      let lastError: unknown;
      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_TASKS_VISION_ASSET_BASE);
          faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: `${MEDIAPIPE_TASKS_VISION_ASSET_BASE}/face_landmarker.task`,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.6,
            minFacePresenceConfidence: 0.6,
            minTrackingConfidence: 0.6,
            outputFaceBlendshapes: true,
          });
          return true;
        } catch (error) {
          lastError = error;
          faceLandmarker = null;
          if (attempt === 1) {
            console.warn("MediaPipe FaceLandmarker initialization failed; retrying once:", error);
          }
        }
      }

      const detail =
        lastError instanceof Error
          ? `${lastError.name}: ${lastError.message}`
          : String(lastError || "Unknown initialization error");
      console.error("MediaPipe FaceLandmarker load error (gaze tracking disabled for this session):", lastError);
      updateGazeRuntime({
        phase: "error",
        cameraActive: isActive,
        message: `Gaze model failed to load — ${detail}`,
      });
      return false;
    })();

    try {
      return await faceLandmarkerLoad;
    } finally {
      faceLandmarkerLoad = null;
    }
  }

  async function activateGazeTracking(): Promise<void> {
    if (!isActive || gazeTrackingActive || !$settings.vision?.gaze_tracking_enabled) return;
    const loaded = await loadFaceLandmarker();
    if (!isActive || !$settings.vision?.gaze_tracking_enabled) {
      if (faceLandmarker) {
        try {
          faceLandmarker.close();
        } catch {
          /* ignore */
        }
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
      try {
        faceLandmarker.close();
      } catch {
        /* ignore */
      }
      faceLandmarker = null;
    }
    gazeTrackingActive = false;
    lastGazeRegion = null;
    lastGazeSentAt = 0;
    lastGazeInferenceAt = 0;
    lastGazeVideoTime = -1;
    gazeStabilizer.reset();
    resetGazeRuntime();
  }

  async function retryGazeTracking(): Promise<void> {
    if (!isActive || !$settings.vision?.gaze_tracking_enabled || faceLandmarkerLoad) return;
    deactivateGazeTracking();
    await activateGazeTracking();
  }

  async function toggleGestures() {
    if (isStarting) return;
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
  let workflowFeedbackTimer: ReturnType<typeof setTimeout> | null = null;

  function showWorkflowFeedback(message: string, isError = false) {
    if (workflowFeedbackTimer) clearTimeout(workflowFeedbackTimer);
    workflowStatus = isError ? "" : message;
    workflowError = isError ? message : "";
    workflowFeedbackTimer = setTimeout(() => {
      workflowStatus = "";
      workflowError = "";
      workflowFeedbackTimer = null;
    }, 5000);
  }

  async function loadGestureWorkflowBindings() {
    try {
      const policy = (await call("gesture_workflow_bindings_get")) as {
        status?: string;
        message?: string;
        error?: string;
        enabled: boolean;
        bindings: Array<{ gesture_name: string; goal_template: string; enabled: boolean }>;
      };
      requireResultStatus(policy, "ok", "Gesture workflow bindings are unavailable.");
      gestureWorkflowBindings = activeGestureWorkflowBindings(policy);
    } catch (cause) {
      gestureWorkflowBindings = {};
      showWorkflowFeedback(cause instanceof Error ? cause.message : "Gesture workflow bindings are unavailable.", true);
    }
  }

  async function subscribeToWorkflowState() {
    workflowNotificationHandler = (method, params) => {
      if (method === "gesture_workflow_bindings_updated") {
        void loadGestureWorkflowBindings();
        return;
      }
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
        status?: string;
        message?: string;
        error?: string;
        workflows: Array<{ workflow_id: string; invocation_source: string; state: string }>;
      };
      requireResultStatus(result, "ok", "Voice and gesture workflow state is unavailable.");
      const existing = result.workflows?.find(
        (w) => w.invocation_source === "gesture" && (w.state === "paused" || w.state === "waiting_for_trigger"),
      );
      if (existing) pendingWorkflowId = existing.workflow_id;
    } catch (cause) {
      // Daemon not ready yet -- fine, future voice_gesture_workflow_state
      // notifications will still populate pendingWorkflowId.
      showWorkflowFeedback(
        cause instanceof Error ? cause.message : "Voice and gesture workflow state is unavailable.",
        true,
      );
    }

    await loadGestureWorkflowBindings();
  }

  async function unsubscribeFromWorkflowState() {
    if (workflowNotificationHandler) {
      offNotification(workflowNotificationHandler);
      workflowNotificationHandler = null;
    }
    pendingWorkflowId = null;
    gestureWorkflowBindings = {};
    if (workflowFeedbackTimer) {
      clearTimeout(workflowFeedbackTimer);
      workflowFeedbackTimer = null;
    }
    workflowStatus = "";
    workflowError = "";
  }

  async function dispatchWorkflowControl(intent: "continue" | "cancel", workflowId: string) {
    try {
      const message = await controlGestureWorkflow((method, params) => call(method, params), intent, workflowId);
      if (pendingWorkflowId === workflowId) pendingWorkflowId = null;
      showWorkflowFeedback(message);
    } catch (cause) {
      showWorkflowFeedback(cause instanceof Error ? cause.message : "Gesture workflow control failed", true);
    }
  }

  async function startBoundWorkflow(goalTemplate: string, gestureName: string) {
    try {
      const message = await submitGestureWorkflow((method, params) => call(method, params), gestureName, goalTemplate);
      showWorkflowFeedback(message);
    } catch (cause) {
      showWorkflowFeedback(cause instanceof Error ? cause.message : "Gesture workflow could not be started", true);
    }
  }

  async function startGestures() {
    if (isStarting || isActive) return;
    isStarting = true;
    cameraError = "";
    resetGazeRuntime();
    if ($settings.vision?.gaze_tracking_enabled) {
      updateGazeRuntime({
        phase: "loading",
        message: "Preparing camera controls and on-device models…",
      });
    }
    // @mediapipe/hands (legacy) and @mediapipe/tasks-vision both install an
    // Emscripten `Module` global. Mixing them aborts FaceLandmarker with
    // "Module.noExitRuntime has been replaced". Keep gaze sessions wholly
    // on Tasks-Vision; the user's configured backend still applies when
    // gaze is off.
    activeBackend = resolveHandBackend(
      $settings.vision?.mediapipe_backend,
      $settings.vision?.gaze_tracking_enabled ?? false,
    );
    const loaded = activeBackend === "tasks" ? await loadHandLandmarker() : await loadMediaPipe();
    if (!loaded) {
      if ($settings.vision?.gaze_tracking_enabled) {
        updateGazeRuntime({
          phase: "error",
          message: cameraError || "Camera controls failed to load.",
        });
      }
      isStarting = false;
      return;
    }

    await subscribeToWorkflowState();

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // A face and hand sharing a 320×240 frame leaves too few pixels for
        // reliable hand detection. Keep the compact PiP display, but analyse
        // a 640×480 stream so gaze and gestures can remain active together.
        video: { width: 640, height: 480, facingMode: "user" },
      });
    } catch (e: any) {
      cameraError = `Camera error: ${e.name || e.message || "Access denied or no device found"}`;
      console.error("Camera error:", e);
      if ($settings.vision?.gaze_tracking_enabled) {
        updateGazeRuntime({
          phase: "error",
          cameraActive: false,
          message: cameraError,
        });
      }
      void unsubscribeFromWorkflowState();
      isStarting = false;
      return;
    }

    isActive = true;
    onActiveChange(true);
    setStudyPointerAvailable(true);
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
    isStarting = false;
  }

  let stopping = false;

  function stopGestures() {
    if (stopping) return; // Guard against double-fire
    stopping = true;

    // 1. Stop the animation frame loop FIRST (prevents new MediaPipe sends)
    isActive = false;
    onActiveChange(false);
    // The study sees the hand pointer go away: a hand it was following is reported lost, so no press survives.
    if (studyHandSeen) {
      studyHandSeen = false;
      studyPinch.reset();
      reportStudyPointerHandLost();
    }
    setStudyPointerAvailable(false);
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = 0;
    }

    // 2. Close MediaPipe (whichever backend was active) to release the
    // video element reference
    if (hands) {
      try {
        hands.close();
      } catch {
        /* ignore */
      }
      hands = null;
    }
    if (handLandmarker) {
      try {
        handLandmarker.close();
      } catch {
        /* ignore */
      }
      handLandmarker = null;
    }
    deactivateGazeTracking();
    lastWorldLandmarks = null;
    lastHandVideoTime = -1;
    worldTracking = false;
    handAcquiring = false;
    handAcquireFrames = 0;
    handDiagnostic = "Show one complete hand";
    handSignalQuality = 0;

    // 3. Stop camera tracks AFTER MediaPipe is closed
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
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
    handDetected = false;
    fingerTrail = [];
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

  let cameraBackendRestarting = false;

  async function restartCameraForBackendChange(): Promise<void> {
    if (cameraBackendRestarting || !isActive || isStarting) return;
    cameraBackendRestarting = true;
    stopGestures();
    try {
      await startGestures();
    } finally {
      cameraBackendRestarting = false;
    }
  }

  async function detectFrame() {
    if (!isActive || !videoEl || stopping) return;

    const videoTime = videoEl.currentTime;
    const hasNewHandFrame = shouldRunVideoInference(videoTime, lastHandVideoTime);

    if (hasNewHandFrame) {
      lastHandVideoTime = videoTime;
      if (activeBackend === "tasks" && handLandmarker) {
        try {
          const result = handLandmarker.detectForVideo(videoEl, performance.now());
          const landmarks = (result.landmarks?.[0] as Landmark[] | undefined) ?? null;
          const worldLandmarks = (result.worldLandmarks?.[0] as Landmark[] | undefined) ?? null;
          const handednessScore = result.handedness?.[0]?.[0]?.score;
          handleFrameResult(landmarks, worldLandmarks, handednessScore);
        } catch {
          /* ignore */
        }
      } else if (activeBackend === "legacy" && hands) {
        try {
          await hands.send({ image: videoEl });
        } catch {
          /* ignore */
        }
      }
    }

    const gazeVideoTime = videoEl.currentTime;
    if (gazeTrackingActive && faceLandmarker && shouldRunVideoInference(gazeVideoTime, lastGazeVideoTime)) {
      const gazeNow = performance.now();
      if (shouldRunGazeInference(gazeNow, lastGazeInferenceAt)) {
        // Record before the synchronous FaceLandmarker call so an expensive
        // detected-face pass cannot trigger another run immediately after it
        // returns and starve the hand/cursor path.
        lastGazeInferenceAt = gazeNow;
        lastGazeVideoTime = gazeVideoTime;
        try {
          const faceResult = faceLandmarker.detectForVideo(videoEl, gazeNow);
          const faceLandmarks = faceResult.faceLandmarks?.[0] as { x: number; y: number; z?: number }[] | undefined;
          const blendshapes = faceResult.faceBlendshapes?.[0]?.categories as
            { categoryName: string; score: number }[] | undefined;
          const stabilized = gazeStabilizer.observe(assessGazeFrame(faceLandmarks ?? null, blendshapes));
          if (stabilized.state === "stable" && stabilized.estimate) {
            const estimate = stabilized.estimate;
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
              // During a selection study gaze stays local (gazeRuntime only): no gaze_event RPC, no ledger row.
              if (gazeEventAllowed($studyActive)) void sendGazeEvent(estimate.region, estimate.confidence);
            }
          } else if (stabilized.state === "settling") {
            updateGazeRuntime({
              phase: "stabilizing",
              cameraActive: true,
              region: null,
              confidence: null,
              message: `Confirming gaze ${stabilized.candidateRegion ?? "direction"}…`,
            });
          } else {
            updateGazeRuntime({
              phase: "scanning",
              cameraActive: true,
              region: null,
              confidence: null,
              message: stabilized.reason
                ? gazeRejectionMessage(stabilized.reason)
                : "Gaze is on, but no usable eye reading is available.",
            });
          }
        } catch {
          /* ignore */
        }
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
    const frameAt = performance.now();
    const handAssessment = assessHandFrame(landmarks, handednessScore);
    handSignalQuality = handAssessment.quality;
    handDiagnostic = handFrameRejectionMessage(handAssessment);
    lastWorldLandmarks = handAssessment.reliable ? worldLandmarks : null;

    // Both arms watch every frame, so a hand that leaves and comes back re-arms them (spec 17.4).
    const handVerified = Boolean(landmarks && handAssessment.reliable);
    syncStudyPointerRequested($studyPointer.requested);
    discreteArm.frame(handVerified, frameAt);
    approveArm.frame(handVerified, frameAt);
    const studyMode = studyPointerModeActive();

    if (!landmarks || !handAssessment.reliable) {
      if (studyMode) noteStudyHandAbsent(frameAt);
      clearLandmarks();
      handDetected = false;
      handAcquiring = false;
      handAcquireFrames = 0;
      worldTracking = false;
      if (cursorModeActive && cursorRuntimePhase !== "error") {
        cursorRuntimePhase = "waiting";
        cursorRuntimeMessage = "Show one hand to the camera";
      }
      currentGesture = "";
      confidence = 0;
      candidateGesture = "";
      candidateCount = 0;
      landmarkFilter.reset(); // avoid smearing stale filter state into the next detected hand
      worldModelFilter.reset();
      temporalGestureVerifier.reset();
      wristHistory = [];
      indexHistory = [];
      worldWristHistory = [];
      fingerTrail = [];
      drawTrail();
      return;
    }

    handAcquireFrames++;
    if (handAcquireFrames < HAND_ACQUIRE_FRAMES) {
      handDetected = false;
      handAcquiring = true;
      handDiagnostic = `Verifying hand (${handAcquireFrames}/${HAND_ACQUIRE_FRAMES})`;
      worldTracking = false;
      currentGesture = "";
      confidence = 0;
      candidateGesture = "";
      candidateCount = 0;
      // The study store runs its own acquisition (`ready`) and pinch-open gate (`pressArmed`) on these frames.
      if (studyMode) publishStudyPointer(landmarkFilter.filter(landmarks, Date.now()), frameAt);
      drawLandmarks(landmarks);
      return;
    }

    handDetected = true;
    handAcquiring = false;
    handDiagnostic = `Hand verified · ${Math.round(handSignalQuality * 100)}% signal quality`;
    worldTracking = Boolean(worldLandmarks && worldLandmarks.length >= 21);

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

    const gesture = classifyGesture(filteredLandmarks, use3DPushPull, pushPull3D, filteredWorldLandmarks);

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

    const temporalVerification = temporalGestureVerifier.observe({
      mediaPipeHandPresent: true,
      landmarks: filteredLandmarks,
      worldLandmarks: filteredWorldLandmarks,
      candidate: gesture.name,
      timestampMs: now,
    });
    if (gesture.name) {
      gesture.confidence *= temporalVerification.confidenceMultiplier;
      if (!temporalVerification.accepted || gesture.confidence < QUALITY_CONFIDENCE_FLOOR) {
        gesture.name = "";
        gesture.confidence = 0;
      }
    }

    if (studyMode) {
      // Nothing is recognised as a pending gesture while the study holds the hand pointer.
      candidateGesture = "";
      candidateCount = 0;
      currentGesture = "";
      confidence = 0;
      // Open palm keeps first priority, as in Cursor Mode: it pauses the study and publishes no pointer frame.
      if (gesture.name === "palm") {
        if (!studyPalmLatched) {
          studyPalmLatched = true;
          reportStudyPointerPalmExit();
        }
        drawLandmarks(landmarks);
        return;
      }
      studyPalmLatched = false;
      publishStudyPointer(filteredLandmarks, frameAt);
      drawLandmarks(landmarks);
      return;
    }

    if (cursorModeActive) {
      const workflowId = pendingWorkflowId;
      const controlIntent = workflowId ? classifyControlGesture(gesture.name) : "unknown";
      const handoffCommand = airHandoffGestureCommand(
        $airHandoff.gestureArmed,
        Boolean($airHandoff.draft),
        gesture.name,
      );
      const owner = selectGestureDispatchOwner({
        hasPendingWorkflow: Boolean(workflowId),
        controlIntent,
        handoffCommand,
        hasBoundWorkflow: false,
      });
      if (ownerOverridesCursorMode(owner)) {
        // A paused workflow control or explicitly armed Air Handoff command
        // owns this gesture. Exit cursor mode and let the normal debounced
        // dispatch path below deliver it exactly once.
        exitCursorMode();
        // Cursor mode displays recognized gestures without dispatching them.
        // Release that display latch or the normal path below will mistake the
        // override gesture for one that has already fired.
        currentGesture = latchedGestureAfterCursorOverride(owner, currentGesture);
      } else {
        // Open palm is the hands-only escape hatch — checked before anything
        // else so it always wins over cursor tracking/pinch-click.
        if (gesture.name === "palm") {
          exitCursorMode();
          drawLandmarks(landmarks);
          return;
        }
        // Cursor mode pauses gesture actions to prevent accidental commands,
        // but recognition remains visible so users can tell that the hand
        // model is working.
        currentGesture = gesture.name;
        confidence = gesture.confidence;
        updateGestureCursor(filteredLandmarks);
        // Every other discrete gesture is suppressed while pointing/clicking —
        // reaching for a swipe/peace/thumbs-up mid-point would otherwise
        // misfire constantly.
        drawLandmarks(landmarks);
        return;
      }
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
            const workflowId = pendingWorkflowId;
            const controlIntent = workflowId ? classifyControlGesture(gesture.name) : "unknown";
            const boundGoal = gestureWorkflowBindings[gesture.name];
            void dispatchRecognizedGesture(gesture, candidateStartedAt, boundGoal, controlIntent, workflowId);
          }
        }
      } else {
        candidateGesture = gesture.name;
        candidateCount = 1;
        candidateStartedAt = frameAt;
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
    const isPointing = landmarks[8].y < landmarks[6].y && landmarks[12].y > landmarks[10].y; // Index up, middle down

    if (isPointing && trailCanvas) {
      const x = indexTip.x * trailCanvas.width;
      const y = indexTip.y * trailCanvas.height;
      fingerTrail.push({ x, y, t: now });
      if (fingerTrail.length > MAX_TRAIL_LENGTH) fingerTrail.shift();
      drawTrail();
    } else {
      // Decay trail
      if (fingerTrail.length > 0) {
        fingerTrail = fingerTrail.filter((p) => now - p.t < 2000);
        drawTrail();
      }
    }
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
    worldLandmarks: Landmark[] | null = null,
  ): Gesture {
    const THUMB_TIP = 4,
      INDEX_TIP = 8,
      MIDDLE_TIP = 12,
      RING_TIP = 16,
      PINKY_TIP = 20;
    const INDEX_PIP = 6,
      MIDDLE_PIP = 10,
      RING_PIP = 14,
      PINKY_PIP = 18;
    const THUMB_MCP = 2,
      INDEX_MCP = 5;
    const WRIST = 0;

    const isExtended2D = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y;
    // Orientation/handedness-invariant — see spatialModel.ts for why the old
    // `landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x` check broke for left
    // hands and rotated wrists. Threshold may be personalized by the
    // on-device calibration loop (calibration.ts) — falls back to the
    // shipped THUMB_EXTENDED_RATIO default until enough confirmed samples
    // exist.
    const effectiveThumbRatio = $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectiveThumbRatio(THUMB_EXTENDED_RATIO)
      : THUMB_EXTENDED_RATIO;
    const thumbState3D = worldLandmarks ? classifyThumbState3D(worldLandmarks) : null;
    const thumbExtended =
      thumbState3D === null
        ? isThumbExtended(landmarks, handSize(landmarks), effectiveThumbRatio)
        : thumbState3D === "extended";
    const thumbTucked =
      thumbState3D === null
        ? !isThumbExtended(landmarks, handSize(landmarks), effectiveThumbRatio)
        : thumbState3D === "tucked";

    // Same calibration treatment for the pinch/OK-sign distance threshold.
    const effectivePinchThreshold = $settings.adaptive_calibration?.gesture_enabled
      ? gestureCalibration.getEffectivePinchThreshold(PINCH_DISTANCE_THRESHOLD)
      : PINCH_DISTANCE_THRESHOLD;

    const extension3D = worldLandmarks ? fingerExtensionStates3D(worldLandmarks) : null;
    const indexUp = extension3D?.index ?? isExtended2D(INDEX_TIP, INDEX_PIP);
    const middleUp = extension3D?.middle ?? isExtended2D(MIDDLE_TIP, MIDDLE_PIP);
    const ringUp = extension3D?.ring ?? isExtended2D(RING_TIP, RING_PIP);
    const pinkyUp = extension3D?.pinky ?? isExtended2D(PINKY_TIP, PINKY_PIP);

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
    const wristMotion = measureRecentMotion(wristHistory);

    // Two-finger swipe (peace sign + horizontal motion)
    if (wristMotion && indexUp && middleUp && !ringUp && !pinkyUp) {
      const dx = wristMotion.dx;
      const predictedDx = predictedMotion ? predictedMotion[WRIST].x - landmarks[WRIST].x : dx;
      if (dx < -0.09) {
        wristHistory = [];
        return { name: "two_finger_swipe_left", confidence: 0.75 * trajectoryAgreement(dx, predictedDx) };
      }
      if (dx > 0.09) {
        wristHistory = [];
        return { name: "two_finger_swipe_right", confidence: 0.75 * trajectoryAgreement(dx, predictedDx) };
      }
    }

    // Full-hand swipe (all fingers up + horizontal motion)
    if (wristMotion && indexUp && middleUp && ringUp && pinkyUp) {
      const dx = wristMotion.dx;
      const dy = wristMotion.dy;
      const predictedDx = predictedMotion ? predictedMotion[WRIST].x - landmarks[WRIST].x : dx;
      const predictedDy = predictedMotion ? predictedMotion[WRIST].y - landmarks[WRIST].y : dy;
      if (Math.abs(dx) > 0.08) {
        wristHistory = [];
        if (dx < -0.08) return { name: "swipe_left", confidence: 0.7 * trajectoryAgreement(dx, predictedDx) };
        if (dx > 0.08) return { name: "swipe_right", confidence: 0.7 * trajectoryAgreement(dx, predictedDx) };
      }
      if (Math.abs(dy) > 0.08) {
        wristHistory = [];
        if (dy < -0.08) return { name: "swipe_up", confidence: 0.7 * trajectoryAgreement(dy, predictedDy) };
        if (dy > 0.08) return { name: "swipe_down", confidence: 0.7 * trajectoryAgreement(dy, predictedDy) };
      }
    }

    // ═══════════════════════════════════════════
    // STATIC POSE GESTURES (most specific first)
    // ═══════════════════════════════════════════

    // 🫳 Palm Down — all fingers extended, wrist higher than fingertips
    if (indexUp && middleUp && ringUp && pinkyUp && !thumbTucked) {
      const avgTipY =
        (landmarks[INDEX_TIP].y + landmarks[MIDDLE_TIP].y + landmarks[RING_TIP].y + landmarks[PINKY_TIP].y) / 4;
      if (avgTipY > landmarks[WRIST].y + 0.15) {
        return { name: "palm_down", confidence: 0.8 };
      }
      // 🫴 Palm Up — fingertips above wrist significantly
      if (avgTipY < landmarks[WRIST].y - 0.15) {
        return { name: "palm_up", confidence: 0.8 };
      }
    }

    // 🖖 Vulcan Salute — all 4 fingers up, gap between middle+ring
    if (indexUp && middleUp && ringUp && pinkyUp && thumbTucked) {
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
    if (!indexUp && middleUp && !ringUp && !pinkyUp && thumbTucked) {
      return { name: "middle_finger", confidence: 0.9 };
    }

    // 🌸 Pinky Up — only pinky extended
    if (!indexUp && !middleUp && !ringUp && pinkyUp && thumbTucked) {
      return { name: "pinky_up", confidence: 0.85 };
    }

    // 🤘 Devil Horns — index + pinky up, middle + ring down, thumb tucked
    if (indexUp && !middleUp && !ringUp && pinkyUp && thumbTucked) {
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
    if (!indexUp && !middleUp && !ringUp && !pinkyUp && thumbTucked) {
      return { name: "fist", confidence: 0.85 };
    }

    // ✋ Open Palm — everything extended (default orientation)
    if (indexUp && middleUp && ringUp && pinkyUp && !thumbTucked) {
      return { name: "palm", confidence: 0.9 };
    }

    // 🔆 Three Up — index + middle + ring, no pinky
    if (indexUp && middleUp && ringUp && !pinkyUp && thumbTucked) {
      return { name: "three_up", confidence: 0.78 };
    }

    // 🔅 Four Up — all 4 fingers, no thumb
    if (indexUp && middleUp && ringUp && pinkyUp && thumbTucked) {
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
    const radii = recent.map((p) => Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2));
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

  /** A refused approve gesture says why, in the chat and next to the camera controls (visible on every tab). */
  function showApprovalGestureNotice(emoji: string, messageKey: string, message: string) {
    const text = $_(messageKey, { default: message });
    showWorkflowFeedback(text, true);
    session.addSystemMessage(`${emoji} ${text}`);
  }

  function executeGestureAction(gesture: string, gestureStartedAt: number) {
    const emoji = GESTURE_EMOJIS[gesture] || "🖐️";
    const action = defaultGestureAction(gesture);
    if (action === "abort") {
      void session.abort();
      return;
    }
    if (action === "confirm" || action === "deny") {
      const visibleSince = $session.confirmVisibleSince;
      const decision = gestureApprovalDecision({
        action,
        confirmRequired: $session.confirmRequired,
        planId: $session.confirmPlanId,
        confirmVisible: $session.confirmVisible,
        visibleForMs: visibleSince === null ? null : performance.now() - visibleSince,
        armed: approveArm.armed,
        gestureStartedAfterArm: approveArm.allows(gestureStartedAt),
        selection: $session.confirmSelection,
      });
      if (decision.kind === "approve" || decision.kind === "deny") {
        // The chat line waits for the daemon's reply and says what it recorded: a gesture that sent nothing (an
        // answer already in flight) or an answer the daemon refused never reads "accepted" or "denied".
        void sendGestureAnswer(
          decision.kind === "approve",
          (accepted) => session.confirm(accepted, undefined, { inputChannel: "gesture" }),
          (text) => session.addSystemMessage(`${emoji} ${text}`),
        );
      } else if (decision.kind === "refuse") {
        showApprovalGestureNotice(emoji, GESTURE_APPROVAL_MESSAGE_KEYS[decision.reason], decision.message);
      } else {
        session.addSystemMessage(`${emoji} ${gesture.replace(/_/g, " ")} recognized · no approval is pending.`);
      }
      return;
    }
    session.addSystemMessage(
      `${emoji} ${gesture.replace(/_/g, " ")} recognized · no default action runs. Bind this gesture in Settings to use it.`,
    );
  }

  async function dispatchRecognizedGesture(
    gesture: Gesture,
    gestureStartedAt: number,
    boundGoal: string | undefined,
    controlIntent: "continue" | "cancel" | "unknown",
    workflowId: string | null,
  ) {
    // Every discrete dispatch (workflow control, Air Handoff, bound workflows, built-in actions and onGesture
    // navigation) passes here. After the study pointer is released nothing runs until the hand left and came back,
    // and a pose recognised before that re-arm is dropped even if it completes later (spec 17.4).
    if (!discreteArm.allows(gestureStartedAt)) {
      if (defaultGestureAction(gesture.name) === "confirm" && $session.confirmRequired) {
        const emoji = GESTURE_EMOJIS[gesture.name] || "🖐️";
        showApprovalGestureNotice(emoji, GESTURE_APPROVAL_MESSAGE_KEYS.rearm, GESTURE_APPROVAL_MESSAGES.rearm);
      }
      return;
    }
    const handoffCommand = airHandoffGestureCommand($airHandoff.gestureArmed, Boolean($airHandoff.draft), gesture.name);
    const owner = selectGestureDispatchOwner({
      hasPendingWorkflow: Boolean(workflowId),
      controlIntent,
      handoffCommand,
      hasBoundWorkflow: Boolean(boundGoal),
    });

    await dispatchGestureToOwner(owner, {
      workflow_control: async () => {
        if (workflowId && controlIntent !== "unknown") await dispatchWorkflowControl(controlIntent, workflowId);
      },
      air_handoff: async () => {
        try {
          const consumed = await airHandoff.handleGesture(gesture.name);
          if (!consumed) {
            session.addSystemMessage("Air Handoff is busy; the gesture was ignored safely.");
            return;
          }
          session.addSystemMessage($airHandoff.message || "Air Handoff gesture accepted.");
          gestureHistory = [...gestureHistory.slice(-4), gesture.name];
          onGesture(gesture.name);
          trackGestureForCalibration(gesture.name, gesture.metricValue ?? 0);
        } catch (cause) {
          session.addSystemMessage(
            `Air Handoff stopped safely: ${cause instanceof Error ? cause.message : "unknown error"}`,
          );
        }
      },
      bound_workflow: async () => {
        if (!boundGoal) return;
        // User-authored workflow bindings retain normal priority, except when
        // the user has explicitly armed a one-shot Air Handoff gesture.
        await startBoundWorkflow(boundGoal, gesture.name);
        gestureHistory = [...gestureHistory.slice(-4), gesture.name];
      },
      default: () => {
        // A claimed handoff/workflow gesture must not remain in the fusion
        // buffer and modify a later voice command or world-model observation.
        multimodal.onGestureInput(gesture.name, gesture.confidence);
        void call("gesture_event", {
          gesture: gesture.name,
          confidence: gesture.confidence,
          data: { source: "on_device_camera", raw_landmarks_excluded: true },
        }).catch((cause) => console.warn("Gesture fusion context unavailable.", cause));

        executeGestureAction(gesture.name, gestureStartedAt);
        gestureHistory = [...gestureHistory.slice(-4), gesture.name];
        onGesture(gesture.name);
        trackGestureForCalibration(gesture.name, gesture.metricValue ?? 0);
      },
    });
  }

  // ── Canvas Drawing ──
  function clearLandmarks() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }

  function drawLandmarks(landmarks: any[]) {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const connections = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [0, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [0, 9],
      [9, 10],
      [10, 11],
      [11, 12],
      [0, 13],
      [13, 14],
      [14, 15],
      [15, 16],
      [0, 17],
      [17, 18],
      [18, 19],
      [19, 20],
      [5, 9],
      [9, 13],
      [13, 17],
    ];

    // Neon connections
    ctx.lineWidth = 1.5;
    connections.forEach(([a, b]) => {
      const grad = ctx.createLinearGradient(
        landmarks[a].x * canvasEl!.width,
        landmarks[a].y * canvasEl!.height,
        landmarks[b].x * canvasEl!.width,
        landmarks[b].y * canvasEl!.height,
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
    const configuredBackend = $settings.vision?.mediapipe_backend;
    const gazeEnabled = $settings.vision?.gaze_tracking_enabled ?? false;
    if (cameraBackendNeedsRestart(isActive, isStarting, activeBackend, configuredBackend, gazeEnabled)) {
      void restartCameraForBackendChange();
    }
  });

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
    class:loading={mpLoading || isStarting}
    onclick={toggleGestures}
    disabled={isStarting}
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
    {#if mpLoading || isStarting}
      <span class="loading-dot"></span>
    {/if}
  </button>

  {#if $settings.vision?.gaze_tracking_enabled}
    <button
      class="gaze-runtime-chip"
      class:active={$gazeRuntime.phase === "active"}
      class:error={$gazeRuntime.phase === "error" || $gazeRuntime.daemonStatus === "error"}
      class:scanning={$gazeRuntime.phase === "loading" ||
        $gazeRuntime.phase === "scanning" ||
        $gazeRuntime.phase === "stabilizing"}
      onclick={() => {
        if (!isActive) void startGestures();
        else if ($gazeRuntime.phase === "error") void retryGazeTracking();
      }}
      disabled={isStarting}
      title={$gazeRuntime.message ||
        (isActive ? "Live coarse gaze region" : "Start the camera to activate gaze tracking")}
    >
      <span class="gaze-runtime-dot"></span>
      {#if $gazeRuntime.phase === "active" && $gazeRuntime.region}
        Gaze: {$gazeRuntime.region} {Math.round(($gazeRuntime.confidence ?? 0) * 100)}%
      {:else if $gazeRuntime.phase === "loading"}
        Loading gaze…
      {:else if $gazeRuntime.phase === "scanning"}
        Gaze on · find face
      {:else if $gazeRuntime.phase === "stabilizing"}
        Gaze stabilizing…
      {:else if $gazeRuntime.phase === "error"}
        Gaze error · retry
      {:else}
        Gaze ready · start camera
      {/if}
    </button>
  {/if}

  {#if isActive && $settings.gesture_cursor?.enabled}
    <button
      class="cursor-mode-btn"
      class:active={cursorModeActive}
      class:error={cursorRuntimePhase === "error"}
      class:waiting={cursorRuntimePhase === "waiting"}
      onclick={toggleCursorMode}
      disabled={studyPointerRequested}
      title={cursorModeActive
        ? `${cursorRuntimeMessage}. Gesture commands are paused; show an open palm to exit.`
        : studyPointerRequested
          ? "Cursor Mode is off while the selection study uses the hand pointer"
          : "Enable gesture cursor control — point to move, pinch to click"}
    >
      {#if !cursorModeActive}
        🖱️ Cursor Mode
      {:else if cursorRuntimePhase === "error"}
        🖱️ Cursor unavailable
      {:else if cursorRuntimePhase === "waiting"}
        🖱️ Cursor: show hand
      {:else}
        🖱️ Cursor: tracking
      {/if}
    </button>
  {/if}

  {#if currentGesture}
    <div class="gesture-label" class:high-conf={confidence > 0.8}>
      <span class="gesture-emoji">{GESTURE_EMOJIS[currentGesture] || "🖐️"}</span>
      <span class="gesture-name">{currentGesture.replace(/_/g, " ")}</span>
    </div>
  {/if}

  {#if isActive}
    <div
      class="spatial-status"
      class:active={worldTracking}
      class:fallback={activeBackend === "legacy"}
      role="status"
      title={worldTracking
        ? "MediaPipe Tasks is supplying real 3D world landmarks for orientation-independent finger detection"
        : handAcquiring
          ? handDiagnostic
          : activeBackend === "tasks"
            ? handDiagnostic
            : "Compatibility backend active; enable Enhanced 3D Hand Tracking in Settings"}
    >
      <span class="spatial-status-dot"></span>
      {worldTracking
        ? "3D hand tracking"
        : handAcquiring
          ? "3D verifying hand"
          : activeBackend === "tasks"
            ? "3D ready · show hand"
            : "2D compatibility mode"}
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
    <div class="camera-pip" class:gesture-detected={!!currentGesture} class:hand-detected={handDetected}>
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
      <div class="pip-hand-badge" class:active={handDetected} title={handDiagnostic}>
        {handDetected ? "Hand detected" : handAcquiring ? "Hold hand steady" : "Show hand"}
      </div>
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
          {:else if $gazeRuntime.phase === "stabilizing"}
            Gaze stabilizing
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
  {#if workflowStatus}
    <div class="gesture-workflow-feedback" role="status">{workflowStatus}</div>
  {/if}
  {#if workflowError}
    <div class="gesture-error" role="alert">{workflowError}</div>
  {/if}
  {#if $gazeRuntime.phase === "error" && $gazeRuntime.message}
    <div class="gesture-error gaze-error-detail">{$gazeRuntime.message}</div>
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
    0%,
    100% {
      box-shadow: 0 0 8px rgba(0, 255, 136, 0.15);
    }
    50% {
      box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
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

  .cursor-mode-btn.active.waiting {
    border-color: rgba(245, 158, 11, 0.75);
    background: rgba(245, 158, 11, 0.14);
    color: rgba(255, 220, 150, 1);
  }

  .cursor-mode-btn.active.error {
    border-color: rgba(255, 70, 80, 0.85);
    background: rgba(180, 35, 45, 0.22);
    color: rgba(255, 180, 185, 1);
    animation: none;
  }

  .hand-icon {
    width: 18px;
    height: 18px;
    z-index: 1;
  }

  .loading-dot {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(255, 200, 0, 0.8);
    animation: blink 0.8s infinite;
  }

  @keyframes blink {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.2;
    }
  }

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

  .gesture-btn:disabled,
  .gaze-runtime-chip:disabled {
    cursor: wait;
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

  .spatial-status {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 12px;
    border: 1px solid rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.08);
    color: rgba(255, 205, 110, 0.95);
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
  }

  .spatial-status.active {
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(0, 255, 136, 0.1);
    color: rgba(130, 255, 195, 0.95);
  }

  .spatial-status.fallback {
    border-color: rgba(160, 170, 190, 0.35);
    background: rgba(160, 170, 190, 0.07);
    color: rgba(190, 200, 220, 0.8);
  }

  .spatial-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 7px currentColor;
  }

  .gesture-label {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 12px;
    background: rgba(180, 120, 255, 0.1);
    border: 1px solid rgba(180, 120, 255, 0.3);
    font-size: 11px;
    color: rgba(180, 120, 255, 0.9);
    white-space: nowrap;
    animation: fadeIn 0.2s ease;
  }

  .gesture-label.high-conf {
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(0, 255, 136, 0.08);
    color: rgba(0, 255, 136, 0.9);
  }

  .gesture-emoji {
    font-size: 14px;
  }
  .gesture-name {
    text-transform: capitalize;
    letter-spacing: 0.3px;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: scale(0.9);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  /* Gesture History */
  .gesture-history {
    display: flex;
    gap: 2px;
    padding: 2px 6px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
  }

  .history-emoji {
    font-size: 12px;
    opacity: 0.5;
    transition: opacity 0.3s;
  }
  .history-emoji:last-child {
    opacity: 1;
  }

  /* Camera PiP */
  .camera-pip {
    position: fixed;
    bottom: 80px;
    right: 16px;
    width: 220px;
    height: 165px;
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid rgba(180, 120, 255, 0.3);
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.5),
      0 0 20px rgba(180, 120, 255, 0.1);
    z-index: 1000;
    transition: border-color 0.3s;
  }

  .camera-pip.gesture-detected,
  .camera-pip.hand-detected {
    border-color: rgba(0, 255, 136, 0.6);
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.5),
      0 0 20px rgba(0, 255, 136, 0.15);
  }

  .cam-video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
  }

  .cam-overlay,
  .cam-trail {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    transform: scaleX(-1);
    pointer-events: none;
  }

  .pip-close {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: none;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    font-size: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }

  .pip-gesture-tag {
    position: absolute;
    bottom: 6px;
    left: 50%;
    transform: translateX(-50%);
    padding: 2px 10px;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.7);
    color: rgba(0, 255, 136, 0.9);
    font-size: 11px;
    font-family: "Inter", sans-serif;
    text-transform: capitalize;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .pip-badge {
    position: absolute;
    top: 4px;
    left: 4px;
    padding: 1px 6px;
    border-radius: 6px;
    background: rgba(180, 120, 255, 0.3);
    color: rgba(255, 255, 255, 0.7);
    font-size: 8px;
    font-family: "Inter", sans-serif;
    letter-spacing: 0.5px;
    z-index: 2;
  }

  .pip-hand-badge {
    position: absolute;
    top: 45px;
    left: 4px;
    padding: 2px 6px;
    border-radius: 6px;
    background: rgba(60, 70, 90, 0.82);
    color: rgba(255, 255, 255, 0.85);
    font-size: 9px;
    font-family: "Inter", sans-serif;
    z-index: 2;
  }

  .pip-hand-badge.active {
    background: rgba(0, 130, 80, 0.85);
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
    font-size: 10px;
    color: rgba(255, 80, 80, 0.8);
    max-width: 160px;
  }

  .gesture-workflow-feedback {
    max-width: 190px;
    color: var(--success);
    font-size: 10px;
    line-height: 1.3;
  }

  .gaze-error-detail {
    max-width: 260px;
    line-height: 1.3;
  }
</style>

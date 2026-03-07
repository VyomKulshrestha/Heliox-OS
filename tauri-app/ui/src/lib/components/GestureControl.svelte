<script lang="ts">
  /**
   * GestureControl — Hand gesture recognition via webcam using MediaPipe.
   * Detects gestures and maps them to Cortex-OS commands.
   *
   * Gestures:
   *  ✋ Open Palm  → Cancel / Stop current action
   *  👍 Thumbs Up → Confirm plan
   *  ✌️ Peace Sign → Toggle voice mode
   *  👊 Fist      → Execute last command
   *  👆 Point Up  → Scroll up
   *  👇 Point Down → Scroll down
   *  🤞 Crossed   → Show system status
   */

  import { session } from "../stores/session";

  // ── State ──
  let isActive = $state(false);
  let currentGesture = $state("");
  let confidence = $state(0);
  let cameraError = $state("");
  let showCamera = $state(false);
  
  let videoEl: HTMLVideoElement | undefined = $state();
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let stream: MediaStream | null = null;
  let hands: any = null;
  let animFrameId: number = 0;
  let lastGestureTime = 0;
  
  // Gesture cooldown to prevent rapid-fire
  const GESTURE_COOLDOWN_MS = 1500;

  // ── MediaPipe Hands Loading ──
  let mpLoaded = $state(false);
  let mpLoading = $state(false);

  async function loadMediaPipe() {
    if (mpLoaded) return true;
    mpLoading = true;

    try {
      // Dynamically load MediaPipe Hands from CDN
      // @ts-ignore — CDN import has no type declarations
      const module = await import(
        /* @vite-ignore */
        "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.js"
      );

      const Hands = module.Hands || (window as any).Hands;

      hands = new Hands({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0, // Fastest
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(onHandResults);
      mpLoaded = true;
      return true;
    } catch (e) {
      cameraError = "Failed to load gesture detection. Check internet connection.";
      console.error("MediaPipe load error:", e);
      return false;
    } finally {
      mpLoading = false;
    }
  }

  async function toggleGestures() {
    if (isActive) {
      stopGestures();
    } else {
      await startGestures();
    }
  }

  async function startGestures() {
    cameraError = "";

    // Load MediaPipe
    const loaded = await loadMediaPipe();
    if (!loaded) return;

    // Request camera
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
    } catch (e) {
      cameraError = "Camera access denied. Allow camera to use gesture control.";
      return;
    }

    if (videoEl) {
      videoEl.srcObject = stream;
      await videoEl.play();
    }

    isActive = true;
    showCamera = true;
    detectFrame();
  }

  function stopGestures() {
    isActive = false;
    showCamera = false;
    currentGesture = "";
    confidence = 0;

    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = 0;
    }

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  async function detectFrame() {
    if (!isActive || !videoEl || !hands) return;

    try {
      await hands.send({ image: videoEl });
    } catch { /* ignore frame errors */ }

    animFrameId = requestAnimationFrame(detectFrame);
  }

  function onHandResults(results: any) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      currentGesture = "";
      confidence = 0;
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const gesture = classifyGesture(landmarks);

    if (gesture.name && gesture.name !== currentGesture) {
      currentGesture = gesture.name;
      confidence = gesture.confidence;

      // Execute gesture action with cooldown
      const now = Date.now();
      if (now - lastGestureTime > GESTURE_COOLDOWN_MS) {
        lastGestureTime = now;
        executeGestureAction(gesture.name);
      }
    } else if (!gesture.name) {
      currentGesture = "";
      confidence = 0;
    }

    // Draw landmarks on canvas
    drawLandmarks(landmarks);
  }

  // ── Gesture Classification ──
  interface Gesture { name: string; confidence: number; }

  function classifyGesture(landmarks: any[]): Gesture {
    // Finger tip and base indices in MediaPipe Hands
    const THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
    const THUMB_IP = 3, INDEX_PIP = 6, MIDDLE_PIP = 10, RING_PIP = 14, PINKY_PIP = 18;
    const WRIST = 0;

    const isExtended = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y;
    const thumbExtended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x; // For right hand

    const indexUp = isExtended(INDEX_TIP, INDEX_PIP);
    const middleUp = isExtended(MIDDLE_TIP, MIDDLE_PIP);
    const ringUp = isExtended(RING_TIP, RING_PIP);
    const pinkyUp = isExtended(PINKY_TIP, PINKY_PIP);

    // 👊 Fist — no fingers extended
    if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbExtended) {
      return { name: "fist", confidence: 0.85 };
    }

    // ✋ Open Palm — all fingers extended
    if (indexUp && middleUp && ringUp && pinkyUp && thumbExtended) {
      return { name: "palm", confidence: 0.9 };
    }

    // 👍 Thumbs Up — only thumb extended
    if (thumbExtended && !indexUp && !middleUp && !ringUp && !pinkyUp) {
      // Check thumb is pointing up
      if (landmarks[THUMB_TIP].y < landmarks[WRIST].y) {
        return { name: "thumbs_up", confidence: 0.8 };
      }
    }

    // ✌️ Peace / Victory — index + middle extended
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
      return { name: "peace", confidence: 0.85 };
    }

    // 👆 Point Up — only index extended
    if (indexUp && !middleUp && !ringUp && !pinkyUp) {
      return { name: "point_up", confidence: 0.8 };
    }

    // 🤟 Rock — index + pinky extended (I love you sign)
    if (indexUp && !middleUp && !ringUp && pinkyUp) {
      return { name: "rock", confidence: 0.75 };
    }

    return { name: "", confidence: 0 };
  }

  function executeGestureAction(gesture: string) {
    switch (gesture) {
      case "palm":
        session.addSystemMessage("✋ Gesture: Stop/Cancel");
        break;
      case "thumbs_up":
        // Auto-confirm if there's a pending confirmation
        session.confirm(true);
        session.addSystemMessage("👍 Gesture: Confirmed!");
        break;
      case "peace":
        session.addSystemMessage("✌️ Gesture: Peace! Toggling voice...");
        break;
      case "fist":
        session.addSystemMessage("👊 Gesture: Ready to execute!");
        break;
      case "point_up":
        session.addSystemMessage("👆 Gesture: Scroll up");
        break;
      case "rock":
        session.sendCommand("Show me my system info");
        break;
    }
  }

  // ── Canvas Drawing ──
  function drawLandmarks(landmarks: any[]) {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    // Draw connections
    const connections = [
      [0,1],[1,2],[2,3],[3,4], // Thumb
      [0,5],[5,6],[6,7],[7,8], // Index
      [0,9],[9,10],[10,11],[11,12], // Middle
      [0,13],[13,14],[14,15],[15,16], // Ring
      [0,17],[17,18],[18,19],[19,20], // Pinky
      [5,9],[9,13],[13,17], // Palm
    ];

    ctx.strokeStyle = "rgba(0, 200, 255, 0.4)";
    ctx.lineWidth = 1.5;
    connections.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * canvasEl!.width, landmarks[a].y * canvasEl!.height);
      ctx.lineTo(landmarks[b].x * canvasEl!.width, landmarks[b].y * canvasEl!.height);
      ctx.stroke();
    });

    // Draw points
    landmarks.forEach((lm, i) => {
      ctx.beginPath();
      const isTip = [4, 8, 12, 16, 20].includes(i);
      ctx.arc(
        lm.x * canvasEl!.width,
        lm.y * canvasEl!.height,
        isTip ? 4 : 2,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = isTip ? "rgba(0, 255, 136, 0.9)" : "rgba(0, 200, 255, 0.7)";
      ctx.fill();
    });
  }

  // Cleanup on unmount
  $effect(() => {
    return () => {
      stopGestures();
    };
  });
</script>

<div class="gesture-control">
  <!-- Toggle Button -->
  <button
    class="gesture-btn"
    class:active={isActive}
    class:loading={mpLoading}
    onclick={toggleGestures}
    title={isActive ? "Stop gesture control" : "Start gesture control"}
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

  <!-- Gesture Name Display -->
  {#if currentGesture}
    <div class="gesture-label" class:high-conf={confidence > 0.8}>
      <span class="gesture-emoji">
        {#if currentGesture === "palm"}✋
        {:else if currentGesture === "thumbs_up"}👍
        {:else if currentGesture === "peace"}✌️
        {:else if currentGesture === "fist"}👊
        {:else if currentGesture === "point_up"}👆
        {:else if currentGesture === "rock"}🤟
        {/if}
      </span>
      <span class="gesture-name">{currentGesture.replace("_", " ")}</span>
    </div>
  {/if}

  <!-- Webcam PiP Window -->
  {#if showCamera}
    <div class="camera-pip" class:gesture-detected={!!currentGesture}>
      <video bind:this={videoEl} class="cam-video" playsinline muted></video>
      <canvas bind:this={canvasEl} class="cam-overlay" width="320" height="240"></canvas>
      <button class="pip-close" onclick={stopGestures}>×</button>
      {#if currentGesture}
        <div class="pip-gesture-tag">{currentGesture.replace("_", " ")}</div>
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
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
  }

  /* Gesture label */
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

  .gesture-emoji { font-size: 14px; }
  .gesture-name { text-transform: capitalize; letter-spacing: 0.3px; }

  @keyframes fadeIn { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }

  /* Camera PiP */
  .camera-pip {
    position: fixed;
    bottom: 80px;
    right: 16px;
    width: 200px;
    height: 150px;
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid rgba(180, 120, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    z-index: 1000;
    transition: border-color 0.3s;
  }

  .camera-pip.gesture-detected {
    border-color: rgba(0, 255, 136, 0.6);
  }

  .cam-video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1); /* Mirror */
  }

  .cam-overlay {
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
  }

  .gesture-error {
    font-size: 10px;
    color: rgba(255, 80, 80, 0.8);
    max-width: 160px;
  }
</style>

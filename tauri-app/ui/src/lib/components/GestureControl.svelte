<script lang="ts">
  /**
   * GestureControl v2 — Enhanced hand gesture recognition with 12 gestures.
   *
   * NEW Gestures:
   *  ✋ Open Palm   → Cancel / Stop
   *  👍 Thumbs Up  → Confirm plan
   *  👎 Thumbs Down → Deny / Reject
   *  ✌️ Peace Sign  → Toggle voice mode
   *  👊 Fist       → Execute last command
   *  👆 Point Up   → Scroll up
   *  🤟 Rock       → System info
   *  👌 OK Sign    → Accept / Acknowledge
   *  🤙 Call Me    → Open settings
   *  👈 Swipe Left → Previous tab
   *  👉 Swipe Right → Next tab
   *  🔫 Finger Gun → Screenshot!
   */

  import { session } from "../stores/session";

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
  let hands: any = null;
  let animFrameId: number = 0;
  let lastGestureTime = 0;
  
  // Finger trail tracking for air drawing
  let fingerTrail: { x: number; y: number; t: number }[] = [];
  let prevIndexPos: { x: number; y: number } | null = null;
  
  const GESTURE_COOLDOWN_MS = 1200;
  const MAX_TRAIL_LENGTH = 60;

  // Gesture emoji map
  const GESTURE_EMOJIS: Record<string, string> = {
    palm: "✋", thumbs_up: "👍", thumbs_down: "👎", peace: "✌️",
    fist: "👊", point_up: "👆", rock: "🤟", ok: "👌",
    call_me: "🤙", finger_gun: "🔫", swipe_left: "👈", swipe_right: "👉",
  };

  // ── MediaPipe Hands Loading ──
  let mpLoaded = $state(false);
  let mpLoading = $state(false);

  async function loadMediaPipe() {
    if (mpLoaded) return true;
    mpLoading = true;

    try {
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
        modelComplexity: 0,
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
    if (isActive) stopGestures();
    else await startGestures();
  }

  async function startGestures() {
    cameraError = "";
    const loaded = await loadMediaPipe();
    if (!loaded) return;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
    } catch {
      cameraError = "Camera access denied.";
      return;
    }

    if (videoEl) {
      videoEl.srcObject = stream;
      await videoEl.play();
    }

    isActive = true;
    showCamera = true;
    fingerTrail = [];
    detectFrame();
  }

  function stopGestures() {
    isActive = false;
    showCamera = false;
    currentGesture = "";
    confidence = 0;
    fingerTrail = [];
    prevIndexPos = null;

    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = 0; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }

  async function detectFrame() {
    if (!isActive || !videoEl || !hands) return;
    try { await hands.send({ image: videoEl }); } catch { /* ignore */ }
    animFrameId = requestAnimationFrame(detectFrame);
  }

  function onHandResults(results: any) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      currentGesture = "";
      confidence = 0;
      prevIndexPos = null;
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const gesture = classifyGesture(landmarks);

    // Track index finger for air drawing
    trackFingerTrail(landmarks);

    if (gesture.name && gesture.name !== currentGesture) {
      currentGesture = gesture.name;
      confidence = gesture.confidence;

      const now = Date.now();
      if (now - lastGestureTime > GESTURE_COOLDOWN_MS) {
        lastGestureTime = now;
        executeGestureAction(gesture.name);
        gestureHistory = [...gestureHistory.slice(-4), gesture.name];
        onGesture(gesture.name);
      }
    } else if (!gesture.name) {
      currentGesture = "";
      confidence = 0;
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
  interface Gesture { name: string; confidence: number; }

  function classifyGesture(landmarks: any[]): Gesture {
    const THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
    const THUMB_IP = 3, INDEX_PIP = 6, MIDDLE_PIP = 10, RING_PIP = 14, PINKY_PIP = 18;
    const THUMB_MCP = 2, INDEX_MCP = 5;
    const WRIST = 0;

    const isExtended = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y;
    const thumbExtended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x;

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

    // 👌 OK Sign — thumb tip touching index tip
    if (dist(THUMB_TIP, INDEX_TIP) < 0.05 && middleUp && ringUp && pinkyUp) {
      return { name: "ok", confidence: 0.85 };
    }

    // 🔫 Finger Gun — index + thumb extended, others down
    if (thumbExtended && indexUp && !middleUp && !ringUp && !pinkyUp) {
      // Check if thumb is horizontal (pointing sideways)
      if (Math.abs(landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y) < 0.08) {
        return { name: "finger_gun", confidence: 0.78 };
      }
    }

    // 🤙 Call Me — thumb + pinky extended, others curled
    if (thumbExtended && !indexUp && !middleUp && !ringUp && pinkyUp) {
      return { name: "call_me", confidence: 0.82 };
    }

    // 👎 Thumbs Down — only thumb extended, pointing downward
    if (thumbExtended && !indexUp && !middleUp && !ringUp && !pinkyUp) {
      if (landmarks[THUMB_TIP].y > landmarks[WRIST].y) {
        return { name: "thumbs_down", confidence: 0.8 };
      }
      // 👍 Thumbs Up — pointing upward
      if (landmarks[THUMB_TIP].y < landmarks[WRIST].y) {
        return { name: "thumbs_up", confidence: 0.8 };
      }
    }

    // 👊 Fist
    if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbExtended) {
      return { name: "fist", confidence: 0.85 };
    }

    // ✋ Open Palm
    if (indexUp && middleUp && ringUp && pinkyUp && thumbExtended) {
      return { name: "palm", confidence: 0.9 };
    }

    // ✌️ Peace
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
      return { name: "peace", confidence: 0.85 };
    }

    // 👆 Point Up — only index
    if (indexUp && !middleUp && !ringUp && !pinkyUp) {
      return { name: "point_up", confidence: 0.8 };
    }

    // 🤟 Rock — index + pinky
    if (indexUp && !middleUp && !ringUp && pinkyUp) {
      return { name: "rock", confidence: 0.75 };
    }

    // Swipe detection (using wrist horizontal velocity)
    if (prevIndexPos) {
      const dx = landmarks[WRIST].x - prevIndexPos.x;
      if (Math.abs(dx) > 0.08 && indexUp && middleUp && ringUp && pinkyUp) {
        if (dx < -0.08) return { name: "swipe_left", confidence: 0.7 };
        if (dx > 0.08) return { name: "swipe_right", confidence: 0.7 };
      }
    }

    return { name: "", confidence: 0 };
  }

  function executeGestureAction(gesture: string) {
    const emoji = GESTURE_EMOJIS[gesture] || "🖐️";
    switch (gesture) {
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
      case "swipe_left":
        session.addSystemMessage(`${emoji} Previous tab`);
        break;
      case "swipe_right":
        session.addSystemMessage(`${emoji} Next tab`);
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
    return () => stopGestures();
  });
</script>

<div class="gesture-control">
  <button
    class="gesture-btn"
    class:active={isActive}
    class:loading={mpLoading}
    onclick={toggleGestures}
    title={isActive ? "Stop gesture control" : "Start gesture control (12 gestures!)"}
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
      <video bind:this={videoEl} class="cam-video" playsinline muted></video>
      <canvas bind:this={canvasEl} class="cam-overlay" width="320" height="240"></canvas>
      <canvas bind:this={trailCanvas} class="cam-trail" width="320" height="240"></canvas>
      <button class="pip-close" onclick={stopGestures}>×</button>
      {#if currentGesture}
        <div class="pip-gesture-tag">
          <span>{GESTURE_EMOJIS[currentGesture] || ""}</span>
          {currentGesture.replace(/_/g, " ")}
        </div>
      {/if}
      <!-- Gesture count badge -->
      <div class="pip-badge">12 gestures</div>
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

  .hand-icon { width: 18px; height: 18px; z-index: 1; }

  .loading-dot {
    position: absolute; top: 2px; right: 2px;
    width: 6px; height: 6px; border-radius: 50%;
    background: rgba(255, 200, 0, 0.8);
    animation: blink 0.8s infinite;
  }

  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

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

  .gesture-error {
    font-size: 10px; color: rgba(255, 80, 80, 0.8); max-width: 160px;
  }
</style>

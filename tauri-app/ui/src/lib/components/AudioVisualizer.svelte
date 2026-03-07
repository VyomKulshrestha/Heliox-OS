<script lang="ts">
  /**
   * AudioVisualizer — Real-time audio waveform / frequency visualization.
   * Shows a circular JARVIS-style audio spectrum while listening.
   */

  let { active = false }: { active: boolean } = $props();

  let canvasEl: HTMLCanvasElement | undefined = $state();
  let animId = 0;
  let analyser: AnalyserNode | null = null;
  let audioCtx: AudioContext | null = null;
  let dataArray: Uint8Array | null = null;
  let streamRef: MediaStream | null = null;

  async function startAudio() {
    try {
      audioCtx = new AudioContext();
      streamRef = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioCtx.createMediaStreamSource(streamRef);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      dataArray = new Uint8Array(analyser.frequencyBinCount);
      draw();
    } catch (e) {
      console.warn("Audio visualizer failed:", e);
      drawIdle();
    }
  }

  function stopAudio() {
    if (streamRef) {
      streamRef.getTracks().forEach(t => t.stop());
      streamRef = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    analyser = null;
    dataArray = null;
    cancelAnimationFrame(animId);
  }

  function draw() {
    if (!canvasEl || !analyser || !dataArray) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    analyser.getByteFrequencyData(dataArray);

    const w = canvasEl.width;
    const h = canvasEl.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(cx, cy) - 4;
    const minR = maxR * 0.4;

    ctx.clearRect(0, 0, w, h);

    const bars = dataArray.length;

    // Circular frequency bars
    for (let i = 0; i < bars; i++) {
      const angle = (Math.PI * 2 * i) / bars - Math.PI / 2;
      const val = dataArray[i] / 255;
      const barLen = minR + val * (maxR - minR);

      const x1 = cx + Math.cos(angle) * minR;
      const y1 = cy + Math.sin(angle) * minR;
      const x2 = cx + Math.cos(angle) * barLen;
      const y2 = cy + Math.sin(angle) * barLen;

      const hue = 190 + val * 40;
      ctx.strokeStyle = `hsla(${hue}, 100%, ${50 + val * 20}%, ${0.4 + val * 0.5})`;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Mirror on opposite side
      const mirrorAngle = angle + Math.PI;
      const mx1 = cx + Math.cos(mirrorAngle) * minR;
      const my1 = cy + Math.sin(mirrorAngle) * minR;
      const mx2 = cx + Math.cos(mirrorAngle) * barLen;
      const my2 = cy + Math.sin(mirrorAngle) * barLen;
      ctx.beginPath();
      ctx.moveTo(mx1, my1);
      ctx.lineTo(mx2, my2);
      ctx.stroke();
    }

    // Inner circle glow
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, minR);
    const avgVol = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
    glow.addColorStop(0, `rgba(0, 200, 255, ${0.05 + avgVol * 0.15})`);
    glow.addColorStop(1, "rgba(0, 200, 255, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, minR, 0, Math.PI * 2);
    ctx.fill();

    // Inner ring
    ctx.strokeStyle = `rgba(0, 200, 255, ${0.2 + avgVol * 0.3})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, minR, 0, Math.PI * 2);
    ctx.stroke();

    animId = requestAnimationFrame(draw);
  }

  function drawIdle() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    const w = canvasEl.width;
    const h = canvasEl.height;
    const cx = w / 2, cy = h / 2;

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(0, 200, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 20, 0, Math.PI * 2);
    ctx.stroke();
  }

  $effect(() => {
    if (active) {
      startAudio();
    } else {
      stopAudio();
      if (canvasEl) drawIdle();
    }
    return () => stopAudio();
  });
</script>

{#if active}
  <div class="audio-viz-container">
    <canvas bind:this={canvasEl} class="audio-viz" width="80" height="80"></canvas>
  </div>
{/if}

<style>
  .audio-viz-container {
    position: absolute;
    top: -90px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 100;
    pointer-events: none;
  }

  .audio-viz {
    width: 80px;
    height: 80px;
    filter: drop-shadow(0 0 10px rgba(0, 200, 255, 0.2));
  }
</style>

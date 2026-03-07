<script lang="ts">
  /**
   * ArcReactor — Iron Man Arc Reactor animated logo.
   * Pulses with system activity, spins during loading,
   * and emits particles on command completion.
   */

  import { session } from "../stores/session";

  let isLoading = $derived($session.loading);
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let animId = 0;
  let rotation = 0;
  let particlesArr: Particle[] = [];
  let glowIntensity = 0;
  let targetGlow = 0.4;

  interface Particle {
    x: number; y: number;
    vx: number; vy: number;
    life: number; maxLife: number;
    size: number; hue: number;
  }

  function spawnParticles(count: number = 12) {
    const cx = 20, cy = 20;
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.5;
      const speed = 1 + Math.random() * 2;
      particlesArr.push({
        x: cx, y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1, maxLife: 30 + Math.random() * 20,
        size: 1 + Math.random() * 2,
        hue: 190 + Math.random() * 30,
      });
    }
  }

  function draw() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    const w = canvasEl.width;
    const h = canvasEl.height;
    const cx = w / 2, cy = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Smooth glow transition
    glowIntensity += (targetGlow - glowIntensity) * 0.08;

    const spinRate = isLoading ? 0.04 : 0.008;
    rotation += spinRate;

    // Outer glow
    const outerGrad = ctx.createRadialGradient(cx, cy, 4, cx, cy, 18);
    outerGrad.addColorStop(0, `rgba(0, 200, 255, ${0.15 + glowIntensity * 0.3})`);
    outerGrad.addColorStop(0.6, `rgba(0, 200, 255, ${0.03 + glowIntensity * 0.08})`);
    outerGrad.addColorStop(1, "rgba(0, 200, 255, 0)");
    ctx.fillStyle = outerGrad;
    ctx.fillRect(0, 0, w, h);

    // ── Outer ring ──
    ctx.strokeStyle = `rgba(0, 200, 255, ${0.3 + glowIntensity * 0.4})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.stroke();

    // ── Segmented spinning ring ──
    const segments = 8;
    for (let i = 0; i < segments; i++) {
      const startAngle = rotation + (Math.PI * 2 * i) / segments;
      const endAngle = startAngle + (Math.PI * 2) / (segments * 1.8);
      ctx.strokeStyle = `rgba(0, 200, 255, ${0.4 + glowIntensity * 0.4})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 11, startAngle, endAngle);
      ctx.stroke();
    }

    // ── Inner ring ──
    ctx.strokeStyle = `rgba(0, 180, 255, ${0.5 + glowIntensity * 0.3})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.stroke();

    // ── Inner spokes ──
    for (let i = 0; i < 3; i++) {
      const angle = -rotation * 1.5 + (Math.PI * 2 * i) / 3;
      ctx.strokeStyle = `rgba(0, 200, 255, ${0.3 + glowIntensity * 0.3})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * 3, cy + Math.sin(angle) * 3);
      ctx.lineTo(cx + Math.cos(angle) * 7, cy + Math.sin(angle) * 7);
      ctx.stroke();
    }

    // ── Core ──
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 4);
    coreGrad.addColorStop(0, `rgba(200, 240, 255, ${0.8 + glowIntensity * 0.2})`);
    coreGrad.addColorStop(0.6, `rgba(0, 200, 255, ${0.5 + glowIntensity * 0.3})`);
    coreGrad.addColorStop(1, "rgba(0, 100, 200, 0)");
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();

    // ── Particles ──
    particlesArr = particlesArr.filter(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.97;
      p.vy *= 0.97;
      p.life++;
      const alpha = 1 - p.life / p.maxLife;
      if (alpha <= 0) return false;

      ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${alpha * 0.8})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * alpha, 0, Math.PI * 2);
      ctx.fill();
      return true;
    });

    animId = requestAnimationFrame(draw);
  }

  // React to loading state changes
  $effect(() => {
    targetGlow = isLoading ? 1.0 : 0.4;
  });

  // Spawn particles on new result messages
  let prevMsgCount = 0;
  $effect(() => {
    const msgs = $session.messages;
    if (msgs.length > prevMsgCount) {
      const last = msgs[msgs.length - 1];
      if (last.type === "result") {
        spawnParticles(16);
        targetGlow = 1.0;
        setTimeout(() => { targetGlow = 0.4; }, 800);
      }
    }
    prevMsgCount = msgs.length;
  });

  $effect(() => {
    if (canvasEl) {
      draw();
    }
    return () => { cancelAnimationFrame(animId); };
  });
</script>

<div class="arc-reactor-wrap">
  <canvas bind:this={canvasEl} class="arc-reactor" width="40" height="40"></canvas>
</div>

<style>
  .arc-reactor-wrap {
    width: 28px;
    height: 28px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .arc-reactor {
    width: 28px;
    height: 28px;
    filter: drop-shadow(0 0 6px rgba(0, 200, 255, 0.3));
  }
</style>

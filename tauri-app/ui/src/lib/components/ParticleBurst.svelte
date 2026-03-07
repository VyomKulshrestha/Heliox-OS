<script lang="ts">
  /**
   * ParticleBurst — Explosive particle effects triggered by events.
   * Used for gesture detection, command completion, and confirmations.
   */

  let canvasEl: HTMLCanvasElement | undefined = $state();
  let animId = 0;

  interface Particle {
    x: number; y: number;
    vx: number; vy: number;
    life: number; maxLife: number;
    size: number; hue: number; sat: number;
    type: "spark" | "ring" | "trail";
  }

  let particles: Particle[] = [];

  export function burst(x: number, y: number, config: {
    count?: number;
    hue?: number;
    spread?: number;
    size?: number;
    type?: "spark" | "ring" | "trail";
  } = {}) {
    const {
      count = 20,
      hue = 190,
      spread = 4,
      size = 2,
      type = "spark"
    } = config;

    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.5;
      const speed = spread * (0.3 + Math.random() * 0.7);
      particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 1, // Slight upward bias
        life: 0,
        maxLife: 30 + Math.random() * 30,
        size: size * (0.5 + Math.random()),
        hue: hue + (Math.random() - 0.5) * 40,
        sat: 80 + Math.random() * 20,
        type,
      });
    }
  }

  // Preset bursts
  export function confirmBurst() {
    if (!canvasEl) return;
    const cx = canvasEl.width / 2;
    const cy = canvasEl.height / 2;
    burst(cx, cy, { count: 30, hue: 140, spread: 6, size: 3 }); // Green
  }

  export function errorBurst() {
    if (!canvasEl) return;
    const cx = canvasEl.width / 2;
    const cy = canvasEl.height / 2;
    burst(cx, cy, { count: 15, hue: 0, spread: 3, size: 2 }); // Red
  }

  export function gestureBurst(gesture: string) {
    if (!canvasEl) return;
    const cx = canvasEl.width / 2;
    const cy = canvasEl.height / 2;
    const hueMap: Record<string, number> = {
      palm: 50,        // Gold
      thumbs_up: 140,  // Green
      peace: 270,      // Purple
      fist: 20,        // Orange
      point_up: 190,   // Cyan
      rock: 300,       // Pink
      ok: 160,         // Teal
      finger_gun: 30,  // Orange-gold
      thumbs_down: 0,  // Red
      swipe_left: 210, // Blue
      swipe_right: 210, // Blue
      call_me: 280,    // Violet
    };
    burst(cx, cy, { count: 25, hue: hueMap[gesture] ?? 190, spread: 5, size: 2.5 });
  }

  function draw() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    const w = canvasEl.width;
    const h = canvasEl.height;
    ctx.clearRect(0, 0, w, h);

    particles = particles.filter(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.05; // Gravity
      p.vx *= 0.98;
      p.vy *= 0.98;
      p.life++;

      const progress = p.life / p.maxLife;
      if (progress >= 1) return false;

      const alpha = 1 - progress;
      const currentSize = p.size * (1 - progress * 0.5);

      if (p.type === "spark") {
        // Glowing dot
        ctx.fillStyle = `hsla(${p.hue}, ${p.sat}%, 65%, ${alpha})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = `hsla(${p.hue}, ${p.sat}%, 50%, ${alpha * 0.5})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, currentSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      } else if (p.type === "ring") {
        // Expanding ring
        const ringSize = currentSize + progress * 20;
        ctx.strokeStyle = `hsla(${p.hue}, ${p.sat}%, 65%, ${alpha * 0.5})`;
        ctx.lineWidth = 1.5 * alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, ringSize, 0, Math.PI * 2);
        ctx.stroke();
      } else if (p.type === "trail") {
        // Line trail
        ctx.strokeStyle = `hsla(${p.hue}, ${p.sat}%, 65%, ${alpha * 0.6})`;
        ctx.lineWidth = currentSize;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.vx * 3, p.y - p.vy * 3);
        ctx.stroke();
      }

      return true;
    });

    if (particles.length > 0) {
      animId = requestAnimationFrame(draw);
    } else {
      animId = requestAnimationFrame(draw); // Keep alive
    }
  }

  function handleResize() {
    if (!canvasEl) return;
    const parent = canvasEl.parentElement;
    if (!parent) return;
    canvasEl.width = parent.clientWidth;
    canvasEl.height = parent.clientHeight;
  }

  $effect(() => {
    if (canvasEl) {
      handleResize();
      draw();
      window.addEventListener("resize", handleResize);
    }
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
    };
  });
</script>

<canvas bind:this={canvasEl} class="particle-canvas"></canvas>

<style>
  .particle-canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9999;
  }
</style>

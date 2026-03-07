<script lang="ts">
  /**
   * NeuralBackground — Animated neural network / data flow background.
   * Nodes float and connect with glowing lines, creating a living
   * JARVIS-style holographic backdrop.
   */

  let canvasEl: HTMLCanvasElement | undefined = $state();
  let animId = 0;

  interface Node {
    x: number; y: number;
    vx: number; vy: number;
    radius: number;
    pulse: number; pulseSpeed: number;
  }

  const NODE_COUNT = 35;
  const CONNECTION_DIST = 120;
  let nodes: Node[] = [];
  let mouseX = -100, mouseY = -100;

  function initNodes(w: number, h: number) {
    nodes = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: 1 + Math.random() * 1.5,
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.02 + Math.random() * 0.02,
      });
    }
  }

  function draw() {
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    const w = canvasEl.width;
    const h = canvasEl.height;
    ctx.clearRect(0, 0, w, h);

    // Update nodes
    for (const node of nodes) {
      node.x += node.vx;
      node.y += node.vy;
      node.pulse += node.pulseSpeed;

      // Bounce off edges
      if (node.x < 0 || node.x > w) node.vx *= -1;
      if (node.y < 0 || node.y > h) node.vy *= -1;
      node.x = Math.max(0, Math.min(w, node.x));
      node.y = Math.max(0, Math.min(h, node.y));

      // Mouse repulsion
      const dx = node.x - mouseX;
      const dy = node.y - mouseY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 80 && dist > 0) {
        const force = (80 - dist) / 80 * 0.5;
        node.vx += (dx / dist) * force;
        node.vy += (dy / dist) * force;
      }

      // Damping
      node.vx *= 0.99;
      node.vy *= 0.99;
    }

    // Draw connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECTION_DIST) {
          const alpha = (1 - dist / CONNECTION_DIST) * 0.12;
          ctx.strokeStyle = `rgba(0, 200, 255, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();

          // Traveling data pulse on some connections
          if (Math.random() < 0.002) {
            const t = (Date.now() % 2000) / 2000;
            const px = nodes[i].x + (nodes[j].x - nodes[i].x) * t;
            const py = nodes[i].y + (nodes[j].y - nodes[i].y) * t;
            ctx.fillStyle = `rgba(0, 255, 200, ${0.6 * (1 - t)})`;
            ctx.beginPath();
            ctx.arc(px, py, 1.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const pulseSize = Math.sin(node.pulse) * 0.5 + 0.5;
      const r = node.radius + pulseSize * 0.8;

      // Glow
      const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 3);
      glow.addColorStop(0, `rgba(0, 200, 255, ${0.08 + pulseSize * 0.04})`);
      glow.addColorStop(1, "rgba(0, 200, 255, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 3, 0, Math.PI * 2);
      ctx.fill();

      // Core
      ctx.fillStyle = `rgba(0, 200, 255, ${0.3 + pulseSize * 0.3})`;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hex grid overlay (very subtle)
    drawHexGrid(ctx, w, h);

    animId = requestAnimationFrame(draw);
  }

  function drawHexGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
    const size = 40;
    const sqrt3 = Math.sqrt(3);
    ctx.strokeStyle = "rgba(0, 200, 255, 0.015)";
    ctx.lineWidth = 0.5;

    for (let row = -1; row < h / (size * 1.5) + 1; row++) {
      for (let col = -1; col < w / (size * sqrt3) + 1; col++) {
        const x = col * size * sqrt3 + (row % 2 === 0 ? 0 : size * sqrt3 * 0.5);
        const y = row * size * 1.5;
        drawHex(ctx, x, y, size);
      }
    }
  }

  function drawHex(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      const hx = x + size * Math.cos(angle);
      const hy = y + size * Math.sin(angle);
      if (i === 0) ctx.moveTo(hx, hy);
      else ctx.lineTo(hx, hy);
    }
    ctx.closePath();
    ctx.stroke();
  }

  function handleResize() {
    if (!canvasEl) return;
    const parent = canvasEl.parentElement;
    if (!parent) return;
    canvasEl.width = parent.clientWidth;
    canvasEl.height = parent.clientHeight;
    if (nodes.length === 0) initNodes(canvasEl.width, canvasEl.height);
  }

  function handleMouseMove(e: MouseEvent) {
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
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

<div class="neural-bg" onmousemove={handleMouseMove}>
  <canvas bind:this={canvasEl} class="neural-canvas"></canvas>
</div>

<style>
  .neural-bg {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
  }

  .neural-bg:hover {
    pointer-events: auto;
  }

  .neural-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>

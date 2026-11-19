<script lang="ts">
  // The talking face (Settings, Appearance, Face): one face pack drawn on a canvas, its mouth moved by the daemon's
  // own timing for each spoken piece (voice_piece, voice_end, voice_cut; IPC_MESSAGE_FORMATS.md). Loaded in its own
  // chunk by FaceHost only while a face is chosen. It draws only while the window and the conversation are visible:
  // at most 30 frames a second while Heliox speaks, 8 while idle, and nothing in between under reduced motion
  // (a still face whose mouth only opens and closes).
  import { onDestroy, onMount } from "svelte";
  import { onConnectionState, offConnectionState, onNotification, offNotification } from "../api/daemon";
  import { session } from "../stores/session";
  import { MouthPlayer, type VoicePieceNotice } from "../face/mouthPlayer";
  import { FaceRenderer, type FaceMode } from "../face/faceRenderer";
  import { packShapes, readManifest, type FaceManifest } from "../face/pack";

  let { pack }: { pack: string } = $props();

  const SPEAKING_MS = 33;
  const IDLE_MS = 125;
  const REDUCED_SPEAKING_MS = 125;
  const THINKING = new Set(["understanding", "planning", "awaiting_approval", "acting", "verifying", "correcting"]);

  let canvas: HTMLCanvasElement | undefined = $state();
  let card: HTMLDivElement | undefined = $state();
  let problem = $state("");
  let manifest: FaceManifest | null = $state(null);
  let shown: FaceMode = $state("idle");

  const player = new MouthPlayer();
  let renderer: FaceRenderer | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let onScreen = true;
  let pageVisible = typeof document === "undefined" || document.visibilityState === "visible";
  let reducedMotion = false;
  let destroyed = false;

  const mode: FaceMode = $derived.by(() => {
    const phase = $session.interaction?.phase ?? "";
    if (phase === "listening") return "listening";
    if ($session.loading || THINKING.has(phase)) return "thinking";
    return "idle";
  });

  function visible(): boolean {
    return onScreen && pageVisible && !destroyed;
  }

  function stop() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  }

  function tick() {
    timer = null;
    if (!renderer || !canvas || !visible()) return;
    const now = performance.now();
    const busy = player.busy(now);
    const frame = player.sample(now, reducedMotion);
    const current: FaceMode = frame.speaking ? "speaking" : mode;
    if (current !== shown) shown = current;
    renderer.draw(now, current, frame);
    let delay: number | null;
    if (busy) delay = reducedMotion ? REDUCED_SPEAKING_MS : SPEAKING_MS;
    else if (renderer.animating(now, current)) delay = SPEAKING_MS;
    else if (reducedMotion)
      delay = null; // a still face: drawn again when something changes
    else delay = Math.max(16, Math.min(IDLE_MS, renderer.untilBlink(now)));
    if (delay !== null) timer = setTimeout(tick, delay);
  }

  function wake() {
    stop();
    tick();
  }

  function onSpeech(method: string, params: unknown) {
    const p = (params ?? {}) as Record<string, unknown>;
    if (method === "voice_piece") {
      player.piece(p as unknown as VoicePieceNotice, performance.now(), Date.now());
      wake();
    } else if (method === "voice_end") {
      player.end(Number(p.reply));
    } else if (method === "voice_cut") {
      player.cut(typeof p.reply === "number" ? p.reply : undefined);
      wake(); // closed on this frame, not the next scheduled one
    }
  }

  function onConnection(connected: boolean) {
    if (!connected) {
      player.reset();
      wake();
    }
  }

  function readToken(token: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(token).trim();
  }

  function sizeCanvas() {
    if (!canvas || !card || !manifest) return;
    const css = card.clientWidth || 112;
    const ratio = Math.min(3, window.devicePixelRatio || 1);
    const width = Math.round(css * ratio);
    const height = Math.round((width * manifest.size[1]) / manifest.size[0]);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  $effect(() => {
    void mode; // a change of state is drawn at once
    if (renderer) wake();
  });

  onMount(() => {
    onNotification(onSpeech);
    onConnectionState(onConnection);
    const motion = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    reducedMotion = Boolean(motion?.matches);
    const onMotion = () => {
      reducedMotion = Boolean(motion?.matches);
      if (renderer) renderer.reducedMotion = reducedMotion;
      wake();
    };
    motion?.addEventListener?.("change", onMotion);
    const onVisibility = () => {
      pageVisible = document.visibilityState === "visible";
      if (pageVisible) wake();
      else stop();
    };
    document.addEventListener("visibilitychange", onVisibility);
    let observer: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== "undefined" && card) {
      observer = new IntersectionObserver((entries) => {
        onScreen = entries.some((entry) => entry.isIntersecting);
        if (onScreen) wake();
        else stop();
      });
      observer.observe(card);
    }
    // The theme changes the colours a drawn face is tinted with.
    const theme = new MutationObserver(() => {
      renderer?.setColours(readToken);
      wake();
    });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style", "data-theme"] });

    void (async () => {
      try {
        const loaded = await readManifest(pack);
        if (destroyed || !canvas) return;
        manifest = loaded;
        player.setShapes(packShapes(loaded)); // a shape without a picture is drawn calmly
        sizeCanvas();
        const made = new FaceRenderer(canvas, loaded);
        made.reducedMotion = reducedMotion;
        await made.load();
        if (destroyed) return;
        made.setColours(readToken);
        renderer = made;
        wake();
      } catch (error) {
        if (!destroyed) problem = error instanceof Error ? error.message : "This face could not be shown.";
      }
    })();

    return () => {
      offNotification(onSpeech);
      offConnectionState(onConnection);
      motion?.removeEventListener?.("change", onMotion);
      document.removeEventListener("visibilitychange", onVisibility);
      observer?.disconnect();
      theme.disconnect();
    };
  });

  onDestroy(() => {
    destroyed = true;
    stop();
    renderer = null;
  });
</script>

<div
  class="face-card"
  class:listening={shown === "listening"}
  class:speaking={shown === "speaking"}
  data-testid="face"
  data-face-state={shown}
  bind:this={card}
>
  <canvas bind:this={canvas} aria-hidden="true"></canvas>
  {#if problem}
    <p class="face-problem" role="status">{problem}</p>
  {/if}
</div>

<style>
  .face-card {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--bg-elevated);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    transition: border-color var(--dur) var(--ease-out);
  }

  .face-card.listening {
    border-color: var(--accent-line);
  }

  .face-card.speaking {
    border-color: var(--border-strong);
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  .face-problem {
    position: absolute;
    inset: auto 8px 8px;
    margin: 0;
    font-size: var(--text-2xs);
    color: var(--text-muted);
    line-height: 1.3;
  }
</style>

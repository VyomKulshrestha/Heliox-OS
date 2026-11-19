/**
 * Draws one face pack on a 2D canvas (no WebGL: WebView2, WKWebView and WebKitGTK all have it). The face component
 * decides when to draw; this file decides what a frame looks like: the idle frame (with breathing when the pack asks
 * for it), the eyes (open, a blink, or the listening and thinking eyes), the mouth (one shape, blended into the next),
 * and a small turn of the head for listening and thinking. Packs marked with a tint are drawn in the window's own
 * colour tokens, so a drawn face follows the light and dark themes.
 */

import type { MouthFrame } from "./mouthPlayer";
import { imageForShape, packUrl, type FaceManifest, type FaceState } from "./pack";

export type FaceMode = "idle" | "listening" | "thinking" | "speaking";

type Layer = "idle" | "eyes" | "mouth";

export class FaceRenderer {
  private images = new Map<string, HTMLImageElement>();
  private tinted = new Map<string, HTMLCanvasElement>();
  private colours: Record<string, string> = {};
  private tilt = 0;
  private lift = 0;
  private lastDraw = 0;
  private blinkAt = 0;
  private blinkUntil = 0;
  reducedMotion = false;

  constructor(
    private canvas: HTMLCanvasElement,
    private manifest: FaceManifest,
    private random: () => number = Math.random,
  ) {}

  /** Every file the pack names, loaded once. Rejects when one cannot be loaded. */
  async load(): Promise<void> {
    const m = this.manifest;
    const files = new Set<string>([
      ...m.idle.frames,
      ...Object.values(m.mouth.shapes).filter((f): f is string => typeof f === "string"),
      ...[m.eyes?.open, m.eyes?.listening, m.eyes?.thinking, ...(m.eyes?.blink ?? [])].filter(
        (f): f is string => typeof f === "string",
      ),
    ]);
    await Promise.all(
      [...files].map(
        (file) =>
          new Promise<void>((resolve, reject) => {
            const image = new Image();
            image.decoding = "async";
            image.onload = () => {
              this.images.set(file, image);
              resolve();
            };
            image.onerror = () => reject(new Error(`The face pack's ${file} could not be loaded.`));
            image.src = packUrl(m.id, file);
          }),
      ),
    );
  }

  /** The colours of the window's tokens the pack is tinted with (read again when the theme changes). */
  setColours(read: (token: string) => string): void {
    const next: Record<string, string> = {};
    for (const token of Object.values(this.manifest.tint ?? {})) {
      if (token) next[token] = read(token) || "#888";
    }
    if (JSON.stringify(next) !== JSON.stringify(this.colours)) {
      this.colours = next;
      this.tinted.clear();
    }
  }

  /** Something is moving on its own (a blink, the head turning): draw at the speaking rate until it settles. */
  animating(now: number, mode: FaceMode): boolean {
    if (this.reducedMotion) return false;
    const target = this.stateFor(mode);
    const settled = Math.abs((target.tilt ?? 0) - this.tilt) < 0.05 && Math.abs((target.lift ?? 0) - this.lift) < 0.05;
    return !settled || (now >= this.blinkAt && now < this.blinkUntil);
  }

  /** Milliseconds until the next blink starts (Infinity when the pack does not blink or motion is reduced). */
  untilBlink(now: number): number {
    if (this.reducedMotion || !this.manifest.eyes?.blink?.length || !this.blinkAt) return Infinity;
    return Math.max(0, this.blinkAt - now);
  }

  draw(now: number, mode: FaceMode, mouth: MouthFrame): void {
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    const m = this.manifest;
    const [width, height] = m.size;
    const scale = this.canvas.width / width;
    const dt = this.lastDraw ? Math.min(250, now - this.lastDraw) : 250;
    this.lastDraw = now;
    const target = this.stateFor(mode);
    if (this.reducedMotion) {
      this.tilt = 0;
      this.lift = 0;
    } else {
      const ease = Math.min(1, dt / 160);
      this.tilt += ((target.tilt ?? 0) - this.tilt) * ease;
      this.lift += ((target.lift ?? 0) - this.lift) * ease;
    }
    const breathing =
      this.reducedMotion || !m.breathe
        ? 1
        : 1 + m.breathe.amount * Math.sin((now / 1000 / Math.max(1, m.breathe.seconds)) * Math.PI * 2);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    // The head turns about the neck, breathing about the same point.
    const pivotX = width / 2;
    const pivotY = height * 0.9;
    ctx.translate(pivotX, pivotY + this.lift);
    ctx.rotate((this.tilt * Math.PI) / 180);
    ctx.scale(breathing * (target.scale ?? 1), breathing * (target.scale ?? 1));
    ctx.translate(-pivotX, -pivotY);

    const frameIndex = this.idleFrame(now);
    this.layer(ctx, "idle", m.idle.frames[frameIndex], [0, 0, width, height], 1);
    this.layer(ctx, "eyes", this.eyesFor(now, mode), m.eyes?.rect ?? [0, 0, width, height], 1);
    const offset = m.idle.mouth_offsets?.[frameIndex] ?? [0, 0];
    const rect = m.mouth.rect ?? [0, 0, width, height];
    const at: [number, number, number, number] = [rect[0] + offset[0], rect[1] + offset[1], rect[2], rect[3]];
    this.layer(ctx, "mouth", imageForShape(m, mouth.shape), at, 1);
    if (mouth.mix > 0.02 && mouth.next !== mouth.shape) {
      this.layer(ctx, "mouth", imageForShape(m, mouth.next), at, mouth.mix);
    }
  }

  private stateFor(mode: FaceMode): FaceState {
    const states = this.manifest.states ?? {};
    if (mode === "listening") return states.listening ?? {};
    if (mode === "thinking") return states.thinking ?? {};
    if (mode === "speaking") return states.speaking ?? {};
    return {};
  }

  private idleFrame(now: number): number {
    const frames = this.manifest.idle.frames.length;
    if (frames <= 1 || this.reducedMotion) return 0;
    const step = Math.floor((now / 1000) * (this.manifest.idle.fps ?? 12));
    if (this.manifest.idle.loop === "forward") return step % frames;
    const cycle = 2 * frames - 2; // forward then back, without repeating the ends
    const position = step % cycle;
    return position < frames ? position : cycle - position;
  }

  private eyesFor(now: number, mode: FaceMode): string | undefined {
    const eyes = this.manifest.eyes;
    if (!eyes) return undefined;
    const steady = (mode === "listening" && eyes.listening) || (mode === "thinking" && eyes.thinking) || eyes.open;
    const blink = eyes.blink ?? [];
    if (this.reducedMotion || !blink.length) return steady;
    if (!this.blinkAt) this.scheduleBlink(now);
    if (now >= this.blinkUntil && now >= this.blinkAt) this.scheduleBlink(now);
    if (now >= this.blinkAt && now < this.blinkUntil) {
      const frame = Math.floor(((now - this.blinkAt) / 1000) * (eyes.blink_fps ?? 24));
      return blink[Math.min(blink.length - 1, frame)];
    }
    return steady;
  }

  private scheduleBlink(now: number): void {
    const [low, high] = this.manifest.eyes?.blink_every ?? [2.5, 6];
    const frames = this.manifest.eyes?.blink?.length ?? 0;
    this.blinkAt = now + (low + (high - low) * this.random()) * 1000;
    this.blinkUntil = this.blinkAt + (frames / (this.manifest.eyes?.blink_fps ?? 24)) * 1000;
  }

  private layer(
    ctx: CanvasRenderingContext2D,
    layer: Layer,
    file: string | undefined,
    rect: [number, number, number, number],
    alpha: number,
  ): void {
    if (!file) return;
    const image = this.images.get(file);
    if (!image) return;
    const source = this.tintedImage(layer, file, image, rect);
    ctx.globalAlpha = alpha;
    ctx.drawImage(source, rect[0], rect[1], rect[2], rect[3]);
    ctx.globalAlpha = 1;
  }

  /** A drawn pack's layer in the token's colour, at the canvas's own resolution so lines stay sharp. */
  private tintedImage(
    layer: Layer,
    file: string,
    image: HTMLImageElement,
    rect: [number, number, number, number],
  ): CanvasImageSource {
    const token = this.manifest.tint?.[layer];
    if (!token) return image;
    const colour = this.colours[token] ?? "#888";
    const scale = this.canvas.width / this.manifest.size[0];
    const key = `${layer}|${file}|${colour}|${scale}`;
    let cached = this.tinted.get(key);
    if (!cached) {
      cached = document.createElement("canvas");
      cached.width = Math.max(1, Math.round(rect[2] * scale));
      cached.height = Math.max(1, Math.round(rect[3] * scale));
      const ctx = cached.getContext("2d");
      if (!ctx) return image;
      ctx.drawImage(image, 0, 0, cached.width, cached.height);
      ctx.globalCompositeOperation = "source-in";
      ctx.fillStyle = colour;
      ctx.fillRect(0, 0, cached.width, cached.height);
      this.tinted.set(key, cached);
    }
    return cached;
  }
}

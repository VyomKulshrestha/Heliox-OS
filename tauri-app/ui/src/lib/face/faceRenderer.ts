/**
 * Draws one face pack on a 2D canvas (no WebGL: WebView2, WKWebView and WebKitGTK all have it). The face component
 * decides when to draw; this file decides what a frame looks like: the idle frame (with breathing when the pack asks
 * for it), the eyes (open, a blink, a glance, or the listening and thinking eyes), the mouth (each shape formed part
 * of the way through its pictures, mouth_steps), and the head (a small turn for listening and thinking, and a nod and
 * sway with the stress of the speech). Packs marked with a tint are drawn in the window's own colour tokens, so a
 * drawn face follows the light and dark themes.
 *
 * Every motion eases: tilt, lift, scale, and the resting mouth a state names. A photograph fills its card, so it is
 * drawn a little larger than the card (overscan), by just enough that no pose the pack or the speech asks for shows
 * the card behind it.
 */

import type { MouthFrame } from "./mouthPlayer";
import { imageForShape, packUrl, shapeShown, STATE_NAMES, stepsOf, type FaceManifest, type FaceState } from "./pack";

export type FaceMode = "idle" | "listening" | "thinking" | "speaking";

type Layer = "idle" | "eyes" | "mouth";
type Rect = [number, number, number, number];

/** How fast every motion eases toward its target: the whole way in this many milliseconds at one frame. */
export const EASE_MS = 160;
/**
 * The head with the speech's stress, at its largest: a nod of this share of the picture's height and a sway of this
 * many degrees. At the 132 px the face is shown at, 0.019 is 2.5 px down and 0.8 degrees moves the top of the head
 * about 1.7 px, in a rise and fall of about 600 ms (the player's STRESS_WIDTH_MS): enough to be seen, at a dozen
 * pixels a second at most. Stage 3's 1 px (0.009, 0.45 degrees) could not be seen at that size.
 */
export const NOD_SHARE = 0.019;
export const SWAY_DEG = 0.8;
/**
 * Two pictures of different jaw positions blended half and half show both chins and both lower lips, softly doubled.
 * A blend from one picture to the next stays on the first until DOMINANT_FROM of the way and is on the second from
 * DOMINANT_TO (a smooth step between), so it is half and half only for a moment. A lip closure's pictures differ in
 * the lips alone and cross in a narrower band still.
 */
export const DOMINANT_FROM = 0.25;
export const DOMINANT_TO = 0.75;
export const CLOSURE_FROM = 0.4;
export const CLOSURE_TO = 0.6;

/** 0 below `from`, 1 above `to`, a smooth step between. */
export function dominant(x: number, from = DOMINANT_FROM, to = DOMINANT_TO): number {
  const t = Math.max(0, Math.min(1, (x - from) / (to - from)));
  return t * t * (3 - 2 * t);
}
/**
 * Blinks while Heliox speaks wait for a pause. At the end of a phrase (a pause of PHRASE_PAUSE_MS or more, or the end
 * of speech) a blink follows with PHRASE_BLINK_CHANCE if the last one was at least BLINK_GAP_MS ago; otherwise a
 * pause of GLANCE_PAUSE_MS or more may bring a glance of GLANCE_MS (eyes.glance) with GLANCE_CHANCE.
 */
export const PHRASE_BLINK_CHANCE = 0.6;
export const BLINK_GAP_MS = 1200;
export const PHRASE_PAUSE_MS = 200;
export const GLANCE_PAUSE_MS = 500;
export const GLANCE_CHANCE = 0.3;
export const GLANCE_MS = 360;

/** Where the head turns and breathes about: the neck, low in the middle of the picture. */
function pivotOf(width: number, height: number): [number, number] {
  return [width / 2, height * 0.9];
}

/**
 * The overscan that keeps every corner of the card covered by the picture in one pose (a tilt in degrees, a lift in
 * pixels and a scale, all about the pivot). The overscan enlarges the picture about its centre before the pose, so a
 * corner is covered when the pose's inverse takes it no further from the centre than the enlarged picture reaches.
 */
export function coverScale(width: number, height: number, tilt: number, lift: number, scale: number): number {
  const [px, py] = pivotOf(width, height);
  const angle = (-tilt * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  let needed = 1;
  for (const [x, y] of [
    [0, 0],
    [width, 0],
    [0, height],
    [width, height],
  ]) {
    const dx = x - px;
    const dy = y - (py + lift);
    const ux = (dx * cos - dy * sin) / scale + px;
    const uy = (dx * sin + dy * cos) / scale + py;
    needed = Math.max(needed, Math.abs(ux - width / 2) / (width / 2), Math.abs(uy - height / 2) / (height / 2));
  }
  return needed;
}

/**
 * How much larger than the card the picture is drawn, as a factor: 1 + the manifest's `overscan` when it gives one;
 * else, for a photo, enough for the largest pose any state asks for (and, while speaking, the nod and sway) with
 * breathing at its smallest, and half a percent to spare. A drawing has its own background colour behind it, so it
 * gets none.
 */
export function overscanFor(manifest: FaceManifest): number {
  if (typeof manifest.overscan === "number") return 1 + manifest.overscan;
  if (manifest.kind !== "photo") return 1;
  const [width, height] = manifest.size;
  const breathe = 1 - (manifest.breathe?.amount ?? 0);
  let needed = 1;
  for (const name of STATE_NAMES) {
    const state: FaceState = manifest.states?.[name] ?? {};
    const sways = name === "speaking" ? [-SWAY_DEG, 0, SWAY_DEG] : [0];
    const nods = name === "speaking" ? [0, NOD_SHARE * height] : [0];
    for (const sway of sways) {
      for (const nod of nods) {
        const pose = coverScale(
          width,
          height,
          (state.tilt ?? 0) + sway,
          (state.lift ?? 0) + nod,
          (state.scale ?? 1) * breathe,
        );
        needed = Math.max(needed, pose);
      }
    }
  }
  return needed > 1 ? needed * 1.005 : 1;
}

export class FaceRenderer {
  private images = new Map<string, HTMLImageElement>();
  private tinted = new Map<string, HTMLCanvasElement>();
  private colours: Record<string, string> = {};
  private tilt = 0;
  private lift = 0;
  private scale = 1;
  private overscan = 1;
  /** The resting mouth: the image shown, the one it is easing away from, and how far (1 = all `restTo`). */
  private restTo = "";
  private restFrom = "";
  private restMix = 1;
  private shapeCanvases: HTMLCanvasElement[] = [];
  private lastDraw = 0;
  private blinkAt = 0;
  private blinkUntil = 0;
  private lastBlinkEnd = 0;
  private glanceAt = 0;
  private glanceUntil = 0;
  private wasSpeaking = false;
  private wasPausing = false;
  reducedMotion = false;

  constructor(
    private canvas: HTMLCanvasElement,
    private manifest: FaceManifest,
    private random: () => number = Math.random,
  ) {
    this.overscan = overscanFor(manifest);
    this.restTo = this.restFor("idle");
    this.restFrom = this.restTo;
  }

  /** Every file the pack names, loaded once. Rejects when one cannot be loaded. */
  async load(): Promise<void> {
    const m = this.manifest;
    const files = new Set<string>([
      ...m.idle.frames,
      ...Object.values(m.mouth.shapes).filter((f): f is string => typeof f === "string"),
      ...[
        m.eyes?.open,
        m.eyes?.listening,
        m.eyes?.thinking,
        ...(m.eyes?.blink ?? []),
        ...(m.eyes?.glance ?? []),
      ].filter((f): f is string => typeof f === "string"),
      ...STATE_NAMES.map((name) => m.states?.[name]?.mouth).filter((f): f is string => typeof f === "string"),
      ...Object.values(m.mouth_steps ?? {}).flatMap((files) => files ?? []),
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

  /** Something is moving on its own (a blink, a glance, the head turning): draw at the speaking rate until it settles. */
  animating(now: number, mode: FaceMode): boolean {
    if (this.reducedMotion) return false;
    const target = this.stateFor(mode);
    const settled =
      Math.abs((target.tilt ?? 0) - this.tilt) < 0.05 &&
      Math.abs((target.lift ?? 0) - this.lift) < 0.05 &&
      Math.abs((target.scale ?? 1) - this.scale) < 0.0005 &&
      this.restTo === this.restFor(mode) &&
      this.restMix >= 1;
    return (
      !settled || (now >= this.blinkAt && now < this.blinkUntil) || (now >= this.glanceAt && now < this.glanceUntil)
    );
  }

  /** Milliseconds until the next blink starts (Infinity when the pack does not blink or motion is reduced). */
  untilBlink(now: number): number {
    if (this.reducedMotion || !this.manifest.eyes?.blink?.length || !this.blinkAt) return Infinity;
    return Math.max(0, this.blinkAt - now);
  }

  /** The head's pose now (for tests and for the face's own checks). */
  pose(): { tilt: number; lift: number; scale: number; overscan: number; rest: string; restMix: number } {
    return {
      tilt: this.tilt,
      lift: this.lift,
      scale: this.scale,
      overscan: this.overscan,
      rest: this.restTo,
      restMix: this.restMix,
    };
  }

  /** When the next blink is planned, [start, end) (for tests). */
  blinkPlan(): [number, number] {
    return [this.blinkAt, this.blinkUntil];
  }

  draw(now: number, mode: FaceMode, mouth: MouthFrame): void {
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    const m = this.manifest;
    const [width, height] = m.size;
    const pixels = this.canvas.width / width;
    const dt = this.lastDraw ? Math.min(250, now - this.lastDraw) : 250;
    this.lastDraw = now;
    this.step(dt, mode);
    this.speechCues(now, mouth);
    // The head with the speech's stress (the player's nod and sway are already smooth).
    const nod = this.reducedMotion ? 0 : Math.max(0, Math.min(1, mouth.nod ?? 0)) * NOD_SHARE * height;
    const sway = this.reducedMotion ? 0 : Math.max(-1, Math.min(1, mouth.sway ?? 0)) * SWAY_DEG;
    const breathing =
      this.reducedMotion || !m.breathe
        ? 1
        : 1 + m.breathe.amount * Math.sin((now / 1000 / Math.max(1, m.breathe.seconds)) * Math.PI * 2);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(pixels, 0, 0, pixels, 0, 0);
    // The head turns about the neck, and breathes and scales about the same point.
    const [pivotX, pivotY] = pivotOf(width, height);
    ctx.translate(pivotX, pivotY + this.lift + nod);
    ctx.rotate(((this.tilt + sway) * Math.PI) / 180);
    const size = breathing * this.scale;
    ctx.scale(size, size);
    ctx.translate(-pivotX, -pivotY);
    if (this.overscan !== 1) {
      // A photograph is drawn larger than the card, about its centre, so no pose shows the card behind it.
      ctx.translate(width / 2, height / 2);
      ctx.scale(this.overscan, this.overscan);
      ctx.translate(-width / 2, -height / 2);
    }

    const frameIndex = this.idleFrame(now);
    this.layer(ctx, "idle", m.idle.frames[frameIndex], [0, 0, width, height], 1);
    const eyes = this.eyesFor(now, mode);
    const eyesRect: Rect = m.eyes?.rect ?? [0, 0, width, height];
    if (eyes.under) this.layer(ctx, "eyes", eyes.under, eyesRect, 1);
    this.layer(ctx, "eyes", eyes.file, eyesRect, eyes.alpha);
    const offset = m.idle.mouth_offsets?.[frameIndex] ?? [0, 0];
    const rect = m.mouth.rect ?? [0, 0, width, height];
    this.drawMouth(ctx, [rect[0] + offset[0], rect[1] + offset[1], rect[2], rect[3]], mouth);
  }

  /** Moves every eased motion toward the state's target by `dt` milliseconds. */
  private step(dt: number, mode: FaceMode): void {
    const target = this.stateFor(mode);
    const rest = this.restFor(mode);
    if (rest !== this.restTo) {
      // Mostly showing the old target already: ease from it; else from where the last change started.
      if (this.restMix >= 0.5) this.restFrom = this.restTo;
      this.restTo = rest;
      this.restMix = this.restFrom === rest ? 1 : 0;
    }
    if (this.reducedMotion) {
      this.tilt = 0;
      this.lift = 0;
      this.scale = 1;
      this.restMix = 1;
      return;
    }
    const ease = Math.min(1, dt / EASE_MS);
    this.tilt += ((target.tilt ?? 0) - this.tilt) * ease;
    this.lift += ((target.lift ?? 0) - this.lift) * ease;
    this.scale += ((target.scale ?? 1) - this.scale) * ease;
    this.restMix = Math.min(1, this.restMix + dt / EASE_MS);
  }

  /**
   * The resting mouth (the state's own, or sil), then the spoken shape over it as far as it is formed. A shape's
   * pictures run from closed to full: the rest, its in-between steps (mouth_steps), its own picture. A weight picks a
   * place on that line and only the two neighbouring pictures there are blended. Two shapes at once (one giving way
   * to the next) are each drawn over the rest on a canvas of their own, and the second laid over the first.
   */
  private drawMouth(ctx: CanvasRenderingContext2D, at: Rect, mouth: MouthFrame): void {
    const fading = this.restMix < 1 && this.restFrom !== this.restTo;
    if (fading) this.layer(ctx, "mouth", this.restFrom, at, 1);
    this.layer(ctx, "mouth", this.restTo, at, fading ? this.restMix : 1);
    if (!mouth.speaking) return;
    const legacy = Math.max(0, Math.min(1, mouth.amount ?? 1));
    const first = mouth.weight ?? legacy;
    const two = mouth.mix > 0.02 && mouth.next !== mouth.shape && mouth.next !== "sil";
    if (!two) {
      this.drawShape(ctx, mouth.shape, first, at, 1);
      return;
    }
    const second = this.atSameChin(mouth.shape, first, mouth.next, mouth.nextWeight ?? legacy);
    // mix is at most a half (the more formed shape is `shape`): the second shape shows only as it nears the first,
    // and the two are half and half only where they swap, so one jaw is seen at a time.
    const over = 0.5 * dominant(Math.min(0.5, mouth.mix) * 2, 0.4, 1);
    const a = this.shapeCanvas(0, at);
    const b = this.shapeCanvas(1, at);
    const actx = a?.getContext("2d");
    const bctx = b?.getContext("2d");
    if (!a || !b || !actx || !bctx) {
      this.drawShape(ctx, mouth.shape, first, at, 1);
      this.drawShape(ctx, mouth.next, second, at, over);
      return;
    }
    const local: Rect = [0, 0, a.width, a.height];
    const parts: [HTMLCanvasElement, CanvasRenderingContext2D, MouthFrame["shape"], number][] = [
      [a, actx, mouth.shape, first],
      [b, bctx, mouth.next, second],
    ];
    for (const [target, tctx, shape, weight] of parts) {
      tctx.setTransform(1, 0, 0, 1, 0, 0);
      tctx.clearRect(0, 0, target.width, target.height);
      this.layer(tctx, "mouth", this.restTo, local, 1, at);
      this.drawShape(tctx, shape, weight, local, 1, at);
    }
    ctx.drawImage(a, at[0], at[1], at[2], at[3]);
    ctx.globalAlpha = over;
    ctx.drawImage(b, at[0], at[1], at[2], at[3]);
    ctx.globalAlpha = 1;
  }

  /**
   * The weight to draw `next` at beside `shape` at `weight`: where the pack gives both chins (mouth.chin), the one
   * whose chin sits where `shape`'s does (as far as `next` reaches), so the two pictures share one chin; else
   * `nextWeight` as it is.
   */
  private atSameChin(
    shape: MouthFrame["shape"],
    weight: number,
    next: MouthFrame["shape"],
    nextWeight: number,
  ): number {
    const chin = this.manifest.mouth.chin;
    if (!chin) return nextWeight;
    const here = chin[shapeShown(this.manifest, shape)];
    const there = chin[shapeShown(this.manifest, next)];
    if (typeof here !== "number" || typeof there !== "number" || Math.abs(there) < 1 || here * there <= 0)
      return nextWeight;
    return Math.max(0, Math.min(1, (here * Math.max(0, Math.min(1, weight))) / there));
  }

  /** One shape formed `weight` of the way from the rest (already drawn under it) to its full picture. */
  private drawShape(
    ctx: CanvasRenderingContext2D,
    shape: MouthFrame["shape"],
    weight: number,
    rect: Rect,
    alpha: number,
    packRect: Rect = rect,
  ): void {
    if (shape === "sil") return;
    const steps = stepsOf(this.manifest, shape);
    const w = Math.max(0, Math.min(1, weight));
    if (!steps.length || w <= 0.01) return;
    const position = w * steps.length; // 0 is the rest, steps.length the full picture
    const below = Math.floor(position);
    // Between two pictures one stays dominant (dominant); a shape with its full picture alone fades it in as it is.
    const raw = position - below;
    const toward = steps.length < 2 ? raw : shape === "PP" ? dominant(raw, CLOSURE_FROM, CLOSURE_TO) : dominant(raw);
    if (below >= 1 && toward < 0.99) this.layer(ctx, "mouth", steps[below - 1], rect, alpha, packRect);
    if (below < steps.length && toward > 0.01) this.layer(ctx, "mouth", steps[below], rect, alpha * toward, packRect);
  }

  private shapeCanvas(which: number, rect: Rect): HTMLCanvasElement | null {
    if (typeof document === "undefined") return null;
    const pixels = (this.canvas.width / this.manifest.size[0]) * this.overscan;
    const width = Math.max(1, Math.round(rect[2] * pixels));
    const height = Math.max(1, Math.round(rect[3] * pixels));
    let canvas = this.shapeCanvases[which];
    if (!canvas) {
      canvas = document.createElement("canvas");
      this.shapeCanvases[which] = canvas;
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return canvas;
  }

  /**
   * Blinks and glances with the speech (see PHRASE_BLINK_CHANCE): a blink due while a word sounds waits for a
   * pause; the end of a phrase may bring a blink, or in a long pause a glance.
   */
  private speechCues(now: number, mouth: MouthFrame): void {
    const eyes = this.manifest.eyes;
    if (this.reducedMotion || !eyes) return;
    const blinking = now >= this.blinkAt && now < this.blinkUntil;
    if (!blinking && this.blinkUntil && now >= this.blinkUntil) this.lastBlinkEnd = this.blinkUntil;
    const pause = mouth.speaking ? (mouth.pause ?? 0) : Infinity;
    const length = this.blinkUntil - this.blinkAt;
    const blinks = Boolean(eyes.blink?.length);
    // A blink about to start, or started less than a frame ago, while a word sounds: put it off.
    const starting = this.blinkAt > 0 && now >= this.blinkAt - 40 && now < this.blinkAt + 34;
    if (blinks && mouth.speaking && starting && pause < length + 40) {
      this.blinkAt = now + 60; // never mid-word
      this.blinkUntil = this.blinkAt + length;
    }
    const pausing = mouth.speaking && pause >= PHRASE_PAUSE_MS;
    const phraseEnd = (pausing && !this.wasPausing) || (!mouth.speaking && this.wasSpeaking);
    if (phraseEnd && !blinking) {
      if (blinks && now - this.lastBlinkEnd > BLINK_GAP_MS && this.random() < PHRASE_BLINK_CHANCE) {
        const frames = eyes.blink?.length ?? 0;
        this.blinkAt = now + 40;
        this.blinkUntil = this.blinkAt + (frames / (eyes.blink_fps ?? 24)) * 1000;
      } else if (eyes.glance?.length && pausing && pause >= GLANCE_PAUSE_MS && this.random() < GLANCE_CHANCE) {
        this.glanceAt = now + 60;
        this.glanceUntil = this.glanceAt + Math.min(GLANCE_MS, pause - 120);
      }
    }
    this.wasPausing = pausing;
    this.wasSpeaking = mouth.speaking;
  }

  /** The image of the mouth at rest in a state: the state's own `mouth`, else the closed mouth. */
  private restFor(mode: FaceMode): string {
    return this.stateFor(mode).mouth ?? imageForShape(this.manifest, "sil");
  }

  private stateFor(mode: FaceMode): FaceState {
    return this.manifest.states?.[mode] ?? {};
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

  /**
   * The eyes now. A blink of several images plays them in turn. A blink of one image (a photo pack with only a
   * closed-eyes still) fades that image in over the first third of the blink and out over the last third, over the
   * steady eyes, so it does not snap shut. A glance goes out through eyes.glance, rests at the last picture and
   * comes back, each picture fading into the next.
   */
  private eyesFor(now: number, mode: FaceMode): { file?: string; under?: string; alpha: number } {
    const eyes = this.manifest.eyes;
    if (!eyes) return { alpha: 1 };
    const steady = (mode === "listening" && eyes.listening) || (mode === "thinking" && eyes.thinking) || eyes.open;
    const blink = eyes.blink ?? [];
    if (this.reducedMotion || !blink.length) return { file: steady, alpha: 1 };
    if (!this.blinkAt) this.scheduleBlink(now);
    if (now >= this.blinkUntil && now >= this.blinkAt) this.scheduleBlink(now);
    if (now >= this.blinkAt && now < this.blinkUntil) {
      if (blink.length === 1) {
        const through = (now - this.blinkAt) / Math.max(1, this.blinkUntil - this.blinkAt);
        return { file: blink[0], under: steady, alpha: Math.min(1, 3 * Math.min(through, 1 - through) + 0.34) };
      }
      const frame = Math.floor(((now - this.blinkAt) / 1000) * (eyes.blink_fps ?? 24));
      return { file: blink[Math.min(blink.length - 1, frame)], alpha: 1 };
    }
    const glance = eyes.glance ?? [];
    if (glance.length && now >= this.glanceAt && now < this.glanceUntil) {
      // Out and back over 90 ms each way: a place along steady, glance[0], ..., glance[last].
      const ramp = Math.min(now - this.glanceAt, this.glanceUntil - now) / 90;
      const position = Math.max(0, Math.min(1, ramp)) * glance.length;
      const below = Math.floor(position);
      const toward = position - below;
      if (below >= glance.length) return { file: glance[glance.length - 1], under: steady, alpha: 1 };
      const under = below === 0 ? steady : glance[below - 1];
      return { file: glance[below], under, alpha: toward };
    }
    return { file: steady, alpha: 1 };
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
    rect: Rect,
    alpha: number,
    packRect: Rect = rect,
  ): void {
    if (!file || alpha <= 0) return;
    const image = this.images.get(file);
    if (!image) return;
    const source = this.tintedImage(layer, file, image, packRect);
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.drawImage(source, rect[0], rect[1], rect[2], rect[3]);
    ctx.globalAlpha = 1;
  }

  /** A drawn pack's layer in the token's colour, at the canvas's own resolution so lines stay sharp. */
  private tintedImage(layer: Layer, file: string, image: HTMLImageElement, rect: Rect): CanvasImageSource {
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

/**
 * Plays the daemon's speech notices (voice_piece, voice_end, voice_cut; IPC_MESSAGE_FORMATS.md) as a mouth: for any
 * moment on a clock the caller owns, which mouth shapes show, how far each is formed, and how the head moves with the
 * speech.
 *
 * Plain module, no DOM: the face component (Face.svelte) feeds it notices and asks it for frames. Rules that hold
 * whatever the motion below does:
 * - A piece plays from the moment its first sample is heard (receipt + lead - transit) for its own length.
 * - A cut closes the mouth on the next frame drawn, whatever was planned.
 * - A notice that arrives too late to be heard, never arrives, or carries no timing leaves the mouth closed: every
 *   piece ends by its own length, so the mouth cannot stay open waiting for an end notice.
 *
 * How it moves (planTrack): the mouth is a weight per shape, 0 (at rest) to 1 (fully formed). Each moment of the
 * timeline sets a target, one shape at a strength that follows the sound's loudness, and every weight moves toward
 * its target like a lightly damped spring (SPRING_MS), so a short sound pulls the mouth only part of the way and a
 * quiet syllable opens it less. A lip closure (p, b, m) springs faster (CLOSURE_MS) and always closes fully, on time.
 * The targets run TARGET_LEAD_MS ahead of the sound, which is about how long the spring takes, so the picture lands
 * on the sound instead of after it. Where the sound is quiet for QUIET_MIN_MS or more the target is rest
 * (soundGate). A shape the pack has no picture for never jumps to an unrelated wide shape: a consonant keeps the
 * opening of the vowel before it, a little smaller, or a small neutral opening (setShapes).
 *
 * The head: the loudest syllables (stress) give a small nod and an alternating sway, peaking just before the syllable;
 * `pause` says how long the sound will now rest, so blinks and glances fall between phrases and not mid-word.
 */

export const SHAPES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U"] as const;
export type Shape = (typeof SHAPES)[number];

/** Kept for the reduced-motion face and for callers that blend two whole shapes. */
export const CROSSFADE_MS = 70;
/** A transit longer than this is a clock that cannot be trusted: the notice is taken as just sent. */
export const MAX_TRANSIT_MS = 1000;
/** A spoken piece the system voice plays by itself has no length: "speaking" ends after this without an end notice. */
export const UNTIMED_LIMIT_MS = 30_000;

/**
 * The spring: time constant for a shape forming, for a shape giving way (faster: a mouth lets go of a shape quicker
 * than it reaches one), and for a lip closure, and its damping (1 = no overshoot).
 */
export const SPRING_MS = 32;
export const RELEASE_MS = 26;
export const CLOSURE_MS = 14;
export const DAMPING = 0.8;
/** Targets run this far ahead of the sound: about the spring's own delay plus half a frame at 30 frames a second. */
export const TARGET_LEAD_MS = 45;
export const CLOSURE_LEAD_MS = 25;
/** A lip closure presses the lips (PP) for at most this long, then rests closed (sil). */
export const PRESS_MS = 140;
/** How strongly a shape is formed: from MIN_STRENGTH for a quiet sound to 1 at FULL_LEVEL (envelope, 0 to 15). */
export const MIN_STRENGTH = 0.45;
export const FULL_LEVEL = 9;
/** A consonant with a picture of its own is formed at most this much (the vowels carry the opening). */
export const CONSONANT_STRENGTH = 0.7;
/** Loudness (envelope, 0 to 15) at or below which the sound has died away. */
export const QUIET_LEVEL = 3;
/** A quiet stretch this long or longer rests the mouth; a shorter one (a stop closure inside a word) does not. */
export const QUIET_MIN_MS = 80;
/** The mouth closes from CLOSE_AHEAD_MS before the quiet starts to SETTLE_MS after it, half closed as it starts. */
export const CLOSE_AHEAD_MS = 10;
export const SETTLE_MS = 30;
/** It reopens over this long before the sound returns. */
export const REOPEN_AHEAD_MS = 10;
/** Stress: a syllable at least this loud (0 to 1 of the piece's loudest) and this far from the last one nods. */
export const STRESS_LEVEL = 0.75;
export const STRESS_GAP_MS = 220;
/** The nod and sway peak this long before the stressed syllable and last about twice STRESS_WIDTH_MS. */
export const STRESS_LEAD_MS = 60;
export const STRESS_WIDTH_MS = 170;

const STEP_MS = 4;

export interface VoicePieceNotice {
  reply: number;
  index: number;
  seconds: number | null;
  lead_ms: number | null;
  sent_ms?: number;
  voice?: string;
  mouth: "visemes" | "envelope" | "none" | "off";
  visemes?: [string, number, number][];
  envelope?: number[];
  envelope_rate?: number;
}

export interface MouthFrame {
  /** A piece is being heard now (its mouth may still be closed between words). */
  speaking: boolean;
  /** Where the mouth shape comes from at this moment. */
  source: "visemes" | "envelope" | "none";
  /** The two most formed shapes, and how far the picture has blended from `shape` to `next` (0 to 1). */
  shape: Shape;
  next: Shape;
  mix: number;
  /** How open, 0 to 1. */
  open: number;
  /** How strongly the pair is formed over the resting mouth, 0 to 1 (missing: 1). */
  amount?: number;
  /** How far `shape` and `next` are each formed, 0 to 1 (missing: `amount`). */
  weight?: number;
  nextWeight?: number;
  /** The head with the speech's stress: nod 0 to 1 (down), sway -1 to 1. Missing: still. */
  nod?: number;
  sway?: number;
  /** Milliseconds the sound will now rest (a pause between phrases); 0 while it sounds. */
  pause?: number;
}

interface Track {
  /** Weights of each shape every STEP_MS from the piece's start, and the head's nod and sway. */
  weights: Map<Shape, Float32Array>;
  nod: Float32Array;
  sway: Float32Array;
  quiet: [number, number][];
}

interface Planned {
  reply: number;
  index: number;
  start: number;
  end: number;
  notice: VoicePieceNotice;
  track: Track | null;
}

const VOWELS: ReadonlySet<Shape> = new Set(["aa", "E", "I", "O", "U"]);
const CLOSURES: ReadonlySet<Shape> = new Set(["PP"]);
/** A missing vowel is drawn as the nearest vowel the pack has. */
const VOWEL_STAND_INS: Record<string, Shape[]> = {
  aa: ["E", "O"],
  E: ["aa", "I"],
  I: ["E"],
  O: ["U", "aa"],
  U: ["O"],
};
const IDLE: MouthFrame = {
  speaking: false,
  source: "none",
  shape: "sil",
  next: "sil",
  mix: 0,
  open: 0,
  amount: 0,
  weight: 0,
  nextWeight: 0,
  nod: 0,
  sway: 0,
  pause: 0,
};

function isShape(value: string): value is Shape {
  return (SHAPES as readonly string[]).includes(value);
}

/** Where the piece's first sample is heard, on the caller's clock: receipt + lead - time already spent in transit. */
export function pieceStart(notice: VoicePieceNotice, receivedAt: number, wallNow: number): number {
  const lead = typeof notice.lead_ms === "number" && Number.isFinite(notice.lead_ms) ? notice.lead_ms : 0;
  let transit = 0;
  if (typeof notice.sent_ms === "number" && Number.isFinite(notice.sent_ms)) {
    const spent = wallNow - notice.sent_ms;
    transit = spent >= 0 && spent <= MAX_TRANSIT_MS ? spent : 0;
  }
  return receivedAt + lead - transit;
}

/** Quiet stretches of QUIET_MIN_MS or more, [start, end) in ms from the piece's start. */
export function quietStretches(levels: number[] | undefined, rate: number): [number, number][] {
  if (!Array.isArray(levels) || !levels.length || !(rate > 0)) return [];
  const bin = 1000 / rate;
  const out: [number, number][] = [];
  for (let i = 0; i < levels.length;) {
    if ((Number(levels[i]) || 0) > QUIET_LEVEL) {
      i++;
      continue;
    }
    let j = i;
    while (j < levels.length && (Number(levels[j]) || 0) <= QUIET_LEVEL) j++;
    if ((j - i) * bin >= QUIET_MIN_MS) out.push([i * bin, j >= levels.length ? Infinity : j * bin]);
    i = j;
  }
  return out;
}

/**
 * How far the sound lets the mouth be formed at `at` (1 = as planned, 0 = at rest). The whole envelope is known in
 * advance, so the mouth closes with the sound rather than after it: through every quiet stretch it ramps down from
 * CLOSE_AHEAD_MS before the stretch to SETTLE_MS into it, stays at rest, and ramps up over the REOPEN_AHEAD_MS before
 * the sound returns.
 */
export function soundGate(levels: number[] | undefined, rate: number, at: number): number {
  return gateFrom(quietStretches(levels, rate), at);
}

function gateFrom(quiet: [number, number][], at: number): number {
  let gate = 1;
  for (const [start, end] of quiet) {
    if (at >= end) continue;
    if (at < start - CLOSE_AHEAD_MS) break;
    let here: number;
    if (at < start + SETTLE_MS) here = 1 - (at - (start - CLOSE_AHEAD_MS)) / (CLOSE_AHEAD_MS + SETTLE_MS);
    else if (at < end - REOPEN_AHEAD_MS) here = 0;
    else here = (at - (end - REOPEN_AHEAD_MS)) / REOPEN_AHEAD_MS;
    gate = Math.min(gate, Math.max(0, Math.min(1, here)));
  }
  return gate;
}

/** How strongly the sound asks a shape to be formed: quiet sounds less, from MIN_STRENGTH to 1. */
export function strengthOf(level: number): number {
  const x = Math.max(0, Math.min(1, (level - QUIET_LEVEL) / (FULL_LEVEL - QUIET_LEVEL)));
  return MIN_STRENGTH + (1 - MIN_STRENGTH) * Math.pow(x, 0.8);
}

/**
 * The shape a pack can show for `shape`, and how strongly: its own picture, the nearest vowel for a vowel, the
 * vowel before it (a little smaller) or a small neutral opening for a consonant. Never a wide shape the sound does
 * not have.
 */
export function calmShape(shape: Shape, has: (s: Shape) => boolean, vowelBefore: Shape | null): [Shape, number] {
  if (shape === "sil") return ["sil", 0];
  if (has(shape)) return [shape, VOWELS.has(shape) || CLOSURES.has(shape) ? 1 : CONSONANT_STRENGTH];
  if (shape === "PP" || shape === "FF") return has("PP") ? ["PP", 1] : ["sil", 0];
  if (VOWELS.has(shape)) {
    for (const other of VOWEL_STAND_INS[shape] ?? []) if (has(other)) return [other, 0.9];
    return has("E") ? ["E", 0.6] : ["sil", 0];
  }
  // The tongue-tip sounds (t, d, th, and n, l) share the pack's calm small opening when it has one; s-like ch shares s.
  if ((shape === "DD" || shape === "TH") && has("nn")) return ["nn", CONSONANT_STRENGTH];
  if (shape === "CH" && has("SS")) return ["SS", CONSONANT_STRENGTH];
  if (vowelBefore && has(vowelBefore)) return [vowelBefore, 0.55];
  if (has("E")) return ["E", 0.3];
  if (has("I")) return ["I", 0.25];
  return ["sil", 0];
}

/** The loudest syllables of a piece: [time ms, height 0 to 1], at least STRESS_GAP_MS apart. */
export function stressPeaks(levels: number[] | undefined, rate: number): [number, number][] {
  if (!Array.isArray(levels) || levels.length < 3 || !(rate > 0)) return [];
  const bin = 1000 / rate;
  const top = Math.max(...levels.map((v) => Number(v) || 0));
  if (top <= QUIET_LEVEL) return [];
  const peaks: [number, number][] = [];
  for (let i = 1; i < levels.length - 1; i++) {
    const v = Number(levels[i]) || 0;
    if (v < STRESS_LEVEL * top || v < (Number(levels[i - 1]) || 0) || v < (Number(levels[i + 1]) || 0)) continue;
    const at = (i + 0.5) * bin;
    const last = peaks.at(-1);
    if (last && at - last[0] < STRESS_GAP_MS) {
      if (v / top > last[1]) peaks[peaks.length - 1] = [at, v / top];
      continue;
    }
    peaks.push([at, v / top]);
  }
  return peaks;
}

/** Simulates the mouth and the head for one piece (see the module comment). */
export function planTrack(notice: VoicePieceNotice, has: (s: Shape) => boolean): Track | null {
  const seconds = typeof notice.seconds === "number" && notice.seconds > 0 ? notice.seconds : 0;
  const levels = Array.isArray(notice.envelope) && notice.envelope.length ? notice.envelope : undefined;
  const rate = notice.envelope_rate || 25;
  const visemes =
    notice.mouth === "visemes" && Array.isArray(notice.visemes)
      ? notice.visemes.filter(
          (v): v is [string, number, number] => Array.isArray(v) && typeof v[1] === "number" && v[2] > v[1],
        )
      : [];
  if (!seconds || (!visemes.length && !levels)) return null;
  const length = seconds * 1000;
  const steps = Math.ceil(length / STEP_MS) + 1;
  const quiet = quietStretches(levels, rate);
  const bin = 1000 / rate;
  const levelAt = (t: number) => {
    if (!levels) return FULL_LEVEL;
    const position = t / bin - 0.5; // each value is its window's loudness: placed at the window's middle
    const i = Math.max(0, Math.floor(position));
    const a = Number(levels[Math.min(levels.length - 1, i)]) || 0;
    const b = Number(levels[Math.min(levels.length - 1, i + 1)]) || 0;
    return a + (b - a) * Math.max(0, Math.min(1, position - i));
  };

  // The target at each step: one shape and its strength.
  const target = (t: number): [Shape, number, boolean] => {
    if (!visemes.length) {
      // Loudness alone: half open (E) for a moderate sound, open (aa) for a loud one.
      const level = levelAt(t + TARGET_LEAD_MS);
      const strength = level <= QUIET_LEVEL ? 0 : strengthOf(level) * gateFrom(quiet, t + TARGET_LEAD_MS);
      const shape: Shape = level >= FULL_LEVEL * 0.7 && has("aa") ? "aa" : has("E") ? "E" : has("aa") ? "aa" : "sil";
      return [shape, strength, false];
    }
    // A closure looks a little less far ahead than an opening shape: it is quick, and it must land on time.
    const find = (at: number) => visemes.findIndex(([, s, e]) => s <= at && at < e);
    const closing = find(t + CLOSURE_LEAD_MS);
    if (closing >= 0 && visemes[closing][0] === "PP") {
      // Pressed lips for the closure itself; a long one (an m held at the end of a word) relaxes to the rest.
      const pressed = t + CLOSURE_LEAD_MS - visemes[closing][1] < PRESS_MS;
      return pressed && has("PP") ? ["PP", 1, true] : ["sil", 0, true];
    }
    const at = t + TARGET_LEAD_MS;
    const index = find(at);
    if (index < 0) return ["sil", 0, false];
    const name = visemes[index][0];
    const shape: Shape = isShape(name) ? name : "E";
    let vowelBefore: Shape | null = null;
    for (let k = index - 1; k >= 0 && visemes[index][1] - visemes[k][2] < 300; k--) {
      const before = visemes[k][0];
      if (isShape(before) && VOWELS.has(before)) {
        vowelBefore = before;
        break;
      }
    }
    const [shown, scale] = calmShape(shape, has, vowelBefore);
    if (shown === "sil") return ["sil", 0, false];
    const loud = shown === "PP" ? 1 : strengthOf(levelAt(at)) * gateFrom(quiet, at);
    return [shown, scale * loud, shown === "PP"];
  };

  const weights = new Map<Shape, Float32Array>();
  const velocity = new Map<Shape, number>();
  const series = (shape: Shape) => {
    let s = weights.get(shape);
    if (!s) {
      s = new Float32Array(steps);
      weights.set(shape, s);
      velocity.set(shape, 0);
    }
    return s;
  };
  const dt = STEP_MS / 1000;
  for (let k = 1; k < steps; k++) {
    const [shape, strength, closure] = target(k * STEP_MS);
    if (shape !== "sil") series(shape);
    for (const [name, s] of weights) {
      const goal = name === shape ? strength : 0;
      // A closure pulls every other shape shut as fast as it closes itself; a lip shape also lets go that fast.
      const fast = closure || name === "PP";
      const om = 1000 / (fast ? CLOSURE_MS : goal < s[k - 1] ? RELEASE_MS : SPRING_MS);
      let v = velocity.get(name) ?? 0;
      v += (om * om * (goal - s[k - 1]) - 2 * DAMPING * om * v) * dt;
      s[k] = Math.max(0, Math.min(1.05, s[k - 1] + v * dt));
      velocity.set(name, v);
    }
  }

  // The head: a raised-cosine bump before each stressed syllable. The sway, smaller, changes side every second
  // stressed syllable, so the head leans one way through a phrase-sized stretch instead of wobbling each syllable.
  const nod = new Float32Array(steps);
  const sway = new Float32Array(steps);
  stressPeaks(levels, rate).forEach(([at, height], n) => {
    const centre = at - STRESS_LEAD_MS;
    const side = Math.floor(n / 2) % 2 === 0 ? 0.7 : -0.7;
    const first = Math.max(0, Math.floor((centre - STRESS_WIDTH_MS) / STEP_MS));
    const last = Math.min(steps - 1, Math.ceil((centre + STRESS_WIDTH_MS) / STEP_MS));
    for (let k = first; k <= last; k++) {
      const x = (k * STEP_MS - centre) / STRESS_WIDTH_MS;
      if (Math.abs(x) >= 1) continue;
      const bump = 0.5 * (1 + Math.cos(Math.PI * x)) * height;
      nod[k] = Math.min(1, nod[k] + bump);
      sway[k] = Math.max(-1, Math.min(1, sway[k] + side * bump));
    }
  });
  return { weights, nod, sway, quiet };
}

export class MouthPlayer {
  private pieces: Planned[] = [];
  private closedReplies = new Set<number>();
  private newestReply = -1;
  private has: (s: Shape) => boolean = () => true;

  /** The shapes the face has pictures for (its pack): the others are drawn calmly (calmShape). Default: all. */
  setShapes(shapes: Iterable<string> | null): void {
    if (!shapes) {
      this.has = () => true;
      return;
    }
    const set = new Set(shapes);
    this.has = (s) => set.has(s);
  }

  /** A voice_piece notice arrived at `receivedAt` (caller's clock, ms); `wallNow` is Date.now() at that moment. */
  piece(notice: VoicePieceNotice, receivedAt: number, wallNow: number = Date.now()): void {
    if (!notice || typeof notice.reply !== "number") return;
    if (this.closedReplies.has(notice.reply) || notice.reply < this.newestReply) return;
    if (notice.reply > this.newestReply) {
      // A newer reply took the voice: anything older still planned would talk over it.
      this.pieces = [];
      this.newestReply = notice.reply;
    }
    const start = pieceStart(notice, receivedAt, wallNow);
    const seconds = typeof notice.seconds === "number" && notice.seconds > 0 ? notice.seconds : null;
    const end = seconds === null ? start + UNTIMED_LIMIT_MS : start + seconds * 1000;
    let track: Track | null = null;
    try {
      track = planTrack(notice, this.has);
    } catch {
      track = null; // a malformed notice leaves the mouth still
    }
    this.pieces = this.pieces.filter((p) => p.index !== notice.index);
    this.pieces.push({ reply: notice.reply, index: notice.index, start, end, notice, track });
    this.pieces.sort((a, b) => a.start - b.start);
  }

  /** The reply played to its end. Its pieces still end by their own length (the end notice may be early or late). */
  end(reply: number): void {
    // An untimed piece (the system voice) has no length of its own: the end notice is its end.
    this.pieces = this.pieces.filter((p) => p.reply !== reply || p.notice.seconds !== null);
    this.closedReplies.add(reply);
    this.trimClosed();
  }

  /** The reply is being cut off now: the mouth closes on the next frame. */
  cut(reply?: number): void {
    this.pieces = reply === undefined ? [] : this.pieces.filter((p) => p.reply !== reply);
    if (reply !== undefined) this.closedReplies.add(reply);
    this.trimClosed();
  }

  /** Forget everything (the face was switched off or the daemon went away). */
  reset(): void {
    this.pieces = [];
  }

  /** Something is planned for now or later (the caller keeps drawing at the speaking rate). */
  busy(now: number): boolean {
    this.pieces = this.pieces.filter((p) => p.end > now);
    return this.pieces.length > 0;
  }

  /** The mouth at `now`. `binary`: reduced motion, only open or closed, no blending and no head motion. */
  sample(now: number, binary = false): MouthFrame {
    const playing = this.pieces.find((p) => p.start <= now && now < p.end);
    if (!playing) return IDLE;
    const at = now - playing.start;
    const source = playing.notice.mouth === "visemes" ? "visemes" : "envelope";
    const frame = playing.track
      ? sampleTrack(playing.track, at, playing.notice.visemes?.length ? "visemes" : source)
      : { ...IDLE, speaking: true };
    if (!binary) return frame;
    const open = frame.open >= 0.4;
    const shape: Shape = open ? "aa" : "sil";
    return { ...frame, shape, next: shape, mix: 0, open: open ? 1 : 0, amount: open ? 1 : 0, weight: open ? 1 : 0,
      nextWeight: 0, nod: 0, sway: 0 }; // prettier-ignore
  }

  private trimClosed(): void {
    if (this.closedReplies.size > 64) {
      const keep = [...this.closedReplies].slice(-32);
      this.closedReplies = new Set(keep);
    }
  }
}

function sampleTrack(track: Track, at: number, source: "visemes" | "envelope"): MouthFrame {
  const position = Math.max(0, at / STEP_MS);
  const k = Math.floor(position);
  const f = position - k;
  const read = (s: Float32Array) => {
    const a = s[Math.min(s.length - 1, k)] ?? 0;
    const b = s[Math.min(s.length - 1, k + 1)] ?? a;
    return Math.max(0, Math.min(1, a + (b - a) * f));
  };
  let first: [Shape, number] = ["sil", 0];
  let second: [Shape, number] = ["sil", 0];
  for (const [shape, series] of track.weights) {
    const w = read(series);
    if (w > first[1]) {
      second = first;
      first = [shape, w];
    } else if (w > second[1]) second = [shape, w];
  }
  if (second[1] < 0.02) second = ["sil", 0];
  if (first[1] < 0.01) first = ["sil", 0];
  const total = Math.min(1, first[1] + second[1]);
  const mix = second[0] === "sil" || total <= 0 ? 0 : second[1] / (first[1] + second[1]);
  let pause = 0;
  for (const [start, end] of track.quiet) {
    if (start <= at && at < end) pause = end === Infinity ? 1e9 : end - at;
  }
  return {
    speaking: true,
    source,
    shape: first[0],
    next: second[0] === "sil" ? first[0] : second[0],
    mix,
    open: first[0] === "PP" ? 0 : total,
    amount: total,
    weight: first[1],
    nextWeight: second[1],
    nod: read(track.nod),
    sway: Math.max(-1, Math.min(1, track.sway[Math.min(track.sway.length - 1, k)] ?? 0)),
    pause,
  };
}

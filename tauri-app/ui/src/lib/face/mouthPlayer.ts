/**
 * Plays the daemon's speech notices (voice_piece, voice_end, voice_cut; IPC_MESSAGE_FORMATS.md) as a mouth: for any
 * moment on a clock the caller owns, which mouth shape shows and how far it has blended into the next one.
 *
 * Plain module, no DOM: the face component (Face.svelte) feeds it notices and asks it for frames. Rules:
 * - A piece plays from the moment its first sample is heard (receipt + lead - transit) for its own length.
 * - A cut closes the mouth on the next frame drawn, whatever was planned.
 * - A notice that arrives too late to be heard, never arrives, or carries no timing leaves the mouth closed: every
 *   piece ends by its own length, so the mouth cannot stay open waiting for an end notice.
 */

export const SHAPES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U"] as const;
export type Shape = (typeof SHAPES)[number];

/** How long two neighbouring shapes blend into each other, centred on the change. */
export const CROSSFADE_MS = 70;
/** A transit longer than this is a clock that cannot be trusted: the notice is taken as just sent. */
export const MAX_TRANSIT_MS = 1000;
/** A spoken piece the system voice plays by itself has no length: "speaking" ends after this without an end notice. */
export const UNTIMED_LIMIT_MS = 30_000;

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
  shape: Shape;
  /** The shape it is blending into, and how far (0 = all `shape`, 1 = all `next`). */
  next: Shape;
  mix: number;
  /** For the envelope: how open, 0 to 1. For visemes: 0 for a closed shape, else 1. */
  open: number;
}

interface Planned {
  reply: number;
  index: number;
  start: number;
  end: number;
  notice: VoicePieceNotice;
}

const CLOSED: ReadonlySet<string> = new Set(["sil", "PP"]);
const IDLE: MouthFrame = { speaking: false, source: "none", shape: "sil", next: "sil", mix: 0, open: 0 };

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

export class MouthPlayer {
  private pieces: Planned[] = [];
  private closedReplies = new Set<number>();
  private newestReply = -1;

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
    this.pieces = this.pieces.filter((p) => p.index !== notice.index);
    this.pieces.push({ reply: notice.reply, index: notice.index, start, end, notice });
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

  /** The mouth at `now`. `binary`: reduced motion, only open or closed, no blending. */
  sample(now: number, binary = false): MouthFrame {
    const playing = this.pieces.find((p) => p.start <= now && now < p.end);
    if (!playing) return IDLE;
    const at = now - playing.start;
    const notice = playing.notice;
    let frame: MouthFrame;
    if (notice.mouth === "visemes" && Array.isArray(notice.visemes) && notice.visemes.length) {
      frame = sampleVisemes(notice.visemes, at);
    } else if (notice.mouth === "envelope" && Array.isArray(notice.envelope) && notice.envelope.length) {
      frame = sampleEnvelope(notice.envelope, notice.envelope_rate || 25, at);
    } else {
      frame = { ...IDLE, speaking: true };
    }
    if (!binary) return frame;
    const open = frame.source === "envelope" ? frame.open >= 0.35 : frame.open > 0;
    const shape: Shape = open ? "aa" : "sil";
    return { ...frame, shape, next: shape, mix: 0, open: open ? 1 : 0 };
  }

  private trimClosed(): void {
    if (this.closedReplies.size > 64) {
      const keep = [...this.closedReplies].slice(-32);
      this.closedReplies = new Set(keep);
    }
  }
}

function sampleVisemes(visemes: [string, number, number][], at: number): MouthFrame {
  const index = visemes.findIndex(([, start, end]) => start <= at && at < end);
  if (index < 0) {
    return { ...IDLE, speaking: true, source: "visemes" };
  }
  const [name, start, end] = visemes[index];
  const shape: Shape = isShape(name) ? name : "E";
  let next: Shape = shape;
  let mix = 0;
  let from: Shape = shape;
  const half = Math.min(CROSSFADE_MS / 2, (end - start) / 2);
  if (at - start < half && index > 0) {
    const previous = visemes[index - 1][0];
    from = isShape(previous) ? previous : "E";
    next = shape;
    mix = 0.5 + (at - start) / (2 * half);
  } else if (end - at < half && index < visemes.length - 1) {
    const following = visemes[index + 1][0];
    next = isShape(following) ? following : "E";
    mix = 0.5 - (end - at) / (2 * half);
    from = shape;
  }
  mix = Math.min(1, Math.max(0, mix));
  if (from === next) mix = 0;
  return { speaking: true, source: "visemes", shape: from, next, mix, open: CLOSED.has(shape) ? 0 : 1 };
}

/** Envelope levels (0 to 15) to a mouth: closed, half open (E) and open (aa), blended by level. */
function sampleEnvelope(levels: number[], rate: number, at: number): MouthFrame {
  const position = (at / 1000) * rate;
  const i = Math.floor(position);
  if (i >= levels.length) return { ...IDLE, speaking: true, source: "envelope" };
  const a = levels[i] ?? 0;
  const b = levels[Math.min(levels.length - 1, i + 1)] ?? a;
  const level = Math.max(0, Math.min(15, a + (b - a) * (position - i))) / 15;
  if (level < 0.5) {
    return { speaking: true, source: "envelope", shape: "sil", next: "E", mix: level * 2, open: level };
  }
  return { speaking: true, source: "envelope", shape: "E", next: "aa", mix: (level - 0.5) * 2, open: level };
}

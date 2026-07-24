/**
 * Lightweight spatial/world-model layer for hand gesture recognition.
 *
 * Sits between raw MediaPipe Hands landmark output and the existing gesture
 * classifiers in GestureControl.svelte. Uses only the 21 landmarks MediaPipe
 * already provides — no new models, no MediaPipe Pose/Holistic, no depth
 * camera. Pure functions, no MediaPipe/DOM dependency, so this is testable
 * in isolation from the camera/component.
 *
 * What this adds, and deliberately what it does NOT touch:
 *  - Temporal filtering (One Euro filter) of raw landmark positions, so
 *    classifiers see smoothed input instead of raw per-frame jitter. This
 *    operates in the SAME coordinate space as the raw landmarks, so it's a
 *    safe drop-in replacement for the existing distance/position thresholds
 *    in classifyGesture() — no recalibration needed.
 *  - An orientation/handedness-invariant thumb-extension check, replacing
 *    the old `landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x` heuristic that
 *    silently assumed a right hand facing the camera and broke for left
 *    hands or a rotated wrist.
 *  - A visibility/quality score (MediaPipe's own handedness confidence
 *    combined with a geometric self-consistency check) that scales down
 *    reported gesture confidence instead of letting brittle rules misfire
 *    silently on a degenerate/occluded hand pose.
 *
 *  - A short-horizon predictive layer ("world model") — the One Euro filter
 *    already tracks a smoothed velocity estimate internally; `predict()`/
 *    `predictAhead()` expose a constant-velocity extrapolation of that state
 *    a few tens of milliseconds into the future. This is a kinematic
 *    extrapolator, NOT a generative model — it does not predict screen
 *    pixels/video frames (a real Genie-3/Cosmos-style world model), just
 *    where the already-tracked hand landmarks are kinematically heading.
 *    Used to let the gesture-cursor bridge (GestureControl.svelte) feel
 *    lower-latency, and to fire a pinch-to-click a few frames before the
 *    pinch pose fully closes.
 *
 * What this does NOT do: fully re-express every classifyGesture() threshold
 * in a hand-local, scale-normalized coordinate frame. That would touch ~20
 * empirically-tuned distance constants with no way to validate the
 * recalibration against real camera input in this environment, and would
 * risk silently breaking gestures that work today. If a future pass wants
 * that (e.g. for true camera-distance invariance beyond what filtering and
 * the ratio-based thumb check already provide), it should land behind a
 * recorded-landmark-sequence regression fixture per gesture so the
 * recalibration can actually be verified.
 */

export interface Landmark {
  x: number;
  y: number;
  z?: number;
}

// MediaPipe Hands 21-landmark topology.
const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;
const THUMB_TIP = 4;

function dist3d(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Wrist-to-middle-MCP distance — a stable per-frame estimate of hand size
 * in image space, used to scale distance thresholds so they degrade
 * gracefully as the hand moves closer to/further from the camera. */
export function handSize(landmarks: Landmark[]): number {
  return dist3d(landmarks[WRIST], landmarks[MIDDLE_MCP]) || 1e-6;
}

/**
 * Orientation- and handedness-invariant thumb-extension check.
 *
 * Replaces `landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x`, which encodes
 * "thumb is to the left of its own IP joint" — only true for a right hand
 * facing the camera in a roughly upright pose. Instead, use the thumb tip's
 * distance from the palm (index MCP), normalized by hand size: tucked
 * against the palm (fist, pinch, etc.) puts the tip close to the index MCP
 * regardless of hand rotation or left/right handedness; extended (thumbs
 * up/down, open palm, etc.) puts it clearly further away.
 *
 * THUMB_EXTENDED_RATIO is an empirical threshold (distance / handSize) —
 * approximate, not physically derived; revisit if real-camera testing shows
 * false positives/negatives.
 */
export const THUMB_EXTENDED_RATIO = 0.55;

/** The raw thumb-tip-to-index-MCP distance ratio `isThumbExtended()` compares
 * against a threshold — exposed separately so the calibration layer
 * (calibration.ts) can record what this specific hand actually measures at
 * the moment a thumbs_up/thumbs_down gesture fires, without duplicating the
 * distance math. */
export function thumbExtensionRatio(landmarks: Landmark[], size: number = handSize(landmarks)): number {
  return dist3d(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / size;
}

export function isThumbExtended(
  landmarks: Landmark[],
  size: number = handSize(landmarks),
  threshold: number = THUMB_EXTENDED_RATIO
): boolean {
  return thumbExtensionRatio(landmarks, size) > threshold;
}

// ── Temporal filtering (One Euro filter) ──
//
// Casiez et al. "1€ Filter" — adapts smoothing to velocity, so a still hand
// gets heavy smoothing (kills jitter) while a fast motion (a swipe) gets low
// lag. This filters landmark *positions* before classification; it is
// distinct from and complementary to the existing REQUIRED_FRAMES/cooldown
// debounce in GestureControl.svelte, which only debounces the classifier's
// *output*, not its input.

function lowPassAlpha(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuroScalarFilter {
  private mincutoff: number;
  private beta: number;
  private dcutoff: number;
  private xPrev: number | null = null;
  private dxPrev = 0;

  constructor(mincutoff = 1.0, beta = 0.3, dcutoff = 1.0) {
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
  }

  filter(x: number, dt: number): number {
    if (this.xPrev === null || dt <= 0) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / dt;
    const aD = lowPassAlpha(this.dcutoff, dt);
    const dxFiltered = this.dxPrev + aD * (dx - this.dxPrev);

    const cutoff = this.mincutoff + this.beta * Math.abs(dxFiltered);
    const a = lowPassAlpha(cutoff, dt);
    const xFiltered = this.xPrev + a * (x - this.xPrev);

    this.xPrev = xFiltered;
    this.dxPrev = dxFiltered;
    return xFiltered;
  }

  /** Constant-velocity extrapolation from the current filtered position and
   * smoothed velocity — a lightweight kinematic prediction, not a learned
   * model. `filter()` always sets `xPrev` on its very first call (even
   * before a velocity estimate exists, `dxPrev` starts at 0), so this only
   * returns the degenerate `0` if called before any frame was ever filtered
   * — callers should go through `LandmarkFilterBank.predictAhead()`, which
   * guards that case at the bank level instead. */
  predict(aheadMs: number): number {
    if (this.xPrev === null) return 0;
    return this.xPrev + this.dxPrev * (aheadMs / 1000);
  }

  reset(): void {
    this.xPrev = null;
    this.dxPrev = 0;
  }
}

/** Filters all 21 landmarks (x, y, z) frame-to-frame. One instance per
 * gesture session — call `reset()` when hand tracking is lost so stale
 * state doesn't smear into a freshly-detected hand. */
export class LandmarkFilterBank {
  private filters: OneEuroScalarFilter[] | null = null;
  private lastT: number | null = null;

  constructor(
    private mincutoff = 1.0,
    private beta = 0.3,
    private dcutoff = 1.0
  ) {}

  filter(landmarks: Landmark[], tNowMs: number): Landmark[] {
    if (!this.filters) {
      this.filters = landmarks.flatMap(() => [
        new OneEuroScalarFilter(this.mincutoff, this.beta, this.dcutoff),
        new OneEuroScalarFilter(this.mincutoff, this.beta, this.dcutoff),
        new OneEuroScalarFilter(this.mincutoff, this.beta, this.dcutoff),
      ]);
    }

    const dt = this.lastT === null ? 1 / 30 : Math.max((tNowMs - this.lastT) / 1000, 1 / 120);
    this.lastT = tNowMs;

    return landmarks.map((lm, i) => ({
      x: this.filters![i * 3].filter(lm.x, dt),
      y: this.filters![i * 3 + 1].filter(lm.y, dt),
      z: this.filters![i * 3 + 2].filter(lm.z ?? 0, dt),
    }));
  }

  /** Constant-velocity extrapolation of every landmark `aheadMs` milliseconds
   * into the future, from the current filtered position + smoothed velocity.
   * Returns `null` if no frame has been filtered yet (nothing to
   * extrapolate from). */
  predictAhead(aheadMs: number): Landmark[] | null {
    if (!this.filters) return null;
    const n = this.filters.length / 3;
    return Array.from({ length: n }, (_, i) => ({
      x: this.filters![i * 3].predict(aheadMs),
      y: this.filters![i * 3 + 1].predict(aheadMs),
      z: this.filters![i * 3 + 2].predict(aheadMs),
    }));
  }

  reset(): void {
    this.filters = null;
    this.lastT = null;
  }
}

/** Blends a currently-filtered landmark position with its predicted future
 * position — `blend = 0` is purely current, `1` is purely predicted. Used
 * to feed the gesture-cursor bridge a lower-perceived-latency target without
 * fully committing to the (noisier, longer-horizon) raw prediction. */
export function predictCursorTarget(filtered: Landmark, predicted: Landmark, blend: number): Landmark {
  const t = Math.max(0, Math.min(1, blend));
  return {
    x: filtered.x + (predicted.x - filtered.x) * t,
    y: filtered.y + (predicted.y - filtered.y) * t,
    z: (filtered.z ?? 0) + ((predicted.z ?? 0) - (filtered.z ?? 0)) * t,
  };
}

/** Map a normalized, unmirrored MediaPipe target to an absolute screen point.
 *
 * Sensitivity is a gain around screen centre: 1 preserves the original
 * full-screen mapping, values below 1 reduce travel, and values above 1
 * expand travel until it clamps at the screen edge.
 */
export function mapCursorTargetToScreen(
  target: Landmark,
  screenWidth: number,
  screenHeight: number,
  sensitivity: number
): { x: number; y: number } {
  const width = Math.max(1, Math.floor(screenWidth));
  const height = Math.max(1, Math.floor(screenHeight));
  const gain = Math.max(0.1, Math.min(3, sensitivity));
  const mirroredX = 1 - target.x;
  const normalizedX = 0.5 + (mirroredX - 0.5) * gain;
  const normalizedY = 0.5 + (target.y - 0.5) * gain;

  return {
    x: Math.round(Math.max(0, Math.min(width - 1, normalizedX * width))),
    y: Math.round(Math.max(0, Math.min(height - 1, normalizedY * height))),
  };
}

/**
 * Scores how much a predicted next position agrees with an observed
 * per-axis motion direction (e.g. a swipe's dx, or a circular gesture's
 * tangential direction) — used to scale down confidence when the predicted
 * trajectory contradicts what a motion classifier just detected (reducing
 * misfires), without touching the classifiers' own thresholds.
 *
 * Returns 1 when the predicted delta's sign matches `observedDelta`'s sign
 * (or `observedDelta` is ~0 — nothing to contradict), and decays toward a
 * floor (never fully zero — a single noisy prediction shouldn't veto an
 * otherwise-solid classification) when they disagree.
 */
const TRAJECTORY_DISAGREEMENT_FLOOR = 0.5;

export function trajectoryAgreement(observedDelta: number, predictedDelta: number): number {
  if (Math.abs(observedDelta) < 1e-6) return 1;
  const agrees = Math.sign(observedDelta) === Math.sign(predictedDelta);
  return agrees ? 1 : TRAJECTORY_DISAGREEMENT_FLOOR;
}

// ── Occlusion/visibility-aware quality score ──
//
// Two cheap, combinable signals: MediaPipe's own per-hand confidence (base
// detection quality), and a geometric self-consistency check (do finger
// tip-to-MCP distances fall in a plausible range for a clearly-visible,
// non-edge-on hand). Neither requires a new model or dependency.

const FINGERS: Array<[mcp: number, tip: number]> = [
  [5, 8], // index
  [9, 12], // middle
  [13, 16], // ring
  [17, 20], // pinky
];

// Empirical bounds for tip-to-MCP distance / handSize. A clearly visible,
// front-facing hand's fingers (extended or curled) fall roughly in this
// range; a badly foreshortened/occluded/edge-on hand tends to fall outside
// it (fingers compressed to near-zero apparent length, or geometrically
// implausible stretch from tracking noise). Approximate, not derived from
// real MediaPipe data — widened after spatialModel.test.ts's synthetic
// open-palm fixture caught the initial bounds (0.15-1.6) penalizing an
// ordinary extended-finger pose (ratios of 1.6-1.8 are normal when fingers
// are longer than the wrist-to-middle-MCP reference segment). Revisit
// against real camera landmarks if false-positive quality suppression
// shows up in testing.
const PLAUSIBLE_RATIO_MIN = 0.12;
const PLAUSIBLE_RATIO_MAX = 2.5;

export function geometricQuality(landmarks: Landmark[], size: number = handSize(landmarks)): number {
  let penalized = 0;
  for (const [mcp, tip] of FINGERS) {
    const ratio = dist3d(landmarks[mcp], landmarks[tip]) / size;
    if (ratio < PLAUSIBLE_RATIO_MIN || ratio > PLAUSIBLE_RATIO_MAX) {
      penalized++;
    }
  }
  // Each implausible finger knocks down quality; four implausible fingers
  // (a truly degenerate frame) bottoms out near 0 rather than exactly 0, so
  // a single bad frame doesn't fully zero out confidence.
  return Math.max(0, 1 - penalized / FINGERS.length);
}

/** Combines MediaPipe's own detection confidence with the geometric
 * self-consistency check. `handednessScore` should come from
 * `results.multiHandedness[0].score` (already computed by @mediapipe/hands,
 * currently unused by GestureControl.svelte). */
export function computeHandQuality(
  landmarks: Landmark[],
  handednessScore: number | undefined,
  size: number = handSize(landmarks)
): number {
  const qDetection = handednessScore ?? 1;
  const qGeometric = geometricQuality(landmarks, size);
  return qDetection * qGeometric;
}

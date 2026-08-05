/**
 * Real-metric-scale 3D world-model layer for hand gesture recognition.
 *
 * Additive sibling to spatialModel.ts, kept in a separate file on purpose:
 * spatialModel.ts's own docstring explicitly scopes it to the normalized
 * (image-space) landmark layer that every existing classifyGesture()
 * threshold in GestureControl.svelte is tuned against — nothing in this
 * file touches that path. This module only consumes `worldLandmarks`,
 * which the "tasks" backend (@mediapipe/tasks-vision's HandLandmarker)
 * exposes in addition to (not instead of) the normalized `landmarks` array
 * spatialModel.ts already handles. worldLandmarks are real-world 3D
 * coordinates in meters, roughly centered on the hand's geometric center —
 * the legacy @mediapipe/hands API never exposes this.
 *
 * What this adds:
 *  - toWristRelative3D(): re-anchors worldLandmarks (hand-center-relative
 *    by default) to the wrist, matching the wrist-relative convention the
 *    existing 2D handSize()/thumbExtensionRatio() already use.
 *  - handSize3D()/pinchDistance3D(): metric (meters) analogs of
 *    handSize()/the discrete pinch gesture's 2D distance check — new,
 *    camera-distance-invariant capabilities, not replacements.
 *  - detectPushPull3D(): a metric-threshold alternative to
 *    GestureControl.svelte's ad hoc ±0.06 normalized-z detectPushPull(),
 *    active only under the "tasks" backend.
 *  - WorldModelFilterBank: a temporal filter bank for worldLandmarks, like
 *    spatialModel.ts's LandmarkFilterBank, but COUPLED across x/y/z per
 *    landmark — one shared 3D velocity vector and one adaptive cutoff
 *    derived from combined 3D speed, rather than three independently
 *    extrapolated axes. This keeps predictAhead() aligned to the true 3D
 *    motion direction instead of drifting per-axis.
 *
 * All thresholds here are approximate (not tuned against real camera
 * input) — see worldModel.test.ts for the numeric fixtures they're
 * expected to satisfy, and revisit once this backend gets live testing.
 */

import type { Landmark } from "./spatialModel";

export type { Landmark };

// MediaPipe hand 21-landmark topology (shared with spatialModel.ts).
const WRIST = 0;
const MIDDLE_MCP = 9;
const THUMB_TIP = 4;
const INDEX_TIP = 8;

export interface FingerExtensionStates3D {
  index: boolean;
  middle: boolean;
  ring: boolean;
  pinky: boolean;
}

export type ThumbState3D = "extended" | "tucked" | "uncertain";

function dist3d(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function angleDegrees(a: Landmark, vertex: Landmark, c: Landmark): number {
  const ab = {
    x: a.x - vertex.x,
    y: a.y - vertex.y,
    z: (a.z ?? 0) - (vertex.z ?? 0),
  };
  const cb = {
    x: c.x - vertex.x,
    y: c.y - vertex.y,
    z: (c.z ?? 0) - (vertex.z ?? 0),
  };
  const abLength = Math.sqrt(ab.x * ab.x + ab.y * ab.y + ab.z * ab.z);
  const cbLength = Math.sqrt(cb.x * cb.x + cb.y * cb.y + cb.z * cb.z);
  if (abLength <= 1e-8 || cbLength <= 1e-8) return 0;
  const cosine = (ab.x * cb.x + ab.y * cb.y + ab.z * cb.z) / (abLength * cbLength);
  return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
}

function isFiniteLandmark(landmark: Landmark | undefined): landmark is Landmark {
  return Boolean(
    landmark && Number.isFinite(landmark.x) && Number.isFinite(landmark.y) && Number.isFinite(landmark.z ?? 0),
  );
}

function isFingerExtended3D(
  landmarks: Landmark[],
  mcpIndex: number,
  pipIndex: number,
  dipIndex: number,
  tipIndex: number,
): boolean {
  const wrist = landmarks[WRIST];
  const mcp = landmarks[mcpIndex];
  const pip = landmarks[pipIndex];
  const dip = landmarks[dipIndex];
  const tip = landmarks[tipIndex];
  if (![wrist, mcp, pip, dip, tip].every(isFiniteLandmark)) return false;

  // A straight finger is close to 180 degrees at both joints. Combining
  // joint angles with wrist distance keeps the test orientation-independent
  // while rejecting a folded finger whose joints happen to align in depth.
  const pipAngle = angleDegrees(mcp, pip, dip);
  const dipAngle = angleDegrees(pip, dip, tip);
  const tipFromWrist = dist3d(tip, wrist);
  const pipFromWrist = dist3d(pip, wrist);
  return pipAngle >= 145 && dipAngle >= 145 && tipFromWrist >= pipFromWrist * 1.08;
}

/**
 * Classifies the four non-thumb fingers from real 3D joint geometry.
 *
 * Unlike the legacy `tip.y < pip.y` rule, this works when a hand is tilted,
 * rotated sideways, or pointed partly toward the camera. `null` means the
 * world-landmark frame is incomplete/invalid and callers should use their
 * established normalized-landmark fallback.
 */
export function fingerExtensionStates3D(worldLandmarks: Landmark[]): FingerExtensionStates3D | null {
  if (worldLandmarks.length < 21 || !worldLandmarks.every(isFiniteLandmark)) return null;
  return {
    index: isFingerExtended3D(worldLandmarks, 5, 6, 7, 8),
    middle: isFingerExtended3D(worldLandmarks, 9, 10, 11, 12),
    ring: isFingerExtended3D(worldLandmarks, 13, 14, 15, 16),
    pinky: isFingerExtended3D(worldLandmarks, 17, 18, 19, 20),
  };
}

/**
 * Uses metric 3D spacing to distinguish an extended thumb from a confidently
 * tucked one. The middle band is intentionally "uncertain": callers must not
 * turn an ambiguous thumb into a high-impact four-finger/fist action.
 */
export function classifyThumbState3D(worldLandmarks: Landmark[]): ThumbState3D | null {
  if (worldLandmarks.length < 21 || !worldLandmarks.every(isFiniteLandmark)) return null;
  const wristRelative = toWristRelative3D(worldLandmarks);
  const size = handSize3D(wristRelative);
  const thumbToIndexMcpRatio = dist3d(wristRelative[THUMB_TIP], wristRelative[5]) / size;
  if (thumbToIndexMcpRatio >= 0.62) return "extended";
  if (thumbToIndexMcpRatio <= 0.42) return "tucked";
  return "uncertain";
}

/** Re-anchors worldLandmarks (hand-center-relative by default, per the
 * MediaPipe Tasks HandLandmarker docs) to the wrist, so downstream metric
 * functions share the same wrist-relative convention as the existing 2D
 * handSize()/thumbExtensionRatio() in spatialModel.ts. */
export function toWristRelative3D(worldLandmarks: Landmark[]): Landmark[] {
  const wrist = worldLandmarks[WRIST];
  const wx = wrist.x;
  const wy = wrist.y;
  const wz = wrist.z ?? 0;
  return worldLandmarks.map((lm) => ({
    x: lm.x - wx,
    y: lm.y - wy,
    z: (lm.z ?? 0) - wz,
  }));
}

/** Wrist-to-middle-MCP distance in real-world meters — the metric analog of
 * spatialModel.ts's handSize(), which only measures this in normalized
 * image space (so it shrinks/grows with camera distance). Expects
 * wrist-relative input (see toWristRelative3D()). */
export function handSize3D(wristRelative: Landmark[]): number {
  return dist3d(wristRelative[WRIST], wristRelative[MIDDLE_MCP]) || 1e-6;
}

/** Real metric (meters) thumb-tip-to-index-tip distance — a
 * camera-distance-invariant complement to the existing normalized-space
 * `dist(THUMB_TIP, INDEX_TIP) < PINCH_DISTANCE_THRESHOLD` pinch check in
 * GestureControl.svelte's classifyGesture(). This is a new capability, not
 * a replacement: the existing 2D pinch threshold stays exactly as tuned. */
export function pinchDistance3D(wristRelative: Landmark[]): number {
  return dist3d(wristRelative[THUMB_TIP], wristRelative[INDEX_TIP]);
}

// Empirical metric (meters) depth-change threshold for a push/pull motion,
// analogous to GestureControl.svelte's existing ±0.06 normalized-z
// threshold in its inline detectPushPull() -- expressed here in real-world
// meters using worldLandmarks instead of normalized image-space z.
// Approximate, not tuned against real camera data.
const PUSH_PULL_DEPTH_METERS = 0.05;

/** Metric-threshold push/pull detector over a short buffer of past wrist
 * worldLandmark readings (oldest first, newest last) — mirrors
 * GestureControl.svelte's wristHistory-based detectPushPull(), but
 * threshold in real meters instead of normalized z. Negative z is closer to
 * the camera (push); positive is further away (pull), matching the sign
 * convention of the existing normalized-z check. */
export function detectPushPull3D(worldWristHistory: Landmark[]): "push" | "pull" | null {
  if (worldWristHistory.length < 2) return null;
  const first = worldWristHistory[0];
  const last = worldWristHistory.at(-1);
  const dz = (last.z ?? 0) - (first.z ?? 0);

  if (dz < -PUSH_PULL_DEPTH_METERS) return "push";
  if (dz > PUSH_PULL_DEPTH_METERS) return "pull";
  return null;
}

// ── Coupled 3D temporal filtering ──
//
// Unlike spatialModel.ts's LandmarkFilterBank (three independent
// OneEuroScalarFilter instances per landmark, one per axis, each adapting
// its cutoff from its OWN per-axis velocity), this filter derives a single
// adaptive cutoff per landmark from the COMBINED 3D speed and applies it
// uniformly across x/y/z. That keeps the smoothed velocity a true 3D vector
// -- so predictAhead() extrapolates along the actual motion direction
// instead of each axis drifting off at its own independently-smoothed rate.

function lowPassAlpha(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuroVector3Filter {
  private prev: Landmark | null = null;
  private dPrev: Landmark = { x: 0, y: 0, z: 0 };

  constructor(
    private mincutoff = 1.0,
    private beta = 0.3,
    private dcutoff = 1.0,
  ) {}

  filter(p: Landmark, dt: number): Landmark {
    const z = p.z ?? 0;
    if (this.prev === null || dt <= 0) {
      this.prev = { x: p.x, y: p.y, z };
      this.dPrev = { x: 0, y: 0, z: 0 };
      return this.prev;
    }

    const dx = (p.x - this.prev.x) / dt;
    const dy = (p.y - this.prev.y) / dt;
    const dz = (z - (this.prev.z ?? 0)) / dt;

    const aD = lowPassAlpha(this.dcutoff, dt);
    const prevD = this.dPrev;
    const dxF = prevD.x + aD * (dx - prevD.x);
    const dyF = prevD.y + aD * (dy - prevD.y);
    const dzF = (prevD.z ?? 0) + aD * (dz - (prevD.z ?? 0));

    // Single coupled cutoff from combined 3D speed, not per-axis.
    const speed = Math.sqrt(dxF * dxF + dyF * dyF + dzF * dzF);
    const cutoff = this.mincutoff + this.beta * speed;
    const a = lowPassAlpha(cutoff, dt);

    const xF = this.prev.x + a * (p.x - this.prev.x);
    const yF = this.prev.y + a * (p.y - this.prev.y);
    const zF = (this.prev.z ?? 0) + a * (z - (this.prev.z ?? 0));

    this.prev = { x: xF, y: yF, z: zF };
    this.dPrev = { x: dxF, y: dyF, z: dzF };
    return this.prev;
  }

  /** Constant-velocity extrapolation using the coupled 3D velocity vector.
   * Returns the origin if `filter()` has never been called — callers
   * should go through `WorldModelFilterBank.predictAhead()`, which guards
   * that case at the bank level instead. */
  predict(aheadMs: number): Landmark {
    if (this.prev === null) return { x: 0, y: 0, z: 0 };
    const t = aheadMs / 1000;
    const d = this.dPrev;
    return {
      x: this.prev.x + d.x * t,
      y: this.prev.y + d.y * t,
      z: (this.prev.z ?? 0) + (d.z ?? 0) * t,
    };
  }

  reset(): void {
    this.prev = null;
    this.dPrev = { x: 0, y: 0, z: 0 };
  }
}

/** Filters all worldLandmarks frame-to-frame with a coupled 3D velocity
 * per landmark (see module docstring). One instance per gesture session —
 * call `reset()` when hand tracking is lost so stale state doesn't smear
 * into a freshly-detected hand. */
export class WorldModelFilterBank {
  private filters: OneEuroVector3Filter[] | null = null;
  private lastT: number | null = null;

  constructor(
    private mincutoff = 1.0,
    private beta = 0.3,
    private dcutoff = 1.0,
  ) {}

  filter(worldLandmarks: Landmark[], tNowMs: number): Landmark[] {
    if (!this.filters) {
      this.filters = worldLandmarks.map(() => new OneEuroVector3Filter(this.mincutoff, this.beta, this.dcutoff));
    }

    const dt = this.lastT === null ? 1 / 30 : Math.max((tNowMs - this.lastT) / 1000, 1 / 120);
    this.lastT = tNowMs;

    return worldLandmarks.map((lm, i) => this.filters![i].filter(lm, dt));
  }

  /** Constant-velocity extrapolation of every landmark `aheadMs`
   * milliseconds into the future, along its coupled 3D velocity vector.
   * Returns `null` if no frame has been filtered yet. */
  predictAhead(aheadMs: number): Landmark[] | null {
    if (!this.filters) return null;
    return this.filters.map((f) => f.predict(aheadMs));
  }

  reset(): void {
    this.filters = null;
    this.lastT = null;
  }
}

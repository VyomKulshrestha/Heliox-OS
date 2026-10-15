import { mapCursorTargetToScreen, type Landmark } from "./spatialModel";

/**
 * Pure helpers for the in-window study pointer and for re-arming discrete
 * gestures (spec sections 16.3, 17.2 F2 and 17.4). GestureControl owns the
 * camera loop; these functions hold no DOM, store or Tauri state, so the
 * rules they encode are unit-tested on their own.
 */

/** Consecutive frames below the pinch threshold before a pinch counts as closed. */
export const PINCH_CLOSE_FRAMES = 2;
/** A closed pinch opens again only above this multiple of the threshold. */
export const PINCH_OPEN_FACTOR = 1.3;
/** Same value as GestureControl's HAND_ACQUIRE_FRAMES. */
export const STUDY_ACQUIRE_FRAMES = 8;

export interface PinchHysteresis {
  /** Feeds one measured thumb-to-index distance and returns whether the pinch is closed. `threshold` defaults to
   * the one the hysteresis was created with; GestureControl passes its calibrated threshold every frame. */
  update(distance: number, threshold?: number): boolean;
  /** Back to open, as on a hand loss or when the study pointer is requested again. */
  reset(): void;
  readonly closed: boolean;
}

export function createPinchHysteresis(threshold: number): PinchHysteresis {
  let closed = false;
  let framesBelow = 0;
  return {
    update(distance: number, currentThreshold = threshold): boolean {
      if (!Number.isFinite(distance)) return closed;
      if (closed) {
        if (distance > currentThreshold * PINCH_OPEN_FACTOR) {
          closed = false;
          framesBelow = 0;
        }
      } else if (distance < currentThreshold) {
        framesBelow += 1;
        if (framesBelow >= PINCH_CLOSE_FRAMES) closed = true;
      } else {
        framesBelow = 0;
      }
      return closed;
    },
    reset(): void {
      closed = false;
      framesBelow = 0;
    },
    get closed() {
      return closed;
    },
  };
}

/**
 * Index-tip position in [0, 1] on both axes, mirrored on x and scaled by the
 * cursor sensitivity exactly as the OS gesture cursor maps it, but on a
 * 1000 x 1000 grid instead of the screen. The tip is the filtered landmark;
 * no prediction is blended in.
 */
export function normalizedPointer(tip: Landmark, sensitivity: number): { nx: number; ny: number } {
  const { x, y } = mapCursorTargetToScreen(tip, 1000, 1000, sensitivity);
  return { nx: x / 999, ny: y / 999 };
}

export interface AcquireCounter {
  /** Counts one frame; a frame without a verified hand restarts the count. Returns whether the hand is acquired. */
  frame(handVerified: boolean): boolean;
  reset(): void;
  readonly count: number;
  readonly acquired: boolean;
}

export function createAcquireCounter(frames: number = STUDY_ACQUIRE_FRAMES): AcquireCounter {
  let count = 0;
  return {
    frame(handVerified: boolean): boolean {
      count = handVerified ? count + 1 : 0;
      return count >= frames;
    },
    reset(): void {
      count = 0;
    },
    get count() {
      return count;
    },
    get acquired() {
      return count >= frames;
    },
  };
}

export interface GestureArmOptions {
  /** A hand absence counts once no verified hand was seen for this long. */
  absentMs: number;
  /** No verified hand for this long re-arms while the hand is still away. */
  idleMs: number;
  /** Verified frames after an absence that re-arm. */
  acquireFrames: number;
}

export const GESTURE_ARM_DEFAULTS: GestureArmOptions = {
  absentMs: 200,
  idleMs: 500,
  acquireFrames: STUDY_ACQUIRE_FRAMES,
};

export interface GestureArm {
  /** Disarms until the hand has left and come back, or no hand was seen for `idleMs`. */
  disarm(now: number): void;
  /** Observes one camera frame. Absence is learned only from frames that show no verified hand, so a gap with
   * no frames at all (a hidden window, a stalled camera) never counts as the hand having left. */
  frame(handVerified: boolean, now: number): boolean;
  /** True when armed and the gesture's recognition began at or after the re-arm. */
  allows(gestureStartedAt: number): boolean;
  readonly armed: boolean;
  /** When the arm last re-armed; null while disarmed. Armed from the start before any disarm. */
  readonly armedAt: number | null;
}

/**
 * Re-arm gate for discrete gestures (17.4, F2(d)). After `disarm`, a hand seen
 * continuously keeps it disarmed. It re-arms after the hand was absent for at
 * least `absentMs` and then verified on `acquireFrames` consecutive frames, or
 * after `idleMs` with no verified hand. A gesture must begin at or after the
 * re-arm, so a pose held across the disarm never fires.
 */
export function createGestureArm(options: Partial<GestureArmOptions> = {}): GestureArm {
  const { absentMs, idleMs, acquireFrames } = { ...GESTURE_ARM_DEFAULTS, ...options };
  let armed = true;
  let armedAt: number | null = Number.NEGATIVE_INFINITY;
  let disarmedAt = Number.NEGATIVE_INFINITY;
  let lastVerifiedAt = Number.NEGATIVE_INFINITY;
  let sawNoHandSinceVerified = false;
  let handLeft = false;
  let presentFrames = 0;

  function arm(now: number): void {
    armed = true;
    armedAt = Math.max(now, disarmedAt);
    handLeft = false;
    presentFrames = 0;
  }

  return {
    disarm(now: number): void {
      armed = false;
      armedAt = null;
      disarmedAt = now;
      handLeft = false;
      presentFrames = 0;
    },
    frame(handVerified: boolean, now: number): boolean {
      if (handVerified) {
        if (!armed) {
          if (sawNoHandSinceVerified && now - lastVerifiedAt >= absentMs) handLeft = true;
          presentFrames = handLeft ? presentFrames + 1 : 0;
          if (handLeft && presentFrames >= acquireFrames) arm(now);
        }
        lastVerifiedAt = now;
        sawNoHandSinceVerified = false;
        return armed;
      }
      sawNoHandSinceVerified = true;
      presentFrames = 0;
      if (!armed) {
        const absentFor = now - lastVerifiedAt;
        if (absentFor >= absentMs) handLeft = true;
        if (absentFor >= idleMs) arm(now);
      }
      return armed;
    },
    allows(gestureStartedAt: number): boolean {
      return armed && armedAt !== null && gestureStartedAt >= armedAt;
    },
    get armed() {
      return armed;
    },
    get armedAt() {
      return armedAt;
    },
  };
}

/** No gaze_event RPC while a study session is active, in any modality, paused or not (17.4). */
export function gazeEventAllowed(studyActive: boolean): boolean {
  return !studyActive;
}

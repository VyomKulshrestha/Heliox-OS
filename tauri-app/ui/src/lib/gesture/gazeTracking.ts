/**
 * Coarse, on-device gaze-region estimation from MediaPipe Tasks-Vision's
 * FaceLandmarker output (478-point face mesh with iris refinement).
 *
 * Deliberately NOT pixel-precise pointing — a consumer webcam has no IR
 * eye-tracker hardware, and pixel-accurate gaze estimation from RGB alone
 * needs a real per-user calibration routine this module doesn't attempt.
 * Instead, this reports one of five coarse regions (center/left/right/up/
 * down) from the iris's position within each eye socket, relative to that
 * eye's own corner/eyelid landmarks — a "which rough direction is the user
 * looking" signal, not "which pixel." See GESTURES.md's gaze-tracking
 * section for how this feeds the multimodal fusion engine as a passive
 * disambiguating signal alongside voice + gesture, never as a standalone
 * command trigger on its own.
 *
 * Pure geometry, no MediaPipe/DOM dependency — testable in isolation from
 * the camera/component, same pattern as spatialModel.ts/worldModel.ts.
 *
 * Privacy: this function's only output is a coarse region label + a
 * confidence float — never raw landmarks. GestureControl.svelte sends only
 * that label to the backend, never face geometry or video frames,
 * mirroring how gesture events already send just a gesture name.
 */

export interface FaceLandmark {
  x: number;
  y: number;
  z?: number;
}

export type GazeRegion = "center" | "left" | "right" | "up" | "down";

export interface GazeEstimate {
  region: GazeRegion;
  confidence: number;
}

export type HandTrackingBackend = "legacy" | "tasks";

/** Legacy Hands and Tasks-Vision cannot safely coexist because both WASM
 * bundles install an Emscripten `Module` global. A gaze session therefore
 * keeps both hand and face models on Tasks-Vision. */
export function resolveHandBackend(
  configured: HandTrackingBackend | undefined,
  gazeEnabled: boolean,
): HandTrackingBackend {
  return gazeEnabled || configured === "tasks" ? "tasks" : "legacy";
}

// Fusion only considers readings inside its short correlation window. A
// steady gaze therefore needs a heartbeat as well as change-based updates;
// otherwise an unchanged region silently expires after the first event.
export const GAZE_HEARTBEAT_MS = 750;

export function shouldSendGazeUpdate(
  region: GazeRegion,
  previousRegion: GazeRegion | null,
  nowMs: number,
  previousSentAtMs: number,
): boolean {
  return region !== previousRegion || nowMs - previousSentAtMs >= GAZE_HEARTBEAT_MS;
}

// MediaPipe's 478-point face mesh topology (iris refinement enabled) —
// fixed indices, not derived at runtime.
const LEFT_EYE = { iris: 468, outer: 33, inner: 133, top: 159, bottom: 145 };
const RIGHT_EYE = { iris: 473, outer: 263, inner: 362, top: 386, bottom: 374 };
const MIN_LANDMARK_COUNT = 478;

// Approximate — not tuned against a real webcam/real users. A dead zone
// around the eye socket's geometric center absorbs normal jitter and head
// micro-movement so "center" doesn't flicker to a side reading on every
// frame. Revisit against real usage, same caveat as every other
// empirically-tuned threshold in this gesture pipeline (spatialModel.ts,
// worldModel.ts) that hasn't been validated against real camera data yet.
const HORIZONTAL_DEADZONE = 0.15;
const VERTICAL_DEADZONE = 0.15;

interface EyeRatio {
  horizontal: number; // 0..1 across the eye socket, not screen-relative
  vertical: number; // 0..1 across the eye socket, not screen-relative
}

function eyeGazeRatio(
  landmarks: FaceLandmark[],
  eye: { iris: number; outer: number; inner: number; top: number; bottom: number }
): EyeRatio {
  const iris = landmarks[eye.iris];
  const outer = landmarks[eye.outer];
  const inner = landmarks[eye.inner];
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];

  const minX = Math.min(outer.x, inner.x);
  const width = Math.abs(inner.x - outer.x) || 1e-6;
  const horizontal = (iris.x - minX) / width;

  const minY = Math.min(top.y, bottom.y);
  const height = Math.abs(bottom.y - top.y) || 1e-6;
  const vertical = (iris.y - minY) / height;

  return { horizontal, vertical };
}

/** Estimates a coarse gaze region from a full 478-point FaceLandmarker
 * reading (raw, unmirrored camera-frame landmark space — same convention
 * GestureControl.svelte's cursor bridge already flips for display, see
 * its "Coordinate mapping" note). Returns null if the landmark array isn't
 * a full face mesh reading (e.g. no face detected this frame). */
export function estimateGazeRegion(landmarks: FaceLandmark[] | null | undefined): GazeEstimate | null {
  if (!landmarks || landmarks.length < MIN_LANDMARK_COUNT) return null;

  const left = eyeGazeRatio(landmarks, LEFT_EYE);
  const right = eyeGazeRatio(landmarks, RIGHT_EYE);

  // Average both eyes — more robust than trusting one alone, since a head
  // turn can foreshorten/occlude one eye's landmarks more than the other.
  const horizontal = (left.horizontal + right.horizontal) / 2;
  const vertical = (left.vertical + right.vertical) / 2;

  const dx = horizontal - 0.5;
  const dy = vertical - 0.5;

  if (Math.abs(dx) < HORIZONTAL_DEADZONE && Math.abs(dy) < VERTICAL_DEADZONE) {
    const maxDeadzone = Math.max(HORIZONTAL_DEADZONE, VERTICAL_DEADZONE);
    const confidence = 1 - Math.max(Math.abs(dx), Math.abs(dy)) / maxDeadzone;
    return { region: "center", confidence };
  }

  // Plus-shaped discretization (not a full 3x3 grid): whichever axis
  // deviates further from center wins, deliberately not trying to resolve
  // diagonal corners precisely — simpler and more robust to webcam noise.
  if (Math.abs(dx) > Math.abs(dy)) {
    return { region: dx > 0 ? "right" : "left", confidence: Math.min(1, Math.abs(dx) / 0.5) };
  }
  return { region: dy > 0 ? "down" : "up", confidence: Math.min(1, Math.abs(dy) / 0.5) };
}

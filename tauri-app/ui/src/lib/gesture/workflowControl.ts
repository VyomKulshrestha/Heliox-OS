/**
 * Gesture control signals for a paused/waiting-for-trigger
 * VoiceGestureWorkflow (see daemon/pilot/agents/voice_gesture_workflow.py).
 *
 * Deliberately simple and NOT modeled on calibration.ts's pending-event/
 * timeout shape — there's no ambiguity to resolve here (calibration infers
 * an outcome from whatever gesture happens to follow within a window).
 * Here, a specific gesture is only ever consulted as a control signal when
 * the caller already knows a gesture-sourced workflow is currently
 * PAUSED/WAITING_FOR_TRIGGER (tracked via the voice_gesture_workflow_state
 * notification) — this module is just the gesture-name-to-intent mapping,
 * kept separate so it's trivially unit-testable.
 *
 * `thumbs_up` -> continue and `palm` -> cancel are deliberate choices:
 * palm already universally means "Cancel/Stop" everywhere else in this
 * app (the cursor-mode escape hatch, checked before anything else in
 * GestureControl.svelte), so reusing it here is consistent, not a new
 * meaning to learn.
 */

export type GestureControlIntent = "continue" | "cancel" | "unknown";

const CONTINUE_GESTURES = new Set(["thumbs_up"]);
const CANCEL_GESTURES = new Set(["palm", "thumbs_down"]);

export function classifyControlGesture(gestureName: string): GestureControlIntent {
  if (CONTINUE_GESTURES.has(gestureName)) return "continue";
  if (CANCEL_GESTURES.has(gestureName)) return "cancel";
  return "unknown";
}

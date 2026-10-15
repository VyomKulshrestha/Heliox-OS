import type { ConfirmOutcome } from "../stores/session";
import type { DefaultGestureAction } from "./actionPolicy";

/**
 * Decides what an approve or deny gesture may do with a pending approval
 * (spec 17.2 F1 and F2, 17.4). Approval by gesture is narrower than approval
 * in the dialog: the dialog must have been visible on the Command tab for at
 * least a second, the hand must have left the camera view and come back since
 * it appeared, and only the actions the dialog shows as checked are sent.
 * Denying is always allowed once discrete gestures are armed, so this
 * function never refuses a deny.
 */

/** How long the dialog must have been visible before a gesture may approve it. */
export const GESTURE_APPROVAL_MIN_VISIBLE_MS = 1000;

export const GESTURE_APPROVAL_MESSAGES = {
  notVisible: "Open the Command tab to approve. Gestures can approve only a dialog you have had time to see.",
  rearm: "Lower your hand out of view, then raise it to approve by gesture.",
  nothingChecked: "Nothing is checked. Check the actions you want to approve, or deny.",
  noPending: "No approval is pending.",
} as const;

/** i18n keys of the messages above; the English text is the fallback. */
export const GESTURE_APPROVAL_MESSAGE_KEYS = {
  notVisible: "gesture.approve_not_visible",
  rearm: "gesture.approve_rearm",
  nothingChecked: "confirm.nothing_checked",
  noPending: "gesture.no_approval_pending",
} as const;

export type GestureApprovalMessage = keyof typeof GESTURE_APPROVAL_MESSAGES;

export interface GestureApprovalInput {
  /** The gesture's default action (`defaultGestureAction`). */
  action: DefaultGestureAction;
  confirmRequired: boolean;
  /** The pending plan's id; a selection reported for any other plan is not used. */
  planId: string;
  confirmVisible: boolean;
  /** Milliseconds the dialog has been visible, or null while it is not. */
  visibleForMs: number | null;
  /** The approve arm is armed (F2(d)). */
  armed: boolean;
  /** The gesture began at or after the approve arm's last re-arm. */
  gestureStartedAfterArm: boolean;
  /** What the dialog last reported as checked. */
  selection: { planId: string; indices: number[] } | null;
}

export type GestureApprovalDecision =
  | { kind: "approve"; indices: number[] }
  | { kind: "deny" }
  | { kind: "refuse"; message: string; reason: GestureApprovalMessage }
  | { kind: "none"; message: string; reason: GestureApprovalMessage | "not_an_approval_gesture" };

function withMessage<K extends "refuse" | "none">(kind: K, reason: GestureApprovalMessage) {
  return { kind, message: GESTURE_APPROVAL_MESSAGES[reason], reason };
}

export function gestureApprovalDecision(input: GestureApprovalInput): GestureApprovalDecision {
  if (input.action !== "confirm" && input.action !== "deny") {
    return { kind: "none", message: "", reason: "not_an_approval_gesture" };
  }
  if (!input.confirmRequired || !input.planId) return withMessage("none", "noPending");
  if (input.action === "deny") return { kind: "deny" };

  if (
    !input.confirmVisible ||
    input.visibleForMs === null ||
    !Number.isFinite(input.visibleForMs) ||
    input.visibleForMs < GESTURE_APPROVAL_MIN_VISIBLE_MS
  ) {
    return withMessage("refuse", "notVisible");
  }
  if (!input.armed || !input.gestureStartedAfterArm) return withMessage("refuse", "rearm");

  const selection = input.selection;
  if (!selection || selection.planId !== input.planId || selection.indices.length === 0) {
    return withMessage("refuse", "nothingChecked");
  }
  return { kind: "approve", indices: [...selection.indices] };
}

/**
 * Chat lines for an approve or deny gesture, chosen from what the daemon recorded. "Approval accepted." and
 * "Approval denied." appear only when the daemon replied ok with that same decision, so the chat never says
 * denied for a plan that runs, or accepted for one that does not (SECURITY.md, spec 17.2 F6).
 */
export const GESTURE_ANSWER_MESSAGES = {
  approved: "Approval accepted.",
  denied: "Approval denied.",
  approveInFlight: "Another answer to this approval is already being sent, so this approval gesture sent nothing.",
  denyInFlight:
    "Another answer to this approval is already being sent, so this denial gesture sent nothing. To stop the task, say or press Stop.",
  missingPlan: "This approval request is missing its plan ID, so the gesture sent nothing.",
  alreadyApproved:
    "This plan was already approved, so the denial gesture changed nothing. To stop the task, say or press Stop.",
  alreadyDenied: "This plan was already denied, so the approval gesture changed nothing.",
  unknownDecision: "Heliox did not say which answer it recorded. Check the task before relying on it.",
} as const;

/** The chat line for a gesture answer, or null when the store already shows the reason ("Approval failed: …"). */
export function gestureAnswerMessage(accepted: boolean, outcome: ConfirmOutcome): string | null {
  switch (outcome.kind) {
    case "recorded":
      if (outcome.confirmed === accepted) {
        return accepted ? GESTURE_ANSWER_MESSAGES.approved : GESTURE_ANSWER_MESSAGES.denied;
      }
      if (outcome.confirmed === null) return GESTURE_ANSWER_MESSAGES.unknownDecision;
      return accepted ? GESTURE_ANSWER_MESSAGES.alreadyDenied : GESTURE_ANSWER_MESSAGES.alreadyApproved;
    case "not_sent":
      if (outcome.reason === "submitting") {
        return accepted ? GESTURE_ANSWER_MESSAGES.approveInFlight : GESTURE_ANSWER_MESSAGES.denyInFlight;
      }
      if (outcome.reason === "nothing_checked") return GESTURE_APPROVAL_MESSAGES.nothingChecked;
      return GESTURE_ANSWER_MESSAGES.missingPlan;
    case "rejected":
      if (outcome.errorShown) return null;
      return `${accepted ? "The approval gesture" : "The denial gesture"} was not recorded: ${outcome.message}`;
  }
}

/**
 * Sends a gesture's answer through `confirm` and posts the chat line for what the daemon actually recorded,
 * once its reply is in. `confirm` is `session.confirm` with the gesture channel; an approval sends exactly the
 * selection the dialog reported for the pending plan, never "everything".
 */
export async function sendGestureAnswer(
  accepted: boolean,
  confirm: (accepted: boolean) => Promise<ConfirmOutcome>,
  post: (text: string) => void,
): Promise<ConfirmOutcome> {
  const outcome = await confirm(accepted);
  const text = gestureAnswerMessage(accepted, outcome);
  if (text) post(text);
  return outcome;
}

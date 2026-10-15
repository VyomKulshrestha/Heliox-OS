import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { ConfirmOutcome } from "../stores/session";
import {
  GESTURE_ANSWER_MESSAGES,
  GESTURE_APPROVAL_MESSAGES,
  gestureAnswerMessage,
  gestureApprovalDecision,
  sendGestureAnswer,
  type GestureApprovalInput,
} from "./approvalGesture";

function input(overrides: Partial<GestureApprovalInput> = {}): GestureApprovalInput {
  return {
    action: "confirm",
    confirmRequired: true,
    planId: "plan-1",
    confirmVisible: true,
    visibleForMs: 1500,
    armed: true,
    gestureStartedAfterArm: true,
    selection: { planId: "plan-1", indices: [0, 2] },
    ...overrides,
  };
}

describe("gestureApprovalDecision", () => {
  it("approves exactly the reported selection of the pending plan", () => {
    const selection = { planId: "plan-1", indices: [2] };
    const decision = gestureApprovalDecision(input({ selection }));
    expect(decision).toEqual({ kind: "approve", indices: [2] });
    if (decision.kind === "approve") expect(decision.indices).not.toBe(selection.indices);
  });

  it("approves once the dialog has been visible for 1000 ms", () => {
    expect(gestureApprovalDecision(input({ visibleForMs: 1000 })).kind).toBe("approve");
  });

  it("refuses a dialog that is not visible", () => {
    expect(gestureApprovalDecision(input({ confirmVisible: false, visibleForMs: null }))).toMatchObject({
      kind: "refuse",
      message: "Open the Command tab to approve. Gestures can approve only a dialog you have had time to see.",
    });
  });

  it("refuses a dialog visible for less than a second", () => {
    expect(gestureApprovalDecision(input({ visibleForMs: 999 }))).toMatchObject({
      kind: "refuse",
      message: GESTURE_APPROVAL_MESSAGES.notVisible,
    });
    expect(gestureApprovalDecision(input({ visibleForMs: null }))).toMatchObject({ kind: "refuse" });
    expect(gestureApprovalDecision(input({ visibleForMs: Number.NaN }))).toMatchObject({ kind: "refuse" });
  });

  it("refuses until the approve gesture is re-armed", () => {
    expect(gestureApprovalDecision(input({ armed: false, gestureStartedAfterArm: false }))).toMatchObject({
      kind: "refuse",
      message: "Lower your hand out of view, then raise it to approve by gesture.",
    });
  });

  it("refuses a gesture that began before the re-arm", () => {
    expect(gestureApprovalDecision(input({ gestureStartedAfterArm: false }))).toMatchObject({
      kind: "refuse",
      message: GESTURE_APPROVAL_MESSAGES.rearm,
    });
  });

  it("refuses an empty, missing or other-plan selection", () => {
    const nothingChecked = {
      kind: "refuse",
      message: "Nothing is checked. Check the actions you want to approve, or deny.",
    };
    expect(gestureApprovalDecision(input({ selection: { planId: "plan-1", indices: [] } }))).toMatchObject(
      nothingChecked,
    );
    expect(gestureApprovalDecision(input({ selection: null }))).toMatchObject(nothingChecked);
    expect(gestureApprovalDecision(input({ selection: { planId: "plan-0", indices: [0] } }))).toMatchObject(
      nothingChecked,
    );
  });

  it("denies whenever an approval is pending, whatever the dialog or approve arm state", () => {
    expect(
      gestureApprovalDecision(
        input({
          action: "deny",
          confirmVisible: false,
          visibleForMs: null,
          armed: false,
          gestureStartedAfterArm: false,
          selection: null,
        }),
      ),
    ).toEqual({ kind: "deny" });
  });

  it("returns none when no approval is pending", () => {
    expect(gestureApprovalDecision(input({ confirmRequired: false }))).toMatchObject({
      kind: "none",
      message: GESTURE_APPROVAL_MESSAGES.noPending,
    });
    expect(gestureApprovalDecision(input({ action: "deny", confirmRequired: false, planId: "" }))).toMatchObject({
      kind: "none",
    });
  });

  it("returns none for gestures that are not approve or deny", () => {
    expect(gestureApprovalDecision(input({ action: "abort" })).kind).toBe("none");
    expect(gestureApprovalDecision(input({ action: "unbound" })).kind).toBe("none");
  });
});

describe("gestureAnswerMessage", () => {
  it("says accepted or denied only when the daemon recorded that same decision", () => {
    expect(gestureAnswerMessage(true, { kind: "recorded", confirmed: true })).toBe("Approval accepted.");
    expect(gestureAnswerMessage(false, { kind: "recorded", confirmed: false })).toBe("Approval denied.");
  });

  it("does not say denied when the daemon recorded an approval, or accepted when it recorded a denial", () => {
    const deny = gestureAnswerMessage(false, { kind: "recorded", confirmed: true });
    expect(deny).toBe(GESTURE_ANSWER_MESSAGES.alreadyApproved);
    expect(deny).toContain("Stop");
    expect(gestureAnswerMessage(true, { kind: "recorded", confirmed: false })).toBe(
      GESTURE_ANSWER_MESSAGES.alreadyDenied,
    );
    expect(gestureAnswerMessage(false, { kind: "recorded", confirmed: null })).toBe(
      GESTURE_ANSWER_MESSAGES.unknownDecision,
    );
  });

  it("says nothing was sent while another answer is in flight", () => {
    const outcome: ConfirmOutcome = { kind: "not_sent", reason: "submitting" };
    expect(gestureAnswerMessage(false, outcome)).toBe(GESTURE_ANSWER_MESSAGES.denyInFlight);
    expect(gestureAnswerMessage(true, outcome)).toBe(GESTURE_ANSWER_MESSAGES.approveInFlight);
  });

  it("explains the other answers that sent nothing", () => {
    expect(gestureAnswerMessage(true, { kind: "not_sent", reason: "nothing_checked" })).toBe(
      GESTURE_APPROVAL_MESSAGES.nothingChecked,
    );
    expect(gestureAnswerMessage(false, { kind: "not_sent", reason: "missing_plan" })).toBe(
      GESTURE_ANSWER_MESSAGES.missingPlan,
    );
  });

  it("leaves a refused answer to the chat's 'Approval failed' line, or names the refusal when there is none", () => {
    expect(gestureAnswerMessage(false, { kind: "rejected", message: "already resolved", errorShown: true })).toBeNull();
    expect(gestureAnswerMessage(false, { kind: "rejected", message: "already resolved", errorShown: false })).toBe(
      "The denial gesture was not recorded: already resolved",
    );
    expect(gestureAnswerMessage(true, { kind: "rejected", message: "offline", errorShown: false })).toBe(
      "The approval gesture was not recorded: offline",
    );
  });

  it("never says accepted or denied for an outcome other than the matching recorded decision", () => {
    const outcomes: ConfirmOutcome[] = [
      { kind: "not_sent", reason: "submitting" },
      { kind: "not_sent", reason: "missing_plan" },
      { kind: "not_sent", reason: "nothing_checked" },
      { kind: "recorded", confirmed: null },
      { kind: "rejected", message: "x", errorShown: false },
      { kind: "rejected", message: "x", errorShown: true },
    ];
    for (const accepted of [true, false]) {
      for (const outcome of [...outcomes, { kind: "recorded", confirmed: !accepted } as ConfirmOutcome]) {
        const text = gestureAnswerMessage(accepted, outcome) ?? "";
        expect(text).not.toBe(GESTURE_ANSWER_MESSAGES.approved);
        expect(text).not.toBe(GESTURE_ANSWER_MESSAGES.denied);
      }
    }
  });
});

describe("sendGestureAnswer", () => {
  it("posts the chat line only after the daemon's reply, from what it recorded", async () => {
    let reply!: (outcome: ConfirmOutcome) => void;
    const confirm = vi.fn(() => new Promise<ConfirmOutcome>((done) => (reply = done)));
    const post = vi.fn();

    const sent = sendGestureAnswer(false, confirm, post);
    expect(confirm).toHaveBeenCalledWith(false);
    await Promise.resolve();
    expect(post).not.toHaveBeenCalled();

    reply({ kind: "recorded", confirmed: false });
    expect(await sent).toEqual({ kind: "recorded", confirmed: false });
    expect(post).toHaveBeenCalledWith("Approval denied.");
  });

  it("posts nothing when the store already shows why the answer failed", async () => {
    const post = vi.fn();
    await sendGestureAnswer(true, async () => ({ kind: "rejected", message: "x", errorShown: true }), post);
    expect(post).not.toHaveBeenCalled();
  });

  it("is how GestureControl answers an approval: it never posts accepted or denied before the reply", () => {
    const source = readFileSync(resolve(process.cwd(), "src/lib/components/GestureControl.svelte"), "utf-8");
    expect(source).toContain("sendGestureAnswer(");
    expect(source).not.toContain("Approval accepted.");
    expect(source).not.toContain("Approval denied.");
    expect(source).not.toMatch(/void session\.confirm\(/);
  });
});

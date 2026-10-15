import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let notificationHandler: NotificationHandler | null = null;

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: vi.fn(),
  onNotification: (handler: NotificationHandler) => {
    notificationHandler = handler;
  },
}));

vi.mock("./settings", () => ({
  settings: writable({ model: { cloud_model: "ollama" } }),
}));

vi.mock("./companion", () => ({
  companion: { speak: vi.fn() },
}));

vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function requireConfirmation(planId: string) {
  notificationHandler!("confirm_required", {
    plan_id: planId,
    actions: [{ index: 0, action_type: "browser_navigate", target: "https://example.test" }],
  });
}

describe("confirmation acknowledgement", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("keeps the dialog pending until the daemon acknowledges the decision", async () => {
    const acknowledgement = deferred<Record<string, unknown>>();
    daemonMocks.call.mockReturnValueOnce(acknowledgement.promise);
    const { session } = await import("./session");
    requireConfirmation("plan-approve");

    const decision = session.confirm(true, [0]);
    expect(get(session)).toMatchObject({
      confirmRequired: true,
      confirmPlanId: "plan-approve",
      confirmSubmitting: true,
    });

    acknowledgement.resolve({ status: "ok", confirmed: true });
    await decision;

    expect(daemonMocks.call).toHaveBeenCalledWith("confirm", {
      plan_id: "plan-approve",
      confirmed: true,
      approved_indices: [0],
    });
    expect(get(session)).toMatchObject({ confirmRequired: false, confirmSubmitting: false });
  });

  it("ignores duplicate submissions and does not let a stale acknowledgement dismiss a newer plan", async () => {
    const acknowledgement = deferred<Record<string, unknown>>();
    daemonMocks.call.mockReturnValueOnce(acknowledgement.promise);
    const { session } = await import("./session");
    requireConfirmation("plan-old");

    const oldDecision = session.confirm(true, [0]);
    await session.confirm(false);
    expect(daemonMocks.call).toHaveBeenCalledTimes(1);

    requireConfirmation("plan-new");
    acknowledgement.resolve({ status: "ok", confirmed: true });
    await oldDecision;

    expect(get(session)).toMatchObject({
      confirmRequired: true,
      confirmPlanId: "plan-new",
      confirmSubmitting: false,
    });
  });
});

describe("confirmation input channels and the reported selection", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    daemonMocks.call.mockResolvedValue({ status: "ok", confirmed: true });
    vi.resetModules();
  });

  function requirePlan(planId: string, indices: number[] = [0, 1, 2]) {
    notificationHandler!("confirm_required", {
      plan_id: planId,
      actions: indices.map((index) => ({ index, action_type: "file_write", target: `C:/Temp/${index}.txt` })),
    });
  }

  it("sends input_channel for the dialog with the dialog's own indices", async () => {
    const { session } = await import("./session");
    requirePlan("plan-dialog");

    await session.confirm(true, [0, 2], { inputChannel: "dialog" });

    expect(daemonMocks.call).toHaveBeenCalledWith("confirm", {
      plan_id: "plan-dialog",
      confirmed: true,
      approved_indices: [0, 2],
      input_channel: "dialog",
    });
  });

  it("sends input_channel for a dialog denial", async () => {
    const { session } = await import("./session");
    requirePlan("plan-deny");

    await session.confirm(false, undefined, { inputChannel: "dialog" });

    expect(daemonMocks.call).toHaveBeenCalledWith("confirm", {
      plan_id: "plan-deny",
      confirmed: false,
      input_channel: "dialog",
    });
  });

  it("approves by gesture exactly the selection reported for the pending plan", async () => {
    const { session } = await import("./session");
    requirePlan("plan-gesture");
    session.setConfirmSelection("plan-gesture", [2]);

    await session.confirm(true, undefined, { inputChannel: "gesture" });

    expect(daemonMocks.call).toHaveBeenCalledWith("confirm", {
      plan_id: "plan-gesture",
      confirmed: true,
      approved_indices: [2],
      input_channel: "gesture",
    });
  });

  it("denies by gesture without needing a selection", async () => {
    const { session } = await import("./session");
    requirePlan("plan-gesture-deny");

    await session.confirm(false, undefined, { inputChannel: "gesture" });

    expect(daemonMocks.call).toHaveBeenCalledWith("confirm", {
      plan_id: "plan-gesture-deny",
      confirmed: false,
      input_channel: "gesture",
    });
  });

  it.each([
    ["no reported selection", null],
    ["an empty selection", { planId: "plan-now", indices: [] }],
    ["a selection for another plan", { planId: "plan-before", indices: [0, 1, 2] }],
  ])("refuses a gesture approval locally with %s", async (_label, selection) => {
    const { session } = await import("./session");
    requirePlan("plan-now");
    if (selection) session.setConfirmSelection(selection.planId, selection.indices);

    await session.confirm(true, undefined, { inputChannel: "gesture" });

    expect(daemonMocks.call).not.toHaveBeenCalled();
    expect(get(session)).toMatchObject({
      confirmRequired: true,
      confirmPlanId: "plan-now",
      confirmSubmitting: false,
      confirmError: "Nothing is checked. Check the actions you want to approve, or deny.",
    });
  });

  it("resets the reported selection on every confirm_required", async () => {
    const { session } = await import("./session");
    requirePlan("plan-a");
    session.setConfirmSelection("plan-a", [0]);
    expect(get(session).confirmSelection).toEqual({ planId: "plan-a", indices: [0] });

    requirePlan("plan-a");
    expect(get(session).confirmSelection).toBeNull();

    session.setConfirmSelection("plan-a", [1]);
    requirePlan("plan-b");
    expect(get(session).confirmSelection).toBeNull();
    await session.confirm(true, undefined, { inputChannel: "gesture" });
    expect(daemonMocks.call).not.toHaveBeenCalled();
  });

  it("resets the reported selection when a resumed task is awaiting approval", async () => {
    localStorage.setItem(
      "heliox_chat_sessions_v1",
      JSON.stringify([
        {
          id: "chat-1",
          title: "Chat",
          createdAt: 1,
          updatedAt: 1,
          messages: [],
          totalTokens: 0,
          estimatedCost: 0,
          durableTask: { taskId: "task-1", resumeToken: "token-1" },
        },
      ]),
    );
    localStorage.setItem("heliox_active_chat_session", "chat-1");
    daemonMocks.call.mockResolvedValueOnce({
      status: "awaiting_approval",
      plan_id: "plan-resumed",
      approval: { actions: [{ index: 0, action_type: "file_write", target: "C:/Temp/a.txt" }] },
    });
    const { session } = await import("./session");
    session.setConfirmSelection("plan-resumed", [0]);

    await session.resumeDurableTask();

    expect(get(session)).toMatchObject({
      confirmRequired: true,
      confirmPlanId: "plan-resumed",
      confirmSelection: null,
    });
  });

  it("tracks when the dialog became visible and restarts the clock for a new approval", async () => {
    const clock = vi.spyOn(performance, "now").mockReturnValue(10_000);
    try {
      const { session } = await import("./session");
      expect(get(session)).toMatchObject({ confirmVisible: false, confirmVisibleSince: null });

      requirePlan("plan-visible");
      session.setConfirmVisible(true);
      expect(get(session)).toMatchObject({ confirmVisible: true, confirmVisibleSince: 10_000 });

      clock.mockReturnValue(12_000);
      session.setConfirmVisible(true);
      expect(get(session).confirmVisibleSince).toBe(10_000);

      requirePlan("plan-next");
      expect(get(session).confirmVisibleSince).toBe(12_000);

      session.setConfirmVisible(false);
      expect(get(session)).toMatchObject({ confirmVisible: false, confirmVisibleSince: null });
      requirePlan("plan-hidden");
      expect(get(session).confirmVisibleSince).toBeNull();
    } finally {
      clock.mockRestore();
    }
  });

  it("sends nothing to co-adaptation after a decision", async () => {
    const { session } = await import("./session");
    requirePlan("plan-quiet");
    session.setConfirmSelection("plan-quiet", [1]);
    await session.confirm(true, undefined, { inputChannel: "gesture" });
    requirePlan("plan-quiet-2");
    await session.confirm(false, undefined, { inputChannel: "dialog" });

    const methods = daemonMocks.call.mock.calls.map(([method]) => String(method));
    expect(methods).toEqual(["confirm", "confirm"]);
    expect(methods.some((method) => method.startsWith("co_adaptation"))).toBe(false);
  });
});

describe("confirmation outcome", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  function requirePlan(planId: string) {
    notificationHandler!("confirm_required", {
      plan_id: planId,
      actions: [{ index: 0, action_type: "file_write", target: "C:/Temp/0.txt" }],
    });
  }

  function chatTexts(session: { subscribe: (run: (value: { messages: { text: string }[] }) => void) => () => void }) {
    return get(session).messages.map((message) => message.text);
  }

  it("reports the decision the daemon says it recorded", async () => {
    daemonMocks.call.mockResolvedValueOnce({ status: "ok", confirmed: false });
    const { session } = await import("./session");
    requirePlan("plan-recorded");

    expect(await session.confirm(false, undefined, { inputChannel: "dialog" })).toEqual({
      kind: "recorded",
      confirmed: false,
    });
  });

  it("reports an answer that sent nothing", async () => {
    const { session } = await import("./session");
    expect(await session.confirm(false)).toEqual({ kind: "not_sent", reason: "missing_plan" });

    requirePlan("plan-unchecked");
    expect(await session.confirm(true, undefined, { inputChannel: "gesture" })).toEqual({
      kind: "not_sent",
      reason: "nothing_checked",
    });
    expect(daemonMocks.call).not.toHaveBeenCalled();
  });

  it("reports a refused answer and whether the chat already shows it", async () => {
    daemonMocks.call.mockRejectedValueOnce(new Error("Confirmation already resolved for plan_id: plan-refused"));
    const { session } = await import("./session");
    requirePlan("plan-refused");

    expect(await session.confirm(false, undefined, { inputChannel: "gesture" })).toEqual({
      kind: "rejected",
      message: "Confirmation already resolved for plan_id: plan-refused",
      errorShown: true,
    });

    const reply = deferred<Record<string, unknown>>();
    daemonMocks.call.mockReturnValueOnce(reply.promise);
    requirePlan("plan-replaced");
    const decision = session.confirm(false, undefined, { inputChannel: "gesture" });
    requirePlan("plan-newer");
    reply.resolve({ status: "error", message: "No pending confirmation for plan_id: plan-replaced" });
    expect(await decision).toEqual({
      kind: "rejected",
      message: "No pending confirmation for plan_id: plan-replaced",
      errorShown: false,
    });
  });

  it("answers before a resumed durable task finishes", async () => {
    localStorage.setItem(
      "heliox_chat_sessions_v1",
      JSON.stringify([
        {
          id: "chat-1",
          title: "Chat",
          createdAt: 1,
          updatedAt: 1,
          messages: [],
          totalTokens: 0,
          estimatedCost: 0,
          durableTask: { taskId: "task-1", resumeToken: "token-1" },
        },
      ]),
    );
    localStorage.setItem("heliox_active_chat_session", "chat-1");
    const resume = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) =>
      method === "resume_task"
        ? resume.promise
        : Promise.resolve({ status: "ok", confirmed: true, resume_required: true, task_id: "task-1" }),
    );
    const { session } = await import("./session");
    requirePlan("plan-durable");

    const outcome = await Promise.race([
      session.confirm(true, [0], { inputChannel: "dialog" }),
      new Promise((done) => setTimeout(() => done("still waiting for the resumed task"), 50)),
    ]);

    expect(outcome).toEqual({ kind: "recorded", confirmed: true });
    expect(daemonMocks.call.mock.calls.map(([method]) => method)).toEqual(["confirm", "resume_task"]);
    resume.resolve({ status: "error", message: "stopped by test", task_id: "task-1" });
  });

  it("does not say 'Approval denied.' for a gesture deny dropped while an approval is in flight", async () => {
    const acknowledgement = deferred<Record<string, unknown>>();
    daemonMocks.call.mockReturnValueOnce(acknowledgement.promise);
    const { session } = await import("./session");
    const { GESTURE_ANSWER_MESSAGES, sendGestureAnswer } = await import("../gesture/approvalGesture");
    requirePlan("plan-race");

    const approval = session.confirm(true, [0], { inputChannel: "dialog" });
    const denial = await sendGestureAnswer(
      false,
      (accepted) => session.confirm(accepted, undefined, { inputChannel: "gesture" }),
      (text) => session.addSystemMessage(text),
    );

    expect(denial).toEqual({ kind: "not_sent", reason: "submitting" });
    expect(chatTexts(session)).toContain(GESTURE_ANSWER_MESSAGES.denyInFlight);
    expect(chatTexts(session)).not.toContain("Approval denied.");

    acknowledgement.resolve({ status: "ok", confirmed: true });
    expect(await approval).toEqual({ kind: "recorded", confirmed: true });
    expect(daemonMocks.call).toHaveBeenCalledTimes(1);
  });

  it("does not say 'Approval denied.' when the daemon refuses a late gesture deny", async () => {
    daemonMocks.call.mockResolvedValueOnce({
      status: "error",
      message: "Confirmation already resolved for plan_id: plan-late",
    });
    const { session } = await import("./session");
    const { sendGestureAnswer } = await import("../gesture/approvalGesture");
    requirePlan("plan-late");

    await sendGestureAnswer(
      false,
      (accepted) => session.confirm(accepted, undefined, { inputChannel: "gesture" }),
      (text) => session.addSystemMessage(text),
    );

    expect(chatTexts(session)).toEqual(["Approval failed: Confirmation already resolved for plan_id: plan-late"]);
  });

  it("does not say 'Approval denied.' when the daemon recorded an approval for the plan", async () => {
    daemonMocks.call.mockResolvedValueOnce({ status: "ok", confirmed: true, resume_required: false });
    const { session } = await import("./session");
    const { GESTURE_ANSWER_MESSAGES, sendGestureAnswer } = await import("../gesture/approvalGesture");
    requirePlan("plan-approved-first");

    await sendGestureAnswer(
      false,
      (accepted) => session.confirm(accepted, undefined, { inputChannel: "gesture" }),
      (text) => session.addSystemMessage(text),
    );

    expect(chatTexts(session)).toEqual([GESTURE_ANSWER_MESSAGES.alreadyApproved]);
  });

  it("says 'Approval denied.' when the daemon recorded the gesture's denial", async () => {
    daemonMocks.call.mockResolvedValueOnce({ status: "ok", confirmed: false });
    const { session } = await import("./session");
    const { sendGestureAnswer } = await import("../gesture/approvalGesture");
    requirePlan("plan-denied");

    await sendGestureAnswer(
      false,
      (accepted) => session.confirm(accepted, undefined, { inputChannel: "gesture" }),
      (text) => session.addSystemMessage(text),
    );

    expect(chatTexts(session)).toEqual(["Approval denied."]);
  });
});

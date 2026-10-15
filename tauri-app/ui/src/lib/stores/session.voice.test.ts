import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let notificationHandler: NotificationHandler | null = null;

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn().mockResolvedValue(undefined),
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

describe("voice command session notifications", () => {
  it("tracks the daemon's unified interaction state", async () => {
    const { session } = await import("./session");

    notificationHandler!("interaction_state", {
      interaction_id: "interaction-1",
      source: "voice",
      phase: "planning",
      message: "Planning the safest useful action",
      active: true,
      elapsed_ms: 42,
      sequence: 2,
    });

    expect(get(session)).toMatchObject({
      phase: "Planning the safest useful action",
      interaction: {
        interactionId: "interaction-1",
        source: "voice",
        phase: "planning",
        active: true,
        elapsedMs: 42,
        sequence: 2,
      },
    });
  });

  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockClear();
    vi.resetModules();
  });

  it("renders the spoken command and its terminal result", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_command", {
      command: "show system information",
      status: "executing",
    });

    let state = get(session);
    expect(state.loading).toBe(true);
    expect(state.messages.at(-1)).toMatchObject({
      type: "user",
      text: "show system information",
    });

    notificationHandler!("voice_result", {
      command: "show system information",
      status: "success",
      result: "Windows 11, 32 GB RAM",
    });

    state = get(session);
    expect(state.loading).toBe(false);
    expect(state.currentPlan).toBeNull();
    expect(state.terminalStatus).toBe("success");
    expect(state.messages.at(-1)).toMatchObject({
      type: "result",
      text: "Windows 11, 32 GB RAM",
    });
  });

  it("maps a partial voice result to a visible failure state", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_result", {
      command: "inspect the system",
      status: "partial",
      result: "One action could not be verified.",
    });

    const state = get(session);
    expect(state.loading).toBe(false);
    expect(state.terminalStatus).toBe("partial_failure");
    expect(state.messages.at(-1)).toMatchObject({
      type: "error",
      text: "One action could not be verified.",
    });
  });

  it("keeps the visible confirmation open when voice approval is refused", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_command", {
      command: "approve",
      status: "executing",
    });
    notificationHandler!("confirm_required", {
      plan_id: "plan-visible",
      actions: [{ action_type: "browser_click", target: "Launch" }],
      reason: "This browser action needs approval.",
    });
    notificationHandler!("voice_result", {
      command: "approve",
      status: "approval_required",
      plan_id: "plan-visible",
      message: "For safety, approve this request in the visible Heliox confirmation dialog.",
    });

    const state = get(session);
    expect(state).toMatchObject({
      loading: true,
      phase: "awaiting visible approval",
      confirmRequired: true,
      confirmPlanId: "plan-visible",
      confirmReason: "This browser action needs approval.",
      terminalStatus: "",
    });
    expect(state.confirmActions).toEqual([{ action_type: "browser_click", target: "Launch" }]);
    expect(state.messages.at(-1)).toMatchObject({
      type: "system",
      text: "For safety, approve this request in the visible Heliox confirmation dialog.",
    });
  });

  it("adds optional companion ideas after the terminal result", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_result", {
      command: "show system information",
      status: "success",
      result: "Windows 11",
    });
    notificationHandler!("companion_follow_up", {
      message: "The system report is ready.",
      suggestions: ["Compare it with the app requirements", "Save a baseline"],
    });

    const state = get(session);
    expect(state.messages.at(-2)).toMatchObject({ type: "result", text: "Windows 11" });
    expect(state.messages.at(-1)).toMatchObject({
      type: "assistant",
      text: "The system report is ready.\n\nPossible next steps:\n- Compare it with the app requirements\n- Save a baseline",
    });
  });

  it("queues an early companion follow-up until the terminal result is rendered", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_command", { command: "show battery status" });
    notificationHandler!("companion_follow_up", {
      message: "The battery check is complete.",
      suggestions: ["Compare the charge with your planned work time"],
    });

    expect(get(session).messages.at(-1)).toMatchObject({ type: "user", text: "show battery status" });

    notificationHandler!("voice_result", {
      command: "show battery status",
      status: "success",
      result: "Charge: 100%",
    });

    const state = get(session);
    expect(state.messages.at(-2)).toMatchObject({ type: "result", text: "Charge: 100%" });
    expect(state.messages.at(-1)).toMatchObject({
      type: "assistant",
      text: "The battery check is complete.\n\nPossible next steps:\n- Compare the charge with your planned work time",
    });
  });

  it("discards an early companion follow-up when the task fails", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_command", { command: "show battery status" });
    notificationHandler!("companion_follow_up", {
      message: "The battery check is complete.",
      suggestions: ["Compare the charge with your planned work time"],
    });
    notificationHandler!("voice_result", {
      command: "show battery status",
      status: "error",
      error: "Battery telemetry is unavailable",
    });

    const state = get(session);
    expect(state.messages.at(-1)).toMatchObject({ type: "error", text: "Battery telemetry is unavailable" });
    expect(state.messages.some((message) => message.type === "assistant")).toBe(false);
  });
});

describe("voice workflow notes", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("shows a started voice workflow as a note, not an unexpected-status error", async () => {
    const { session } = await import("./session");

    notificationHandler!("confirm_required", {
      plan_id: "background-plan",
      actions: [{ action_type: "shell_script", target: "powershell", parameters: {} }],
      reason: "An autonomous task reached an action that requires explicit approval.",
    });
    notificationHandler!("voice_result", {
      command: "open notepad, write a list and save it",
      status: "submitted",
      result: "I’m on it. Keep speaking if you want to correct or redirect me.",
    });

    const state = get(session);
    expect(state.messages.at(-1)).toMatchObject({
      type: "system",
      text: "I’m on it. Keep speaking if you want to correct or redirect me.",
    });
    expect(state.confirmRequired).toBe(true); // an unrelated pending approval stays visible
  });

  it("shows what a mid-task utterance did", async () => {
    const { session } = await import("./session");

    notificationHandler!("voice_result", {
      command: "also save it",
      status: "queued",
      result: "Got it, I'll include that.",
    });
    expect(get(session).messages.at(-1)).toMatchObject({ type: "system", text: "Got it, I'll include that." });

    notificationHandler!("voice_result", {
      command: "what are you doing",
      status: "answered",
      message: "I'm working on: write the shopping list. Step 2 of 3.",
    });
    expect(get(session).messages.at(-1)).toMatchObject({
      type: "result",
      text: "I'm working on: write the shopping list. Step 2 of 3.",
    });
  });
});

describe("background approvals", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("stay visible when the person types a new command", async () => {
    const { session } = await import("./session");
    notificationHandler!("confirm_required", {
      plan_id: "autonomous-j1:0:0",
      source: "scheduled",
      actions: [{ action_type: "api_send_email", target: "team@example.com", parameters: {} }],
      reason: "An autonomous task reached an action that requires explicit approval.",
    });

    await session.sendCommand("open notepad");

    expect(get(session)).toMatchObject({ confirmRequired: true, confirmPlanId: "autonomous-j1:0:0" });
  });

  it("an interactive approval still resets with its own task", async () => {
    const { session } = await import("./session");
    notificationHandler!("confirm_required", {
      plan_id: "interactive-plan",
      actions: [{ action_type: "shell_command", target: "whoami", parameters: {} }],
      reason: "Needs approval.",
    });

    await session.sendCommand("open notepad");

    expect(get(session).confirmRequired).toBe(false);
  });
});

describe("typed or spoken text during an approval", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    daemonMocks.call.mockResolvedValue(undefined);
    vi.resetModules();
  });

  async function pendingApproval() {
    const { session } = await import("./session");
    notificationHandler!("voice_command", { command: "delete the old logs" });
    notificationHandler!("confirm_required", {
      plan_id: "plan-typed",
      actions: [{ index: 0, action_type: "file_delete", target: "C:/Temp/old.log", parameters: {} }],
      reason: "Deleting files needs approval.",
    });
    notificationHandler!("status", { phase: "awaiting approval" });
    return session;
  }

  it.each([
    [
      "approval_requires_dialog",
      "Approve in the visible confirmation dialog. Speech and typed text can deny but never approve.",
    ],
    ["denied", "Okay, I won't run that."],
    ["not_denied", "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task."],
  ])("handles the %s interject status: phase restored and the daemon's text shown", async (status, message) => {
    const session = await pendingApproval();
    daemonMocks.call.mockResolvedValueOnce({ status, plan_id: "plan-typed", message });

    await session.sendCommand(status === "approval_requires_dialog" ? "yes" : "no");

    expect(daemonMocks.call).toHaveBeenCalledWith("interject", {
      input: status === "approval_requires_dialog" ? "yes" : "no",
      mode: "correct",
    });
    const state = get(session);
    expect(state.phase).toBe("awaiting approval");
    expect(state.messages.at(-1)).toMatchObject({ type: "system", text: message });
    expect(state.messages.some((entry) => entry.type === "error")).toBe(false);
    // The approval itself is untouched here; the daemon resolves it and the task result clears it.
    expect(state).toMatchObject({ confirmRequired: true, confirmPlanId: "plan-typed" });
  });

  it("falls back to the documented text when the daemon sends none", async () => {
    const session = await pendingApproval();
    daemonMocks.call.mockResolvedValueOnce({ status: "denied", plan_id: "plan-typed" });

    await session.sendCommand("cancel");

    expect(get(session).messages.at(-1)).toMatchObject({ type: "system", text: "Okay, I won't run that." });
  });
});

import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
type ConnectionHandler = (connected: boolean) => void;
let notify: NotificationHandler | null = null;
let connection: ConnectionHandler | null = null;

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: (handler: ConnectionHandler) => {
    connection = handler;
  },
  onNotification: (handler: NotificationHandler) => {
    notify = handler;
  },
}));

vi.mock("./settings", () => ({ settings: writable({ model: { cloud_model: "ollama" } }) }));
vi.mock("./companion", () => ({ companion: { speak: vi.fn() } }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

const OFFER = {
  status: "interrupted",
  task_id: "task-1",
  request: "send the weekly report",
  phase: "executing",
  steps_done: 1,
  steps_left: 2,
  uncertain_steps: 0,
};

function savedTask() {
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
}

function resumeCalls() {
  return daemonMocks.call.mock.calls.filter(([method]) => method === "resume_task").map(([, params]) => params);
}

describe("a task Heliox stopped under", () => {
  beforeEach(() => {
    localStorage.clear();
    notify = null;
    connection = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
    savedTask();
  });

  it("is offered after the daemon came back, and the reconnect never asks to continue it", async () => {
    daemonMocks.call.mockResolvedValue(OFFER);
    const { session } = await import("./session");

    connection!(true);
    await vi.waitFor(() => expect(get(session).interruptedTask).not.toBeNull());

    expect(resumeCalls()).toEqual([{ task_id: "task-1", resume_token: "token-1" }]);
    expect(get(session)).toMatchObject({
      loading: false,
      interruptedTask: { taskId: "task-1", phase: "executing", stepsDone: 1, stepsLeft: 2 },
    });
    expect(get(session).durableTask?.taskId).toBe("task-1"); // kept: the card stays until answered
  });

  it("continues only from the Continue button, with the explicit flag", async () => {
    daemonMocks.call.mockResolvedValueOnce(OFFER).mockResolvedValueOnce({ status: "success", message: "Sent." });
    const { session } = await import("./session");
    connection!(true);
    await vi.waitFor(() => expect(get(session).interruptedTask).not.toBeNull());

    await session.continueInterruptedTask();

    expect(resumeCalls()[1]).toEqual({ task_id: "task-1", resume_token: "token-1", continue: true });
    expect(get(session).interruptedTask).toBeNull();
    expect(get(session).durableTask).toBeUndefined();
  });

  it("leaves the task with Leave it, and nothing more runs", async () => {
    daemonMocks.call.mockResolvedValueOnce(OFFER).mockResolvedValueOnce({
      status: "cancelled",
      task_id: "task-1",
      message: "Left as it was after Heliox stopped.",
    });
    const { session } = await import("./session");
    connection!(true);
    await vi.waitFor(() => expect(get(session).interruptedTask).not.toBeNull());

    await session.leaveInterruptedTask();

    expect(resumeCalls()[1]).toEqual({ task_id: "task-1", resume_token: "token-1", leave: true });
    expect(get(session).interruptedTask).toBeNull();
    expect(get(session).durableTask).toBeUndefined();
    expect(get(session).messages.at(-1)?.text).toContain("Left as it was");
  });

  it("goes away when the person says 'leave it'", async () => {
    daemonMocks.call.mockResolvedValue(OFFER);
    const { session } = await import("./session");
    connection!(true);
    await vi.waitFor(() => expect(get(session).interruptedTask).not.toBeNull());

    notify!("interrupted_task_left", { status: "cancelled", task_id: "task-1", message: "Left as it was." });

    expect(get(session).interruptedTask).toBeNull();
    expect(get(session).durableTask).toBeUndefined();
  });

  it("is still offered after another reconnect until it is answered", async () => {
    daemonMocks.call.mockResolvedValue(OFFER);
    const { session } = await import("./session");
    connection!(true);
    await vi.waitFor(() => expect(get(session).interruptedTask).not.toBeNull());
    connection!(false);
    connection!(true);
    await vi.waitFor(() => expect(resumeCalls()).toHaveLength(2));

    expect(resumeCalls().every((params) => !("continue" in params))).toBe(true);
    expect(get(session).interruptedTask?.taskId).toBe("task-1");
  });

  it("resumes as before after a reconnect to the same daemon", async () => {
    daemonMocks.call.mockResolvedValue({ status: "success", task_id: "task-1", message: "Done." });
    const { session } = await import("./session");
    connection!(true);
    await vi.waitFor(() => expect(get(session).durableTask).toBeUndefined());

    expect(get(session).interruptedTask).toBeNull();
  });
});

describe("the card's sentence", () => {
  it("says what was done, what was left and what is uncertain", async () => {
    const { describeInterruptedTask } = await import("./session");
    const base = { taskId: "t", request: "send the weekly report", uncertainSteps: 0 };
    expect(describeInterruptedTask({ ...base, phase: "executing", stepsDone: 1, stepsLeft: 2 })).toBe(
      'Heliox stopped while it was doing "send the weekly report". 1 step was done, 2 were left.',
    );
    expect(
      describeInterruptedTask({ ...base, phase: "executing", stepsDone: 2, stepsLeft: 1, uncertainSteps: 1 }),
    ).toBe(
      'Heliox stopped while it was doing "send the weekly report". 2 steps were done, 1 was left. ' +
        "1 step may or may not have happened and will not be repeated.",
    );
    expect(describeInterruptedTask({ ...base, phase: "planning", stepsDone: 0, stepsLeft: 0 })).toBe(
      'Heliox stopped while it was planning "send the weekly report". Nothing was done yet.',
    );
  });
});

describe("a storage problem", () => {
  beforeEach(() => {
    localStorage.clear();
    notify = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("is shown once as a plain line in the conversation", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok" });
    const { session } = await import("./session");
    const text =
      "The disk is full, so Heliox could not save some data. Free some space; Heliox keeps working meanwhile.";

    notify!("storage_problem", { kind: "full", component: "memory", message: text });
    notify!("storage_problem", { kind: "full", message: "" });

    expect(
      get(session)
        .messages.filter((m) => m.type === "system")
        .map((m) => m.text),
    ).toEqual([text]);
  });
});

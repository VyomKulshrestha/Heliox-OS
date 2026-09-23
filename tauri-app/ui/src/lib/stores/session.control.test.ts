import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: vi.fn(),
  onNotification: vi.fn(),
}));

vi.mock("./settings", () => ({
  settings: writable({ model: { cloud_model: "ollama" } }),
}));
vi.mock("./companion", () => ({ companion: { speak: vi.fn() } }));
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

describe("session execution controls", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("reports an unacknowledged stop as an error", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "Cancellation channel unavailable" });
    const { session } = await import("./session");

    await session.abort();

    expect(get(session).messages.at(-1)).toMatchObject({
      type: "error",
      text: "Stop failed: Cancellation channel unavailable",
    });
  });

  it("restores the active phase when a live correction is rejected", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "interject") return Promise.resolve({ status: "disabled", message: "Corrections are disabled" });
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");

    const activeCommand = session.sendCommand("Inspect the system");
    await session.sendCommand("Check memory first");

    expect(get(session)).toMatchObject({ loading: true, phase: "" });
    expect(get(session).messages.at(-1)).toMatchObject({
      type: "error",
      text: "Live correction failed: Corrections are disabled",
    });

    execution.resolve({ status: "cancelled", message: "Stopped for test" });
    await activeCommand;
  });

  it("answers an in-task question without stopping the active command", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "interject") return Promise.resolve({ status: "answered", message: "Still checking the system." });
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");

    const activeCommand = session.sendCommand("Inspect the system");
    await session.sendCommand("What are you doing?", [], "ask");

    expect(daemonMocks.call).toHaveBeenCalledWith("interject", {
      input: "What are you doing?",
      mode: "ask",
    });
    expect(get(session).loading).toBe(true);
    expect(get(session).messages.at(-1)).toMatchObject({
      type: "result",
      text: "Still checking the system.",
    });

    execution.resolve({ status: "success", message: "Finished checking." });
    await activeCommand;
  });

  it("queues and cancels a second task without interrupting the first", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "queue_execute") {
        return Promise.resolve({ status: "queued", task_id: "later-1", resume_token: "secret-token" });
      }
      if (method === "cancel_queued_execute") return Promise.resolve({ status: "cancelled" });
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");

    const activeCommand = session.sendCommand("Inspect the system");
    await session.sendCommand("Summarize the report", [], "next");

    expect(daemonMocks.call).toHaveBeenCalledWith("queue_execute", {
      input: "Summarize the report",
      session_id: get(session).activeSessionId,
    });
    expect(get(session).loading).toBe(true);
    expect(get(session).queuedTasks).toMatchObject([{ taskId: "later-1", status: "queued" }]);

    await session.cancelQueuedTask("later-1");
    expect(daemonMocks.call).toHaveBeenCalledWith("cancel_queued_execute", {
      task_id: "later-1",
      resume_token: "secret-token",
    });

    execution.resolve({ status: "success", message: "Done" });
    await activeCommand;
  });

  it("keeps the queued task active when the previous execute response arrives late", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "queue_execute") {
        return Promise.resolve({ status: "queued", task_id: "later-2", resume_token: "token-2" });
      }
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");
    const { onNotification } = await import("../api/daemon");
    const notify = (method: string, params: Record<string, unknown>) => {
      for (const [handler] of vi.mocked(onNotification).mock.calls) handler(method, params);
    };

    const first = session.sendCommand("First task");
    await session.sendCommand("Second task", [], "next");
    notify("queued_task_started", {
      task_id: "later-2",
      session_id: get(session).activeSessionId,
      input: "Second task",
    });
    execution.resolve({ status: "success", message: "First finished" });
    await first;

    expect(get(session).loading).toBe(true);
    expect(get(session).queuedTasks[0].status).toBe("running");

    notify("queued_task_result", {
      task_id: "later-2",
      session_id: get(session).activeSessionId,
      status: "success",
      message: "Second finished",
      result: { status: "success", message: "Second finished" },
    });
    expect(get(session).loading).toBe(false);
    expect(get(session).queuedTasks).toEqual([]);
  });

  it("shows a voice-queued task and keeps its result in the originating chat", async () => {
    const { session } = await import("./session");
    const { onNotification } = await import("../api/daemon");
    const notify = (method: string, params: Record<string, unknown>) => {
      for (const [handler] of vi.mocked(onNotification).mock.calls) handler(method, params);
    };
    const originalChat = get(session).activeSessionId;
    notify("queued_task_added", {
      task_id: "voice-next",
      session_id: originalChat,
      input: "Summarize the open page",
    });
    expect(get(session).queuedTasks).toMatchObject([{
      taskId: "voice-next",
      input: "Summarize the open page",
      status: "queued",
    }]);

    expect(session.newChat()).toBe(true);
    notify("queued_task_result", {
      task_id: "voice-next",
      session_id: originalChat,
      status: "success",
      result: { status: "success", message: "The page is about Heliox." },
    });
    expect(get(session).messages).toEqual([]);
    expect(session.switchChatSession(originalChat)).toBe(true);
    expect(get(session).messages.at(-1)?.text).toContain("The page is about Heliox.");
  });
});

import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let notificationHandler: NotificationHandler | null = null;

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return {
    ...actual,
    call: daemonMocks.call,
    connect: vi.fn().mockResolvedValue(true),
    isConnected: vi.fn(() => true),
    onConnectionState: vi.fn(),
    onNotification: (handler: NotificationHandler) => {
      notificationHandler = handler;
    },
  };
});

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

const preview = {
  task_key: "task-7",
  undone: false,
  ready: 1,
  items: [{ seq: 1, type: "file_write", target: "notes.txt", state: "ready", description: "Remove notes.txt." }],
  not_undoable: [],
  preview_token: "token-1",
};

describe("per-task undo", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("offers undo only when the finished task left something undoable", async () => {
    const { session } = await import("./session");

    notificationHandler!("task_complete", { status: "completed", undo: { task_key: "a", available: false } });
    expect(get(session).undoOffer).toBeNull();

    notificationHandler!("task_complete", {
      status: "completed",
      undo: { task_key: "task-7", available: true, undoable: 2, not_undoable: 1 },
    });
    expect(get(session).undoOffer).toEqual({ taskKey: "task-7", undoable: 2, notUndoable: 1 });
  });

  it("undoes only after the preview is confirmed, sending its token", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "undo_preview") return { status: "ok", preview };
      if (method === "undo_task") return { status: "ok", undone: 1, skipped: 0, failed: 0 };
      return { status: "ok" };
    });
    const { session } = await import("./session");
    notificationHandler!("task_complete", {
      status: "completed",
      undo: { task_key: "task-7", available: true, undoable: 1, not_undoable: 0 },
    });

    await session.openUndoPreview();
    expect(get(session).undoPreview?.preview_token).toBe("token-1");
    expect(daemonMocks.call).not.toHaveBeenCalledWith("undo_task", expect.anything());

    await session.confirmUndo();

    expect(daemonMocks.call).toHaveBeenCalledWith("undo_task", {
      task_key: "task-7",
      preview_token: "token-1",
      confirmed: true,
    });
    const state = get(session);
    expect(state.undoPreview).toBeNull();
    expect(state.undoOffer).toBeNull();
    expect(state.messages.at(-1)?.text).toBe("Undid 1 change.");
  });

  it("shows the fresh preview when the files changed after the first one", async () => {
    const fresh = { ...preview, ready: 0, preview_token: "token-2" };
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "undo_preview") return { status: "ok", preview };
      return { status: "changed", message: "Something changed since the preview.", preview: fresh };
    });
    const { session } = await import("./session");

    await session.openUndoPreview("task-7");
    await session.confirmUndo();

    const state = get(session);
    expect(state.undoPreview?.preview_token).toBe("token-2");
    expect(state.undoError).toBe("Something changed since the preview.");
  });

  it("opens the preview when the daemon offers undo after a spoken request", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", preview });
    const { session } = await import("./session");

    notificationHandler!("undo_offer", { task_key: "task-7" });
    await vi.waitFor(() => expect(get(session).undoPreview).not.toBeNull());

    expect(daemonMocks.call).toHaveBeenCalledWith("undo_preview", { task_key: "task-7" });
    expect(daemonMocks.call).not.toHaveBeenCalledWith("undo_task", expect.anything());
  });

  it("clears the offer when another window finished the undo", async () => {
    const { session } = await import("./session");
    notificationHandler!("task_complete", {
      status: "completed",
      undo: { task_key: "task-7", available: true, undoable: 1, not_undoable: 0 },
    });

    notificationHandler!("undo_complete", { task_key: "task-7", status: "ok", undone: 1 });

    expect(get(session).undoOffer).toBeNull();
  });
});

describe("task footer in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  const executeReply = {
    status: "success",
    explanation: "Wrote notes.",
    results: [{ action: { action_type: "file_write", target: "notes.txt" }, success: true, output: "ok" }],
    receipt_id: "abc123",
    undo: { task_key: "task-7", available: true, undoable: 1, not_undoable: 0 },
  };

  it("puts Undo and the receipt on the task's own message", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "execute") return executeReply;
      if (method === "receipt_verify")
        return { status: "ok", result: { valid: true, this_computer: true, reasons: [] } };
      return { status: "ok" };
    });
    const { session } = await import("./session");

    await session.sendCommand("write my notes");
    const message = get(session).messages.at(-1);
    expect(message?.task).toEqual({
      receiptId: "abc123",
      undo: { taskKey: "task-7", available: true, undoable: 1, notUndoable: 0 },
    });

    await session.verifyReceipt("abc123");
    expect(get(session).messages.at(-1)?.task?.receiptCheck).toBe("Signed by this computer, unchanged");

    notificationHandler!("undo_complete", { task_key: "task-7", status: "ok", undone: 1 });
    expect(get(session).messages.at(-1)?.task?.undo).toMatchObject({ available: false, undone: true });
  });
});

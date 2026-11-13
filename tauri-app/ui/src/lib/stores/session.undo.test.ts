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

  it("sends the undo at once with the preview's token: the daemon's own dialog asks, not the window", async () => {
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

    expect(daemonMocks.call).toHaveBeenCalledWith("undo_task", { task_key: "task-7", preview_token: "token-1" });
    const state = get(session);
    expect(state.undoPreview).toBeNull();
    expect(state.undoOffer).toBeNull();
    expect(state.messages.at(-1)?.text).toBe("Undid 1 change.");
  });

  it("a no in Heliox's dialog undoes nothing and is not an error", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "undo_preview") return { status: "ok", preview };
      if (method === "undo_task") return { status: "cancelled", message: "Nothing was undone." };
      return { status: "ok" };
    });
    const { session } = await import("./session");

    await session.openUndoPreview("task-7");

    const state = get(session);
    expect(state.undoPreview).toBeNull();
    expect(state.undoError).toBe("");
    expect(state.undoSubmitting).toBe(false);
    expect(state.messages.at(-1)?.text).toBe("Nothing was undone.");
  });

  it("shows the fresh preview when the files changed, and Undo there sends its new token", async () => {
    const fresh = { ...preview, preview_token: "token-2" };
    let first = true;
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "undo_preview") return { status: "ok", preview };
      if (first) {
        first = false;
        return { status: "changed", message: "Something changed since the preview.", preview: fresh };
      }
      return { status: "ok", undone: 1, skipped: 0, failed: 0 };
    });
    const { session } = await import("./session");

    await session.openUndoPreview("task-7");
    const state = get(session);
    expect(state.undoPreview?.preview_token).toBe("token-2");
    expect(state.undoError).toBe("Something changed since the preview.");

    await session.confirmUndo();
    expect(daemonMocks.call).toHaveBeenLastCalledWith("undo_task", { task_key: "task-7", preview_token: "token-2" });
    expect(get(session).undoPreview).toBeNull();
  });

  it("an undo the daemon offers after a spoken request goes to its dialog", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "undo_preview") return { status: "ok", preview };
      return { status: "cancelled", message: "Nothing was undone." };
    });
    const { session } = await import("./session");

    notificationHandler!("undo_offer", { task_key: "task-7" });
    await vi.waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("undo_task", { task_key: "task-7", preview_token: "token-1" }),
    );
    expect(daemonMocks.call).toHaveBeenCalledWith("undo_preview", { task_key: "task-7" });
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

describe("undo for several tasks", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("asks about each task in turn, newest first, and a no stops the rest", async () => {
    const sent: string[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: { task_key?: string }) => {
      if (method === "undo_preview") return { status: "ok", preview: { ...preview, task_key: params.task_key } };
      if (method === "undo_task") {
        sent.push(params.task_key ?? "");
        return params.task_key === "old"
          ? { status: "cancelled", message: "Nothing was undone." }
          : { status: "ok", undone: 1, skipped: 0, failed: 0 };
      }
      return { status: "ok" };
    });
    const { session } = await import("./session");

    notificationHandler!("undo_offer", { task_key: "new", task_keys: ["new", "old", "older"] });
    await vi.waitFor(() => expect(sent).toEqual(["new", "old"]));
    await vi.waitFor(() => expect(get(session).undoQueue).toEqual([]));
    expect(sent).toEqual(["new", "old"]);
    expect(get(session).messages.at(-1)?.text).toBe("Nothing was undone.");
  });

  it("shows a daily brief in chat", async () => {
    const { session } = await import("./session");

    notificationHandler!("daily_brief", { day: "2026-10-23", text: "Today Heliox ran no tasks." });

    expect(get(session).messages.at(-1)?.text).toBe("Daily brief: Today Heliox ran no tasks.");
  });
});

describe("routines and shortcuts in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  const offer = {
    receipt_id: "r1",
    name: "tidy my downloads",
    goal: "tidy my downloads",
    allowed_actions: ["file_list", "file_move"],
    daily_at: "09:00",
    weekdays: [],
    label: "every day at 9:00 AM",
  };

  it("creates a routine only when Create is pressed, with the task's own action list", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok" });
    const { session } = await import("./session");

    notificationHandler!("routine_offer", offer);
    expect(get(session).routineOffer?.name).toBe("tidy my downloads");
    expect(daemonMocks.call).not.toHaveBeenCalled();

    session.setRoutineTime("07:30");
    await session.createRoutine();

    expect(daemonMocks.call).toHaveBeenCalledWith("scheduled_run_create", {
      name: "tidy my downloads",
      goal: "tidy my downloads",
      allowed_actions: ["file_list", "file_move"],
      daily_at: "07:30",
      weekdays: [],
    });
    expect(get(session).routineOffer).toBeNull();
    expect(get(session).messages.at(-1)?.text).toContain("may use only file_list, file_move");
  });

  it("asks the daemon for a routine from a task's receipt", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", routine: offer });
    const { session } = await import("./session");

    await session.suggestRoutine("r1");

    expect(daemonMocks.call).toHaveBeenCalledWith("routine_suggest", { receipt_id: "r1" });
    expect(get(session).routineOffer?.allowed_actions).toEqual(["file_list", "file_move"]);
  });

  it("notes in chat what a shortcut or 'do that again' expanded to", async () => {
    const { session } = await import("./session");

    notificationHandler!("request_expanded", {
      input: "open VS Code",
      note: 'Shortcut "start work": open VS Code',
    });

    expect(get(session).messages.at(-1)?.text).toBe('Shortcut "start work": open VS Code');
  });
});

describe("reminders in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("shows a due reminder, and marks a late one", async () => {
    const { session } = await import("./session");

    notificationHandler!("reminder", { id: "a1", text: "stretch", late: false });
    expect(get(session).messages.at(-1)?.text).toBe("Reminder: stretch");
    notificationHandler!("reminder", { id: "b2", text: "call mom", late: true });
    expect(get(session).messages.at(-1)?.text).toBe("Reminder (late): call mom");
  });
});

describe("focus in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("shows the focus summary and the reminders it held", async () => {
    const { session } = await import("./session");

    notificationHandler!("focus_ended", { text: "Focus session done: 25 minutes.", held: 1 });
    expect(get(session).messages.at(-1)?.text).toBe("Focus session done: 25 minutes.");
    notificationHandler!("reminder", { id: "a", text: "stretch", held: true });
    expect(get(session).messages.at(-1)?.text).toBe("Reminder (held while you focused): stretch");
  });
});

describe("nudges in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("shows a nudge", async () => {
    const { session } = await import("./session");

    notificationHandler!("nudge", { id: "u1", text: "You have spent 1 h on YouTube today." });
    expect(get(session).messages.at(-1)?.text).toBe("Nudge: You have spent 1 h on YouTube today.");
  });
});

describe("watchers in chat", () => {
  beforeEach(() => {
    localStorage.clear();
    notificationHandler = null;
    vi.resetModules();
  });

  it("shows what a watcher saw", async () => {
    const { session } = await import("./session");

    notificationHandler!("focus_distraction", {
      word: "YouTube",
      app: "msedge",
      text: "Heads-up: YouTube is open, and you're focusing for another 25 minutes.",
    });
    expect(get(session).messages.at(-1)?.text).toBe(
      "Heads-up: YouTube is open, and you're focusing for another 25 minutes.",
    );

    notificationHandler!("watch_fired", {
      id: "w1",
      kind: "process",
      label: "chrome closes",
      text: "Chrome has closed.",
    });
    expect(get(session).messages.at(-1)?.text).toBe("Chrome has closed.");
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => ({ updateSectionDetailed: vi.fn() }));
const sessionMocks = vi.hoisted(() => ({ openUndoPreview: vi.fn(), sendCommand: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});
vi.mock("../stores/settings", () => {
  const store = writable({ brief: { enabled: false, time: "18:00", speak: false }, clipboard: { history: true } });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: settingsMocks.updateSectionDetailed } };
});
vi.mock("../stores/session", () => ({
  session: { openUndoPreview: sessionMocks.openUndoPreview, sendCommand: sessionMocks.sendCommand },
}));

import TodayPanel from "./TodayPanel.svelte";

const overview = {
  status: "ok",
  label: "today",
  brief: 'Today Heliox ran 1 task; the latest was "tidy downloads".',
  tasks: [
    {
      receipt_id: "r1",
      finished_at: "2026-10-23T09:00:00+00:00",
      status: "success",
      summary: "ok",
      request: "tidy downloads",
      action_count: 3,
    },
  ],
  timeline_enabled: true,
  usage: { total_seconds: 3600, apps: [{ app: "Code", seconds: 3600, top_titles: ["server.py"] }] },
  undoable: [{ task_key: "t1", undoable: 3, targets: ["report.pdf"] }],
};

describe("TodayPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    settingsMocks.updateSectionDetailed.mockReset();
    sessionMocks.openUndoPreview.mockReset();
    daemonMocks.call.mockImplementation(async (method: string) =>
      method === "today_overview" ? overview : { status: "ok", text: overview.brief },
    );
  });

  it("shows the brief, the tasks, app time and what can be undone", async () => {
    render(TodayPanel);

    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());
    expect(screen.getByText("tidy downloads")).toBeTruthy();
    expect(screen.getByText("server.py")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(sessionMocks.openUndoPreview).toHaveBeenCalledWith("t1");
  });

  it("switches to yesterday and saves the brief settings", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Yesterday" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("today_overview", { day: "yesterday" });

    await fireEvent.click(screen.getByLabelText("Send a daily brief at"));
    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("brief", { enabled: true });
    await fireEvent.click(screen.getByRole("button", { name: "Send now" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Sent."));
  });

  it("lists reminders, adds one and cancels it", async () => {
    let reminders = [{ id: "a1", due: 1792760400, text: "stretch" }];
    daemonMocks.call.mockImplementation(async (method: string, params: { text?: string }) => {
      if (method === "today_overview") return overview;
      if (method === "reminder_add") reminders = [...reminders, { id: "b2", due: 1792764000, text: params.text ?? "" }];
      if (method === "reminder_cancel") reminders = reminders.filter((item) => item.id !== "a1");
      return { status: "ok", reminders };
    });
    render(TodayPanel);

    await waitFor(() => expect(screen.getByText("stretch")).toBeTruthy());
    await fireEvent.input(screen.getByLabelText("Reminder"), { target: { value: "call mom" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(screen.getByText("call mom")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("reminder_add", { text: "call mom", in_minutes: 20 });

    await fireEvent.click(screen.getByRole("button", { name: "Cancel stretch" }));
    await waitFor(() => expect(screen.queryByText("stretch")).toBeNull());
  });

  it("shows this week, exports a report and starts focus", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "today_overview") return overview;
      if (method === "report_export") return { status: "ok", path: "C:/Users/me/Downloads/heliox-report.md" };
      return { status: "ok" };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "This week" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("today_overview", { day: "week" });
    await fireEvent.click(screen.getByRole("button", { name: "Export report" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("heliox-report.md"));
    expect(daemonMocks.call).toHaveBeenCalledWith("report_export", { day: "week" });

    await fireEvent.click(screen.getByRole("button", { name: "Start focus" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("focus_start", { minutes: 25 });
  });

  it("saves and opens a setup and searches past tasks", async () => {
    let workspaces = [{ name: "coding", apps: ["Visual Studio Code", "Slack"] }];
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "today_overview") return overview;
      if (method === "workspace_save") workspaces = [...workspaces, { name: "writing", apps: ["Word"] }];
      if (method === "receipts_search") return { status: "ok", receipts: overview.tasks };
      return { status: "ok", workspaces };
    });
    render(TodayPanel);

    await waitFor(() => expect(screen.getByText("Visual Studio Code, Slack")).toBeTruthy());
    await fireEvent.click(screen.getByRole("button", { name: "Open" }));
    expect(sessionMocks.sendCommand).toHaveBeenCalledWith("open my coding setup");

    await fireEvent.input(screen.getByLabelText("Setup name"), { target: { value: "writing" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save open apps" }));
    await waitFor(() => expect(screen.getByText("Word")).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("Search tasks"), { target: { value: "tidy" } });
    await fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("receipts_search", { query: "tidy" }));
  });

  it("adds and deletes notes and nudges", async () => {
    let notes: { id: string; text: string; created_at: number }[] = [];
    let nudges: { id: string; match: string; minutes: number }[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === "today_overview") return overview;
      if (method === "note_add") notes = [{ id: "n1", text: String(params.text), created_at: 1 }];
      if (method === "note_delete") notes = [];
      if (method === "nudge_add") nudges = [{ id: "u1", match: String(params.match), minutes: Number(params.minutes) }];
      if (method === "nudge_delete") nudges = [];
      return { status: "ok", notes, nudges };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("Note"), { target: { value: "buy milk" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add note" }));
    await waitFor(() => expect(screen.getByText("buy milk")).toBeTruthy());
    await fireEvent.click(screen.getByRole("button", { name: "Delete note buy milk" }));
    await waitFor(() => expect(screen.queryByText("buy milk")).toBeNull());

    await fireEvent.input(screen.getByLabelText("App or site"), { target: { value: "YouTube" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add nudge" }));
    await waitFor(() => expect(screen.getByText("after 60 min a day")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("nudge_add", { match: "YouTube", minutes: 60 });
  });

  it("adds, finishes, reopens and clears to-dos", async () => {
    type Todo = { id: string; text: string; created_at: number; done: boolean; done_at: number | null };
    let todos: Todo[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === "today_overview") return overview;
      if (method === "todo_add")
        todos = [{ id: "t1", text: String(params.text), created_at: 1, done: false, done_at: null }];
      if (method === "todo_set_done") todos = todos.map((item) => ({ ...item, done: Boolean(params.done) }));
      if (method === "todo_clear_done") todos = todos.filter((item) => !item.done);
      return { status: "ok", todos };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("New to-do"), { target: { value: "call the bank" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add to-do" }));
    await waitFor(() => expect(screen.getByText("call the bank")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Mark call the bank done" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Mark call the bank not done" })).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("todo_set_done", { id: "t1", done: true });

    await fireEvent.click(screen.getByRole("button", { name: "Clear finished (1)" }));
    await waitFor(() => expect(screen.queryByText("call the bank")).toBeNull());
  });

  it("starts and stops a watcher", async () => {
    let watchers: { id: string; kind: string; label: string; created_at: number }[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === "today_overview") return overview;
      if (method === "watch_add") watchers = [{ id: "w1", kind: "process", label: "chrome closes", created_at: 1 }];
      if (method === "watch_cancel") watchers = [];
      return { status: "ok", watchers, text: params.text };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("Tell me when"), { target: { value: "Chrome closes" } });
    await fireEvent.click(screen.getByRole("button", { name: "Watch" }));
    await waitFor(() => expect(screen.getByText("Until chrome closes")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("watch_add", { text: "Chrome closes" });

    await fireEvent.click(screen.getByRole("button", { name: "Stop watching until chrome closes" }));
    await waitFor(() => expect(screen.queryByText("Until chrome closes")).toBeNull());
  });

  it("lists what was copied and copies an item again", async () => {
    const items = [
      { id: "c1", text: "https://example.com/report", at: 2, app: "msedge" },
      { id: "c2", text: "meeting notes", at: 1, app: "notepad" },
    ];
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "today_overview") return overview;
      return { status: "ok", items };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText("https://example.com/report")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Copy again: meeting notes" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("clipboard_history_copy", { id: "c2" });
  });

  it("starts a pomodoro and manages quiet hours", async () => {
    let windows: { id: string; days: number[]; start: number; end: number }[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
      if (method === "today_overview") return overview;
      if (method === "focus_schedule_add")
        windows = [{ id: "w1", days: params.days as number[], start: Number(params.start), end: Number(params.end) }];
      if (method === "focus_schedule_delete") windows = [];
      return { status: "ok", windows };
    });
    render(TodayPanel);
    await waitFor(() => expect(screen.getByText(overview.brief)).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Start pomodoro" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("focus_start", { minutes: 25, break_minutes: 5 });

    await fireEvent.click(screen.getByRole("button", { name: "Add quiet hours" }));
    await waitFor(() => expect(screen.getByText("09:00 to 12:00")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("focus_schedule_add", {
      days: [0, 1, 2, 3, 4],
      start: 540,
      end: 720,
    });
    await fireEvent.click(screen.getByRole("button", { name: "Remove quiet hours 09:00" }));
    await waitFor(() => expect(screen.queryByText("09:00 to 12:00")).toBeNull());
  });
});

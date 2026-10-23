import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => ({ updateSectionDetailed: vi.fn() }));
const sessionMocks = vi.hoisted(() => ({ openUndoPreview: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});
vi.mock("../stores/settings", () => {
  const store = writable({ brief: { enabled: false, time: "18:00", speak: false } });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: settingsMocks.updateSectionDetailed } };
});
vi.mock("../stores/session", () => ({ session: { openUndoPreview: sessionMocks.openUndoPreview } }));

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
});

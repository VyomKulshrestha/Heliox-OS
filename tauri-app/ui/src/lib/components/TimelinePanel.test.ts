import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => ({ updateSectionDetailed: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

vi.mock("../stores/settings", () => {
  const store = writable({ timeline: { enabled: true, retention_days: 7, excluded_apps: [] } });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: settingsMocks.updateSectionDetailed } };
});

import TimelinePanel from "./TimelinePanel.svelte";

describe("TimelinePanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    settingsMocks.updateSectionDetailed.mockReset();
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "timeline_status") return { status: "ok", span_count: 12, paused_until: null };
      if (method === "timeline_summary")
        return { status: "ok", summary: { apps: [{ app: "Code", seconds: 2400, top_titles: ["server.py"] }] } };
      if (method === "timeline_search")
        return { status: "ok", spans: [{ start: 1792700000, end: 1792700600, app: "Chrome", title: "Report" }] };
      return { status: "ok" };
    });
  });

  it("shows today's apps and finds a window", async () => {
    render(TimelinePanel);

    await waitFor(() => expect(screen.getByText("Code")).toBeTruthy());
    expect(screen.getByText("40 min · server.py")).toBeTruthy();

    await fireEvent.input(screen.getByLabelText("Search timeline"), { target: { value: "report" } });
    await fireEvent.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(screen.getByText("Report")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("timeline_search", { query: "report" });
  });

  it("pauses for an hour and clears only on a second click", async () => {
    render(TimelinePanel);
    await waitFor(() => expect(screen.getByText("Code")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Pause 1 hour" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("timeline_pause", { minutes: 60 });

    await fireEvent.click(screen.getByRole("button", { name: "Clear timeline" }));
    expect(daemonMocks.call).not.toHaveBeenCalledWith("timeline_clear", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Click again to delete" }));
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("timeline_clear", { confirmed: true }));
  });

  it("turns recording off through the saved settings", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    render(TimelinePanel);

    await fireEvent.click(screen.getByRole("button", { name: "Record activity timeline" }));

    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("timeline", { enabled: false });
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ScheduledRunsPanel from "./ScheduledRunsPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));

const morning = {
  run_id: "r1",
  name: "Morning disk report",
  goal: "summarise disk use",
  allowed_actions: ["disk_usage", "notify"],
  every_minutes: 0,
  daily_at: "08:15",
  weekdays: [0, 1, 2, 3, 4],
  enabled: true,
  next_run_at: 1_800_000_000,
  last_status: "started",
  last_job_status: "success",
};

describe("ScheduledRunsPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
  });

  it("lists schedules with their timetable and last result", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", runs: [morning] });
    render(ScheduledRunsPanel);

    expect(await screen.findByText("Morning disk report")).toBeTruthy();
    expect(screen.getByText(/Daily at 08:15, Mon, Tue, Wed, Thu, Fri · Last run: success/)).toBeTruthy();
  });

  it("creates an interval schedule with the approved actions split into a list", async () => {
    daemonMocks.call.mockImplementation((method: string, params: Record<string, unknown>) => {
      if (method === "scheduled_run_list") return Promise.resolve({ status: "ok", runs: [] });
      return Promise.resolve({
        status: "ok",
        run: { ...morning, run_id: "r2", name: params.name, every_minutes: 30, daily_at: "", weekdays: [] },
      });
    });
    render(ScheduledRunsPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "New schedule" }));

    await fireEvent.input(screen.getByLabelText("Name"), { target: { value: "Watch downloads" } });
    await fireEvent.input(screen.getByLabelText("Goal"), { target: { value: "list new files in Downloads" } });
    await fireEvent.change(screen.getByLabelText("Timing"), { target: { value: "interval" } });
    await fireEvent.input(await screen.findByLabelText("Minutes between runs"), { target: { value: "30" } });
    await fireEvent.input(screen.getByLabelText(/Approved actions/), { target: { value: "file_list, notify" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save schedule" }));

    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("scheduled_run_create", {
        name: "Watch downloads",
        goal: "list new files in Downloads",
        allowed_actions: ["file_list", "notify"],
        every_minutes: 30,
      }),
    );
    expect(await screen.findByText("Watch downloads")).toBeTruthy();
    expect(screen.getByText(/Every 30 min/)).toBeTruthy();
  });

  it("shows the daemon's validation message", async () => {
    daemonMocks.call.mockImplementation((method: string) =>
      Promise.resolve(
        method === "scheduled_run_list"
          ? { status: "ok", runs: [] }
          : { status: "error", message: "Unknown action types: launch_rockets" },
      ),
    );
    render(ScheduledRunsPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "New schedule" }));
    await fireEvent.input(screen.getByLabelText("Name"), { target: { value: "x" } });
    await fireEvent.input(screen.getByLabelText("Goal"), { target: { value: "y" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save schedule" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Unknown action types: launch_rockets");
  });

  it("toggles, runs now and deletes through the dedicated RPCs", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "scheduled_run_list") return Promise.resolve({ status: "ok", runs: [morning] });
      if (method === "scheduled_run_set_enabled")
        return Promise.resolve({ status: "ok", run: { ...morning, enabled: false } });
      if (method === "scheduled_run_now") return Promise.resolve({ status: "ok", run: morning });
      return Promise.resolve({ status: "ok", deleted: true });
    });
    render(ScheduledRunsPanel);

    await fireEvent.click(await screen.findByLabelText("Enable Morning disk report"));
    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("scheduled_run_set_enabled", { run_id: "r1", enabled: false }),
    );
    await fireEvent.click(screen.getByRole("button", { name: "Run now" }));
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("scheduled_run_now", { run_id: "r1" }));
    await fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(screen.queryByText("Morning disk report")).toBeNull());
  });
});

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ScheduledRunsPanel from "./ScheduledRunsPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));

afterEach(cleanup);

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
    await fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    await fireEvent.click(screen.getByRole("button", { name: "Delete schedule" }));
    await waitFor(() => expect(screen.queryByText("Morning disk report")).toBeNull());
  });

  it("puts the switch back and shows the daemon's message when turning a schedule off fails", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "scheduled_run_list") return Promise.resolve({ status: "ok", runs: [morning] });
      return Promise.resolve({ status: "error", message: "No such scheduled run" });
    });
    render(ScheduledRunsPanel);
    const toggle = (await screen.findByLabelText("Enable Morning disk report")) as HTMLInputElement;

    await fireEvent.click(toggle);

    expect((await screen.findByRole("alert")).textContent).toContain("No such scheduled run");
    expect(toggle.checked).toBe(true);
    expect(toggle.disabled).toBe(false);
  });

  it("puts the switch back when the toggle request cannot reach the daemon", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "scheduled_run_list")
        return Promise.resolve({ status: "ok", runs: [{ ...morning, enabled: false }] });
      return Promise.reject(new Error("Cannot connect to Heliox OS daemon"));
    });
    render(ScheduledRunsPanel);
    const toggle = (await screen.findByLabelText("Enable Morning disk report")) as HTMLInputElement;

    await fireEvent.click(toggle);

    expect((await screen.findByRole("alert")).textContent).toContain("Cannot connect to Heliox OS daemon");
    expect(toggle.checked).toBe(false);
  });

  it("leaves the run state alone when Run now fails and shows why", async () => {
    const neverRun = { ...morning, last_status: "", last_job_status: "" };
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "scheduled_run_list") return Promise.resolve({ status: "ok", runs: [neverRun] });
      return Promise.resolve({
        status: "error",
        message: "The run could not start",
        run: { ...neverRun, last_status: "started", last_job_status: "running" },
      });
    });
    render(ScheduledRunsPanel);

    await fireEvent.click(await screen.findByRole("button", { name: "Run now" }));

    expect((await screen.findByRole("alert")).textContent).toContain("The run could not start");
    expect(screen.getByText(/Not run yet/)).toBeTruthy();
    expect(screen.queryByText(/Last run/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Starting..." })).toBeNull();
    expect((screen.getByRole("button", { name: "Run now" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("asks before deleting and deletes nothing when the prompt is cancelled", async () => {
    daemonMocks.call.mockImplementation((method: string) =>
      Promise.resolve(
        method === "scheduled_run_list" ? { status: "ok", runs: [morning] } : { status: "ok", deleted: true },
      ),
    );
    render(ScheduledRunsPanel);

    await fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    const prompt = screen.getByRole("alertdialog", { name: "Delete schedule" });
    expect(prompt.textContent).toContain('Delete the schedule "Morning disk report"?');
    await fireEvent.keyDown(prompt, { key: "Escape" });

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(daemonMocks.call).not.toHaveBeenCalledWith("scheduled_run_delete", expect.anything());
    expect(screen.getByText("Morning disk report")).toBeTruthy();

    await fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await fireEvent.click(screen.getByRole("button", { name: "Delete schedule" }));
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("scheduled_run_delete", { run_id: "r1" }));
    await waitFor(() => expect(screen.queryByText("Morning disk report")).toBeNull());
  });
});

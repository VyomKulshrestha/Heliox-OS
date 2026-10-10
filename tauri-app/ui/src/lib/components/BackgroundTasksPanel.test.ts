import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import BackgroundTasksPanel from "./BackgroundTasksPanel.svelte";

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Array<(method: string, params: unknown) => void>,
}));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  onNotification: (handler: (method: string, params: unknown) => void) => daemonMocks.handlers.push(handler),
  offNotification: (handler: (method: string, params: unknown) => void) => {
    daemonMocks.handlers = daemonMocks.handlers.filter((item) => item !== handler);
  },
}));

const interrupted = {
  job_id: "j1",
  goal: "Rename every report in Downloads",
  status: "interrupted",
  current_step: 1,
  total_steps: 3,
  result_summary: "Interrupted when Heliox stopped. Resume it when you are ready.",
  resumable: true,
};

describe("BackgroundTasksPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.handlers = [];
  });

  it("offers Resume for an interrupted job and shows it running afterwards", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "autonomous_jobs") return Promise.resolve({ jobs: [interrupted] });
      if (method === "autonomous_resume") {
        return Promise.resolve({ status: "resumed", job: { ...interrupted, status: "queued", resumable: false } });
      }
      throw new Error(`unexpected ${method}`);
    });
    render(BackgroundTasksPanel);

    expect(await screen.findByText("Interrupted when Heliox stopped")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("autonomous_resume", { job_id: "j1" }));
    expect(await screen.findByText("Starting")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
  });

  it("removes a dismissed job and follows live job notifications", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "autonomous_jobs") return Promise.resolve({ jobs: [interrupted] });
      if (method === "autonomous_cancel") return Promise.resolve({ cancelled: true, job_id: "j1" });
      throw new Error(`unexpected ${method}`);
    });
    render(BackgroundTasksPanel);
    await screen.findByText("Rename every report in Downloads");

    await fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(screen.queryByText("Rename every report in Downloads")).toBeNull());

    daemonMocks.handlers[0]("autonomous_step_start", {
      job_id: "j2",
      goal: "Back up the notes folder",
      status: "running",
      current_step: 0,
      total_steps: 2,
      result_summary: "",
    });
    expect(await screen.findByText("Step 1 of 2")).toBeTruthy();
    daemonMocks.handlers[0]("autonomous_complete", {
      job_id: "j2",
      goal: "Back up the notes folder",
      status: "success",
      current_step: 1,
      total_steps: 2,
      result_summary: "done",
    });
    await waitFor(() => expect(screen.queryByText("Back up the notes folder")).toBeNull());
  });

  it("reports a failed resume instead of pretending it worked", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "autonomous_jobs") return Promise.resolve({ jobs: [interrupted] });
      return Promise.resolve({ status: "error", message: "Only an interrupted job can be resumed" });
    });
    render(BackgroundTasksPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "Resume" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Only an interrupted job can be resumed");
  });
});

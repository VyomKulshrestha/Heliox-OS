import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import RecordedWorkflowsPanel from "./RecordedWorkflowsPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, ...daemonMocks };
});

afterEach(cleanup);

describe("recorded workflow consent", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "recorded_workflow_status") {
        return Promise.resolve({
          status: "ok",
          active: false,
          name: "",
          capture_visual: false,
          step_count: 0,
          sample_count: 0,
          omitted_actions: 0,
        });
      }
      if (method === "recorded_workflow_list") return Promise.resolve({ status: "ok", workflows: [] });
      if (method === "recorded_workflow_start") return Promise.resolve({ status: "ok", active: true });
      throw new Error(`unexpected method: ${method}`);
    });
  });

  it("requires explicit consent and leaves screen features off by default", async () => {
    render(RecordedWorkflowsPanel);
    const start = await screen.findByRole("button", { name: "Start recording" });
    expect((start as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.input(screen.getByLabelText("Workflow name"), { target: { value: "Morning review" } });
    expect((start as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.click(screen.getByLabelText(/I want Heliox to record actions/));
    await fireEvent.click(start);
    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_start", {
      name: "Morning review",
      goal: "",
      consent: true,
      capture_visual: false,
      capture_manual: false,
      manual_consent: false,
    });
  });

  it("requires a second consent for desktop-wide manual input", async () => {
    render(RecordedWorkflowsPanel);
    const start = await screen.findByRole("button", { name: "Start recording" });
    await fireEvent.input(screen.getByLabelText("Workflow name"), { target: { value: "Desktop review" } });
    await fireEvent.click(screen.getByLabelText(/I want Heliox to record actions/));
    await fireEvent.click(screen.getByLabelText(/Also record my manual desktop actions/));
    expect((start as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.click(screen.getByLabelText(/I consent to a temporary desktop-wide/));
    expect((start as HTMLButtonElement).disabled).toBe(false);
    await fireEvent.click(start);
    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_start", {
      name: "Desktop review",
      goal: "",
      consent: true,
      capture_visual: false,
      capture_manual: true,
      manual_consent: true,
    });
  });

  it("explains rolling visual capture when both manual and visual options are selected", async () => {
    render(RecordedWorkflowsPanel);
    await screen.findByRole("button", { name: "Start recording" });
    await fireEvent.input(screen.getByLabelText("Workflow name"), { target: { value: "Visual review" } });
    await fireEvent.click(screen.getByLabelText(/I want Heliox to record actions/));
    await fireEvent.click(screen.getByLabelText(/Also record my manual desktop actions/));
    await fireEvent.click(screen.getByLabelText(/I consent to a temporary desktop-wide/));
    await fireEvent.click(screen.getByLabelText(/Also capture local 16×16/));
    expect(screen.getByText(/rolling in-memory feature supports manual clicks/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_start", {
      name: "Visual review",
      goal: "",
      consent: true,
      capture_visual: true,
      capture_manual: true,
      manual_consent: true,
    });
  });
});

describe("recorded workflow goal feedback", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "recorded_workflow_status") return Promise.resolve({ status: "ok", active: false });
      if (method === "recorded_workflow_list")
        return Promise.resolve({
          status: "ok",
          workflows: [
            {
              id: "flow-1",
              name: "Review",
              goal: "Check a status",
              steps: [],
              step_count: 1,
              sample_count: 0,
              omitted_actions: 0,
              visual_gaps: 0,
              incomplete: false,
              needs_review: false,
              overlap_warning: false,
              placeholders: [],
              schedule: null,
              feedback_count: 0,
              feedback_visual_sample_count: 0,
            },
          ],
        });
      if (method === "recorded_workflow_run")
        return Promise.resolve({
          status: "submitted",
          job: { job_id: "job-1", status: "queued" },
        });
      if (method === "autonomous_job") return Promise.resolve({ job_id: "job-1", status: "success" });
      if (method === "recorded_workflow_feedback") return Promise.resolve({ status: "ok" });
      throw new Error(`unexpected method: ${method}`);
    });
  });

  it("asks for explicit goal outcome only after a completed run", async () => {
    render(RecordedWorkflowsPanel);
    await fireEvent.click(await screen.findByText("Review"));
    await fireEvent.click(screen.getByRole("button", { name: "Run with approvals" }));
    expect(screen.queryByRole("button", { name: "Not achieved" })).toBeNull();
    await fireEvent.click(await screen.findByRole("button", { name: "Check run status" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Not achieved" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_feedback", {
      workflow_id: "flow-1",
      job_id: "job-1",
      outcome: "not_achieved",
    });
    expect(await screen.findByText("Goal-outcome rating saved locally.")).toBeTruthy();
  });

  it("captures replay visuals only when enabled for that run", async () => {
    render(RecordedWorkflowsPanel);
    await fireEvent.click(await screen.findByText("Review"));
    await fireEvent.click(screen.getByLabelText(/Capture local 16×16 before\/after features/));
    await fireEvent.click(screen.getByRole("button", { name: "Run with approvals" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_run", {
      workflow_id: "flow-1",
      reviewed: false,
      text_values: {},
      capture_visual: true,
    });
  });
});

describe("recorded workflow delete", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "recorded_workflow_status") return Promise.resolve({ status: "ok", active: false });
      if (method === "recorded_workflow_list")
        return Promise.resolve({
          status: "ok",
          workflows: [
            {
              id: "flow-1",
              name: "Review",
              goal: "Check a status",
              steps: [],
              step_count: 1,
              sample_count: 0,
              omitted_actions: 0,
              visual_gaps: 0,
              incomplete: false,
              needs_review: false,
              overlap_warning: false,
              placeholders: [],
              schedule: null,
              feedback_count: 0,
              feedback_visual_sample_count: 0,
            },
          ],
        });
      if (method === "recorded_workflow_delete") return Promise.resolve({ status: "ok" });
      throw new Error(`unexpected method: ${method}`);
    });
  });

  it("asks in the app before deleting and deletes nothing when cancelled", async () => {
    const nativeConfirm = vi.spyOn(window, "confirm");
    render(RecordedWorkflowsPanel);
    await fireEvent.click(await screen.findByText("Review"));

    await fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    const prompt = screen.getByRole("alertdialog", { name: "Delete workflow" });
    expect(prompt.textContent).toContain("Delete saved workflow “Review” and its local visual samples?");
    await fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(daemonMocks.call).not.toHaveBeenCalledWith("recorded_workflow_delete", expect.anything());

    await fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await fireEvent.click(screen.getByRole("button", { name: "Delete workflow" }));

    expect(daemonMocks.call).toHaveBeenCalledWith("recorded_workflow_delete", { workflow_id: "flow-1" });
    expect(await screen.findByText("Saved workflow deleted.")).toBeTruthy();
    expect(nativeConfirm).not.toHaveBeenCalled();
    nativeConfirm.mockRestore();
  });
});

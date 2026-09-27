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
        return Promise.resolve({ status: "ok", active: false, name: "", capture_visual: false, step_count: 0, sample_count: 0, omitted_actions: 0 });
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
      name: "Morning review", goal: "", consent: true, capture_visual: false,
      capture_manual: false, manual_consent: false,
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
      name: "Desktop review", goal: "", consent: true, capture_visual: false,
      capture_manual: true, manual_consent: true,
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
      name: "Visual review", goal: "", consent: true, capture_visual: true,
      capture_manual: true, manual_consent: true,
    });
  });
});

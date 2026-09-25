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
    });
  });
});

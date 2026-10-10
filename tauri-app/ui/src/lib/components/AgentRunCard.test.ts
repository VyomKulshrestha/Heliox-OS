import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import AgentRunCard from "./AgentRunCard.svelte";
import { applyAgentRunStatus, applyToolCall } from "../utils/agentRuns";
import type { Message } from "../stores/session";

const start = {
  job_id: "j1",
  goal: "Report free disk space",
  source: "scheduled",
  step: 0,
  round: 1,
  call_id: "c1",
  action_type: "disk_usage",
  target: "C:",
  arguments: '{"path": "C:"}',
  tier: "read_only",
  phase: "start",
};

describe("agent run messages", () => {
  it("creates one card per job and updates a call in place when it completes", () => {
    let messages: Message[] = [{ type: "user", text: "hi", timestamp: 1 }];
    messages = applyToolCall(messages, start);
    messages = applyToolCall(messages, { ...start, call_id: "c2", action_type: "notify", target: "" });
    messages = applyToolCall(messages, {
      ...start,
      phase: "complete",
      success: true,
      output: "C: 19 GB free",
      duration_ms: 1500,
    });

    const runs = messages.filter((message) => message.type === "agent_run");
    expect(runs).toHaveLength(1);
    const run = runs[0].agentRun!;
    expect(run.calls.map((call) => [call.callId, call.status])).toEqual([
      ["c1", "success"],
      ["c2", "running"],
    ]);
    expect(run.calls[0].output).toBe("C: 19 GB free");

    const finished = applyAgentRunStatus(messages, "j1", "success");
    expect(finished.find((message) => message.type === "agent_run")!.agentRun!.status).toBe("success");
    expect(applyAgentRunStatus(messages, "other", "failed")).toBe(messages);
  });

  it("marks a call refused before it ran as not run", () => {
    const messages = applyToolCall([], {
      ...start,
      phase: "complete",
      executed: false,
      success: false,
      error: "denied",
    });

    expect(messages[0].agentRun!.calls[0].status).toBe("skipped");
  });
});

describe("AgentRunCard", () => {
  it("shows each call and reveals arguments and output on demand", async () => {
    const [message] = applyToolCall([], {
      ...start,
      phase: "complete",
      success: true,
      output: "C: 19 GB free",
      duration_ms: 1500,
    });
    render(AgentRunCard, { run: { ...message.agentRun!, status: "success" } });

    expect(screen.getByText("Scheduled run")).toBeTruthy();
    expect(screen.getByText("Done")).toBeTruthy();
    expect(screen.getByText("Report free disk space")).toBeTruthy();
    expect(screen.getByText("1.5s")).toBeTruthy();
    expect(screen.queryByText("C: 19 GB free")).toBeNull();

    await fireEvent.click(screen.getByRole("button", { name: /Disk usage/ }));

    expect(screen.getByText('{"path": "C:"}')).toBeTruthy();
    expect(screen.getByText("C: 19 GB free")).toBeTruthy();
    expect(screen.getByText(/Step 1, round 1 · read only/)).toBeTruthy();
  });

  it("shows an error instead of output for a failed call", async () => {
    const [message] = applyToolCall([], { ...start, phase: "complete", success: false, error: "drive not found" });
    render(AgentRunCard, { run: message.agentRun! });

    await fireEvent.click(screen.getByRole("button", { name: /Disk usage/ }));

    expect(screen.getByText("drive not found")).toBeTruthy();
    expect(screen.getByText("Working")).toBeTruthy();
  });
});

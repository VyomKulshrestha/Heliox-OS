import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let notify: NotificationHandler | null = null;

vi.mock("../api/daemon", () => ({
  call: vi.fn(),
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: vi.fn(),
  onNotification: (handler: NotificationHandler) => {
    notify = handler;
  },
}));

vi.mock("./settings", () => ({ settings: writable({ model: { cloud_model: "ollama" } }) }));
vi.mock("./companion", () => ({ companion: { speak: vi.fn() } }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

const action = { action_type: "file_write", target: "C:/notes/todo.txt", parameters: {} };

async function planCard(planId: string, extra: Record<string, unknown> = {}) {
  const { session } = await import("./session");
  notify!("plan_preview", { plan_id: planId, explanation: "write the note", actions: [action], ...extra });
  const plan = () => get(session).messages.find((m) => m.type === "plan" && m.plan?.plan_id === planId)?.plan;
  return { session, plan };
}

describe("plan checks in the session", () => {
  beforeEach(() => {
    localStorage.clear();
    notify = null;
    vi.resetModules();
  });

  it("attaches the world model summary and Gnomon review to their plan card", async () => {
    const { session, plan } = await planCard("p-1", { world_model_enabled: true });
    expect(plan()?.checks).toBeUndefined();

    const summary = { enabled: true, raised: false, score: 0, observed: null, predicted: null };
    notify!("world_model_assessment", { plan_id: "p-1", world_model_score: 0, summary });
    notify!("gnomon_review", { plan_id: "p-1", state: "reviewed", escalate: false, scores: { in_scope: 0.9 } });

    expect(plan()?.checks?.worldModel).toEqual(summary);
    expect(plan()?.checks?.gnomon).toMatchObject({ state: "reviewed", scores: { in_scope: 0.9 } });
    expect(get(session).currentPlan?.checks?.gnomon?.state).toBe("reviewed");
  });

  it("records a switched-off world model from the plan preview", async () => {
    const { plan } = await planCard("p-2", { world_model_enabled: false });
    expect(plan()?.checks).toEqual({ worldModel: { enabled: false } });
  });

  it("ignores assessments without a summary or for another plan", async () => {
    const { plan } = await planCard("p-3");
    notify!("world_model_assessment", { plan_id: "p-3", combined_score: 0.2 }); // autonomous shape
    notify!("gnomon_review", { plan_id: "other", state: "loading" });
    expect(plan()?.checks).toBeUndefined();
  });

  it("attaches the safety review with only the critic's own text", async () => {
    const { plan } = await planCard("p-4");
    notify!("critic_verdict", {
      plan_id: "p-4",
      verdict: "WARN",
      risk_score: 0.42,
      issues: ["Step 1 overwrites todo.txt.", "A daemon note."],
      findings: [
        { step: 1, severity: "high", text: "Step 1 overwrites todo.txt." },
        { step: "two", severity: "urgent", text: "Odd fields." },
        { step: null, severity: "low", text: "" },
      ],
      recommendation: "Check todo.txt first.",
      ran: true,
      world_model: { predictions: [] },
    });
    expect(plan()?.checks?.critic).toEqual({
      verdict: "WARN",
      ran: true,
      downgraded: false,
      findings: [
        { step: 1, severity: "high", text: "Step 1 overwrites todo.txt." },
        { step: null, severity: null, text: "Odd fields." },
      ],
      recommendation: "Check todo.txt first.",
    });
  });

  it("keeps a skipped or failed review free of other components' text", async () => {
    const { plan } = await planCard("p-5");
    notify!("critic_verdict", {
      plan_id: "p-5",
      verdict: "SKIPPED",
      issues: ["predicted disk usage 96% exceeds the safe threshold"],
      recommendation: "Low-risk heuristic, LLM safety review was skipped.",
      critic_skipped: "low_risk_heuristic",
    });
    expect(plan()?.checks?.critic).toEqual({
      verdict: "SKIPPED",
      ran: false,
      skipped: "low_risk_heuristic",
      findings: [],
      recommendation: "",
    });
    notify!("critic_verdict", {
      plan_id: "p-5",
      verdict: "WARN",
      issues: ["Critic agent encountered an error: timed out"],
      recommendation: "The safety review could not run.",
      ran: false,
    });
    expect(plan()?.checks?.critic).toMatchObject({ verdict: "WARN", ran: false, findings: [], recommendation: "" });
  });

  it("drops a verdict that names no plan", async () => {
    const { plan } = await planCard("p-6");
    notify!("critic_verdict", { verdict: "APPROVE", findings: [], recommendation: "" });
    expect(plan()?.checks).toBeUndefined();
  });

  it("attaches the safety review with only the critic's own text", async () => {
    const { plan } = await planCard("p-4");
    notify!("critic_verdict", {
      plan_id: "p-4",
      verdict: "WARN",
      risk_score: 0.42,
      issues: ["Step 1 overwrites todo.txt.", "A daemon note."],
      findings: [
        { step: 1, severity: "high", text: "Step 1 overwrites todo.txt." },
        { step: "two", severity: "urgent", text: "Odd fields." },
        { step: null, severity: "low", text: "" },
      ],
      recommendation: "Check todo.txt first.",
      ran: true,
      world_model: { predictions: [] },
    });
    expect(plan()?.checks?.critic).toEqual({
      verdict: "WARN",
      ran: true,
      downgraded: false,
      findings: [
        { step: 1, severity: "high", text: "Step 1 overwrites todo.txt." },
        { step: null, severity: null, text: "Odd fields." },
      ],
      recommendation: "Check todo.txt first.",
    });
  });

  it("keeps a skipped or failed review free of other components' text", async () => {
    const { plan } = await planCard("p-5");
    notify!("critic_verdict", {
      plan_id: "p-5",
      verdict: "SKIPPED",
      issues: ["predicted disk usage 96% exceeds the safe threshold"],
      recommendation: "Low-risk heuristic, LLM safety review was skipped.",
      critic_skipped: "low_risk_heuristic",
    });
    expect(plan()?.checks?.critic).toEqual({
      verdict: "SKIPPED",
      ran: false,
      skipped: "low_risk_heuristic",
      findings: [],
      recommendation: "",
    });
    notify!("critic_verdict", {
      plan_id: "p-5",
      verdict: "WARN",
      issues: ["Critic agent encountered an error: timed out"],
      recommendation: "The safety review could not run.",
      ran: false,
    });
    expect(plan()?.checks?.critic).toMatchObject({ verdict: "WARN", ran: false, findings: [], recommendation: "" });
  });

  it("drops a verdict that names no plan", async () => {
    const { plan } = await planCard("p-6");
    notify!("critic_verdict", { verdict: "APPROVE", findings: [], recommendation: "" });
    expect(plan()?.checks).toBeUndefined();
  });
});

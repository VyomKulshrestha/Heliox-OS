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
});

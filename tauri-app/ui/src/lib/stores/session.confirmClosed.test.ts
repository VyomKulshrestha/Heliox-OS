import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
type ConnectionHandler = (connected: boolean) => void;
let notify: NotificationHandler | null = null;
let connection: ConnectionHandler | null = null;

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: (handler: ConnectionHandler) => {
    connection = handler;
  },
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

/** A running task (its execute RPC never returns here) waiting on approval for `planId`. */
async function waitingOnApproval(planId: string, replies: Record<string, unknown> = {}) {
  const execute = deferred<Record<string, unknown>>();
  daemonMocks.call.mockImplementation((method: string) => {
    if (method === "execute") return execute.promise;
    if (method in replies) return replies[method];
    return Promise.resolve({ status: "ok" });
  });
  const { session } = await import("./session");
  void session.sendCommand("back up and ship");
  notify!("confirm_required", {
    plan_id: planId,
    actions: [{ index: 0, action_type: "file_copy", target: "C:/launch", parameters: {} }],
  });
  expect(get(session)).toMatchObject({ confirmRequired: true, confirmPlanId: planId, loading: true });
  return { session, execute };
}

describe("an approval the daemon stops waiting on", () => {
  beforeEach(() => {
    localStorage.clear();
    notify = null;
    connection = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("goes away after a correction, and cannot be answered while the correction is in flight", async () => {
    const reply = deferred<Record<string, unknown>>();
    const { session } = await waitingOnApproval("plan-old", { interject: reply.promise });

    const sending = session.sendCommand("Skip the backup.");
    expect(get(session).confirmSubmitting).toBe(true);
    expect(await session.confirm(true, [0], { inputChannel: "dialog" })).toEqual({
      kind: "not_sent",
      reason: "submitting",
    });

    reply.resolve({ status: "revising", message: "Live correction accepted." });
    await sending;
    // The reply alone does not reopen the dialog for a plan that no longer exists.
    expect(get(session)).toMatchObject({ confirmRequired: true, confirmSubmitting: true });

    // Status updates and a close for another approval never close this one.
    notify!("companion_interjection", { mode: "correct", message: "Correction received." });
    notify!("status", { phase: "planning" });
    notify!("confirmation_closed", { plan_id: "background-job", resolution: "timeout" });
    expect(get(session).confirmPlanId).toBe("plan-old");

    notify!("confirmation_closed", { plan_id: "plan-old", resolution: "interject_correction" });
    expect(get(session)).toMatchObject({ confirmRequired: false, confirmPlanId: "", confirmSubmitting: false });
    expect(get(session).loading).toBe(true);
    expect(
      daemonMocks.call.mock.calls.filter(([method]) => method === "confirm"),
      "nothing was answered",
    ).toEqual([]);
  });

  it("shows the revised plan's approval normally", async () => {
    const { session } = await waitingOnApproval("plan-old", {
      interject: Promise.resolve({ status: "revising" }),
    });
    await session.sendCommand("Skip the backup.");
    notify!("confirmation_closed", { plan_id: "plan-old", resolution: "interject_correction" });

    notify!("confirm_required", {
      plan_id: "plan-new",
      actions: [{ index: 0, action_type: "file_delete", target: "C:/launch/drafts", parameters: {} }],
    });
    // A late close for the old plan leaves the new dialog alone.
    notify!("confirmation_closed", { plan_id: "plan-old", resolution: "interject_correction" });
    expect(get(session)).toMatchObject({
      confirmRequired: true,
      confirmPlanId: "plan-new",
      confirmSubmitting: false,
      confirmError: "",
    });
  });

  it("gives the dialog back when the daemon kept the approval open", async () => {
    const { session } = await waitingOnApproval("plan-1", {
      interject: Promise.resolve({ status: "approval_requires_dialog", plan_id: "plan-1", message: "Approve here." }),
    });
    await session.sendCommand("yes");
    expect(get(session)).toMatchObject({ confirmRequired: true, confirmPlanId: "plan-1", confirmSubmitting: false });
  });

  it("goes away after Stop", async () => {
    const reply = deferred<Record<string, unknown>>();
    const { session } = await waitingOnApproval("plan-1", { abort: reply.promise });
    const stopping = session.abort();
    expect(get(session).confirmSubmitting).toBe(true);
    reply.resolve({ status: "aborted" });
    await stopping;
    expect(get(session).confirmSubmitting).toBe(true);
    notify!("confirmation_closed", { plan_id: "plan-1", resolution: "abort" });
    expect(get(session).confirmRequired).toBe(false);
  });

  it("gives the dialog back when Stop found nothing to stop", async () => {
    const { session } = await waitingOnApproval("plan-1", {
      abort: Promise.resolve({ status: "no_active_execution" }),
    });
    await session.abort();
    expect(get(session)).toMatchObject({ confirmRequired: true, confirmSubmitting: false });
  });

  it("goes away when the daemon's wait times out", async () => {
    const { session } = await waitingOnApproval("plan-1");
    notify!("confirmation_closed", { plan_id: "plan-1", resolution: "timeout" });
    expect(get(session).confirmRequired).toBe(false);
  });

  it("does not stay clickable when the connection to the daemon drops", async () => {
    const { session } = await waitingOnApproval("plan-1");
    connection!(false);
    const state = get(session);
    expect(state).toMatchObject({ confirmRequired: false, confirmPlanId: "", daemonConnected: false });
    expect(state.messages.at(-1)?.text).toBe(
      "Lost the connection to Heliox, so the approval dialog was closed. Nothing in it was approved.",
    );
    connection!(true);
    expect(get(session).confirmRequired).toBe(false);
  });

  it("keeps the normal approve and deny paths", async () => {
    const { session } = await waitingOnApproval("plan-1", {
      confirm: Promise.resolve({ status: "ok", confirmed: true }),
    });
    expect(await session.confirm(true, [0], { inputChannel: "dialog" })).toEqual({ kind: "recorded", confirmed: true });
    expect(get(session).confirmRequired).toBe(false);
    notify!("confirmation_closed", { plan_id: "plan-1", resolution: "confirm_rpc" });
    expect(get(session).confirmRequired).toBe(false);

    notify!("confirm_required", { plan_id: "plan-2", actions: [] });
    daemonMocks.call.mockImplementation(() => Promise.resolve({ status: "ok", confirmed: false }));
    // The daemon's close can arrive before the confirm reply; the reply still reports the decision.
    const denying = session.confirm(false, undefined, { inputChannel: "dialog" });
    notify!("confirmation_closed", { plan_id: "plan-2", resolution: "confirm_rpc" });
    expect(await denying).toEqual({ kind: "recorded", confirmed: false });
    expect(get(session).confirmRequired).toBe(false);
  });
});

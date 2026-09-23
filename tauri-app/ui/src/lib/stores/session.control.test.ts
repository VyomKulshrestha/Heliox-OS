import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: vi.fn(),
  onNotification: vi.fn(),
}));

vi.mock("./settings", () => ({
  settings: writable({ model: { cloud_model: "ollama" } }),
}));
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

describe("session execution controls", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("reports an unacknowledged stop as an error", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "Cancellation channel unavailable" });
    const { session } = await import("./session");

    await session.abort();

    expect(get(session).messages.at(-1)).toMatchObject({
      type: "error",
      text: "Stop failed: Cancellation channel unavailable",
    });
  });

  it("restores the active phase when a live correction is rejected", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "interject") return Promise.resolve({ status: "disabled", message: "Corrections are disabled" });
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");

    const activeCommand = session.sendCommand("Inspect the system");
    await session.sendCommand("Check memory first");

    expect(get(session)).toMatchObject({ loading: true, phase: "" });
    expect(get(session).messages.at(-1)).toMatchObject({
      type: "error",
      text: "Live correction failed: Corrections are disabled",
    });

    execution.resolve({ status: "cancelled", message: "Stopped for test" });
    await activeCommand;
  });

  it("answers an in-task question without stopping the active command", async () => {
    const execution = deferred<Record<string, unknown>>();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "execute") return execution.promise;
      if (method === "interject") return Promise.resolve({ status: "answered", message: "Still checking the system." });
      return Promise.resolve({ status: "ok" });
    });
    const { session } = await import("./session");

    const activeCommand = session.sendCommand("Inspect the system");
    await session.sendCommand("What are you doing?", [], "ask");

    expect(daemonMocks.call).toHaveBeenCalledWith("interject", {
      input: "What are you doing?",
      mode: "ask",
    });
    expect(get(session).loading).toBe(true);
    expect(get(session).messages.at(-1)).toMatchObject({
      type: "result",
      text: "Still checking the system.",
    });

    execution.resolve({ status: "success", message: "Finished checking." });
    await activeCommand;
  });
});

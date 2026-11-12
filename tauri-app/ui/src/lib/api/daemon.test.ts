import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(),
}));
vi.mock("./invoke", () => ({
  invoke: vi.fn(),
}));

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(_url: string) {
    FakeWebSocket.instances.push(this);
    queueMicrotask(() => this.onopen?.());
  }

  send(raw: string) {
    const request = JSON.parse(raw);
    if (request.method === "auth") {
      queueMicrotask(() =>
        this.onmessage?.({
          data: JSON.stringify({
            jsonrpc: "2.0",
            id: request.id,
            result: { status: "authenticated" },
          }),
        }),
      );
    }
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

describe("daemon websocket interruption", () => {
  beforeEach(() => {
    vi.resetModules();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: async () => "local-auth-token",
      }),
    );
  });

  it("rejects in-flight calls immediately when the socket closes", async () => {
    const daemon = await import("./daemon");
    await expect(daemon.connect()).resolves.toBe(true);

    const inFlight = daemon.call("execute", { input: "long task" });
    FakeWebSocket.instances[0].close();

    await expect(inFlight).rejects.toThrow("Daemon connection was interrupted");
    daemon.disconnect();
  });

  it("publishes authenticated and disconnected connection states", async () => {
    const daemon = await import("./daemon");
    const states: boolean[] = [];
    daemon.onConnectionState((connected) => states.push(connected));

    await daemon.connect();
    FakeWebSocket.instances[0].close();

    expect(states).toEqual([true, false]);
    daemon.disconnect();
  });

  it("does not let an obsolete socket clear a newer connection", async () => {
    const daemon = await import("./daemon");
    await expect(daemon.connect()).resolves.toBe(true);
    const original = FakeWebSocket.instances[0];

    original.readyState = 3;
    original.onclose?.();
    await expect(daemon.connect()).resolves.toBe(true);
    const replacement = FakeWebSocket.instances[1];

    original.onerror?.();

    expect(daemon.isConnected()).toBe(true);
    expect(replacement.readyState).toBe(FakeWebSocket.OPEN);
    daemon.disconnect();
  });

  it("does not reconnect after an intentional disconnect", async () => {
    vi.useFakeTimers();
    const daemon = await import("./daemon");
    await expect(daemon.connect()).resolves.toBe(true);

    daemon.disconnect();
    await vi.advanceTimersByTimeAsync(4_000);

    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.useRealTimers();
  });
});

describe("daemon application results", () => {
  it("accepts an operation-specific success status", async () => {
    const { requireResultStatus } = await import("./daemon");
    const result = { status: "candidate", value: 3 };

    expect(requireResultStatus(result, "candidate", "failed")).toBe(result);
  });

  it("accepts one of several operation-specific success statuses", async () => {
    const { requireResultStatus } = await import("./daemon");
    const result = { status: "evaluated" };

    expect(requireResultStatus(result, ["collecting", "evaluated"], "failed")).toBe(result);
  });

  it("rejects an application result outside the expected contract", async () => {
    const { requireResultStatus } = await import("./daemon");

    expect(() =>
      requireResultStatus({ status: "error", message: "Harness is unavailable" }, "evaluated", "failed"),
    ).toThrow("Harness is unavailable");
  });

  it("accepts successful status results", async () => {
    const { requireOkResult } = await import("./daemon");
    const result = { status: "ok", value: 3 };

    expect(requireOkResult(result, "failed")).toBe(result);
  });

  it("surfaces a daemon rejection instead of reporting success", async () => {
    const { requireOkResult } = await import("./daemon");

    expect(() => requireOkResult({ status: "error", message: "Hook could not start" }, "failed")).toThrow(
      "Hook could not start",
    );
  });

  it("uses a bounded fallback when a rejection has no message", async () => {
    const { requireOkResult } = await import("./daemon");

    expect(() => requireOkResult({ status: "error" }, "Configuration was not saved")).toThrow(
      "Configuration was not saved",
    );
  });
});

describe("a request typed while Heliox is still starting", () => {
  class StartingDaemon {
    static OPEN = 1;
    static attempts = 0;
    static listensFromAttempt = 3; // the first two connects find nothing listening yet
    static sent: { method: string; id: number }[] = [];

    readyState = 0;
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(_url: string) {
      StartingDaemon.attempts += 1;
      const listening = StartingDaemon.attempts >= StartingDaemon.listensFromAttempt;
      queueMicrotask(() => {
        if (!listening) {
          this.readyState = 3;
          this.onerror?.();
          this.onclose?.();
          return;
        }
        this.readyState = StartingDaemon.OPEN;
        this.onopen?.();
      });
    }

    send(raw: string) {
      const request = JSON.parse(raw);
      StartingDaemon.sent.push({ method: request.method, id: request.id });
      const result = request.method === "auth" ? { status: "authenticated" } : { status: "success", message: "Done." };
      queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", id: request.id, result }) }));
    }

    close() {
      this.readyState = 3;
      this.onclose?.();
    }
  }

  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    StartingDaemon.attempts = 0;
    StartingDaemon.sent = [];
    vi.stubGlobal("WebSocket", StartingDaemon);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, text: async () => "local-auth-token" }));
  });

  it("is kept, sent once the daemon listens, and answered once", async () => {
    const daemon = await import("./daemon");
    const answer = daemon.call("execute", { input: "open my mail" });
    let settled = false;
    void answer.then(() => (settled = true));
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false); // nothing listens yet: the request waits instead of failing
    await vi.advanceTimersByTimeAsync(3_000); // the reconnect loop tries again: still nothing
    await vi.advanceTimersByTimeAsync(3_000); // now the daemon listens
    await expect(answer).resolves.toEqual({ status: "success", message: "Done." });
    expect(StartingDaemon.sent.filter((request) => request.method === "execute")).toHaveLength(1);
    daemon.disconnect();
    vi.useRealTimers();
  });

  it("keeps trying after a refused connection, as at every start before the daemon listens", async () => {
    const daemon = await import("./daemon");
    const states: boolean[] = [];
    daemon.onConnectionState((connected) => states.push(connected));
    await expect(daemon.connect()).resolves.toBe(false); // refused: error, then close
    await vi.advanceTimersByTimeAsync(6_000);
    expect(daemon.isConnected()).toBe(true); // the window connects by itself, no request needed
    expect(states.at(-1)).toBe(true);
    daemon.disconnect();
    vi.useRealTimers();
  });

  it("says so when Heliox never starts, instead of waiting for ever", async () => {
    StartingDaemon.listensFromAttempt = Number.POSITIVE_INFINITY;
    const daemon = await import("./daemon");
    const answer = daemon.call("execute", { input: "open my mail" });
    const failed = expect(answer).rejects.toThrow(daemon.START_WAIT_FAILED);
    await vi.advanceTimersByTimeAsync(daemon.START_WAIT_MS);
    await failed;
    expect(StartingDaemon.sent.some((request) => request.method === "execute")).toBe(false);
    daemon.disconnect();
    vi.useRealTimers();
    StartingDaemon.listensFromAttempt = 3;
  });
});

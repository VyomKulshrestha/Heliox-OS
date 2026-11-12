import { get } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
vi.mock("../api/invoke", () => ({ invoke: vi.fn() }));

const FULL = "The disk is full, so Heliox could not save that. Free some space; Heliox keeps working meanwhile.";
const READ_ONLY =
  "Heliox's data folder cannot be written to, so it could not save that. Check the folder's permissions or that " +
  "the drive is not read-only; Heliox keeps working meanwhile.";

/** A daemon socket whose replies the test writes: each request is answered by ``reply(method)``. */
class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  static reply: (method: string, id: number) => Record<string, unknown> = () => ({});
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
    const body =
      request.method === "auth"
        ? { result: { status: "authenticated" } }
        : FakeWebSocket.reply(request.method, request.id);
    queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", id: request.id, ...body }) }));
  }

  notify(method: string, params: unknown) {
    this.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", method, params }) });
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

async function connected() {
  const daemon = await import("../api/daemon");
  const notice = await import("./storageNotice");
  await daemon.connect();
  return { daemon, notice, socket: FakeWebSocket.instances[0] };
}

describe("a full disk or read-only data folder reaches the person in one place", () => {
  beforeEach(() => {
    vi.resetModules();
    FakeWebSocket.instances = [];
    FakeWebSocket.reply = () => ({ result: { status: "ok" } });
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, text: async () => "local-auth-token" }));
  });

  it("an error with the storage code: the caller still gets its rejection, the notice gets the sentence", async () => {
    const { daemon, notice } = await connected();
    FakeWebSocket.reply = () => ({
      error: { code: -32010, message: READ_ONLY, data: { storage_problem: true, kind: "read_only" } },
    });

    await expect(daemon.call("note_add", { text: "x" })).rejects.toThrow(READ_ONLY);
    expect(get(notice.storageNotice)).toEqual({ message: READ_ONLY, kind: "read_only" });
    daemon.disconnect();
  });

  it("an execute result with the flag: the result is returned as before, the notice gets the sentence", async () => {
    const { daemon, notice } = await connected();
    const answer = { status: "error", message: FULL, explanation: FULL, storage_problem: true };
    FakeWebSocket.reply = () => ({ result: answer });

    await expect(daemon.call("execute", { input: "note that x" })).resolves.toEqual(answer);
    expect(get(notice.storageNotice)?.message).toBe(FULL);
    daemon.disconnect();
  });

  it("another error is not a storage problem", async () => {
    const { daemon, notice } = await connected();
    FakeWebSocket.reply = () => ({ error: { code: -32603, message: "Internal error: boom" } });

    await expect(daemon.call("todo_add", { text: "x" })).rejects.toThrow("Internal error: boom");
    expect(get(notice.storageNotice)).toBeNull();
    daemon.disconnect();
  });

  it("a burst of failures is shown once", async () => {
    const { daemon, notice } = await connected();
    FakeWebSocket.reply = () => ({
      error: { code: -32010, message: FULL, data: { storage_problem: true, kind: "full" } },
    });
    const seen: unknown[] = [];
    const stop = notice.storageNotice.subscribe((value) => seen.push(value));

    await Promise.allSettled(["note_add", "todo_add", "reminder_add", "note_add"].map((m) => daemon.call(m)));

    expect(seen.filter(Boolean)).toHaveLength(1); // one notice, not one per failed request
    stop();
    daemon.disconnect();
  });

  it("the daemon's notice that it can save again clears it; its first notice shows it", async () => {
    const { daemon, notice, socket } = await connected();
    socket.notify("storage_problem", { kind: "full", component: "memory", message: FULL });
    expect(get(notice.storageNotice)?.message).toBe(FULL);

    socket.notify("storage_problem", { kind: "", component: "memory", message: "Heliox can save again." });
    expect(get(notice.storageNotice)).toBeNull();
    daemon.disconnect();
  });

  it("a later success of the method that failed clears it; other requests succeeding do not", async () => {
    const { daemon, notice } = await connected();
    FakeWebSocket.reply = (method) =>
      method === "note_add"
        ? { error: { code: -32010, message: FULL, data: { storage_problem: true, kind: "full" } } }
        : { result: { pong: true } };
    await expect(daemon.call("note_add", { text: "x" })).rejects.toThrow(FULL);

    await daemon.call("ping");
    expect(get(notice.storageNotice)?.message).toBe(FULL); // a ping saves nothing: the disk may still be full

    FakeWebSocket.reply = () => ({ result: { status: "ok", notes: [] } });
    await daemon.call("note_add", { text: "x" });
    expect(get(notice.storageNotice)).toBeNull();
    daemon.disconnect();
  });
});

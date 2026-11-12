import { get } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  handlers: new Set<(method: string, params: unknown) => void>(),
  call: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: (method: string, params: unknown) => void) => mocks.handlers.add(handler),
  offNotification: (handler: (method: string, params: unknown) => void) => mocks.handlers.delete(handler),
  requireResultStatus: (
    result: { status?: string; message?: string },
    expected: string | string[],
    fallback: string,
  ) => {
    if (![expected].flat().includes(result.status ?? "")) throw new Error(result.message || fallback);
    return result;
  },
}));
vi.mock("../api/invoke", () => ({ invoke: vi.fn().mockResolvedValue({ running: false, pid: null }) }));

const STARTING = { status: "starting", message: "Neural input is still starting. Ask again in a moment." };
const READY = { status: "ok", state: "disconnected", connected: false };

function announce(states: Record<string, string>) {
  for (const handler of [...mocks.handlers]) handler("late_parts", { late_parts: states });
}

describe("neural input that is still starting", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    mocks.handlers.clear();
    mocks.call.mockReset();
  });

  it("is a waiting state with the daemon's sentence, not an error, and is asked again when the part is ready", async () => {
    const { neural } = await import("./neural");
    mocks.call.mockResolvedValueOnce(STARTING).mockResolvedValueOnce(READY);

    await neural.refresh();
    expect(get(neural).waiting).toBe(STARTING.message);
    expect(get(neural).error).toBe("");

    announce({ neural: "ready" });
    await vi.waitFor(() => expect(mocks.call).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(get(neural).waiting).toBe(""));
    expect(get(neural).error).toBe("");
  });

  it("asks again after a short interval when no notice comes", async () => {
    const { LATE_PART_RETRY_MS } = await import("../api/lateParts");
    const { neural } = await import("./neural");
    mocks.call.mockResolvedValueOnce(STARTING).mockResolvedValueOnce(READY);

    await neural.refresh();
    expect(mocks.call).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(LATE_PART_RETRY_MS);
    expect(mocks.call).toHaveBeenCalledTimes(2);
    expect(get(neural).waiting).toBe("");
  });

  it("an action asked while the part starts waits the same way instead of failing", async () => {
    const { neural } = await import("./neural");
    mocks.call.mockResolvedValueOnce(STARTING);

    expect(await neural.stageTask("Open notes", "open my notes")).toBe(false);
    expect(get(neural).waiting).toBe(STARTING.message);
    expect(get(neural).error).toBe("");
  });
});

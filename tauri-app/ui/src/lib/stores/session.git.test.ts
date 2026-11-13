import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return {
    ...actual,
    call: daemonMocks.call,
    connect: vi.fn().mockResolvedValue(true),
    isConnected: vi.fn(() => true),
    onConnectionState: vi.fn(),
    onNotification: vi.fn(),
  };
});

vi.mock("./settings", () => ({
  settings: writable({ model: { cloud_model: "ollama" } }),
}));
vi.mock("./companion", () => ({ companion: { speak: vi.fn() } }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

describe("session git conflict shortcut", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("renders daemon conflict-detection failures as errors", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "The file could not be read" });
    const { session } = await import("./session");

    await session.sendCommand("/git-resolve missing.txt");

    expect(get(session).messages.at(-1)).toMatchObject({
      type: "error",
      text: "The file could not be read",
    });
  });

  it("says nothing was sent, not an error, when the person keeps a file's conflicts on this computer", async () => {
    daemonMocks.call.mockResolvedValue({ status: "cancelled", message: "Nothing was sent." });
    const { session } = await import("./session");

    await session.sendCommand("/git-resolve app.py");

    expect(get(session).messages.at(-1)).toMatchObject({ type: "system", text: "Nothing was sent." });
    expect(get(session).loading).toBe(false);
  });
});

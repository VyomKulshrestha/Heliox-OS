import { writable } from "svelte/store";
import { describe, expect, it, vi } from "vitest";

const events = vi.hoisted(() => ({ handlers: new Map<string, (event: { payload: unknown }) => void>() }));

// Importing the session store must not open a real socket to a daemon that may be running on this machine.
vi.mock("../api/daemon", () => ({
  call: vi.fn(),
  connect: vi.fn().mockResolvedValue(false),
  isConnected: vi.fn(() => false),
  onConnectionState: vi.fn(),
  onNotification: vi.fn(),
}));

vi.mock("./settings", () => ({
  settings: writable({ model: { cloud_model: "ollama" } }),
}));

vi.mock("./companion", () => ({
  companion: { speak: vi.fn() },
}));

vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (name: string, handler: (event: { payload: unknown }) => void) => {
    events.handlers.set(name, handler);
    return () => events.handlers.delete(name);
  }),
}));

import { listenToDaemonSupervisor } from "./session";

describe("daemon supervisor line", () => {
  it("passes each announced line through once and ignores empty ones", async () => {
    const lines: string[] = [];
    await listenToDaemonSupervisor((text) => lines.push(text));
    const handler = events.handlers.get("daemon-supervisor");
    expect(handler).toBeTruthy();

    handler?.({ payload: { message: "Heliox's background service stopped. Starting it again in 2 s." } });
    handler?.({ payload: { message: "  " } });
    handler?.({ payload: null });

    expect(lines).toEqual(["Heliox's background service stopped. Starting it again in 2 s."]);
  });
});

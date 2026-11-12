import { describe, expect, it, vi } from "vitest";

const events = vi.hoisted(() => ({ handlers: new Map<string, (event: { payload: unknown }) => void>() }));

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

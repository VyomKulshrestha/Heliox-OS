import { render, screen } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  handlers: [] as Array<(method: string, params: unknown) => void>,
  session: null as unknown as ReturnType<typeof writable<{ daemonConnected: boolean }>>,
}));

vi.mock("../api/daemon", () => ({
  onNotification: (handler: (method: string, params: unknown) => void) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));
vi.mock("svelte-i18n", async () => {
  const { readable } = await import("svelte/store");
  return { _: readable((key: string, options?: { default?: string }) => options?.default ?? key) };
});
vi.mock("../stores/session", async () => {
  const { writable: makeStore } = await import("svelte/store");
  mocks.session = makeStore({ daemonConnected: true });
  return { session: mocks.session };
});

import ConnectionBadge from "./ConnectionBadge.svelte";

function announce(state: string, stage: string) {
  for (const handler of mocks.handlers) handler("startup_status", { state, stage });
}

describe("ConnectionBadge while the daemon starts", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
    mocks.session?.set({ daemonConnected: true });
  });

  it("shows the stage instead of Online, then Online when ready", async () => {
    render(ConnectionBadge);
    announce("starting", "opening Heliox's records");
    await tick();
    expect(screen.getByTestId("connection-badge").textContent).toContain("Starting: opening Heliox's records");

    announce("ready", "");
    await tick();
    expect(screen.getByTestId("connection-badge").textContent).toContain("Online");
  });

  it("says it is waiting for the data folder, and forgets the stage when the connection drops", async () => {
    render(ConnectionBadge);
    announce("waiting_for_storage", "waiting until the data folder can be written");
    await tick();
    expect(screen.getByTestId("connection-badge").textContent).toContain(
      "Starting: waiting until the data folder can be written",
    );

    mocks.session.set({ daemonConnected: false });
    await tick();
    mocks.session.set({ daemonConnected: true });
    await tick();
    expect(screen.getByTestId("connection-badge").textContent).toContain("Online");
  });
});

import { render, waitFor } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ config: {} as Record<string, unknown> }));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string) => (method === "get_config" ? mocks.config : { status: "ok" })),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
  onConnectionState: vi.fn(),
  offConnectionState: vi.fn(),
}));

vi.mock("../stores/session", async () => {
  const { writable } = await import("svelte/store");
  return { session: writable({ interaction: null, loading: false }) };
});

import { settings } from "../stores/settings";
import FaceHost from "./FaceHost.svelte";

describe("FaceHost", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("with the face off, renders nothing and fetches nothing", async () => {
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    mocks.config = { face: { pack: "off" } };
    await settings.reloadFromDaemon();
    const shown = vi.fn();
    const { container } = render(FaceHost, { onshown: shown });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(container.querySelector("[data-testid=face]")).toBeNull();
    expect(fetcher).not.toHaveBeenCalled();
    expect(shown).toHaveBeenLastCalledWith(false);
  });

  it("ignores a pack name that is not safe to load", async () => {
    mocks.config = { face: { pack: "../../x" } };
    await settings.reloadFromDaemon();
    const { container } = render(FaceHost);
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(container.querySelector("[data-testid=face]")).toBeNull();
  });

  it("with a face chosen, shows it, and says so when the pack cannot be read", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, json: async () => null })),
    );
    mocks.config = { face: { pack: "placeholder" } };
    await settings.reloadFromDaemon();
    const shown = vi.fn();
    const { container } = render(FaceHost, { onshown: shown });
    // The face's code is a chunk of its own, loaded on first use: allow for a slow first import.
    await waitFor(() => expect(container.querySelector("[data-testid=face]")).not.toBeNull(), { timeout: 10_000 });
    expect(shown).toHaveBeenLastCalledWith(true);
    await waitFor(() => expect(container.textContent).toContain("This face is not installed."));
  });
});

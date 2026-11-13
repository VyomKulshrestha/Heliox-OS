import { render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

import PluginsTab from "./PluginsTab.svelte";

const capabilities = {
  filesystem: { read: [], write: [] },
  network_domains: [],
  processes: [],
  credentials: [],
  clipboard: { read: false, write: false },
  media: { camera: false, microphone: false },
  data_retention: { mode: "none", max_days: 0 },
  destructive_actions: false,
};

function plugin(name: string, extra: Record<string, unknown>) {
  return {
    name,
    description: `${name} plugin`,
    version: "1.0.0",
    author: "someone",
    url: "",
    tools: [],
    installed: true,
    capabilities,
    ...extra,
  };
}

function listing(plugins: unknown[]) {
  daemonMocks.call.mockImplementation(async (method: string) =>
    method === "plugin_market_list" ? { plugins, source: "bundled" } : {},
  );
}

describe("PluginsTab: who vouched for an installed plugin", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("says in words that a plugin signed only by its own key is unverified, on its row", async () => {
    listing([plugin("mine", { verified: false, local_only: true, source: "local" })]);
    render(PluginsTab);

    await waitFor(() => expect(screen.getByText("Unverified: signed only by its own key")).toBeTruthy());
    expect(screen.queryByText("Verified")).toBeNull();
  });

  it("marks a plugin a trusted key vouched for as verified", async () => {
    listing([plugin("weather", { verified: true })]);
    render(PluginsTab);

    await waitFor(() => expect(screen.getByText("Verified")).toBeTruthy());
    expect(screen.queryByText(/Unverified/)).toBeNull();
  });

  it("shows neither when an older daemon sends no verified field", async () => {
    listing([plugin("legacy", {})]);
    render(PluginsTab);

    await waitFor(() => expect(screen.getByText("legacy")).toBeTruthy());
    expect(screen.queryByText(/Unverified/)).toBeNull();
    expect(screen.queryByText("Verified")).toBeNull();
  });
});

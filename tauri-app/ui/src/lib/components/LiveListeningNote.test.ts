import { render, screen } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  call: vi.fn(async () => ({ status: "ok", kinds: [] })),
  store: null as null | ReturnType<typeof writable>,
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-clipboard-manager", () => ({ writeText: vi.fn() }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn(async () => true),
  requestPermission: vi.fn(async () => "granted"),
  sendNotification: vi.fn(),
}));
vi.mock("../stores/session", () => ({ session: { sendCommand: vi.fn() } }));
vi.mock("../stores/settings", async () => {
  const { writable: make } = await import("svelte/store");
  mocks.store = make({
    model: { provider: "ollama", cloud_provider: "" },
    live: { enabled: true, level: "balanced", listen_to_speech: true, muted_kinds: [] },
  });
  return { settings: { subscribe: mocks.store.subscribe, updateSectionDetailed: vi.fn(async () => ({ ok: true })) } };
});

import LiveCompanionPanel from "./LiveCompanionPanel.svelte";

describe("what Live listens to, said plainly", () => {
  it("tells a person with a local model that nothing they say leaves the computer", () => {
    mocks.store!.set({
      model: { provider: "ollama", cloud_provider: "" },
      live: { enabled: true, level: "balanced", listen_to_speech: true, muted_kinds: [] },
    });
    render(LiveCompanionPanel);
    const note = screen.getByTestId("live-listening-note").textContent ?? "";
    expect(note).toContain("never stored as conversation or sent to any model");
    expect(note).toContain("Your language model runs on this computer, so nothing you say leaves it.");
  });

  it("tells a person with a cloud model what goes to it and what never does", () => {
    mocks.store!.set({
      model: { provider: "cloud", cloud_provider: "claude" },
      live: { enabled: true, level: "balanced", listen_to_speech: true, muted_kinds: [] },
    });
    render(LiveCompanionPanel);
    const note = screen.getByTestId("live-listening-note").textContent ?? "";
    expect(note).toContain("goes to the cloud model you chose (claude) with your request");
    expect(note).toContain("speech that was not for Heliox never does");
  });

  it("says nothing about listening when listening is off", () => {
    mocks.store!.set({
      model: { provider: "cloud", cloud_provider: "claude" },
      live: { enabled: true, level: "balanced", listen_to_speech: false, muted_kinds: [] },
    });
    render(LiveCompanionPanel);
    expect(screen.queryByTestId("live-listening-note")).toBeNull();
  });
});

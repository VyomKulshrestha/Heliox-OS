import { render, screen } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({ store: null as unknown }));

vi.mock("../stores/live", async () => {
  const { writable: make } = await import("svelte/store");
  const store = make({ enabled: true, paused: false, watching: "Slack", userState: "paused", blindNote: "" });
  state.store = store;
  return { live: { subscribe: store.subscribe, setPaused: vi.fn() } };
});

import LivePresence from "./LivePresence.svelte";

type Live = { enabled: boolean; paused: boolean; watching: string; userState: string; blindNote: string };
const live = () => state.store as ReturnType<typeof writable<Live>>;

describe("LivePresence", () => {
  it("says where it is with you, as before", () => {
    render(LivePresence);
    expect(screen.getByTestId("live-presence").getAttribute("title")).toContain("with you in Slack");
  });

  it("says once, in its tooltip, when it cannot see into the app in front", () => {
    live().set({
      enabled: true,
      paused: false,
      watching: "",
      userState: "away",
      blindNote: "Live doesn't work in Slack yet.",
    });
    render(LivePresence);
    expect(screen.getByTestId("live-presence").getAttribute("title")).toBe(
      "Live doesn't work in Slack yet. Click to pause.",
    );
  });
});

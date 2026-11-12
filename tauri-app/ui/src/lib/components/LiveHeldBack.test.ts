import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({ call: vi.fn(), handlers: [] as Handler[] }));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-clipboard-manager", () => ({ writeText: vi.fn() }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn(async () => true),
  requestPermission: vi.fn(async () => "granted"),
  sendNotification: vi.fn(),
}));
vi.mock("../stores/session", () => ({ session: { sendCommand: vi.fn() } }));
vi.mock("../stores/settings", () => {
  const store = writable({
    model: { provider: "ollama", cloud_provider: "" },
    live: { enabled: true, level: "balanced", listen_to_speech: true, muted_kinds: [] },
  });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: vi.fn(async () => ({ ok: true })) } };
});

import { live } from "../stores/live";
import LiveCards from "./LiveCards.svelte";
import LiveCompanionPanel from "./LiveCompanionPanel.svelte";

const notify = (method: string, params: unknown) => mocks.handlers.forEach((handler) => handler(method, params));
const SENTENCE = "I'll hold back catches about weekdays and dates for the next half hour.";
const card = (over: Record<string, unknown> = {}) => ({
  id: "c1",
  kind: "weekday_date",
  title: "14 November is a Saturday",
  message: "You wrote Tuesday.",
  detail: "",
  evidence: "Tuesday, 14 November",
  fix: "Saturday, 14 November",
  fix_kind: "replace",
  urgency: "normal",
  spoken: false,
  source: "focused_field",
  app: "Notepad",
  ...over,
});

describe("Not now on a card", () => {
  beforeEach(() => {
    mocks.call.mockReset();
    live.reset();
  });

  it("says in the card's place what it held back, and Undo lifts it", async () => {
    mocks.call.mockImplementation(async (method: string) =>
      method === "live_feedback"
        ? { status: "ok", kind: "weekday_date", held: { kind: "weekday_date", sentence: SENTENCE } }
        : { status: "ok" },
    );
    render(LiveCards);
    notify("live_interjection", card());
    await tick();
    await fireEvent.click(screen.getByRole("button", { name: "Not now" }));
    await waitFor(() => expect(screen.getByTestId("live-card-held").textContent).toContain(SENTENCE));
    await fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(mocks.call).toHaveBeenCalledWith("live_set", { unsnooze: "weekday_date" });
    await tick();
    expect(screen.queryByTestId("live-card-held")).toBeNull();
  });

  it("an urgent card's Not now holds nothing back and says nothing", async () => {
    mocks.call.mockResolvedValue({ status: "ok", kind: "risky_command" });
    render(LiveCards);
    notify("live_interjection", card({ id: "u1", kind: "risky_command", urgency: "urgent" }));
    await tick();
    await fireEvent.click(screen.getByRole("button", { name: "Not now" }));
    await tick();
    expect(screen.queryByTestId("live-card-held")).toBeNull();
    expect(screen.queryByTestId("live-card")).toBeNull();
  });
});

describe("Settings, Live: what is held back", () => {
  beforeEach(() => mocks.call.mockReset());

  it("lists a held-back kind with the time left and lifts it with Undo", async () => {
    const held = [{ kind: "weekday_date", label: "Weekdays and dates", sentence: SENTENCE, seconds_left: 1500 }];
    mocks.call.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === "live_set" && params?.unsnooze) return { status: "ok", held_back: [] };
      return { status: "ok", kinds: [], held_back: held };
    });
    render(LiveCompanionPanel);
    await waitFor(() => expect(screen.getByTestId("live-held-back").textContent).toContain("25 minutes left"));
    await fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(screen.queryByTestId("live-held-back")).toBeNull());
    expect(mocks.call).toHaveBeenCalledWith("live_set", { unsnooze: "weekday_date" });
  });
});

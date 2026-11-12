/**
 * The brief's rule for a new component: it renders without throwing when the daemon answers with a bare
 * {status: "ok"}, with a refusal, and not at all. For the passing-notes switch and the "held back" parts of Live.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Handler[],
  update: vi.fn(),
  store: null as null | { set: (value: unknown) => void },
}));

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
vi.mock("../stores/settings", async () => {
  const { writable } = await import("svelte/store");
  const store = writable<unknown>({ live: { enabled: true, listen_to_speech: true, muted_kinds: [] } });
  mocks.store = store as unknown as { set: (value: unknown) => void };
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: mocks.update } };
});

import { live } from "../stores/live";
import LiveCards from "./LiveCards.svelte";
import LiveCompanionPanel from "./LiveCompanionPanel.svelte";
import PassingNotesPanel from "./PassingNotesPanel.svelte";

const notify = (method: string, params: unknown) => mocks.handlers.forEach((handler) => handler(method, params));
const card = {
  id: "c1",
  kind: "weekday_date",
  title: "14 November is a Saturday",
  message: "You wrote Tuesday.",
  detail: "",
  evidence: "Tuesday, 14 November",
  fix: "",
  fix_kind: "",
  urgency: "normal",
  spoken: false,
  source: "focused_field",
  app: "Notepad",
};

const ANSWERS: [string, () => Promise<unknown>][] = [
  ["a bare ok", async () => ({ status: "ok" })],
  ["a refusal", async () => ({ status: "error", message: "Live is not ready yet." })],
  [
    "no answer at all",
    async () => {
      throw new Error("Heliox is not connected");
    },
  ],
];

describe.each(ANSWERS)("with %s from the daemon", (_label, answer) => {
  beforeEach(() => {
    mocks.call.mockReset();
    mocks.update.mockReset();
    live.reset();
    mocks.store!.set({ live: { enabled: true, listen_to_speech: true, muted_kinds: [] } });
  });

  it("the card's Not now still takes the card away and shows no hold", async () => {
    mocks.call.mockImplementation(answer);
    render(LiveCards);
    notify("live_interjection", card);
    await tick();
    await fireEvent.click(screen.getByRole("button", { name: "Not now" }));
    await waitFor(() => expect(screen.queryByTestId("live-card")).toBeNull());
    expect(screen.queryByTestId("live-card-held")).toBeNull();
  });

  it("Settings, Live renders without a held-back list", async () => {
    mocks.call.mockImplementation(answer);
    render(LiveCompanionPanel);
    await tick();
    expect(screen.getByTestId("live-panel")).toBeTruthy();
    expect(screen.queryByTestId("live-held-back")).toBeNull();
  });

  it("the passing-notes switch renders, and a failed save is said plainly", async () => {
    mocks.store!.set(answer === ANSWERS[0][1] ? { status: "ok" } : null);
    mocks.update.mockImplementation(async () => {
      const value = await answer().catch((err: Error) => {
        throw err;
      });
      const reply = value as { status?: string; message?: string };
      return { ok: reply.status === "ok", message: reply.message ?? "" };
    });
    render(PassingNotesPanel);
    const box = screen.getByRole("checkbox", { name: "Keep notes of things I mention in passing" });
    expect((box as HTMLInputElement).checked).toBe(true); // the default when settings say nothing
    await fireEvent.click(box);
    if (answer === ANSWERS[0][1]) {
      await tick();
      expect(screen.queryByRole("alert")).toBeNull();
    } else {
      await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/not ready|not connected/));
    }
  });
});

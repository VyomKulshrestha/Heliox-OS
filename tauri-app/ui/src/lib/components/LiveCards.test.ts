import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Handler[],
  writeText: vi.fn(async () => {}),
  sendCommand: vi.fn(async () => {}),
  updateSectionDetailed: vi.fn(async () => ({ ok: true, message: "" })),
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-clipboard-manager", () => ({ writeText: mocks.writeText }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn(async () => true),
  requestPermission: vi.fn(async () => "granted"),
  sendNotification: vi.fn(),
}));
vi.mock("../stores/session", () => ({ session: { sendCommand: mocks.sendCommand } }));
vi.mock("../stores/settings", () => {
  const store = writable({
    live: {
      enabled: true,
      level: "balanced",
      read_focused_text: true,
      ocr_fallback: false,
      listen_to_speech: true,
      muted_kinds: ["repeated_word"],
    },
  });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: mocks.updateSectionDetailed } };
});

import { live } from "../stores/live";
import LiveCards from "./LiveCards.svelte";
import LivePresence from "./LivePresence.svelte";
import LiveCompanionPanel from "./LiveCompanionPanel.svelte";

const card = (over: Record<string, unknown> = {}) => ({
  id: "c1",
  kind: "weekday_date",
  title: "14 November is a Saturday",
  message: "14 November is a Saturday, not a Tuesday.",
  detail: "If you meant the Tuesday, that's 17 November.",
  evidence: "Tuesday, 14 November",
  fix: "Saturday, 14 November",
  fix_kind: "replace",
  urgency: "normal",
  spoken: true,
  source: "focused_field",
  app: "notepad.exe",
  ...over,
});

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("Live companion UI", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    live.reset();
    mocks.call.mockImplementation(async (method: string) => {
      if (method === "live_explain") return { status: "ok", text: "Explained: 17 November is the Tuesday." };
      if (method === "live_status")
        return {
          status: "ok",
          enabled: true,
          kinds: [
            { kind: "weekday_date", label: "A weekday that doesn't match its date" },
            { kind: "repeated_word", label: "Doubled words" },
          ],
        };
      return { status: "ok" };
    });
  });

  it("shows a correction with its evidence, and copies the fix", async () => {
    render(LiveCards);
    emit("live_interjection", card());
    await tick();
    expect(screen.getByText("14 November is a Saturday, not a Tuesday.")).toBeTruthy();
    expect(screen.getByText("Tuesday, 14 November")).toBeTruthy();

    await fireEvent.click(screen.getByRole("button", { name: "Why?" }));
    await waitFor(() => expect(screen.getByText("Explained: 17 November is the Tuesday.")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Copy fix" }));
    expect(mocks.writeText).toHaveBeenCalledWith("Saturday, 14 November");
    await waitFor(() => expect(screen.getByText("Copied")).toBeTruthy());
  });

  it("acknowledges a fix the person made, then lets the card go", async () => {
    vi.useFakeTimers();
    render(LiveCards);
    emit("live_interjection", card());
    await tick();
    emit("live_interjection_withdrawn", { id: "c1", reason: "fixed" });
    await tick();
    expect(screen.getByText("Fixed")).toBeTruthy();
    vi.advanceTimersByTime(1500);
    await tick();
    expect(screen.queryByTestId("live-card")).toBeNull();
    vi.useRealTimers();
  });

  it("hands an ask fix to Heliox, and urgent cards are alerts", async () => {
    render(LiveCards);
    emit(
      "live_interjection",
      card({ id: "c2", kind: "error_on_screen", urgency: "urgent", fix_kind: "ask", fix: "Show me port 3000" }),
    );
    await tick();
    expect(screen.getByRole("alert")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Ask Heliox" }));
    await waitFor(() => expect(mocks.sendCommand).toHaveBeenCalledWith("Show me port 3000"));
    expect(mocks.call).toHaveBeenCalledWith("live_feedback", { id: "c2", verdict: "used_fix" });
  });

  it("sends not now and don't mention these back as feedback", async () => {
    render(LiveCards);
    emit("live_interjection", card());
    emit("live_interjection", card({ id: "c3", title: "A doubled word", message: '"the the" has the word twice.' }));
    await tick();
    await fireEvent.click(screen.getAllByRole("button", { name: "Not now" })[0]);
    expect(mocks.call).toHaveBeenCalledWith("live_feedback", { id: "c1", verdict: "dismissed" });
    await fireEvent.click(screen.getByRole("button", { name: "Don't mention these" }));
    expect(mocks.call).toHaveBeenCalledWith("live_feedback", { id: "c3", verdict: "mute" });
    await tick();
    expect(screen.queryByTestId("live-card")).toBeNull();
  });

  it("shows presence only while on, and pauses with a click", async () => {
    render(LivePresence);
    expect(screen.queryByTestId("live-presence")).toBeNull();
    emit("live_state", { enabled: true, paused: false, watching: "notepad.exe", user_state: "typing" });
    await tick();
    const pill = screen.getByTestId("live-presence");
    expect(pill.getAttribute("aria-label")).toContain("notepad.exe");
    await fireEvent.click(pill);
    expect(mocks.call).toHaveBeenCalledWith("live_set", { paused: true });
    await tick();
    expect(screen.getByText("Paused")).toBeTruthy();
  });

  it("settings save the level, the sources and which kinds it points out", async () => {
    render(LiveCompanionPanel);
    await waitFor(() => expect(screen.getByText("Doubled words")).toBeTruthy());
    await fireEvent.click(screen.getByRole("radio", { name: /Proactive/ }));
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith("live", { level: "proactive" });
    await fireEvent.click(screen.getByRole("checkbox", { name: /Doubled words/ }));
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith("live", { muted_kinds: [] });
    await fireEvent.click(screen.getByRole("button", { name: "Live companion" }));
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith("live", { enabled: false });
  });
});

import { render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  hears: { state: "on", reason: "" } as { state: string; reason: string },
  call: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));
vi.mock("../stores/session", () => ({ session: writable({ loading: false, daemonConnected: false }) }));
vi.mock("../stores/companion", () => ({
  companion: { subscribe: writable({}).subscribe, humanSpeechStarted: vi.fn(), humanSpeechEnded: vi.fn() },
}));
vi.mock("../stores/multimodal", () => ({ multimodal: writable({}) }));
vi.mock("./AudioVisualizer.svelte", async () => ({ default: (await import("./WidgetCard.svelte")).default }));

import VoiceControl from "./VoiceControl.svelte";

const READY = "Getting ready to listen…";
const FAULT = "The microphone stopped sending sound; reopening it.";
const SAY_AGAIN = "I wasn't ready to listen yet. Please say that again.";

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("the voice bar while Heliox gets ready to hear", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
    mocks.hears = { state: "on", reason: "" };
    mocks.call.mockImplementation(async (method: string) => {
      if (method === "health") return { ready: { hears: mocks.hears } };
      return method === "voice_listener_start" ? { status: "started" } : { status: "ok", running: false };
    });
  });

  it("a microphone fault during loading ends in the fault message, not in getting ready for ever", async () => {
    render(VoiceControl);
    emit("voice_status", { status: "hearing_loading", message: READY });
    await tick();
    expect(screen.getByText(READY)).toBeTruthy();

    emit("voice_status", { status: "microphone_problem", reason: "no_audio", message: FAULT });
    await tick();
    expect(screen.getByText(FAULT)).toBeTruthy();
    expect(screen.queryByText(READY)).toBeNull();

    // While the fault lasts, a sentence held too long does not turn the bar into "say it again".
    emit("voice_status", { status: "hearing_not_ready", message: SAY_AGAIN });
    emit("voice_status", { status: "hearing_loading", message: READY });
    await tick();
    expect(screen.getByText(FAULT)).toBeTruthy();
    expect(screen.queryByText(SAY_AGAIN)).toBeNull();
    expect(screen.queryByText(READY)).toBeNull();

    emit("voice_status", { status: "microphone_recovered", message: "Listening on Desk Mic." });
    emit("voice_status", { status: "hearing_not_ready", message: SAY_AGAIN });
    await tick();
    expect(screen.getByText(SAY_AGAIN)).toBeTruthy();
  });

  it("reads the daemon's readiness on opening: still loading shows getting ready, a fault shows the fault", async () => {
    mocks.hears = { state: "loading", reason: READY };
    const first = render(VoiceControl);
    await waitFor(() => expect(screen.getByText(READY)).toBeTruthy());
    first.unmount();

    mocks.handlers.length = 0;
    mocks.hears = { state: "fault", reason: FAULT };
    render(VoiceControl);
    await waitFor(() => expect(screen.getByText(FAULT)).toBeTruthy());
    expect(screen.queryByText(READY)).toBeNull();
  });
});

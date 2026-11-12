import { cleanup, render, screen, waitFor } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// While Heliox is starting the microphone is not open yet: the voice bar says so and the button shows it is not yet
// available. Once ready the line goes and nothing else changes. The startup_status request has three replies: ok with
// a state, a refusal (any other answer), or no reply.

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  startup: (async () => ({ status: "ok", state: "starting" })) as () => Promise<Record<string, unknown>>,
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string) => {
    if (method === "startup_status") return mocks.startup();
    if (method === "voice_listener_start") return new Promise(() => {}); // held by the daemon until ready
    return { status: "ok", running: false };
  }),
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: (handler: Handler) => {
    mocks.handlers = mocks.handlers.filter((known) => known !== handler);
  },
}));
const sessionState = vi.hoisted(() => ({ store: null as ReturnType<typeof writable> | null }));
vi.mock("../stores/session", async () => {
  const { writable: make } = await import("svelte/store");
  sessionState.store = make({ loading: false, daemonConnected: false });
  return { session: sessionState.store };
});
vi.mock("../stores/companion", () => ({
  companion: { subscribe: writable({}).subscribe, humanSpeechStarted: vi.fn(), humanSpeechEnded: vi.fn() },
}));
vi.mock("../stores/multimodal", () => ({ multimodal: writable({}) }));
vi.mock("./AudioVisualizer.svelte", async () => ({ default: (await import("./WidgetCard.svelte")).default }));

import VoiceControl from "./VoiceControl.svelte";

const LINE = "Listening starts when Heliox is ready.";
const mic = () => screen.getAllByRole("button")[0] as HTMLButtonElement;
const notify = (state: string) => mocks.handlers.forEach((handler) => handler("startup_status", { state }));

describe("the voice bar while Heliox starts", () => {
  afterEach(() => cleanup()); // each test starts the start-up watch afresh
  beforeEach(() => {
    mocks.handlers = [];
    mocks.startup = async () => ({ status: "ok", state: "starting" });
    sessionState.store?.set({ loading: false, daemonConnected: false });
  });

  it("says listening starts when ready, with the microphone not yet available, until ready", async () => {
    render(VoiceControl);
    expect(screen.getByRole("status").textContent).toBe(LINE);
    expect(mic().disabled).toBe(true);
    expect(mic().getAttribute("aria-label")).toBe(LINE);

    sessionState.store?.set({ loading: false, daemonConnected: true }); // connected, still starting
    await waitFor(() => expect(screen.queryByText(LINE)).not.toBeNull());
    notify("ready");
    await waitFor(() => expect(screen.queryByText(LINE)).toBeNull());
    expect(mic().getAttribute("aria-label")).not.toBe(LINE); // the line goes; the button is as it always is
  });

  it("asks once connected: ok and ready ends the wait", async () => {
    mocks.startup = async () => ({ status: "ok", state: "ready" });
    render(VoiceControl);
    sessionState.store?.set({ loading: false, daemonConnected: true });
    await waitFor(() => expect(screen.queryByText(LINE)).toBeNull());
  });

  it("a refusal means Heliox answers, so the microphone is not held back", async () => {
    mocks.startup = async () => {
      throw new Error("Method not found: startup_status");
    };
    render(VoiceControl);
    sessionState.store?.set({ loading: false, daemonConnected: true });
    await waitFor(() => expect(screen.queryByText(LINE)).toBeNull());
  });

  it("no reply leaves it waiting for Heliox", async () => {
    mocks.startup = async () => {
      throw new Error("Daemon connection was interrupted");
    };
    render(VoiceControl);
    sessionState.store?.set({ loading: false, daemonConnected: true });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText(LINE)).not.toBeNull();
    expect(mic().disabled).toBe(true);
  });
});

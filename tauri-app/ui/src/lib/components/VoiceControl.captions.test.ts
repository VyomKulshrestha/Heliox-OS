import { render, screen } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({ handlers: [] as Handler[] }));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async () => ({ status: "ok", running: false })),
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

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("VoiceControl live captions", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
  });

  it("shows words as they are heard, says it is still listening on a pause, and clears on the command", async () => {
    render(VoiceControl);
    emit("voice_partial", { text: "Open chrome and" });
    await tick();
    expect(screen.getByText("Open chrome and")).toBeTruthy();

    emit("voice_status", { status: "holding", message: "Still listening" });
    await tick();
    expect(screen.getByText("Still listening…")).toBeTruthy();

    emit("voice_partial", { text: "Open chrome and search for flights" });
    await tick();
    expect(screen.queryByText("Still listening…")).toBeNull();

    emit("voice_command", { command: "Open chrome and search for flights", status: "executing" });
    await tick();
    expect(screen.queryByText("Open chrome and search for flights")).toBeNull();
  });

  it("says the voice is getting ready while it loads, and stops saying so once it speaks", async () => {
    render(VoiceControl);
    emit("voice_status", { status: "voice_loading", message: "Getting my voice ready…" });
    await tick();
    expect(screen.getByText("Getting my voice ready…")).toBeTruthy();

    emit("voice_status", { status: "voice_ready" });
    await tick();
    expect(screen.queryByText("Getting my voice ready…")).toBeNull();
  });
});

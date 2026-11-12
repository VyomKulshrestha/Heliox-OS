import { render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  call: vi.fn(async (method: string): Promise<Record<string, unknown>> =>
    method === "voice_listener_start" ? { status: "started" } : { status: "ok", running: false },
  ),
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

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("VoiceControl when Heliox's voice did not work", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
  });

  it("shows the reply as text, and what to check only the first time", async () => {
    render(VoiceControl);
    const check = "Heliox's voice did not work, so its replies are shown here as text. Check the output device.";
    emit("voice_status", { status: "voice_output_failed", text: "Which file, the blue one?", message: check });
    await tick();
    expect(screen.getByText("Which file, the blue one?")).toBeTruthy();
    expect(screen.getByText(check)).toBeTruthy();

    emit("voice_status", { status: "voice_output_failed", text: "It is done.", message: "" });
    await tick();
    expect(screen.getByText("It is done.")).toBeTruthy();
    await waitFor(() => expect(screen.queryAllByText(check).length).toBeLessThanOrEqual(1));
  });
});

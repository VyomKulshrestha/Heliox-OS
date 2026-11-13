import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
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

  it("says it is getting ready to listen while the recognisers warm, and stops once they are ready", async () => {
    render(VoiceControl);
    emit("voice_status", { status: "hearing_loading", message: "Getting ready to listen…" });
    await tick();
    expect(screen.getByText("Getting ready to listen…")).toBeTruthy();

    emit("voice_status", { status: "hearing_ready" });
    await tick();
    expect(screen.queryByText("Getting ready to listen…")).toBeNull();
  });

  it("says a sentence came too early to be heard, and to say it again", async () => {
    render(VoiceControl);
    emit("voice_status", { status: "hearing_loading", message: "Getting ready to listen…" });
    emit("voice_status", {
      status: "hearing_not_ready",
      message: "I wasn't ready to listen yet. Say that again in a moment.",
    });
    await tick();
    expect(screen.getByText("I wasn't ready to listen yet. Say that again in a moment.")).toBeTruthy();
    expect(screen.getByText("Getting ready to listen…")).toBeTruthy();
  });

  it("with always-listening on, the microphone button says the next sentence is for Heliox", async () => {
    render(VoiceControl);
    const button = await screen.findByRole("button", { name: "Talk to Heliox now" });
    expect((button as HTMLButtonElement).disabled).toBe(false);
    await fireEvent.click(button);
    await waitFor(() => expect(mocks.call).toHaveBeenCalledWith("voice_listen_now"));
  });

  it("says it is staying quiet while the person is in a call", async () => {
    render(VoiceControl);
    emit("voice_status", { status: "in_call", app: "Zoom", message: "Quiet while you're in a call" });
    await tick();
    expect(screen.getByText("Quiet while you're in a call")).toBeTruthy();
    emit("voice_status", { status: "call_ended" });
    await tick();
    expect(screen.queryByText("Quiet while you're in a call")).toBeNull();
  });

  it("says when the microphone stops and when it works again", async () => {
    render(VoiceControl);
    emit("voice_status", {
      status: "microphone_problem",
      reason: "no_audio",
      message: "The microphone stopped sending sound; reopening it.",
    });
    await tick();
    expect(screen.getByText("The microphone stopped sending sound; reopening it.")).toBeTruthy();
    emit("voice_status", { status: "microphone_recovered", message: "Listening on Microphone (USBAudio1.0)." });
    await tick();
    expect(screen.queryByText("The microphone stopped sending sound; reopening it.")).toBeNull();
    expect(screen.getByText("Listening on Microphone (USBAudio1.0).")).toBeTruthy();
  });

  it("with no speech model, says so and opens the download list only when pressed", async () => {
    const opened = vi.fn();
    window.addEventListener("heliox:open-settings", opened);
    render(VoiceControl);
    emit("voice_status", {
      status: "recogniser_missing",
      message: "Speech recognition needs a model, which Heliox has not downloaded.",
      settings: "voice-models",
    });
    await tick();
    expect(screen.getByText("Speech recognition needs a model, which Heliox has not downloaded.")).toBeTruthy();
    expect(mocks.call).not.toHaveBeenCalledWith("speech_model_download", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Download a speech model" }));
    expect(opened).toHaveBeenCalledTimes(1);
    expect((opened.mock.calls[0][0] as CustomEvent).detail).toEqual({ section: "voice-models" });
    window.removeEventListener("heliox:open-settings", opened);
  });

  it("offers another microphone for a headset in call mode, and switches only when pressed", async () => {
    render(VoiceControl);
    emit("voice_status", {
      status: "microphone_quality",
      reason: "narrowband",
      message: "This microphone is in Bluetooth call mode.",
      headset: "Headset (Rockerz 415 Hands-Free AG Audio)",
      suggested_input: { id: "Windows WASAPI::Microphone Array (Realtek)", name: "Microphone Array (Realtek)" },
    });
    await tick();
    expect(mocks.call).not.toHaveBeenCalledWith("update_config", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Use Microphone Array (Realtek) for input" }));
    await waitFor(() =>
      expect(mocks.call).toHaveBeenCalledWith("update_config", {
        section: "voice",
        values: {
          headset_input_choices: {
            "Headset (Rockerz 415 Hands-Free AG Audio)": "Windows WASAPI::Microphone Array (Realtek)",
          },
        },
      }),
    );
    expect(screen.queryByRole("button", { name: "Use Microphone Array (Realtek) for input" })).toBeNull();
  });
});

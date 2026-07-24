import { beforeEach, describe, expect, it, vi } from "vitest";

const { callMock } = vi.hoisted(() => ({
  callMock: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: callMock,
}));

import { speakText, stopSpeech } from "./tts";

class FakeUtterance {
  rate = 1;
  pitch = 1;
  volume = 1;
  voice: SpeechSynthesisVoice | null = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public text: string) {}
}

const speechSynthesisMock = {
  cancel: vi.fn(),
  getVoices: vi.fn(() => []),
  speak: vi.fn((utterance: FakeUtterance) => utterance.onend?.()),
};

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: speechSynthesisMock,
  });
  vi.stubGlobal("SpeechSynthesisUtterance", FakeUtterance);
});

describe("configured text-to-speech", () => {
  it("uses the daemon so the selected engine and voice are honored", async () => {
    callMock.mockResolvedValue({ status: "spoken" });
    const onStart = vi.fn();
    const onEnd = vi.fn();

    speakText(" hello ", { onStart, onEnd });
    await vi.waitFor(() => expect(onEnd).toHaveBeenCalledOnce());

    expect(callMock).toHaveBeenCalledWith("speak_text", { text: "hello" });
    expect(onStart).toHaveBeenCalledOnce();
    expect(speechSynthesisMock.speak).not.toHaveBeenCalled();
  });

  it("falls back to browser speech when the daemon is unavailable", async () => {
    callMock.mockRejectedValue(new Error("offline"));
    const onEnd = vi.fn();

    speakText("fallback", { onEnd });
    await vi.waitFor(() => expect(onEnd).toHaveBeenCalledOnce());

    expect(speechSynthesisMock.speak).toHaveBeenCalledOnce();
    expect(speechSynthesisMock.speak.mock.calls[0][0].text).toBe("fallback");
  });

  it("stops browser and daemon playback", () => {
    callMock.mockResolvedValue({ status: "stopped" });

    stopSpeech();

    expect(speechSynthesisMock.cancel).toHaveBeenCalled();
    expect(callMock).toHaveBeenCalledWith("stop_speech");
  });
});

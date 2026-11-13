import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  installed: false,
  call: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));

import SpeechModels from "./SpeechModels.svelte";

const PARAKEET = {
  id: "parakeet-tdt-0.6b-v2-int8",
  title: "Accurate English recognition (Parakeet)",
  purpose: "Hears English more accurately.",
  languages: ["en"],
  licence: "CC-BY-4.0",
  source: "NVIDIA Parakeet TDT 0.6B v2, exported by the sherpa-onnx project (GitHub releases)",
  download_bytes: 482440461,
  partial_bytes: 0,
};

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("Speech models in Settings", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
    mocks.installed = false;
    mocks.call.mockReset();
    mocks.call.mockImplementation(async (method: string) => {
      if (method === "speech_models_list")
        return { status: "ok", models: [{ ...PARAKEET, installed: mocks.installed }] };
      if (method === "speech_model_download") return { status: "started" };
      return { status: "ok" };
    });
  });

  it("says what the model is, its size, licence and source, and downloads only when pressed", async () => {
    render(SpeechModels);
    const button = await screen.findByRole("button", { name: "Download 460 MB" });
    expect(screen.getByText(/CC-BY-4.0/)).toBeTruthy();
    expect(screen.getByText(/English only/)).toBeTruthy();
    expect(mocks.call).not.toHaveBeenCalledWith("speech_model_download", expect.anything());
    await fireEvent.click(button);
    expect(mocks.call).toHaveBeenCalledWith("speech_model_download", { id: PARAKEET.id });
    emit("speech_model_progress", {
      id: PARAKEET.id,
      state: "downloading",
      received: 104857600,
      total: 482440461,
      message: "Downloading… 100 of 460 MB",
    });
    await tick();
    expect(screen.getByText("Downloading… 100 of 460 MB")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy();
    mocks.installed = true;
    emit("speech_model_progress", { id: PARAKEET.id, state: "ready", received: 1, total: 1, message: "Ready." });
    await waitFor(() => expect(screen.getByRole("button", { name: "Remove" })).toBeTruthy());
  });

  it("shows the failure sentence and offers the download again", async () => {
    render(SpeechModels);
    await fireEvent.click(await screen.findByRole("button", { name: "Download 460 MB" }));
    emit("speech_model_progress", {
      id: PARAKEET.id,
      state: "failed",
      received: 0,
      total: 1,
      message: "Could not reach the download (URLError). Check the internet connection and try again.",
    });
    await waitFor(() => expect(screen.getByText(/Check the internet connection/)).toBeTruthy());
    expect(await screen.findByRole("button", { name: "Download 460 MB" })).toBeTruthy();
  });

  describe("the recommended set", () => {
    const WHERE =
      "Downloaded once from GitHub and Hugging Face; the models stay on this computer and recognise your speech here.";
    function listWithRecommended(download: () => Promise<unknown>) {
      mocks.call.mockImplementation(async (method: string) => {
        if (method === "speech_models_list")
          return {
            status: "ok",
            models: [{ ...PARAKEET, installed: false }],
            recommended: {
              title: "Recommended for English: Parakeet with Whisper base",
              why: "Parakeet hears the words.",
              ids: [PARAKEET.id],
              missing: [PARAKEET.id],
              missing_bytes: PARAKEET.download_bytes,
              partial_bytes: 0,
              where: WHERE,
            },
          };
        if (method === "speech_model_download") return download();
        return { status: "ok" };
      });
    }

    it("says where it comes from and starts on one press", async () => {
      listWithRecommended(async () => ({ status: "started" }));
      render(SpeechModels);
      const button = await screen.findByRole("button", { name: "Download · 460 MB" });
      expect(screen.getByText(WHERE)).toBeTruthy();
      await fireEvent.click(button);
      expect(mocks.call).toHaveBeenCalledWith("speech_model_download", { id: PARAKEET.id });
    });

    it("a refusal is said under the model", async () => {
      listWithRecommended(async () => ({ status: "error", message: "There is not enough free space." }));
      render(SpeechModels);
      await fireEvent.click(await screen.findByRole("button", { name: "Download · 460 MB" }));
      await waitFor(() => expect(screen.getByText("There is not enough free space.")).toBeTruthy());
    });

    it("no answer from Heliox is said instead of an unhandled rejection", async () => {
      listWithRecommended(async () => {
        throw new Error("socket closed");
      });
      render(SpeechModels);
      await fireEvent.click(await screen.findByRole("button", { name: "Download · 460 MB" }));
      await waitFor(() =>
        expect(screen.getByText("Heliox did not answer, so the download did not start. Try again.")).toBeTruthy(),
      );
    });
  });

  it("offers the voice of the person's language and changes it only when pressed", async () => {
    mocks.call.mockImplementation(async (method: string) => {
      if (method === "speech_models_list")
        return {
          status: "ok",
          models: [],
          voice: {
            language: "es",
            engine: "kokoro_tts",
            voice: "ef_dora",
            change: true,
            say: "Heliox can speak Spanish with Kokoro's voice ef_dora; about half a megabyte is fetched the first time.",
          },
        };
      if (method === "voice_for_language_apply")
        return {
          status: "ok",
          language: "es",
          engine: "kokoro_tts",
          voice: "ef_dora",
          change: false,
          say: "Heliox speaks Spanish with the voice ef_dora.",
        };
      return { status: "ok" };
    });
    render(SpeechModels);
    const button = await screen.findByRole("button", { name: "Use this voice" });
    expect(screen.getByText(/speak Spanish with Kokoro's voice ef_dora/)).toBeTruthy();
    expect(mocks.call).not.toHaveBeenCalledWith("voice_for_language_apply");
    await fireEvent.click(button);
    expect(mocks.call).toHaveBeenCalledWith("voice_for_language_apply");
    await waitFor(() => expect(screen.getByText("Heliox speaks Spanish with the voice ef_dora.")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Use this voice" })).toBeNull();
  });

  it("says plainly what a German speaker hears when there is no German voice", async () => {
    mocks.call.mockImplementation(async (method: string) =>
      method === "speech_models_list"
        ? {
            status: "ok",
            models: [],
            voice: {
              language: "de",
              engine: "kokoro_tts",
              voice: "af_heart",
              change: false,
              say: "Heliox has no German voice on this computer, so it reads German replies with English sounds.",
            },
          }
        : { status: "ok" },
    );
    render(SpeechModels);
    expect(await screen.findByText(/reads German replies with English sounds/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Use this voice" })).toBeNull();
  });
});

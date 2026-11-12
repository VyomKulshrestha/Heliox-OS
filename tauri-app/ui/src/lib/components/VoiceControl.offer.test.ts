import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The voice bar's offer of the recommended speech models when no recogniser is on this computer: it says where they
// come from, downloads only after a second press, shows progress with Stop, and comes back for what is left after a
// stop (here or in Settings), a refusal, or no answer from the daemon.

type Handler = (method: string, params: unknown) => void;
type Reply = Record<string, unknown>;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  download: (async () => ({ status: "started" })) as (id: string) => Promise<Reply>,
  cancel: (async () => ({ status: "cancelling" })) as (id: string) => Promise<Reply>,
  calls: [] as string[],
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params?: { id?: string }): Promise<Reply> => {
    mocks.calls.push(`${method}${params?.id ? `:${params.id}` : ""}`);
    if (method === "speech_model_download") return mocks.download(String(params?.id));
    if (method === "speech_model_cancel") return mocks.cancel(String(params?.id));
    return method === "voice_listener_start" ? { status: "started" } : { status: "ok", running: false };
  }),
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

const MB = 1_048_576;
const IDS = ["parakeet-tdt-0.6b-v2-int8", "whisper-base"];
const WHERE =
  "Downloaded once from GitHub and Hugging Face; the models stay on this computer and recognise your speech here.";

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

async function offered() {
  render(VoiceControl);
  emit("voice_status", {
    status: "recogniser_missing",
    message: "Speech recognition needs a model.",
    recommended: { missing: IDS, missing_bytes: 601 * MB, partial_bytes: 0, where: WHERE },
  });
  await tick();
}

async function confirm() {
  await fireEvent.click(screen.getByRole("button", { name: "Download the recommended speech models (601 MB)" }));
  await fireEvent.click(screen.getByRole("button", { name: "Yes, download 601 MB" }));
}

describe("VoiceControl speech-model offer", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
    mocks.calls.length = 0;
    mocks.download = async () => ({ status: "started" });
    mocks.cancel = async () => ({ status: "cancelling" });
  });

  it("says where the models come from and downloads only after the second press", async () => {
    await offered();
    expect(screen.getByText(WHERE)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Download the recommended speech models (601 MB)" }));
    expect(mocks.calls.filter((c) => c.startsWith("speech_model_download"))).toEqual([]);
    await fireEvent.click(screen.getByRole("button", { name: "Yes, download 601 MB" }));
    await waitFor(() =>
      expect(mocks.calls.filter((c) => c.startsWith("speech_model_download"))).toEqual(
        IDS.map((id) => `speech_model_download:${id}`),
      ),
    );
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy();
  });

  it("after Stop, or a stop in Settings, offers to continue with what is left", async () => {
    await offered();
    await confirm();
    emit("speech_model_progress", { id: IDS[0], state: "downloading", received: 101 * MB, total: 460 * MB });
    await tick();
    expect(screen.getByText(/Downloading speech models… 17%/)).toBeTruthy();
    // Cancelled in Settings, Voice: the bar must not keep showing a download that is not happening.
    emit("speech_model_progress", { id: IDS[0], state: "cancelled", received: 101 * MB, total: 460 * MB });
    await tick();
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Continue downloading the recommended speech models (500 MB)" }),
    ).toBeTruthy();
  });

  it("Stop in the bar cancels both and comes back for what is left", async () => {
    await offered();
    await confirm();
    await fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() =>
      expect(mocks.calls.filter((c) => c.startsWith("speech_model_cancel"))).toEqual(
        IDS.map((id) => `speech_model_cancel:${id}`),
      ),
    );
    expect(screen.getByRole("button", { name: /recommended speech models \(601 MB\)/ })).toBeTruthy();
  });

  it("a refusal from the daemon is said in one sentence and the offer comes back", async () => {
    mocks.download = async () => ({ status: "error", message: "There is not enough free space for the download." });
    await offered();
    await confirm();
    await waitFor(() => expect(screen.getByText("There is not enough free space for the download.")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
  });

  it("no answer from the daemon resets the bar instead of leaving an unhandled rejection", async () => {
    mocks.download = async () => {
      throw new Error("socket closed");
    };
    await offered();
    await confirm();
    await waitFor(() =>
      expect(
        screen.getByText("Heliox did not answer, so the download did not start. Try again in a moment."),
      ).toBeTruthy(),
    );
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
  });

  it("no answer to Stop says the download may still be running", async () => {
    mocks.cancel = async () => {
      throw new Error("socket closed");
    };
    await offered();
    await confirm();
    await fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() =>
      expect(
        screen.getByText("Heliox did not answer; the download may still be running. Settings, Voice shows it."),
      ).toBeTruthy(),
    );
  });
});

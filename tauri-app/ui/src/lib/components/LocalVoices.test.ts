import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  // A rejected vi.fn promise fails the test under vitest 4 even when caught: the mock resolves to an Error instead.
  const call = async (...args: Parameters<typeof actual.call>) => {
    const result = await daemonMocks.call(...args);
    if (result instanceof Error) throw result;
    return result;
  };
  return { ...actual, call };
});

import { pocketVoiceCredit, pocketVoicesOffered } from "../firstRun";
import { settings } from "../stores/settings";
import LocalVoices from "./LocalVoices.svelte";

const pocket = {
  engine: "pocket_tts",
  voice: "alba",
  download_bytes: 225_000_000,
  say: "On this computer Pocket TTS would answer about 0.6 s sooner than Kokoro (measured on one development laptop's processor). It downloads about 225 MB the first time it speaks. Your voice stays as it is unless you switch.",
};

let voiceNow: Record<string, unknown> = { tts_engine: "kokoro_tts", tts_voice: "af_heart" };

function respond(suggestion: unknown, update: unknown = { status: "ok" }) {
  const asked: { method: string; params: unknown }[] = [];
  daemonMocks.call.mockImplementation(async (method: string, params: unknown) => {
    asked.push({ method, params });
    if (method === "speech_models_list")
      return suggestion instanceof Error ? suggestion : { status: "ok", models: [], engine_suggestion: suggestion };
    if (method === "update_config") return update;
    if (method === "get_config") return { voice: voiceNow };
    return { status: "ok", config: { voice: { tts_engine: "pocket_tts", tts_voice: "alba" } } };
    return { status: "ok" };
  });
  return asked;
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("LocalVoices", () => {
  beforeEach(async () => {
    daemonMocks.call.mockReset();
    voiceNow = { tts_engine: "kokoro_tts", tts_voice: "af_heart" };
    respond(null);
    await settings.reloadFromDaemon();
  });

  it("names each voice model's licence and source, with Pocket's attribution, even without the daemon", async () => {
    respond(new Error("Heliox is not running"));
    render(LocalVoices);
    await settle();
    const row = screen.getByTestId("voice-licences");
    expect(row.textContent).toContain(
      "Kokoro (Kokoro-82M): licence Apache-2.0, from hexgrad/Kokoro-82M on Hugging Face.",
    );
    expect(row.textContent).toContain(
      "licence CC BY 4.0, from kyutai/pocket-tts-without-voice-cloning on Hugging Face",
    );
    expect(row.textContent).toContain("Pocket TTS model by Kyutai");
    expect(row.textContent).toContain("licensed under CC BY 4.0");
    expect(row.textContent).toContain("MIT, as its repository states");
    expect(row.textContent).toContain("about 225 MB");
    expect(screen.queryByTestId("voice-suggestion")).toBeNull();
  });

  it("shows the one suggestion and switches only when the person presses it", async () => {
    const asked = respond(pocket);
    render(LocalVoices);
    const row = await screen.findByTestId("voice-suggestion");
    expect(row.textContent).toContain("about 0.6 s sooner");
    expect(row.textContent).toContain("about 225 MB");
    expect(asked.some((entry) => entry.method === "update_config")).toBe(false); // nothing switched by itself
    await fireEvent.click(screen.getByRole("button", { name: "Use Pocket TTS" }));
    await waitFor(() => expect(screen.getByText("Heliox now speaks with Pocket TTS.")).toBeTruthy());
    expect(asked.find((entry) => entry.method === "update_config")?.params).toEqual({
      section: "voice",
      values: { tts_engine: "pocket_tts", tts_voice: "alba" },
    });
    expect(screen.queryByTestId("voice-suggestion")).toBeNull();
  });

  it("shows no suggestion when the daemon has none", async () => {
    respond(null);
    render(LocalVoices);
    await settle();
    expect(screen.queryByTestId("voice-suggestion")).toBeNull();
    expect(screen.getByTestId("voice-licences")).toBeTruthy();
  });

  it("hides the suggestion for good with Not now, which chooses no voice", async () => {
    const asked = respond(pocket);
    render(LocalVoices);
    await screen.findByTestId("voice-suggestion");
    await fireEvent.click(screen.getByRole("button", { name: "Not now" }));
    await waitFor(() => expect(screen.queryByTestId("voice-suggestion")).toBeNull());
    expect(asked.filter((entry) => entry.method === "update_config").map((entry) => entry.params)).toEqual([
      { section: "voice", values: { engine_suggestion_dismissed: true } },
    ]);
  });

  it("credits the Pocket voice that speaks, and says when one is kept though not offered", async () => {
    voiceNow = { tts_engine: "pocket_tts", tts_voice: "alba" };
    respond(null);
    await settings.reloadFromDaemon();
    const shown = render(LocalVoices);
    await settle();
    expect(screen.getByTestId("voice-credit").textContent).toContain("Voice alba: Voice acted by Alba MacKenna");
    shown.unmount();
    voiceNow = { tts_engine: "pocket_tts", tts_voice: "lola" };
    await settings.reloadFromDaemon();
    render(LocalVoices);
    await settle();
    expect(screen.getByTestId("voice-credit").textContent).toContain("Heliox does not offer it");
  });

  it("offers Pocket voices whose terms allow any use and that were listened to or checked, keeping one chosen", () => {
    expect(pocketVoicesOffered()).toEqual(["alba", "michael"]); // michael: read back and measured, not yet heard
    expect(pocketVoicesOffered("lola")).toEqual(["alba", "michael", "lola"]);
    expect(pocketVoicesOffered("alba")).toEqual(["alba", "michael"]);
    expect(pocketVoiceCredit("michael")).toBe(
      "Voice michael: Voice from VCTK (the Voice Cloning Toolkit, datashare.ed.ac.uk/handle/10283/3443), CC BY 4.0.",
    );
    expect(pocketVoiceCredit("bill_boerst")).toBe(
      "Voice bill_boerst: CC0 1.0, from kyutai/tts-voices voice-zero/bill_boerst.wav.",
    );
    expect(pocketVoiceCredit("cosette")).toContain("does not offer it");
  });
});

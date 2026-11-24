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

import LocalVoices from "./LocalVoices.svelte";

const pocket = {
  engine: "pocket_tts",
  voice: "alba",
  download_bytes: 225_000_000,
  say: "On this computer Pocket TTS would answer about 0.6 s sooner than Kokoro (measured on one development laptop's processor). It downloads about 225 MB the first time it speaks. Your voice stays as it is unless you switch.",
};

function respond(suggestion: unknown, update: unknown = { status: "ok" }) {
  const asked: { method: string; params: unknown }[] = [];
  daemonMocks.call.mockImplementation(async (method: string, params: unknown) => {
    asked.push({ method, params });
    if (method === "speech_models_list") return suggestion instanceof Error ? suggestion : { status: "ok", models: [], engine_suggestion: suggestion };
    if (method === "update_config") return update;
    if (method === "get_config") return { status: "ok", config: { voice: { tts_engine: "pocket_tts", tts_voice: "alba" } } };
    return { status: "ok" };
  });
  return asked;
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("LocalVoices", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("names each voice model's licence and source, with Pocket's attribution, even without the daemon", async () => {
    respond(new Error("Heliox is not running"));
    render(LocalVoices);
    await settle();
    const row = screen.getByTestId("voice-licences");
    expect(row.textContent).toContain("Kokoro (Kokoro-82M): licence Apache-2.0, from hexgrad/Kokoro-82M on Hugging Face.");
    expect(row.textContent).toContain("licence CC BY 4.0, from kyutai/pocket-tts-without-voice-cloning on Hugging Face");
    expect(row.textContent).toContain("Pocket TTS by Kyutai, licensed under CC BY 4.0");
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
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  sent: [] as unknown[],
  reply: "ok" as "ok" | "refuse" | "never",
  faceVoice: [] as unknown[],
  // The daemon's face_voice: what it answers for each apply ("" plan only, "accept").
  plans: {} as Record<string, Record<string, unknown>>,
  // What the daemon does by itself when face.pack is set (it applies the face's voice).
  onFace: null as ((pack: string) => void) | null,
  // The handler FaceChoice gives onNotification (face_voice_ready).
  handler: null as ((method: string, params: unknown) => void) | null,
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      if (mocks.reply === "never") return new Promise(() => {});
      if (mocks.reply === "refuse") return { status: "error", message: 'face.pack must be "off" or a face pack id' };
      const { section, values } = params as { section: string; values: Record<string, unknown> };
      mocks.config = { ...mocks.config, [section]: { ...(mocks.config[section] as object), ...values } };
      if (typeof values.pack === "string") mocks.onFace?.(values.pack);
      return { status: "ok" };
    }
    if (method === "get_config") return mocks.config;
    if (method === "face_voice") {
      const asked = params as { apply: string };
      mocks.faceVoice.push(params);
      const answer = mocks.plans[asked.apply] ?? mocks.plans[""] ?? { status: "ok", action: "none", say: "" };
      if (answer.switched)
        mocks.config = { ...mocks.config, voice: { ...(mocks.config.voice as object), tts_voice: answer.voice } };
      return answer;
    }
    return { status: "ok" };
  }),
  onNotification: vi.fn((handler: (method: string, params: unknown) => void) => (mocks.handler = handler)),
  offNotification: vi.fn(),
}));

import { settings } from "../stores/settings";
import FaceChoice from "./FaceChoice.svelte";

const MANIFEST = {
  format: 1,
  id: "placeholder",
  name: "Line drawing",
  credit: "Drawn for Heliox as a placeholder; a drawing of nobody. CC0 1.0.",
  size: [256, 256],
  idle: { frames: ["base.svg"] },
  mouth: { shapes: { sil: "mouth-sil.svg" } },
};

function serve(files: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const key = Object.keys(files).find((path) => url.endsWith(path));
      return { ok: key !== undefined, json: async () => (key ? files[key] : null) };
    }),
  );
}

const MAN = {
  ...MANIFEST,
  id: "portrait-man",
  name: "Portrait, man",
  credit: "Generated; shows no real person.",
  kind: "photo",
  voice: { kokoro: "am_michael", hint: "masculine" },
};

describe("Face (Settings, Appearance)", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    mocks.faceVoice.length = 0;
    mocks.plans = {};
    mocks.onFace = null;
    mocks.reply = "ok";
    serve({
      "faces/packs.json": { packs: ["placeholder", "Bad/Id", "portrait-man"] },
      "faces/placeholder/manifest.json": MANIFEST,
      "faces/portrait-man/manifest.json": MAN,
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("is off by default and offers the installed packs by their names", async () => {
    mocks.config = { face: {} };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    const select = screen.getByRole("combobox", { name: "Face" }) as HTMLSelectElement;
    expect(select.value).toBe("off");
    await waitFor(() => expect(screen.getByRole("option", { name: "Line drawing" })).toBeTruthy());
    await waitFor(() => screen.getByRole("option", { name: "Portrait, man" }));
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["Off", "Line drawing", "Portrait, man"]);
  });

  it("saves the chosen pack under face, and shows its credit", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "kokoro_tts" } };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await waitFor(() => screen.getByRole("option", { name: "Line drawing" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "placeholder" } });
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "face", values: { pack: "placeholder" } }]));
    await waitFor(() => expect(screen.getByText(/a drawing of nobody/)).toBeTruthy());
  });

  it("says plainly that the system voice leaves the face still", async () => {
    mocks.config = { face: { pack: "placeholder" }, voice: { tts_engine: "os_native" } };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    expect(screen.getByText(/With the system voice the face stays still/)).toBeTruthy();
  });

  it("shows the daemon's refusal and keeps the old choice", async () => {
    mocks.config = { face: { pack: "off" } };
    mocks.reply = "refuse";
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await waitFor(() => screen.getByRole("option", { name: "Line drawing" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "placeholder" } });
    await waitFor(() => expect(screen.getByText(/face.pack must be/)).toBeTruthy());
    expect((screen.getByRole("combobox", { name: "Face" }) as HTMLSelectElement).value).toBe("off");
  });

  it("renders when the daemon never answers and when no pack list can be read", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      }),
    );
    mocks.config = { face: { pack: "off" } };
    mocks.reply = "never";
    await settings.reloadFromDaemon();
    render(FaceChoice);
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["Off"]);
  });

  async function chooseMan() {
    render(FaceChoice);
    await waitFor(() => screen.getByRole("option", { name: "Portrait, man" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "portrait-man" } });
  }

  it("a face with no voice of its own asks nothing about the voice", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "kokoro_tts", tts_voice: "af_heart" } };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await waitFor(() => screen.getByRole("option", { name: "Line drawing" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "placeholder" } });
    await waitFor(() => expect(mocks.sent).toHaveLength(1));
    expect(mocks.faceVoice.filter((asked) => (asked as { pack: string }).pack === "placeholder")).toEqual([]);
    expect(screen.queryByTestId("face-voice")).toBeNull();
  });

  it("shows the switch the daemon made itself when the face was set, without asking for one", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "kokoro_tts", tts_voice: "af_heart" } };
    // The daemon applies the face's voice wherever face.pack is set; face_voice "" then reports it.
    mocks.onFace = (pack) => {
      if (pack !== "portrait-man") return;
      mocks.config = { face: { pack }, voice: { tts_engine: "kokoro_tts", tts_voice: "am_michael" } };
      mocks.plans = {
        "": {
          status: "ok",
          switched: true,
          voice: "am_michael",
          action: "same",
          say: "Heliox now speaks with am_michael, this face's voice. Any other voice can be picked under Voice.",
        },
      };
    };
    mocks.plans = {
      "": {
        status: "ok",
        switched: false,
        voice: "am_michael",
        action: "offer",
        say: "This face's voice is am_michael.",
      },
    };
    await settings.reloadFromDaemon();
    await chooseMan();
    await waitFor(() =>
      expect(screen.getByTestId("face-voice").textContent).toMatch(
        /Heliox now speaks with am_michael, this face's voice/,
      ),
    );
    await waitFor(() => {
      let voice = "";
      settings.subscribe((s) => (voice = s.voice?.tts_voice ?? ""))();
      expect(voice).toBe("am_michael"); // the Voice rows show the new voice at once
    });
    expect(mocks.faceVoice.every((asked) => (asked as { apply: string }).apply === "")).toBe(true);
    expect(screen.queryByRole("button", { name: /Use am_michael/ })).toBeNull();
  });

  it("keeps a chosen voice and offers the face's voice with one control", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "kokoro_tts", tts_voice: "bf_emma" } };
    const offer = {
      status: "ok",
      switched: false,
      voice: "am_michael",
      action: "offer",
      say: "Your voice, bf_emma, is kept. This face's voice is am_michael.",
    };
    mocks.plans = {
      "": offer,
      accept: {
        status: "ok",
        switched: true,
        voice: "am_michael",
        action: "same",
        say: "Heliox now speaks with am_michael.",
      },
    };
    await settings.reloadFromDaemon();
    await chooseMan();
    const use = await screen.findByRole("button", { name: "Use am_michael" });
    expect(screen.getByTestId("face-voice").textContent).toMatch(/Your voice, bf_emma, is kept/);
    await fireEvent.click(use);
    await waitFor(() =>
      expect(screen.getByTestId("face-voice").textContent).toMatch(/Heliox now speaks with am_michael[.]/),
    );
    expect(mocks.faceVoice).toContainEqual({
      voice: "am_michael",
      hint: "masculine",
      pack: "portrait-man",
      apply: "accept",
    });
    expect(screen.queryByRole("button", { name: /Use am_michael/ })).toBeNull();
  });

  const MISSING = {
    status: "ok",
    switched: false,
    voice: "am_michael",
    action: "fetch",
    engine: "kokoro_tts",
    download_bytes: 330_000_000,
    download_ok: true,
    download_reason: "",
    say: "Pocket TTS has no approved voice for this face yet, so the face brings its Kokoro voice. Kokoro is not on this computer yet; using am_michael fetches about 330 MB the first time it speaks.",
  };
  const KEPT = {
    status: "ok",
    switched: false,
    voice: "am_michael",
    action: "kept",
    say: "You chose to keep your voice, alba, with this face.",
  };
  const ASKED = (apply: string) => ({ voice: "am_michael", hint: "masculine", pack: "portrait-man", apply });

  it("says in the list which face needs a download for its voice, and asks in place before saving anything", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "pocket_tts", tts_voice: "alba" } };
    mocks.plans = { "": MISSING };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await screen.findByRole("option", { name: "Portrait, man (its voice needs a download of about 330 MB)" });
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "portrait-man" } });
    const question = await screen.findByTestId("face-voice-question");
    expect(question.textContent).toMatch(/Portrait, man's voice, am_michael, is not on this computer/);
    expect(question.textContent).toMatch(/needs a download of\s+about 330 MB/);
    expect(question.textContent).toMatch(/alba keeps speaking/);
    expect(screen.getByRole("button", { name: "Use this face and download its voice" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Use this face with the voice I have" })).toBeTruthy();
    expect(mocks.sent).toEqual([]); // nothing saved, nothing fetched until a button is pressed
    expect(mocks.faceVoice.some((asked) => (asked as { apply: string }).apply !== "")).toBe(false);
  });

  it("downloads the voice only when asked, and shows the switch once it is ready", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "pocket_tts", tts_voice: "alba" } };
    mocks.plans = {
      "": MISSING,
      download: {
        ...MISSING,
        action: "downloading",
        say: "Downloading am_michael (about 330 MB). alba keeps speaking until it is ready; the switch comes between two replies.",
      },
    };
    // While it downloads, the daemon's plan for the chosen face says so.
    mocks.onFace = () => (mocks.plans = { ...mocks.plans, "": mocks.plans.download });
    await settings.reloadFromDaemon();
    await chooseMan();
    await fireEvent.click(await screen.findByRole("button", { name: "Use this face and download its voice" }));
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "face", values: { pack: "portrait-man" } }]));
    await waitFor(() => expect(screen.getByTestId("face-voice").textContent).toMatch(/Downloading am_michael/));
    expect(mocks.faceVoice).toContainEqual(ASKED("download"));
    expect(screen.queryByRole("button", { name: "Use this face and download its voice" })).toBeNull();
    // The daemon says the voice is ready (it switched between two replies).
    mocks.config = { face: { pack: "portrait-man" }, voice: { tts_engine: "kokoro_tts", tts_voice: "am_michael" } };
    mocks.plans = { "": { status: "ok", switched: false, voice: "am_michael", action: "same", say: "" } };
    mocks.handler?.("face_voice_ready", {
      status: "ok",
      pack: "portrait-man",
      voice: "am_michael",
      say: "Heliox now speaks with am_michael, this face's voice.",
    });
    await waitFor(() =>
      expect(screen.getByTestId("face-voice").textContent).toMatch(
        /Heliox now speaks with am_michael, this face's voice/,
      ),
    );
  });

  it("uses the face with the voice the person has, and asks nothing more", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "pocket_tts", tts_voice: "alba" } };
    mocks.plans = { "": MISSING, keep: KEPT };
    mocks.onFace = () => (mocks.plans = { ...mocks.plans, "": KEPT }); // recorded by the keep that follows
    await settings.reloadFromDaemon();
    await chooseMan();
    await fireEvent.click(await screen.findByRole("button", { name: "Use this face with the voice I have" }));
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "face", values: { pack: "portrait-man" } }]));
    expect(mocks.faceVoice).toContainEqual(ASKED("keep"));
    await waitFor(() =>
      expect(screen.getByTestId("face-voice").textContent).toMatch(/You chose to keep your voice, alba/),
    );
    expect(screen.queryByTestId("face-voice-question")).toBeNull();
  });

  it("without a network the download is not offered, says why, and the face can still be used", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "pocket_tts", tts_voice: "alba" } };
    mocks.plans = {
      "": { ...MISSING, download_ok: false, download_reason: "No connection to huggingface.co right now." },
      keep: KEPT,
    };
    await settings.reloadFromDaemon();
    await chooseMan();
    const download = (await screen.findByRole("button", {
      name: "Use this face and download its voice",
    })) as HTMLButtonElement;
    expect(download.disabled).toBe(true);
    expect(screen.getByText("Cannot download now: No connection to huggingface.co right now.")).toBeTruthy();
    const keep = screen.getByRole("button", { name: "Use this face with the voice I have" }) as HTMLButtonElement;
    expect(keep.disabled).toBe(false);
    await fireEvent.click(keep);
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "face", values: { pack: "portrait-man" } }]));
  });

  it("with another engine says how to hear the face's voice and offers no control", async () => {
    mocks.config = { face: { pack: "portrait-man" }, voice: { tts_engine: "os_native", tts_voice: "af_heart" } };
    mocks.plans = {
      "": {
        status: "ok",
        switched: false,
        voice: "am_michael",
        action: "unavailable",
        say: "This face's voice is am_michael, a Kokoro voice. Heliox speaks with the system voice, so the voice stays as it is; to hear this face's voice, choose Kokoro TTS under Voice.",
      },
    };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await waitFor(() => expect(screen.getByTestId("face-voice").textContent).toMatch(/choose Kokoro TTS under Voice/));
    expect(screen.queryByRole("button", { name: /Use/ })).toBeNull();
  });

  it("turning the face off changes no voice", async () => {
    mocks.config = { face: { pack: "portrait-man" }, voice: { tts_engine: "kokoro_tts", tts_voice: "am_michael" } };
    mocks.plans = { "": { status: "ok", switched: false, voice: "am_michael", action: "same", say: "" } };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    await waitFor(() => screen.getByRole("option", { name: "Portrait, man" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Face" }), { target: { value: "off" } });
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "face", values: { pack: "off" } }]));
    expect(mocks.faceVoice.some((asked) => (asked as { apply: string }).apply !== "")).toBe(false);
  });
  it("says in the face row when the face's voice was put aside, and the voice that goes on", async () => {
    mocks.config = { face: { pack: "portrait-man" }, voice: { tts_engine: "pocket_tts", tts_voice: "michael" } };
    mocks.plans = { "": { status: "ok", switched: false, voice: "michael", action: "same", say: "" } };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    mocks.handler?.("voice_status", {
      status: "voice_put_aside",
      voice: "michael",
      speaking: "alba",
      message: "The voice michael made no sound, so I'll keep speaking with alba.",
    });
    expect(await screen.findByText("The voice michael made no sound, so I'll keep speaking with alba.")).toBeTruthy();
  });
});

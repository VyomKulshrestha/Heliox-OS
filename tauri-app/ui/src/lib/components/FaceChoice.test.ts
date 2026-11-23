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
  onNotification: vi.fn(),
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
    expect(mocks.faceVoice).toEqual([]);
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
    expect(mocks.faceVoice).toContainEqual({ voice: "am_michael", hint: "masculine", apply: "accept" });
    expect(screen.queryByRole("button", { name: /Use am_michael/ })).toBeNull();
  });

  it("says a voice not on the computer can be added, and switches nothing by itself", async () => {
    mocks.config = { face: { pack: "off" }, voice: { tts_engine: "kokoro_tts", tts_voice: "af_heart" } };
    const missing = {
      status: "ok",
      switched: false,
      voice: "am_michael",
      action: "fetch",
      say: "This face's voice, am_michael, is not on this computer yet; using it fetches about half a megabyte the first time it speaks.",
    };
    mocks.plans = { "": missing };
    await settings.reloadFromDaemon();
    await chooseMan();
    await screen.findByRole("button", { name: "Use am_michael" });
    expect(screen.getByTestId("face-voice").textContent).toMatch(/not on this computer yet/);
    expect(mocks.faceVoice.some((asked) => (asked as { apply: string }).apply === "accept")).toBe(false);
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
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  sent: [] as unknown[],
  reply: "ok" as "ok" | "refuse" | "never",
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      if (mocks.reply === "never") return new Promise(() => {});
      if (mocks.reply === "refuse") return { status: "error", message: 'face.pack must be "off" or a face pack id' };
      return { status: "ok" };
    }
    if (method === "get_config") return mocks.config;
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

describe("Face (Settings, Appearance)", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    mocks.reply = "ok";
    serve({ "faces/packs.json": { packs: ["placeholder", "Bad/Id"] }, "faces/placeholder/manifest.json": MANIFEST });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("is off by default and offers the installed packs by their names", async () => {
    mocks.config = { face: {} };
    await settings.reloadFromDaemon();
    render(FaceChoice);
    const select = screen.getByRole("combobox", { name: "Face" }) as HTMLSelectElement;
    expect(select.value).toBe("off");
    await waitFor(() => expect(screen.getByRole("option", { name: "Line drawing" })).toBeTruthy());
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["Off", "Line drawing"]);
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
});

import { describe, expect, it } from "vitest";
import { imageForShape, installedPacks, manifestProblems, stepsOf, validPackId, type FaceManifest } from "./pack";
import placeholder from "../../../public/faces/placeholder/manifest.json";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** Every image file a manifest names. */
function filesNamed(manifest: Record<string, any>): string[] {
  const eyes = manifest.eyes ?? {};
  return [
    ...manifest.idle.frames,
    ...Object.values(manifest.mouth.shapes as Record<string, string>),
    ...[eyes.open, eyes.listening, eyes.thinking, ...(eyes.blink ?? [])],
  ].filter((file): file is string => typeof file === "string");
}

const GOOD = {
  format: 1,
  id: "ana",
  name: "Ana",
  credit: "Made for Heliox. CC BY 4.0.",
  size: [512, 512],
  idle: {
    frames: ["idle-00.webp", "idle-01.webp"],
    mouth_offsets: [
      [0, 0],
      [1, -1],
    ],
  },
  mouth: { rect: [176, 300, 160, 112], shapes: { sil: "mouth-sil.webp", aa: "mouth-aa.webp", O: "mouth-O.webp" } },
};

describe("face packs", () => {
  it("accepts the placeholder that ships, and a realistic pack's manifest", () => {
    expect(manifestProblems(placeholder, "placeholder")).toEqual([]);
    expect(manifestProblems(GOOD, "ana")).toEqual([]);
  });

  it("ships every listed pack whole: a valid manifest, every file it names, and terms beside a photograph", () => {
    const faces = join(process.cwd(), "public", "faces"); // vitest runs in tauri-app/ui
    const listed = JSON.parse(readFileSync(join(faces, "packs.json"), "utf-8")).packs as string[];
    expect(listed).toContain("placeholder");
    expect(listed.length).toBeGreaterThan(1);
    for (const id of listed) {
      const manifest = JSON.parse(readFileSync(join(faces, id, "manifest.json"), "utf-8"));
      expect(manifestProblems(manifest, id), id).toEqual([]);
      for (const file of filesNamed(manifest)) expect(existsSync(join(faces, id, file)), `${id}/${file}`).toBe(true);
      if (manifest.kind === "photo") {
        const readme = readFileSync(join(faces, id, "README.md"), "utf-8");
        expect(readme, `${id}/README.md`).toMatch(/under the project's own licence, the MIT License/);
      }
    }
  });

  it("names what is wrong with a manifest that cannot be played", () => {
    expect(manifestProblems(null, "ana")).toEqual(["the manifest is not a JSON object"]);
    expect(manifestProblems({ ...GOOD, id: "bob" }, "ana")).toContain('id must be "ana", the folder\'s name');
    const noClosedMouth = { ...GOOD, mouth: { shapes: { aa: "mouth-aa.webp" } } };
    expect(manifestProblems(noClosedMouth, "ana")).toContain("mouth.shapes.sil (the closed mouth) is required");
    const escape = { ...GOOD, idle: { frames: ["../../secret.png"] } };
    expect(manifestProblems(escape, "ana")).toContain("idle.frames: one or more image files");
    const offsets = {
      ...GOOD,
      idle: {
        frames: ["idle-00.webp"],
        mouth_offsets: [
          [0, 0],
          [1, 1],
        ],
      },
    };
    expect(manifestProblems(offsets, "ana")).toContain("idle.mouth_offsets: one [dx, dy] per idle frame");
    expect(manifestProblems({ ...GOOD, tint: { idle: "red" } }, "ana")[0]).toMatch(/^tint/);
  });

  it("takes in-between pictures for a shape the pack has, from nearly closed to nearly full", () => {
    const stepped = { ...GOOD, mouth_steps: { aa: ["aa-1.webp", "aa-2.webp"] } };
    expect(manifestProblems(stepped, "ana")).toEqual([]);
    const pack = stepped as unknown as FaceManifest;
    expect(stepsOf(pack, "aa")).toEqual(["aa-1.webp", "aa-2.webp", "mouth-aa.webp"]);
    expect(stepsOf(pack, "O")).toEqual(["mouth-O.webp"]); // no steps: the full picture alone
    expect(stepsOf(pack, "E")).toEqual(["aa-1.webp", "aa-2.webp", "mouth-aa.webp"]); // E is drawn as aa, steps too
    expect(stepsOf(pack, "sil")).toEqual([]);
    expect(manifestProblems({ ...GOOD, mouth_steps: { U: ["u-1.webp"] } }, "ana")).toContain(
      "mouth_steps.U: the shape has no picture",
    );
    expect(manifestProblems({ ...GOOD, mouth_steps: { aa: [] } }, "ana")).toContain(
      "mouth_steps.aa: 1 to 8 image files",
    );
    expect(manifestProblems({ ...GOOD, mouth_steps: { aa: ["../x.png"] } }, "ana")[0]).toMatch(/^mouth_steps.aa/);
  });

  it("draws a missing shape with the nearest one the pack has, then E, then the closed mouth", () => {
    const pack = GOOD as unknown as FaceManifest;
    expect(imageForShape(pack, "aa")).toBe("mouth-aa.webp");
    expect(imageForShape(pack, "U")).toBe("mouth-O.webp");
    expect(imageForShape(pack, "E")).toBe("mouth-aa.webp");
    expect(imageForShape(pack, "PP")).toBe("mouth-sil.webp");
  });

  it("only takes pack ids that are safe to put in a path", () => {
    expect(validPackId("placeholder")).toBe(true);
    expect(validPackId("woman-2")).toBe(true);
    for (const bad of ["", "off/../x", "Ana", "-a", "a".repeat(41), 3]) expect(validPackId(bad)).toBe(false);
  });

  it("lists installed packs by name, leaving out any that cannot be read", async () => {
    const files: Record<string, unknown> = {
      "faces/packs.json": { packs: ["ana", "broken", "missing", "../x"] },
      "faces/ana/manifest.json": GOOD,
      "faces/broken/manifest.json": { format: 2 },
    };
    const fetcher = async (url: string) => {
      const key = Object.keys(files).find((path) => url.endsWith(path));
      return { ok: key !== undefined, json: async () => (key ? files[key] : null) };
    };
    expect(await installedPacks(fetcher)).toEqual([{ id: "ana", name: "Ana", credit: "Made for Heliox. CC BY 4.0." }]);
    files["faces/ana/manifest.json"] = { ...GOOD, voice: { kokoro: "af_heart", hint: "feminine" } };
    expect((await installedPacks(fetcher))[0].voice).toEqual({ kokoro: "af_heart", hint: "feminine" });
  });

  it("takes how far each shape moves the chin", () => {
    const chin = (value: unknown) => ({ ...GOOD, mouth: { ...GOOD.mouth, chin: value } });
    expect(manifestProblems(chin({ aa: 10.7, O: 4.9, PP: -2.8 }), "ana")).toEqual([]);
    expect(manifestProblems(chin({ jaw: 3 }), "ana")).toContain("mouth.chin: unknown shape jaw");
    expect(manifestProblems(chin({ aa: "far" }), "ana")).toContain("mouth.chin.aa: pixels, -100 to 100");
    expect(manifestProblems(chin([1]), "ana")).toContain("mouth.chin: an object");
  });

  it("takes the voice a pack names, and the portraits name theirs", () => {
    expect(manifestProblems({ ...GOOD, voice: { kokoro: "am_michael", hint: "masculine" } }, "ana")).toEqual([]);
    expect(manifestProblems({ ...GOOD, voice: { hint: "feminine" } }, "ana")).toEqual([]);
    expect(manifestProblems({ ...GOOD, voice: { kokoro: "Michael" } }, "ana")).toContain(
      "voice.kokoro: a Kokoro voice id, such as af_heart",
    );
    expect(manifestProblems({ ...GOOD, voice: { kokoro: "af_heart", hint: "deep" } }, "ana")).toContain(
      "voice.hint: feminine or masculine",
    );
    expect(manifestProblems({ ...GOOD, voice: "af_heart" }, "ana")).toContain("voice: an object");
    const faces = join(process.cwd(), "public", "faces");
    const voiceOf = (id: string) => JSON.parse(readFileSync(join(faces, id, "manifest.json"), "utf-8")).voice;
    expect(voiceOf("portrait-woman")).toEqual({ kokoro: "af_heart", hint: "feminine" });
    expect(voiceOf("portrait-man")).toEqual({ kokoro: "am_michael", hint: "masculine" });
    expect(voiceOf("placeholder")).toBeUndefined();
  });
});

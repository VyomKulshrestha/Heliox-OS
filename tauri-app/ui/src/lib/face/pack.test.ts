import { describe, expect, it } from "vitest";
import { imageForShape, installedPacks, manifestProblems, validPackId, type FaceManifest } from "./pack";
import placeholder from "../../../public/faces/placeholder/manifest.json";

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
  });
});

import { describe, expect, it } from "vitest";
import { coverScale, FaceRenderer, NOD_SHARE, overscanFor, SWAY_DEG, type FaceMode } from "./faceRenderer";
import type { MouthFrame } from "./mouthPlayer";
import { manifestProblems, type FaceManifest } from "./pack";
import placeholder from "../../../public/faces/placeholder/manifest.json";

/** A canvas whose 2D context records what is drawn: the image's name and the opacity it was drawn with. */
function fakeCanvas() {
  const drawn: { file: string; alpha: number }[] = [];
  const ctx = {
    globalAlpha: 1,
    setTransform() {},
    clearRect() {},
    translate() {},
    rotate() {},
    scale() {},
    drawImage(source: { file?: string }) {
      drawn.push({ file: source.file ?? "(pair)", alpha: ctx.globalAlpha });
    },
  };
  const canvas = { width: 512, height: 512, getContext: () => ctx } as unknown as HTMLCanvasElement;
  return { canvas, drawn };
}

function photo(over: Partial<FaceManifest> = {}): FaceManifest {
  return {
    format: 1,
    id: "photo",
    name: "Photo",
    credit: "Generated; shows nobody real.",
    kind: "photo",
    size: [512, 512],
    idle: { frames: ["idle.webp"] },
    mouth: { rect: [140, 300, 240, 120], shapes: { sil: "sil.webp", aa: "aa.webp", O: "o.webp" } },
    breathe: { amount: 0.004, seconds: 4.5 },
    states: { listening: { tilt: -2, lift: -2, mouth: "smile.webp" }, speaking: { scale: 1.04 } },
    ...over,
  };
}

const FILES = ["idle.webp", "sil.webp", "aa.webp", "o.webp", "smile.webp", "aa-1.webp", "aa-2.webp", "blink.webp"];

/** A renderer with every image of the pack "loaded" (an object carrying its file name). */
function renderer(manifest: FaceManifest) {
  const { canvas, drawn } = fakeCanvas();
  const made = new FaceRenderer(canvas, manifest, () => 0.5);
  const images = (made as unknown as { images: Map<string, unknown> }).images;
  for (const file of FILES) images.set(file, { file });
  return { made, drawn };
}

const QUIET: MouthFrame = { speaking: false, source: "none", shape: "sil", next: "sil", mix: 0, open: 0 };
const SPEAKING: MouthFrame = { speaking: true, source: "visemes", shape: "aa", next: "aa", mix: 0, open: 1 };

function drawFor(made: FaceRenderer, mode: FaceMode, from: number, ms: number, frame = QUIET) {
  for (let t = from; t <= from + ms; t += 33) made.draw(t, mode, frame);
}

describe("face renderer", () => {
  it("draws a photograph just large enough that no pose shows the card behind it", () => {
    expect(coverScale(512, 512, 0, 0, 1)).toBe(1);
    const tilted = coverScale(512, 512, 3, 0, 1);
    expect(tilted).toBeGreaterThan(1.05); // the top corners swing about 24 px about the neck
    expect(tilted).toBeLessThan(1.12);
    expect(coverScale(512, 512, 0, -3, 1)).toBeCloseTo(259 / 256, 6); // a lift uncovers the bottom edge
    const manifest = photo({ states: { listening: { tilt: -3, lift: -2 }, thinking: { tilt: 2.5, scale: 0.98 } } });
    const factor = overscanFor(manifest);
    for (const [tilt, lift, scale] of [
      [-3, -2, 1],
      [2.5, 0, 0.98],
      [0, 0, 1],
    ]) {
      // Covered with breathing at its smallest.
      expect(factor).toBeGreaterThan(coverScale(512, 512, tilt, lift, scale * 0.996));
    }
    expect(factor).toBeLessThan(1.15);
    // The portraits only lift (2 and 3 px), breathe, and nod and sway a little while speaking: a few % larger.
    const lifted = overscanFor(photo({ states: { listening: { lift: -2 }, thinking: { lift: -3 } } }));
    expect(lifted).toBeGreaterThan(1.01);
    expect(lifted).toBeLessThan(1.05);
    // The speech's own nod (down) and sway are covered too.
    expect(lifted).toBeGreaterThan(coverScale(512, 512, SWAY_DEG, NOD_SHARE * 512, 0.996));
  });

  it("gives a drawing no overscan, and takes the manifest's own when it names one", () => {
    expect(overscanFor(placeholder as unknown as FaceManifest)).toBe(1);
    expect(overscanFor(photo({ overscan: 0.06 }))).toBeCloseTo(1.06, 6);
    expect(overscanFor(photo({ overscan: 0 }))).toBe(1);
    const still = overscanFor(photo({ states: {}, breathe: undefined })); // only the speech's nod and sway to hide
    expect(still).toBeGreaterThan(1);
    expect(still).toBeLessThan(1.05);
  });

  it("eases a state's scale like its tilt and lift instead of jumping", () => {
    const { made } = renderer(photo());
    drawFor(made, "idle", 0, 900);
    made.draw(1000, "speaking", QUIET); // one frame later
    const early = made.pose().scale;
    expect(early).toBeGreaterThan(1);
    expect(early).toBeLessThan(1.04);
    drawFor(made, "speaking", 1066, 1000);
    expect(made.pose().scale).toBeCloseTo(1.04, 3);
    drawFor(made, "idle", 2100, 1000);
    expect(made.pose().scale).toBeCloseTo(1, 3);
  });

  it("shows a state's resting mouth, eased in, and the closed mouth where none is named", () => {
    const { made, drawn } = renderer(photo());
    drawFor(made, "idle", 0, 300);
    expect(made.pose().rest).toBe("sil.webp");
    drawn.length = 0;
    made.draw(400, "listening", QUIET);
    made.draw(433, "listening", QUIET);
    // Part of the way: the closed mouth under, the smile over it at less than full strength.
    const smile = drawn.filter((d) => d.file === "smile.webp").at(-1);
    expect(smile?.alpha).toBeGreaterThan(0);
    expect(smile?.alpha).toBeLessThan(1);
    drawFor(made, "listening", 466, 400);
    drawn.length = 0;
    made.draw(900, "listening", QUIET);
    expect(drawn.map((d) => d.file)).toEqual(["idle.webp", "smile.webp"]);
    drawFor(made, "thinking", 933, 400);
    expect(made.pose().rest).toBe("sil.webp");
  });

  it("forms a shape through its in-between pictures, blending only neighbours", () => {
    const { made, drawn } = renderer(photo({ mouth_steps: { aa: ["aa-1.webp", "aa-2.webp"] } }));
    drawFor(made, "speaking", 0, 300);
    const at = (weight: number) => {
      drawn.length = 0;
      made.draw(400, "speaking", { ...SPEAKING, weight });
      return drawn.slice(2); // after the idle frame and the resting mouth
    };
    expect(at(1)).toEqual([{ file: "aa.webp", alpha: 1 }]);
    // Half way of three steps (rest, aa-1, aa-2, aa): between aa-1 and aa-2.
    expect(at(0.5)).toEqual([
      { file: "aa-1.webp", alpha: 1 },
      { file: "aa-2.webp", alpha: 0.5 },
    ]);
    expect(at(0.2)).toEqual([{ file: "aa-1.webp", alpha: expect.closeTo(0.6, 5) }]); // rest toward aa-1
    expect(at(0)).toEqual([]);
  });

  it("without in-between pictures, lays the full shape over the rest as strongly as it is formed", () => {
    const { made, drawn } = renderer(photo());
    drawFor(made, "speaking", 0, 300);
    drawn.length = 0;
    made.draw(400, "speaking", { ...SPEAKING, weight: 0.4 });
    expect(drawn.slice(2)).toEqual([{ file: "aa.webp", alpha: expect.closeTo(0.4, 5) }]);
    drawn.length = 0;
    made.draw(433, "speaking", { ...SPEAKING, amount: 1 }); // a frame from a caller that gives no weights
    expect(drawn.at(-1)).toEqual({ file: "aa.webp", alpha: 1 });
  });

  it("waits with a blink while a word sounds, and blinks at the end of a phrase", () => {
    const eyes = {
      rect: [134, 158, 244, 114] as [number, number, number, number],
      blink: ["blink.webp"],
      blink_fps: 8,
    };
    const { made } = renderer(photo({ eyes }));
    made.draw(0, "speaking", { ...SPEAKING, pause: 0 });
    const [due] = made.blinkPlan();
    made.draw(due, "speaking", { ...SPEAKING, pause: 0 }); // mid-word: put off
    expect(made.blinkPlan()[0]).toBeGreaterThan(due);
    // The phrase ends (a pause of 400 ms begins): a blink starts within the pause.
    made.draw(due + 2000, "speaking", { ...SPEAKING, pause: 400 });
    const [start, end] = made.blinkPlan();
    expect(start).toBeGreaterThanOrEqual(due + 2000);
    expect(end).toBeLessThan(due + 2000 + 400);
  });

  it("still takes a pack written to the first version of the format", () => {
    expect(manifestProblems(placeholder, "placeholder")).toEqual([]);
    const old = { ...photo(), states: { listening: { tilt: -2 } }, overscan: undefined };
    expect(manifestProblems(old, "photo")).toEqual([]);
    expect(manifestProblems({ ...photo(), states: { listening: { mouth: "../x.png" } } }, "photo")).toContain(
      "states.listening.mouth: not an image file name",
    );
    expect(manifestProblems({ ...photo(), overscan: 1 }, "photo")).toContain("overscan: 0 to 0.25");
  });
});

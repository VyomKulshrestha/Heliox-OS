import { describe, expect, it } from "vitest";
import {
  CLOSURE_FROM,
  CLOSURE_TO,
  coverScale,
  dominant,
  FaceRenderer,
  NOD_SHARE,
  overscanFor,
  SWAY_DEG,
  type FaceMode,
} from "./faceRenderer";
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
    expect(lifted).toBeLessThan(1.08);
    // The speech's own nod (down) and sway are covered too.
    expect(lifted).toBeGreaterThan(coverScale(512, 512, SWAY_DEG, NOD_SHARE * 512, 0.996));
  });

  it("gives a drawing no overscan, and takes the manifest's own when it names one", () => {
    expect(overscanFor(placeholder as unknown as FaceManifest)).toBe(1);
    expect(overscanFor(photo({ overscan: 0.06 }))).toBeCloseTo(1.06, 6);
    expect(overscanFor(photo({ overscan: 0 }))).toBe(1);
    const still = overscanFor(photo({ states: {}, breathe: undefined })); // only the speech's nod and sway to hide
    expect(still).toBeGreaterThan(1);
    expect(still).toBeLessThan(1.08);
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
    // Elsewhere one picture stays dominant: two jaws at different heights half and half look soft and doubled.
    expect(at(0.2)).toEqual([{ file: "aa-1.webp", alpha: expect.closeTo(0.784, 3) }]); // rest toward aa-1, mostly aa-1
    expect(at(0.4)).toEqual([{ file: "aa-1.webp", alpha: 1 }]); // a fifth of the way to aa-2: aa-1 alone
    expect(at(0.62)).toEqual([{ file: "aa-2.webp", alpha: 1 }]); // past three quarters of the way: aa-2 alone
    expect(at(0)).toEqual([]);
  });

  it("crosses a lip closure's pictures in a narrow band, and lets the less formed of two shapes show only as they swap", () => {
    expect(dominant(0.5)).toBe(0.5);
    expect(dominant(0.2)).toBe(0);
    expect(dominant(0.8)).toBe(1);
    expect(dominant(0.35, CLOSURE_FROM, CLOSURE_TO)).toBe(0);
    expect(dominant(0.65, CLOSURE_FROM, CLOSURE_TO)).toBe(1);
    const { made, drawn } = renderer(photo());
    drawFor(made, "speaking", 0, 300);
    const second = (mix: number) => {
      drawn.length = 0;
      made.draw(400, "speaking", { ...SPEAKING, next: "O", weight: 0.9, nextWeight: (0.9 * mix) / (1 - mix), mix });
      return drawn.filter((d) => d.file === "o.webp");
    };
    expect(second(0.15)).toEqual([]); // O hardly formed beside aa: aa alone, one jaw
    const swapping = second(0.5);
    expect(swapping).toHaveLength(1); // where they swap: half and half
    expect(swapping[0].alpha).toBeCloseTo(0.45, 5);
  });

  it("draws two shapes straight onto the face, the more formed one exactly as when it is alone", () => {
    // A document that can make canvases, as the window has: none may be made, nothing is laid back from a copy.
    const made2d: unknown[] = [];
    const saved = globalThis.document;
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: { createElement: () => made2d.push(fakeCanvas()) && fakeCanvas().canvas },
    });
    try {
      const { made, drawn } = renderer(photo({ mouth_steps: { aa: ["aa-1.webp", "aa-2.webp"] } }));
      drawFor(made, "speaking", 0, 300);
      const frame = (extra: Partial<MouthFrame>) => {
        drawn.length = 0;
        made.draw(400, "speaking", { ...SPEAKING, weight: 0.5, ...extra });
        return drawn.slice(2);
      };
      const alone = frame({});
      // O beside aa but not yet showing (mix under a fifth): the very pictures of aa alone. Each shape used to be
      // drawn on a canvas of its own and laid back, which changed the whole mouth rectangle (resampled, the rest's
      // soft edge twice) while the mouth did not: measured as a mouth still open after it had closed.
      expect(frame({ next: "O", nextWeight: 0.06, mix: 0.1 })).toEqual(alone);
      // Where they swap: aa as alone, then O over it at half strength (O has no steps: its picture as far as formed).
      expect(frame({ next: "O", nextWeight: 0.5, mix: 0.5 })).toEqual([...alone, { file: "o.webp", alpha: 0.25 }]);
      // Both nearly closed: almost nothing but the rest, however the two are mixed.
      const closing = frame({ weight: 0.04, next: "O", nextWeight: 0.04, mix: 0.5 });
      expect(closing.every((d) => d.alpha <= 0.02)).toBe(true);
      expect(made2d).toHaveLength(0);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: saved });
    }
  });

  it("draws the less formed of two shapes at the chin height of the other, where the pack gives chins", () => {
    const frame = { ...SPEAKING, next: "O" as const, weight: 0.4, nextWeight: 0.1, mix: 0.5 };
    const oOf = (manifest: FaceManifest) => {
      const { made, drawn } = renderer(manifest);
      drawFor(made, "speaking", 0, 300);
      drawn.length = 0;
      made.draw(400, "speaking", frame);
      return drawn.filter((d) => d.file === "o.webp").map((d) => d.alpha);
    };
    const shapes = { sil: "sil.webp", aa: "aa.webp", O: "o.webp" };
    // aa at 0.4 moves the chin 4 px; O reaches 4 px at 0.8 of the way: drawn there (at half, where they swap).
    expect(oOf(photo({ mouth: { rect: [140, 300, 240, 120], shapes, chin: { aa: 10, O: 5 } } }))[0]).toBeCloseTo(
      0.4,
      5,
    );
    expect(oOf(photo())[0]).toBeCloseTo(0.05, 5); // without chins: as formed
    // A chin moving the other way (pressed lips) cannot be matched: as formed.
    expect(oOf(photo({ mouth: { rect: [140, 300, 240, 120], shapes, chin: { aa: 10, O: -3 } } }))[0]).toBeCloseTo(
      0.05,
      5,
    );
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

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { renderedRects } from "./presentation";
import { ambiguityBin, cellId, computePosterior, keyboardEndpoint, type PosteriorGaze } from "./targetPosterior";
import type { Modality, RectV1 } from "./types";

interface FixtureCase {
  modality: Modality;
  candidates: RectV1[];
  expanded_index: number | null;
  scale: number;
  press_index: number | null;
  endpoint: { x: number; y: number };
  r: number | null;
  floor: number | null;
  surface: { w: number; h: number };
  gaze: PosteriorGaze | null;
  expected: {
    rendered: RectV1[];
    posterior: number[];
    ambiguity: number;
    ambiguity_gate: number;
    n_plausible: number;
    has_hit: boolean;
    hit_index: number | null;
  };
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../../daemon/tests/fixtures/coadaptation_target_posterior.json");

function loadCases(): FixtureCase[] {
  const raw = JSON.parse(readFileSync(fixturePath, "utf8")) as { cases?: FixtureCase[] } | FixtureCase[];
  return Array.isArray(raw) ? raw : (raw.cases ?? []);
}

function close(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-12 + 1e-9 * Math.abs(b);
}

const SURFACE = { w: 700, h: 500 };
const row = (n: number, w: number, gap: number, cy = 250): RectV1[] =>
  Array.from({ length: n }, (_, i) => ({ cx: 200 + i * (w + gap), cy, w, h: w }));

describe("computePosterior properties", () => {
  it("sums to one with the null term and gives the null term far endpoints", () => {
    const rects = row(4, 28, 8);
    const near = computePosterior(rects, { x: 200, y: 250 }, { modality: "mouse", r: 1, floor: 1, surface: SURFACE });
    const total = near.posterior.reduce((sum, p) => sum + p, 0) + near.pNull;
    expect(total).toBeCloseTo(1, 12);
    expect(near.hitIndex).toBe(0);
    expect(near.ambiguity).toBeLessThan(0.05);

    const far = computePosterior(rects, { x: 600, y: 60 }, { modality: "mouse", r: 1, floor: 1, surface: SURFACE });
    expect(far.pNull).toBeGreaterThan(0.99);
    expect(far.hasHit).toBe(false);
    expect(far.ambiguity).toBe(1);
    expect(far.nPlausible).toBe(1);
    expect(cellId(ambiguityBin(far.ambiguityGate, 0.25)!, far.nPlausible, far.hasHit)).toBe("b3:p1:h0");
  });

  it("raises A_hit as the endpoint moves away from the hit's centre", () => {
    const rects = row(3, 28, 2);
    let previous = -1;
    for (let dx = 0; dx <= 14; dx += 2) {
      const result = computePosterior(
        rects,
        { x: 230 + dx, y: 250 },
        { modality: "mouse", r: 1, floor: 1, surface: SURFACE },
      );
      expect(result.hitIndex).toBe(1);
      expect(result.ambiguity).toBeGreaterThanOrEqual(previous);
      previous = result.ambiguity;
    }
  });

  it("has no hit when the press landed on a gap (-1) or on another candidate, unlike no press constraint", () => {
    const rects = row(3, 28, 8);
    const endpoint = { x: 236, y: 250 };
    const free = computePosterior(rects, endpoint, { modality: "mouse", r: 1, floor: 1, surface: SURFACE });
    expect(free.hitIndex).toBe(1);
    const gap = computePosterior(rects, endpoint, {
      modality: "mouse",
      r: 1,
      floor: 1,
      surface: SURFACE,
      pressIndex: -1,
    });
    expect(gap.hasHit).toBe(false);
    expect(gap.ambiguity).toBe(1);
    expect(gap.ambiguityGate).toBe(1);
    const other = computePosterior(rects, endpoint, {
      modality: "mouse",
      r: 1,
      floor: 1,
      surface: SURFACE,
      pressIndex: 0,
    });
    expect(other.hitIndex).toBeNull();
    const same = computePosterior(rects, endpoint, {
      modality: "mouse",
      r: 1,
      floor: 1,
      surface: SURFACE,
      pressIndex: 1,
    });
    expect(same.hitIndex).toBe(1);
  });

  it("gives a click inside a wrongly expanded neighbour a high A_hit (B3.7)", () => {
    const cue = row(4, 28, 2);
    const rendered = renderedRects(cue, 1, 1.5);
    const endpoint = { x: Math.round(cue[1].cx + 20), y: 250 };
    const result = computePosterior(rendered, endpoint, { modality: "mouse", r: 1, floor: 1, surface: SURFACE });
    expect(result.hitIndex).toBe(1);
    expect(result.ambiguity).toBeGreaterThan(0.8);
    expect(Math.max(...result.posterior)).toBeGreaterThan(0.8);
  });

  it("caps the gaze weight at 2 and never lowers A_gate below A_hit without gaze", () => {
    const rects: RectV1[] = [
      { cx: 100, cy: 250, w: 28, h: 28 },
      { cx: 130, cy: 250, w: 28, h: 28 },
    ];
    const endpoint = { x: 116, y: 250 };
    const base = { modality: "mouse" as const, r: 1, floor: 1, surface: SURFACE };
    const plain = computePosterior(rects, endpoint, base);
    const gazed = computePosterior(rects, endpoint, { ...base, gaze: { region: "left", conf: 1, acc: 1 } });
    expect(gazed.ambiguityGate).toBeGreaterThanOrEqual(plain.ambiguity);
    expect(gazed.ambiguity).toBeCloseTo(plain.ambiguity, 15);
    const huge = computePosterior(rects, endpoint, { ...base, gaze: { region: "left", conf: 5, acc: 5 } });
    expect(huge.posterior).toEqual(gazed.posterior);
  });

  it("returns the fixed keyboard result whatever the endpoint, without a spread", () => {
    const rects = row(3, 28, 8);
    const result = computePosterior(
      rects,
      { x: 9999, y: -50 },
      {
        modality: "keyboard",
        r: null,
        floor: null,
        surface: SURFACE,
        pressIndex: 2,
        gaze: { region: "left", conf: 1, acc: 1 },
      },
    );
    expect(result.posterior).toEqual([0, 0, 1]);
    expect(result.ambiguity).toBe(0);
    expect(result.ambiguityGate).toBe(0);
    expect(result.nPlausible).toBe(1);
    expect(result.hitIndex).toBe(2);
    expect(result.pHit).toBe(1);
    expect(() =>
      computePosterior(rects, { x: 0, y: 0 }, { modality: "keyboard", r: null, floor: null, surface: SURFACE }),
    ).toThrow();
    expect(() =>
      computePosterior(
        rects,
        { x: 0, y: 0 },
        {
          modality: "keyboard",
          r: null,
          floor: null,
          surface: SURFACE,
          pressIndex: -1,
        },
      ),
    ).toThrow();
  });

  it("rounds keyboard endpoints with floor(c + 0.5), so .5 goes up", () => {
    expect(keyboardEndpoint({ cx: 10.5, cy: 20.5, w: 10, h: 10 })).toEqual({ x: 11, y: 21 });
    expect(keyboardEndpoint({ cx: -2.5, cy: 7.49, w: 10, h: 10 })).toEqual({ x: -2, y: 7 });
  });

  it("bins ambiguity per section 6.3", () => {
    expect(ambiguityBin(0.2, 0.25)).toBeNull();
    expect(ambiguityBin(0.25, 0.25)).toBeNull();
    expect(ambiguityBin(0.3, 0.25)).toBe("b1");
    expect(ambiguityBin(0.4, 0.25)).toBe("b1");
    expect(ambiguityBin(0.5, 0.25)).toBe("b2");
    expect(ambiguityBin(0.61, 0.25)).toBe("b3");
    expect(ambiguityBin(1, 0.25)).toBe("b3");
    expect(cellId("b2", 3, true)).toBe("b2:p3:h1");
  });

  it("ranks the top k by the with-gaze posterior", () => {
    const rects = row(4, 16, 2);
    const result = computePosterior(
      rects,
      { x: 209, y: 250 },
      { modality: "mouse", r: 1.2, floor: 1, surface: SURFACE },
    );
    const top = result.topK(3);
    expect(top).toHaveLength(3);
    for (let i = 1; i < top.length; i += 1) {
      expect(result.posterior[top[i - 1]]).toBeGreaterThanOrEqual(result.posterior[top[i]]);
    }
  });
});

describe("Python fixture parity", () => {
  const cases = loadCases();

  it("has the 44 cases, including press-on-gap, far and keyboard cases", () => {
    expect(cases).toHaveLength(44);
    expect(cases.some((c) => c.press_index === -1 && !c.expected.has_hit && c.expected.ambiguity === 1)).toBe(true);
    expect(cases.some((c) => c.modality !== "keyboard" && c.expected.posterior.every((p) => p < 0.15))).toBe(true);
    expect(cases.filter((c) => c.modality === "keyboard")).toHaveLength(4);
  });

  it.each(cases.map((c, i) => [i, c] as const))("case %i matches", (_, c) => {
    const rendered = renderedRects(c.candidates, c.expanded_index, c.scale);
    expect(rendered).toEqual(c.expected.rendered);
    const result = computePosterior(rendered, c.endpoint, {
      modality: c.modality,
      r: c.r,
      floor: c.floor,
      surface: c.surface,
      gaze: c.gaze,
      pressIndex: c.press_index,
    });
    expect(result.posterior).toHaveLength(c.expected.posterior.length);
    result.posterior.forEach((p, i) => {
      expect(close(p, c.expected.posterior[i]), `posterior[${i}] ${p} vs ${c.expected.posterior[i]}`).toBe(true);
    });
    expect(close(result.ambiguity, c.expected.ambiguity)).toBe(true);
    expect(close(result.ambiguityGate, c.expected.ambiguity_gate)).toBe(true);
    expect(result.nPlausible).toBe(c.expected.n_plausible);
    expect(result.hasHit).toBe(c.expected.has_hit);
    expect(result.hitIndex).toBe(c.expected.hit_index);
    if (c.modality === "keyboard") {
      expect(c.press_index).not.toBeNull();
      expect(keyboardEndpoint(c.candidates[c.press_index!])).toEqual(c.endpoint);
      expect(result.ambiguity).toBe(0);
      expect(result.nPlausible).toBe(1);
    }
    if (c.modality !== "keyboard" && c.expected.posterior.every((p) => p < 0.15)) {
      expect(result.nPlausible).toBe(1);
    }
  });

  it("covers keyboard centres at .5", () => {
    const halves = cases.filter(
      (c) => c.modality === "keyboard" && c.press_index !== null && c.candidates[c.press_index].cx % 1 === 0.5,
    );
    expect(halves.length).toBeGreaterThanOrEqual(1);
  });
});

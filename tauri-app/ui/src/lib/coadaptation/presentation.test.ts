import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { expandedCandidate, isActGuarded, regimePresentation, renderedRects } from "./presentation";
import { isPresentationOnly, type RectV1 } from "./types";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../../daemon/tests/fixtures/coadaptation_target_posterior.json");

const SURFACE = { w: 700, h: 500 };
const rects: RectV1[] = [
  { cx: 200, cy: 250, w: 28, h: 28 },
  { cx: 230, cy: 250, w: 28, h: 28 },
  { cx: 260, cy: 250, w: 28, h: 28 },
];

describe("regimePresentation", () => {
  it("maps every L1 regime to its scale, confirm method and chip", () => {
    expect(regimePresentation("assistance_off")).toEqual({ scale: 1, confirm: null, chip: false });
    expect(regimePresentation("act")).toEqual({ scale: 1, confirm: null, chip: true });
    expect(regimePresentation("enlarge_target:1.5").scale).toBe(1.5);
    expect(regimePresentation("enlarge_target:2.0").scale).toBe(2);
    expect(regimePresentation("confirm_selection:second_click").confirm).toEqual({
      method: "second_click",
      holdMs: 0,
    });
    expect(regimePresentation("confirm_selection:hold_400").confirm).toEqual({ method: "hold", holdMs: 400 });
    expect(regimePresentation("confirm_selection:hold_700").confirm).toEqual({ method: "hold", holdMs: 700 });
    expect(regimePresentation("wait_for_evidence").chip).toBe(false);
    expect(isActGuarded("act")).toBe(true);
    expect(isActGuarded("enlarge_target:2.0")).toBe(true);
    expect(isActGuarded("assistance_off")).toBe(false);
    expect(isActGuarded("confirm_selection:second_click")).toBe(false);
  });
});

describe("renderedRects", () => {
  it("scales only the expanded candidate about its centre", () => {
    const out = renderedRects(rects, 1, 1.5);
    expect(out[0]).toEqual(rects[0]);
    expect(out[1]).toEqual({ cx: 230, cy: 250, w: 42, h: 42 });
    expect(out[2]).toEqual(rects[2]);
    expect(renderedRects(rects, null, 2)).toEqual(rects);
  });

  it("equals the fixture's rendered rects exactly", () => {
    const raw = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      cases: Array<{
        candidates: RectV1[];
        expanded_index: number | null;
        scale: number;
        expected: { rendered: RectV1[] };
      }>;
    };
    for (const c of raw.cases) {
      expect(renderedRects(c.candidates, c.expanded_index, c.scale)).toEqual(c.expected.rendered);
    }
  });
});

describe("expandedCandidate", () => {
  it("expands the most probable candidate near the pointer and keeps it within the hysteresis", () => {
    const first = expandedCandidate({ x: 200, y: 250 }, rects, null, 1.5, 1, 1, SURFACE, false);
    expect(first).toBe(0);
    // Halfway between 0 and 1 neither leads by 0.2: the expansion stays on 0.
    expect(expandedCandidate({ x: 215, y: 250 }, rects, 0, 1.5, 1, 1, SURFACE, false)).toBe(0);
    // On the centre of 1 it clearly leads: switch.
    expect(expandedCandidate({ x: 230, y: 250 }, rects, 0, 1.5, 1, 1, SURFACE, false)).toBe(1);
  });

  it("needs the pointer within 2 * max(w, h) of the new candidate's centre", () => {
    expect(expandedCandidate({ x: 200, y: 330 }, rects, null, 1.5, 4, 1, SURFACE, false)).toBeNull();
  });

  it("does not change while frozen from press to release", () => {
    expect(expandedCandidate({ x: 260, y: 250 }, rects, 0, 1.5, 1, 1, SURFACE, true)).toBe(0);
    expect(expandedCandidate({ x: 260, y: 250 }, rects, null, 1.5, 1, 1, SURFACE, true)).toBeNull();
  });

  it("ignores gaze: its signature takes no gaze and it reads the without-gaze posterior", () => {
    expect(expandedCandidate.length).toBe(8);
    expect(expandedCandidate({ x: 260, y: 250 }, rects, null, 2, 1, 1, SURFACE, false)).toBe(2);
  });

  it("never expands without an enlarge scale", () => {
    expect(expandedCandidate({ x: 200, y: 250 }, rects, null, 1, 1, 1, SURFACE, false)).toBeNull();
  });
});

describe("isPresentationOnly", () => {
  it("is true only for presentation_only records without execution authority", () => {
    expect(isPresentationOnly({ authority: "presentation_only", execution_authority: false })).toBe(true);
    expect(isPresentationOnly({ authority: "presentation_only", execution_authority: true })).toBe(false);
    expect(isPresentationOnly({ authority: "presentation_only" })).toBe(false);
    expect(isPresentationOnly({ authority: "execute", execution_authority: false })).toBe(false);
    expect(isPresentationOnly(null)).toBe(false);
    expect(isPresentationOnly("presentation_only")).toBe(false);
  });
});

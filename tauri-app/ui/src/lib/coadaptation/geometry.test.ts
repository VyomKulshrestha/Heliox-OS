import { describe, expect, it } from "vitest";
import {
  FIT_H,
  FIT_W,
  LAYOUT_GRID,
  STUDY_WORDS,
  generateMenuLayout,
  generatePointingLayout,
  hitTest,
  homeButton,
  idBits,
  layoutSummary,
  menuButton,
  mulberry32,
  pressesMean,
  type StudyLayout,
} from "./geometry";
import { initialFocus } from "./inputAdapter";
import { renderedRects } from "./presentation";

const MIN_SURFACE = { w: 640, h: 420 };

function bounds(layout: StudyLayout) {
  const rects = [...layout.rects, layout.origin];
  return {
    left: Math.min(...rects.map((r) => r.cx - r.w / 2)),
    right: Math.max(...rects.map((r) => r.cx + r.w / 2)),
    top: Math.min(...rects.map((r) => r.cy - r.h / 2)),
    bottom: Math.max(...rects.map((r) => r.cy + r.h / 2)),
  };
}

describe("mulberry32", () => {
  it("is reproducible for a seed and differs across seeds", () => {
    const a = mulberry32(1234567);
    const b = mulberry32(1234567);
    const c = mulberry32(1234568);
    const first = [a(), a(), a()];
    expect([b(), b(), b()]).toEqual(first);
    expect(c()).not.toBe(first[0]);
    for (const value of first) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("layout constants", () => {
  it("equal LAYOUT_GRID of spec section 12.3", () => {
    expect(LAYOUT_GRID).toEqual({
      pointing: { w_px: [16, 28, 44], gap_px: [2, 8], d_px: [120, 240, 360], grids: ["4x3", "6x2"] },
      menu: { n_items: [6, 12, 16], item_h_px: 24, item_w_px: 220, button_h_px: 32 },
      fit_px: [640, 420],
    });
  });

  it("has 64 distinct short words", () => {
    expect(STUDY_WORDS).toHaveLength(64);
    expect(new Set(STUDY_WORDS).size).toBe(64);
    for (const word of STUDY_WORDS) expect(word.length).toBeLessThanOrEqual(8);
  });
});

describe("generatePointingLayout", () => {
  it("reproduces a layout from the same seed", () => {
    const one = generatePointingLayout(mulberry32(42), MIN_SURFACE);
    const two = generatePointingLayout(mulberry32(42), MIN_SURFACE);
    expect(two).toEqual(one);
  });

  it("draws W, gap and D from the grid and fits 640 x 420 with the home button", () => {
    const seen = { w: new Set<number>(), gap: new Set<number>(), d: new Set<number>(), grid: new Set<string>() };
    for (let seed = 0; seed < 600; seed += 1) {
      const layout = generatePointingLayout(mulberry32(seed), MIN_SURFACE);
      seen.w.add(layout.wPx);
      seen.gap.add(layout.gapPx);
      seen.d.add(layout.dPx!);
      seen.grid.add(layout.grid!);
      expect(layout.rects).toHaveLength(12);
      const box = bounds(layout);
      expect(box.left).toBeGreaterThanOrEqual(0);
      expect(box.top).toBeGreaterThanOrEqual(0);
      expect(box.right).toBeLessThanOrEqual(FIT_W);
      expect(box.bottom).toBeLessThanOrEqual(FIT_H);
      const target = layout.rects[layout.intendedIndex];
      const home = homeButton(MIN_SURFACE);
      expect(Math.abs(Math.hypot(target.cx - home.cx, target.cy - home.cy) - layout.dPx!)).toBeLessThan(1.5);
      // No candidate overlaps the home button.
      for (const rect of layout.rects) expect(rect.cy + rect.h / 2).toBeLessThan(home.cy - home.h / 2);
    }
    expect([...seen.w].sort((a, b) => a - b)).toEqual([16, 28, 44]);
    expect([...seen.gap].sort((a, b) => a - b)).toEqual([2, 8]);
    expect([...seen.d].sort((a, b) => a - b)).toEqual([120, 240, 360]);
    expect([...seen.grid].sort()).toEqual(["4x3", "6x2"]);
  });

  it("centres the layout on a larger surface without scaling it", () => {
    const small = generatePointingLayout(mulberry32(9), MIN_SURFACE);
    const large = generatePointingLayout(mulberry32(9), { w: 700, h: 500 });
    expect(large.rects.map((r) => r.w)).toEqual(small.rects.map((r) => r.w));
    expect(large.rects[0].cx - small.rects[0].cx).toBe(30);
    expect(large.rects[0].cy - small.rects[0].cy).toBe(40);
    expect(large.origin.cx).toBe(350);
  });
});

describe("generateMenuLayout", () => {
  it("uses N in {6, 12, 16} only and each size occurs over 1,000 seeds", () => {
    const counts = new Map<number, number>();
    for (let seed = 0; seed < 1000; seed += 1) {
      const layout = generateMenuLayout(mulberry32(seed), MIN_SURFACE);
      counts.set(layout.menuN, (counts.get(layout.menuN) ?? 0) + 1);
      expect(layout.rects).toHaveLength(layout.menuN);
      expect(layout.intendedIndex).toBeGreaterThanOrEqual(0);
      expect(layout.intendedIndex).toBeLessThan(layout.menuN);
      for (const rect of layout.rects) {
        expect(rect.h).toBe(24);
        expect(rect.w).toBe(220);
      }
      const box = bounds(layout);
      expect(box.bottom - box.top).toBeLessThanOrEqual(416);
      expect(box.top).toBeGreaterThanOrEqual(0);
      expect(box.bottom).toBeLessThanOrEqual(420);
      expect(box.left).toBeGreaterThanOrEqual(0);
      expect(box.right).toBeLessThanOrEqual(640);
    }
    expect([...counts.keys()].sort((a, b) => a - b)).toEqual([6, 12, 16]);
  });

  it("places items directly below the 32 px menu button", () => {
    const layout = generateMenuLayout(mulberry32(3), MIN_SURFACE, { n: 16 });
    const button = menuButton(MIN_SURFACE);
    expect(button.h).toBe(32);
    expect(layout.origin).toEqual(button);
    expect(layout.rects[0].cy - layout.rects[0].h / 2).toBe(button.cy + button.h / 2);
    for (let i = 1; i < layout.rects.length; i += 1) expect(layout.rects[i].cy - layout.rects[i - 1].cy).toBe(24);
    expect(layout.rects[15].cy + 12).toBe(418);
  });

  it("favours early items (Zipf) and keeps item words fixed by the label offset", () => {
    const hits = new Array(16).fill(0);
    for (let seed = 0; seed < 2000; seed += 1) {
      hits[generateMenuLayout(mulberry32(seed), MIN_SURFACE, { n: 16 }).intendedIndex] += 1;
    }
    expect(hits[0]).toBeGreaterThan(hits[15] * 4);
    const a = generateMenuLayout(mulberry32(1), MIN_SURFACE, { n: 6, labelOffset: 10 });
    const b = generateMenuLayout(mulberry32(2), MIN_SURFACE, { n: 6, labelOffset: 10 });
    expect(a.labels).toEqual(b.labels);
  });
});

describe("layoutSummary", () => {
  it("measures id_bar from the origin and never names the target", () => {
    const layout = generatePointingLayout(mulberry32(5), MIN_SURFACE);
    const summary = layoutSummary(layout);
    const expected =
      layout.rects.reduce(
        (sum, r) => sum + idBits(Math.hypot(r.cx - layout.origin.cx, r.cy - layout.origin.cy), layout.wPx),
        0,
      ) / 12;
    expect(summary.id_bar).toBeCloseTo(expected, 12);
    expect(summary.origin_x).toBe(layout.origin.cx);
    expect(summary.origin_y).toBe(layout.origin.cy);
    expect(summary.n_candidates).toBe(12);
    expect(summary.menu_n).toBe(0);
    expect(Object.keys(summary).sort()).toEqual(
      [
        "gap_px",
        "h_px",
        "id_bar",
        "menu_n",
        "n_candidates",
        "origin_x",
        "origin_y",
        "presses_mean",
        "task",
        "w_px",
      ].sort(),
    );
    const menu = layoutSummary(generateMenuLayout(mulberry32(5), MIN_SURFACE, { n: 12 }));
    expect(menu.task).toBe("menu");
    expect(menu.menu_n).toBe(12);
    expect(menu.gap_px).toBe(0);
    // Menus use (n + 1) / 2 in the daemon and idle epochs have no keyboard target: neither carries presses_mean.
    expect("presses_mean" in menu).toBe(false);
    const idle = layoutSummary({ ...generatePointingLayout(mulberry32(5), MIN_SURFACE), task: "idle" });
    expect("presses_mean" in idle).toBe(false);
  });

  it("gives a pointing layout the mean keyboard presses from the candidate keyboard focus starts on", () => {
    for (let seed = 0; seed < 60; seed += 1) {
      const layout = generatePointingLayout(mulberry32(seed), { w: 900, h: 640 });
      const cols = layout.grid === "4x3" ? 4 : 6;
      const start = initialFocus(layout.rects, { x: layout.origin.cx, y: layout.origin.cy }, "pointing") ?? 0;
      const moves = layout.rects.map(
        (_, i) => Math.abs((i % cols) - (start % cols)) + Math.abs(Math.floor(i / cols) - Math.floor(start / cols)),
      );
      const expected = moves.reduce((sum, m) => sum + m, 0) / moves.length + 1;
      expect(layoutSummary(layout).presses_mean, `seed ${seed}`).toBeCloseTo(expected, 12);
    }
  });

  it("starts presses_mean at the nearest candidate, the lower index on ties, with columns and rows from the centres", () => {
    const rect = (cx: number, cy: number) => ({ cx, cy, w: 8, h: 8 });
    // Candidates 3 and 4 are equally near the origin: candidate 3, cell (0, 1), is the start.
    const rects = [rect(0, 0), rect(10, 0), rect(20, 0), rect(0, 10), rect(10, 10)];
    expect(initialFocus(rects, { x: 5, y: 20 }, "pointing")).toBe(3);
    expect(pressesMean(rects, { x: 5, y: 20 })).toBeCloseTo(1 + 7 / 5, 12);
    // From cell (1, 1) the mean would be 1 + 6 / 5; nearer to candidate 4 it is.
    expect(pressesMean(rects, { x: 6, y: 20 })).toBeCloseTo(1 + 6 / 5, 12);
    expect(pressesMean([], { x: 0, y: 0 })).toBeNull();
    const layout: StudyLayout = {
      task: "pointing",
      rects,
      labels: rects.map(() => "x"),
      wPx: 8,
      hPx: 8,
      gapPx: 2,
      grid: null,
      dPx: null,
      menuN: 0,
      intendedIndex: 0,
      origin: { cx: 5, cy: 20, w: 64, h: 32 },
    };
    expect(layoutSummary(layout).presses_mean).toBeCloseTo(2.4, 12);
  });

  it("computes index of difficulty in bits", () => {
    expect(idBits(240, 16)).toBeCloseTo(4, 12);
    expect(idBits(0, 16)).toBe(0);
  });
});

describe("hitTest", () => {
  const rects = [
    { cx: 100, cy: 100, w: 28, h: 28 },
    { cx: 130, cy: 100, w: 28, h: 28 },
  ];

  it("uses inclusive bounds and returns null outside every rect", () => {
    expect(hitTest(rects, { x: 114, y: 100 })).toBe(0);
    expect(hitTest(rects, { x: 115, y: 100 })).toBeNull();
    expect(hitTest(rects, { x: 116, y: 114 })).toBe(1);
  });

  it("lets an enlarged rect win where it covers a neighbour", () => {
    const enlarged = renderedRects(rects, 0, 1.5);
    expect(hitTest(enlarged, { x: 118, y: 100 })).toBe(0);
    expect(hitTest(enlarged, { x: 118, y: 100 }, 0)).toBe(0);
    expect(hitTest(enlarged, { x: 125, y: 100 })).toBe(1);
  });
});

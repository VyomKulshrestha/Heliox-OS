import type { LayoutSummaryV1, RectV1, TaskKind } from "./types";

/**
 * Seeded study layouts (spec section 12.4). Every value comes from LAYOUT_GRID (section 12.3), every layout fits
 * the 640x420 minimum surface, and a larger surface centres the layout without scaling it. Pure: no DOM access.
 */

/** Section 12.3 `LAYOUT_GRID`; the daemon hashes the same values into the study protocol. */
export const LAYOUT_GRID = {
  pointing: { w_px: [16, 28, 44], gap_px: [2, 8], d_px: [120, 240, 360], grids: ["4x3", "6x2"] },
  menu: { n_items: [6, 12, 16], item_h_px: 24, item_w_px: 220, button_h_px: 32 },
  fit_px: [640, 420],
} as const;

export const FIT_W = LAYOUT_GRID.fit_px[0];
export const FIT_H = LAYOUT_GRID.fit_px[1];
export const HOME_W = 64;
export const HOME_H = 32;
/** Margin kept free around the grid inside the 640x420 frame. */
const MARGIN = 8;
/** Gap between the bottom of the frame and the home button. */
const HOME_BOTTOM = 8;
/** Distance of the menu button from the top of the frame: (420 - 32 - 16 * 24) / 2. */
const MENU_TOP = 2;

/** Labels for candidates; shown in the cue and on menu items, never stored. */
export const STUDY_WORDS: readonly string[] = [
  "apple",
  "river",
  "chair",
  "cloud",
  "stone",
  "bread",
  "light",
  "paper",
  "glass",
  "plant",
  "music",
  "ocean",
  "green",
  "tiger",
  "horse",
  "piano",
  "lemon",
  "smile",
  "house",
  "train",
  "water",
  "sugar",
  "table",
  "field",
  "beach",
  "candle",
  "forest",
  "garden",
  "window",
  "bottle",
  "pencil",
  "rocket",
  "camera",
  "button",
  "mirror",
  "basket",
  "ladder",
  "carpet",
  "pocket",
  "wallet",
  "jacket",
  "kettle",
  "island",
  "planet",
  "castle",
  "bridge",
  "valley",
  "desert",
  "meadow",
  "harbor",
  "orange",
  "purple",
  "silver",
  "yellow",
  "cotton",
  "marble",
  "copper",
  "velvet",
  "pepper",
  "cherry",
  "banana",
  "tomato",
  "carrot",
  "onion",
];

export type Rng = () => number;

export interface Surface {
  w: number;
  h: number;
}

export interface Point {
  x: number;
  y: number;
}

export type PointingGrid = "4x3" | "6x2";

/** One cue-time layout. `rects` are surface CSS px; `origin` is the home button or the menu button. */
export interface StudyLayout {
  task: TaskKind;
  rects: RectV1[];
  labels: string[];
  wPx: number;
  hPx: number;
  gapPx: number;
  grid: PointingGrid | null;
  dPx: number | null;
  menuN: number;
  intendedIndex: number;
  origin: RectV1;
}

export interface PointingParams {
  wPx?: number;
  gapPx?: number;
  dPx?: number;
  grid?: PointingGrid;
  intendedIndex?: number;
}

export interface MenuParams {
  n?: number;
  intendedIndex?: number;
  /** Index into STUDY_WORDS of item 0, so an item keeps its word for the whole session. */
  labelOffset?: number;
}

/** mulberry32 PRNG: deterministic floats in [0, 1) from a 32-bit seed. */
export function mulberry32(seed: number): Rng {
  let a = Math.trunc(seed) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick<T>(rng: Rng, values: readonly T[]): T {
  return values[Math.min(values.length - 1, Math.floor(rng() * values.length))];
}

/** Top-left offset that centres the 640x420 frame in a larger surface (never scaled). */
export function frameOffset(surface: Surface): Point {
  return {
    x: Math.max(0, Math.floor((surface.w - FIT_W) / 2)),
    y: Math.max(0, Math.floor((surface.h - FIT_H) / 2)),
  };
}

/** Home button (64x32) at the bottom centre of the frame; pointing trials and idle epochs start from it. */
export function homeButton(surface: Surface): RectV1 {
  const offset = frameOffset(surface);
  return { cx: offset.x + FIT_W / 2, cy: offset.y + FIT_H - HOME_BOTTOM - HOME_H / 2, w: HOME_W, h: HOME_H };
}

/** Menu button (220x32) near the top of the frame; the menu opens directly below it. */
export function menuButton(surface: Surface): RectV1 {
  const offset = frameOffset(surface);
  const { item_w_px, button_h_px } = LAYOUT_GRID.menu;
  return { cx: offset.x + FIT_W / 2, cy: offset.y + MENU_TOP + button_h_px / 2, w: item_w_px, h: button_h_px };
}

/**
 * A 4x3 or 6x2 grid of 12 square candidates. The cued candidate lies at distance D from the home button centre
 * (up to the integer rounding of the grid position) and the whole grid stays inside the frame above the home button.
 */
export function generatePointingLayout(rng: Rng, surface: Surface, params: PointingParams = {}): StudyLayout {
  const w = params.wPx ?? pick(rng, LAYOUT_GRID.pointing.w_px);
  const gap = params.gapPx ?? pick(rng, LAYOUT_GRID.pointing.gap_px);
  const d = params.dPx ?? pick(rng, LAYOUT_GRID.pointing.d_px);
  const grid = params.grid ?? (pick(rng, LAYOUT_GRID.pointing.grids) as PointingGrid);
  const firstTarget = params.intendedIndex ?? Math.floor(rng() * 12);
  const firstAngle = rng() * Math.PI;

  const cols = grid === "4x3" ? 4 : 6;
  const rows = grid === "4x3" ? 3 : 2;
  const count = cols * rows;
  const pitch = w + gap;
  const gridW = cols * w + (cols - 1) * gap;
  const gridH = rows * w + (rows - 1) * gap;
  const home = homeButton({ w: FIT_W, h: FIT_H });
  const maxBottom = home.cy - HOME_H / 2 - MARGIN;

  let placement: { left: number; top: number; target: number } | null = null;
  const targets = params.intendedIndex === undefined ? count : 1;
  const steps = 72;
  search: for (let k = 0; k < targets; k += 1) {
    const target = (firstTarget + k) % count;
    const offsetX = (target % cols) * pitch + w / 2;
    const offsetY = Math.floor(target / cols) * pitch + w / 2;
    for (let s = 0; s < steps; s += 1) {
      // Angles above the home button only (0 to pi), walked from a seeded start.
      const theta = (firstAngle + (s * Math.PI) / steps) % Math.PI;
      const left = Math.round(home.cx + d * Math.cos(theta) - offsetX);
      const top = Math.round(home.cy - d * Math.sin(theta) - offsetY);
      if (left >= MARGIN && left + gridW <= FIT_W - MARGIN && top >= MARGIN && top + gridH <= maxBottom) {
        placement = { left, top, target };
        break search;
      }
    }
  }
  if (!placement) {
    // No angle reaches D for this target: centre the grid above the home button (D then differs from the plan).
    placement = {
      left: Math.round((FIT_W - gridW) / 2),
      top: Math.max(MARGIN, Math.round(maxBottom - gridH)),
      target: firstTarget % count,
    };
  }

  const offset = frameOffset(surface);
  const rects: RectV1[] = [];
  for (let i = 0; i < count; i += 1) {
    rects.push({
      cx: offset.x + placement.left + (i % cols) * pitch + w / 2,
      cy: offset.y + placement.top + Math.floor(i / cols) * pitch + w / 2,
      w,
      h: w,
    });
  }
  const wordStart = Math.floor(rng() * STUDY_WORDS.length);
  const labels = rects.map((_, i) => STUDY_WORDS[(wordStart + i) % STUDY_WORDS.length]);
  return {
    task: "pointing",
    rects,
    labels,
    wPx: w,
    hPx: w,
    gapPx: gap,
    grid,
    dPx: d,
    menuN: 0,
    intendedIndex: placement.target,
    origin: homeButton(surface),
  };
}

/** Zipf(1) draw over item indices 0..n-1 (index 0 most frequent), so familiarity grows within a session. */
export function zipfIndex(rng: Rng, n: number): number {
  let total = 0;
  for (let i = 1; i <= n; i += 1) total += 1 / i;
  let u = rng() * total;
  for (let i = 0; i < n; i += 1) {
    u -= 1 / (i + 1);
    if (u < 0) return i;
  }
  return n - 1;
}

/** A vertical menu of N in {6, 12, 16} items of 24 px directly below a 32 px menu button. */
export function generateMenuLayout(rng: Rng, surface: Surface, params: MenuParams = {}): StudyLayout {
  const n = params.n ?? pick(rng, LAYOUT_GRID.menu.n_items);
  const intendedIndex = params.intendedIndex ?? zipfIndex(rng, n);
  const labelOffset = params.labelOffset ?? 0;
  const { item_h_px, item_w_px, button_h_px } = LAYOUT_GRID.menu;
  const button = menuButton(surface);
  const firstTop = button.cy + button_h_px / 2;
  const rects: RectV1[] = [];
  for (let i = 0; i < n; i += 1) {
    rects.push({ cx: button.cx, cy: firstTop + i * item_h_px + item_h_px / 2, w: item_w_px, h: item_h_px });
  }
  const labels = rects.map((_, i) => STUDY_WORDS[(labelOffset + i) % STUDY_WORDS.length]);
  return {
    task: "menu",
    rects,
    labels,
    wPx: item_w_px,
    hPx: item_h_px,
    gapPx: 0,
    grid: null,
    dPx: null,
    menuN: n,
    intendedIndex,
    origin: button,
  };
}

/** Index of difficulty in bits: log2(D / W + 1). */
export function idBits(d: number, w: number): number {
  return Math.log2(d / w + 1);
}

/** A centre coordinate rounded to 1e-6 px, so equal columns and rows group as in the daemon's prior (spec 7.3). */
function gridKey(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/**
 * Mean keyboard presses over a pointing grid's candidates (spec 7.3): |dcolumn| + |drow| + 1 from the start candidate,
 * the one nearest the origin with the lower index on ties (the focus inputAdapter.initialFocus gives at the cue).
 * Columns and rows are the distinct candidate centres. Null without candidates.
 */
export function pressesMean(rects: readonly RectV1[], origin: Point): number | null {
  if (rects.length === 0) return null;
  const columns = [...new Set(rects.map((rect) => gridKey(rect.cx)))].sort((a, b) => a - b);
  const rows = [...new Set(rects.map((rect) => gridKey(rect.cy)))].sort((a, b) => a - b);
  const cells = rects.map((rect) => [columns.indexOf(gridKey(rect.cx)), rows.indexOf(gridKey(rect.cy))] as const);
  let start = 0;
  let nearest = Infinity;
  rects.forEach((rect, i) => {
    const distance = Math.hypot(rect.cx - origin.x, rect.cy - origin.y);
    if (distance < nearest) {
      start = i;
      nearest = distance;
    }
  });
  const [sx, sy] = cells[start];
  const moves = cells.reduce((sum, [x, y]) => sum + Math.abs(x - sx) + Math.abs(y - sy), 0);
  return moves / rects.length + 1;
}

/**
 * Interface-level summary of a layout (never the cued target). `home` defaults to the layout's origin; id_bar is the
 * mean over candidates of log2(D_i / W_i + 1) with D_i measured from that origin and W_i the candidate's smaller side.
 * A pointing layout also carries presses_mean, measured from the same origin.
 */
export function layoutSummary(layout: StudyLayout, home: Point | RectV1 = layout.origin): LayoutSummaryV1 {
  const ox = "cx" in home ? home.cx : home.x;
  const oy = "cy" in home ? home.cy : home.y;
  let bits = 0;
  for (const rect of layout.rects) {
    bits += idBits(Math.hypot(rect.cx - ox, rect.cy - oy), Math.min(rect.w, rect.h));
  }
  const idBar = layout.rects.length > 0 ? bits / layout.rects.length : 0;
  const summary: LayoutSummaryV1 = {
    task: layout.task,
    n_candidates: layout.rects.length,
    w_px: layout.wPx,
    h_px: layout.hPx,
    gap_px: layout.gapPx,
    id_bar: Math.max(0, Math.min(12, idBar)),
    menu_n: layout.task === "menu" ? layout.menuN : 0,
    origin_x: ox,
    origin_y: oy,
  };
  const presses = layout.task === "pointing" ? pressesMean(layout.rects, { x: ox, y: oy }) : null;
  if (presses !== null) summary.presses_mean = presses;
  return summary;
}

export function rectContains(rect: RectV1, point: Point): boolean {
  return Math.abs(point.x - rect.cx) <= rect.w / 2 && Math.abs(point.y - rect.cy) <= rect.h / 2;
}

/**
 * The candidate whose rect contains the point (inclusive bounds), or null. The topmost drawn rect wins: `topIndex`
 * (the expanded candidate, drawn last) when it contains the point, else the largest containing rect, and among
 * equal sizes the later one in drawing order.
 */
export function hitTest(rects: readonly RectV1[], point: Point, topIndex: number | null = null): number | null {
  if (topIndex !== null && topIndex >= 0 && topIndex < rects.length && rectContains(rects[topIndex], point)) {
    return topIndex;
  }
  let best: number | null = null;
  let bestArea = -1;
  for (let i = 0; i < rects.length; i += 1) {
    if (!rectContains(rects[i], point)) continue;
    const area = rects[i].w * rects[i].h;
    if (area >= bestArea) {
      best = i;
      bestArea = area;
    }
  }
  return best;
}

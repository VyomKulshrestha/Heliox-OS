import type { Point, Surface } from "./geometry";
import { computePosterior } from "./targetPosterior";
import type { RectV1 } from "./types";

/**
 * How a regime is presented (spec section 4.3): target scale, confirm method and whether an AssistChip shows. The
 * study surface draws exactly the rects `renderedRects` returns, and the hit test and posterior use the same rects.
 * Pure: no DOM access.
 */

export const EXPAND_HYSTERESIS = 0.2;

export interface RegimePresentation {
  scale: 1 | 1.5 | 2;
  confirm: null | { method: "second_click" | "hold"; holdMs: number };
  chip: boolean;
}

export function regimePresentation(variant: string | null | undefined): RegimePresentation {
  switch (variant) {
    case "act":
      return { scale: 1, confirm: null, chip: true };
    case "enlarge_target:1.5":
      return { scale: 1.5, confirm: null, chip: true };
    case "enlarge_target:2.0":
      return { scale: 2, confirm: null, chip: true };
    case "confirm_selection:second_click":
      return { scale: 1, confirm: { method: "second_click", holdMs: 0 }, chip: true };
    case "confirm_selection:hold_400":
      return { scale: 1, confirm: { method: "hold", holdMs: 400 }, chip: true };
    case "confirm_selection:hold_700":
      return { scale: 1, confirm: { method: "hold", holdMs: 700 }, chip: true };
    case "ask_clarification:2":
    case "ask_clarification:3":
      return { scale: 1, confirm: null, chip: true };
    default:
      // assistance_off (the plain interface) and wait_for_evidence (a dismiss-only hint) show no AssistChip.
      return { scale: 1, confirm: null, chip: false };
  }
}

/** True for regimes whose ambiguous activations go to the L2 table (`act` and `enlarge_target:*`). */
export function isActGuarded(variant: string | null | undefined): boolean {
  return variant === "act" || variant === "enlarge_target:1.5" || variant === "enlarge_target:2.0";
}

/** Cue-time rects with the expanded candidate, if any, scaled about its own centre; the others unchanged. */
export function renderedRects(rects: readonly RectV1[], expandedIndex: number | null, scale: number): RectV1[] {
  return rects.map((rect, i) =>
    i === expandedIndex ? { cx: rect.cx, cy: rect.cy, w: rect.w * scale, h: rect.h * scale } : { ...rect },
  );
}

/**
 * The candidate to draw enlarged for a pointer position (expand-near-pointer, section 4.3). Uses the without-gaze
 * posterior over the cue-time rects, so gaze never chooses the expansion. Switches to another candidate only when
 * its posterior exceeds the current one's by at least 0.2 and the pointer is within 2 * max(w, h) of its centre.
 * While `frozen` (from press to release) the current expansion is returned unchanged.
 */
export function expandedCandidate(
  pointer: Point | null,
  rects: readonly RectV1[],
  current: number | null,
  scale: number,
  r: number,
  floor: number,
  surface: Surface,
  frozen: boolean,
): number | null {
  if (frozen) return current;
  if (scale <= 1 || pointer === null || rects.length === 0) return null;
  const { posteriorWithoutGaze: posterior } = computePosterior(rects, pointer, {
    modality: "mouse",
    r,
    floor,
    surface,
  });
  let best = 0;
  for (let i = 1; i < posterior.length; i += 1) {
    if (posterior[i] > posterior[best]) best = i;
  }
  if (best === current) return current;
  const rect = rects[best];
  const near = Math.hypot(pointer.x - rect.cx, pointer.y - rect.cy) <= 2 * Math.max(rect.w, rect.h);
  const currentPosterior = current !== null && current < posterior.length ? posterior[current] : 0;
  if (near && posterior[best] >= currentPosterior + EXPAND_HYSTERESIS) return best;
  return current !== null && current < rects.length ? current : null;
}

import { hitTest, type Point, type Surface } from "./geometry";
import type { AmbiguityBin, GazeRegion, Modality, RectV1 } from "./types";

/**
 * Target posterior and ambiguity (spec section 6), the TypeScript twin of `core/target_posterior.py`. The daemon
 * recomputes every activation with the same integer endpoint, press index, spread pair and gaze values, so this
 * module must stay numerically identical to the Python side (parity fixture, section 6.4). Pure: no DOM access.
 */

/** Endpoint floor per modality in px (section 6.1); keyboard has no spread and no floor. */
export const FLOORS: Readonly<Record<Exclude<Modality, "keyboard">, number>> = {
  mouse: 1,
  touchpad: 1.5,
  pen: 1.5,
  touch: 3,
  hand_pointer: 6,
};
export const NULL_PRIOR = 0.02;
export const CANDIDATE_PRIOR_MASS = 0.98;
export const PLAUSIBLE_POSTERIOR = 0.15;
export const SPREAD_DIVISOR = 4.133;
export const MAX_GAZE_WEIGHT = 2;

export interface PosteriorGaze {
  region: GazeRegion;
  conf: number;
  /** Gaze-check accuracy of the session (section 6.2). */
  acc: number;
}

export interface PosteriorOptions {
  modality: Modality;
  /** Spread factor of the block's pair; null only for keyboard. */
  r: number | null;
  floor: number | null;
  surface: Surface;
  /** Base candidate prior pi_i summing to 0.98; uniform when omitted, so the cue never leaks in. */
  prior?: readonly number[] | null;
  gaze?: PosteriorGaze | null;
  /**
   * Candidate under the press point: the literal hit is none unless the release lands in it. -1 = the press hit
   * nothing (no literal hit); null or omitted = no press constraint. Keyboard: the focused candidate (required).
   */
  pressIndex?: number | null;
  /** Expanded candidate, drawn on top (optional; the largest containing rect wins without it). */
  topIndex?: number | null;
}

export interface PosteriorResult {
  /** P(i | e) per candidate with gaze when gaze is given (the null term is `pNull`). */
  posterior: number[];
  posteriorWithoutGaze: number[];
  pNull: number;
  /** A_hit without gaze: 1 - P(hit | e), or 1 without a literal hit. */
  ambiguity: number;
  /** A_gate: the larger of A_hit with and without gaze. */
  ambiguityGate: number;
  nPlausible: number;
  hasHit: boolean;
  hitIndex: number | null;
  pHit: number;
  pTop: number;
  /** Indices of the k most probable candidates (with-gaze posterior), lower index first on ties. */
  topK(k: number): number[];
}

const TWO_PI = 2 * Math.PI;

/** Compensated (Neumaier) sum, close to Python's math.fsum used on the daemon side. */
function preciseSum(values: readonly number[]): number {
  let sum = 0;
  let compensation = 0;
  for (const value of values) {
    const total = sum + value;
    compensation += Math.abs(sum) >= Math.abs(value) ? sum - total + value : value - total + sum;
    sum = total;
  }
  return sum + compensation;
}

/** Whether a candidate centre lies in the gazed region of the surface (thirds in each axis, section 6.2). */
export function inGazeRegion(rect: RectV1, region: GazeRegion, surface: Surface): boolean {
  const left = rect.cx < surface.w / 3;
  const right = rect.cx > (2 * surface.w) / 3;
  const up = rect.cy < surface.h / 3;
  const down = rect.cy > (2 * surface.h) / 3;
  switch (region) {
    case "left":
      return left;
    case "right":
      return right;
    case "up":
      return up;
    case "down":
      return down;
    case "center":
      return !left && !right && !up && !down;
  }
}

function basePrior(n: number, prior: readonly number[] | null | undefined): number[] {
  if (prior === null || prior === undefined) return Array.from({ length: n }, () => CANDIDATE_PRIOR_MASS / n);
  if (prior.length !== n) throw new Error("The prior needs one entry per candidate.");
  return [...prior];
}

function gazePrior(rects: readonly RectV1[], prior: number[], gaze: PosteriorGaze, surface: Surface): number[] {
  const kappa = Math.min(MAX_GAZE_WEIGHT, 1 + gaze.acc * gaze.conf);
  const weighted = prior.map((value, i) => (inGazeRegion(rects[i], gaze.region, surface) ? value * kappa : value));
  const scale = CANDIDATE_PRIOR_MASS / preciseSum(weighted);
  return weighted.map((value) => value * scale);
}

/** Gaussian endpoint likelihood per candidate with the quadrature spread s = r * sqrt((W / 4.133)^2 + floor^2). */
function likelihoods(rects: readonly RectV1[], endpoint: Point, r: number, floor: number): number[] {
  return rects.map((rect) => {
    const sx = r * Math.sqrt((rect.w / SPREAD_DIVISOR) ** 2 + floor * floor);
    const sy = r * Math.sqrt((rect.h / SPREAD_DIVISOR) ** 2 + floor * floor);
    const dx = endpoint.x - rect.cx;
    const dy = endpoint.y - rect.cy;
    return Math.exp(-0.5 * ((dx * dx) / (sx * sx) + (dy * dy) / (sy * sy))) / (TWO_PI * sx * sy);
  });
}

function normalisedPosterior(
  prior: readonly number[],
  likes: readonly number[],
  nullTerm: number,
): { posterior: number[]; pNull: number } {
  const weighted = prior.map((p, i) => p * likes[i]);
  const total = nullTerm + preciseSum(weighted);
  return { posterior: weighted.map((w) => w / total), pNull: nullTerm / total };
}

function plausibleCount(posterior: readonly number[]): number {
  const count = posterior.filter((p) => p > PLAUSIBLE_POSTERIOR).length;
  return Math.max(1, Math.min(3, count));
}

function rankedIndices(posterior: readonly number[]): number[] {
  return posterior
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p || a.i - b.i)
    .map((entry) => entry.i);
}

function keyboardResult(n: number, pressIndex: number | null | undefined): PosteriorResult {
  if (pressIndex === null || pressIndex === undefined || !Number.isInteger(pressIndex)) {
    throw new Error("A keyboard activation needs the focused candidate as pressIndex.");
  }
  if (pressIndex < 0 || pressIndex >= n) {
    throw new Error("A keyboard activation's pressIndex must name a candidate.");
  }
  const posterior = Array.from({ length: n }, (_, i) => (i === pressIndex ? 1 : 0));
  return {
    posterior,
    posteriorWithoutGaze: [...posterior],
    pNull: 0,
    ambiguity: 0,
    ambiguityGate: 0,
    nPlausible: 1,
    hasHit: true,
    hitIndex: pressIndex,
    pHit: 1,
    pTop: 1,
    topK: (k: number) => (k > 0 ? [pressIndex] : []),
  };
}

/**
 * Posterior over the rendered candidate rects for an integer endpoint (section 6.1). Keyboard activations return the
 * fixed result (one-hot on the focused candidate, A_hit = A_gate = 0, one plausible candidate) without reading the
 * endpoint, the spread or gaze.
 */
export function computePosterior(
  candidates: readonly RectV1[],
  endpoint: Point,
  options: PosteriorOptions,
): PosteriorResult {
  const n = candidates.length;
  if (n === 0) throw new Error("A posterior needs at least one candidate.");
  if (options.modality === "keyboard") return keyboardResult(n, options.pressIndex);
  if (options.r === null || options.floor === null) {
    throw new Error("A pointer activation needs the spread pair (r, floor).");
  }
  const pressIndex = options.pressIndex;
  if (pressIndex !== null && pressIndex !== undefined && !(pressIndex >= -1 && pressIndex < n)) {
    throw new Error("pressIndex lies outside the candidates.");
  }

  const prior = basePrior(n, options.prior);
  const likes = likelihoods(candidates, endpoint, options.r, options.floor);
  const nullTerm = NULL_PRIOR / (options.surface.w * options.surface.h);
  const plain = normalisedPosterior(prior, likes, nullTerm);
  const withGaze = options.gaze
    ? normalisedPosterior(gazePrior(candidates, prior, options.gaze, options.surface), likes, nullTerm)
    : plain;

  const literal = hitTest(candidates, endpoint, options.topIndex ?? null);
  const hitIndex = pressIndex === null || pressIndex === undefined || pressIndex === literal ? literal : null;
  const hasHit = hitIndex !== null;
  const ambiguity = hasHit ? 1 - plain.posterior[hitIndex] : 1;
  const ambiguityWithGaze = hasHit ? 1 - withGaze.posterior[hitIndex] : 1;
  const ambiguityGate = Math.max(ambiguity, ambiguityWithGaze);
  const posterior = withGaze.posterior;
  const ranked = rankedIndices(posterior);

  return {
    posterior,
    posteriorWithoutGaze: plain.posterior,
    pNull: withGaze.pNull,
    ambiguity,
    ambiguityGate,
    nPlausible: plausibleCount(posterior),
    hasHit,
    hitIndex,
    pHit: hasHit ? 1 - ambiguityGate : 0,
    pTop: posterior[ranked[0]],
    topK: (k: number) => ranked.slice(0, Math.max(0, k)),
  };
}

/** Keyboard endpoint: the focused candidate's cue-time centre rounded with floor(c + 0.5) on both axes. */
export function keyboardEndpoint(rect: RectV1): Point {
  return { x: Math.floor(rect.cx + 0.5), y: Math.floor(rect.cy + 0.5) };
}

/** Ambiguity bin of section 6.3, or null when the activation is not ambiguous (A_gate <= tau). */
export function ambiguityBin(a: number, tau: number): AmbiguityBin | null {
  if (a <= tau) return null;
  if (a <= Math.max(tau, 0.4)) return "b1";
  if (a <= Math.max(tau, 0.6)) return "b2";
  return "b3";
}

export function cellId(bin: AmbiguityBin, nPlausible: number, hasHit: boolean): string {
  return `${bin}:p${nPlausible}:h${hasHit ? 1 : 0}`;
}

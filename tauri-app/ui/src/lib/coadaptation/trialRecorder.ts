import type { TrialActivation, TrialState } from "./commitRules";
import type {
  ActivationV1,
  FallbackReason,
  GazeSummaryV1,
  HandSummaryV1,
  InterruptionReason,
  L1Variant,
  LayoutSummaryV1,
  Modality,
  MovementSummaryV1,
  OfferResponseV1,
  RectV1,
  SpreadV1,
  TaskKind,
  TrialV1,
  VetoV1,
} from "./types";

/**
 * Builds the TrialV1 record of one trial (spec sections 12.7 and 14.4). Raw pointer samples stay inside the recorder
 * while the trial runs; only the movement summary leaves it, and `buildTrial` and `discard` drop the samples.
 * Pure: no DOM access.
 */

/** Movement smaller than this (px) from the first sample is not yet a movement. */
const MOVE_THRESHOLD_PX = 2;
/** Samples kept per trial at most (a 15 s trial at 240 Hz stays below it). */
const MAX_SAMPLES = 6000;

interface Sample {
  x: number;
  y: number;
  t: number;
}

export interface BuildTrialInput {
  trialId: string;
  blockIndex: number;
  trialIndex: number;
  task: TaskKind;
  /** The block's recommended L1 regime (the preceding block's for an idle epoch). */
  regime: L1Variant;
  regimeDecisionId: string;
  tableDecisionIds: string[];
  /** The regime applied at cue: `regime`, or fallbacks.l1 after an L1 fallback. */
  appliedRegime: L1Variant;
  l1FallbackReason: FallbackReason | null;
  spread: SpreadV1 | null;
  layout: LayoutSummaryV1;
  candidates: RectV1[];
  surface: { w: number; h: number };
  dpr: number;
  reversible: boolean[];
  cueAt: Date | string;
  intendedIndex: number | null;
  modality: Modality;
  state: TrialState;
  timeMs: number;
  limitMs: 10000 | 15000 | 30000;
  durationMs: number;
  offers?: OfferResponseV1[];
  hand?: HandSummaryV1 | null;
  gaze?: GazeSummaryV1 | null;
  explanationViews?: number;
  /** Overrides the reducer's interruption (for example a trial excluded by a pause). */
  interrupted?: boolean;
  interruptionReason?: InterruptionReason | null;
  veto?: VetoV1 | null;
}

function clamp(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

function inside(rect: RectV1, x: number, y: number): boolean {
  return Math.abs(x - rect.cx) <= rect.w / 2 && Math.abs(y - rect.cy) <= rect.h / 2;
}

function toActivationV1(activation: TrialActivation): ActivationV1 {
  const hitIndex = activation.hitIndex;
  return {
    index: activation.index,
    t_ms: clamp(activation.tMs, 0, 60000),
    modality: activation.modality,
    x: Math.round(activation.x),
    y: Math.round(activation.y),
    hit_index: hitIndex,
    press_index: Number.isInteger(activation.pressIndex) ? activation.pressIndex : -1,
    a_gate: clamp(activation.aGate, 0, 1),
    n_plausible: Math.round(clamp(activation.nPlausible, 1, 3)),
    expanded_index: activation.expandedIndex,
    hit_rect: hitIndex === null ? null : activation.hitRect,
    gaze: activation.gaze,
    l2: activation.l2
      ? {
          decision_id: activation.l2.decisionId,
          cell_id: activation.l2.cellId,
          draw_index: activation.l2.drawIndex,
          variant: activation.l2.variant,
          source: activation.l2.source,
          fallback_reason: activation.l2.source === "fallback" ? activation.l2.fallbackReason : null,
        }
      : null,
    irreversible_rule: activation.irreversibleRule,
    result: activation.result,
    outcome: activation.outcome,
    committed_index: activation.committedIndex,
  };
}

function isoString(value: Date | string): string {
  return typeof value === "string" ? value : value.toISOString();
}

export class TrialRecorder {
  #samples: Sample[] = [];
  #cueTime: number | null = null;
  #target: RectV1 | null = null;
  #first: Sample | null = null;

  /** Starts a trial at `cueTime` (the clock of the samples); `target` is the cued candidate's cue-time rect. */
  start(cueTime: number, target: RectV1 | null = null): void {
    this.#samples = [];
    this.#cueTime = cueTime;
    this.#target = target;
    this.#first = null;
  }

  /** Keeps one pointer sample in memory for the movement summary. */
  sample(x: number, y: number, t: number): void {
    if (this.#cueTime === null || this.#samples.length >= MAX_SAMPLES) return;
    this.#samples.push({ x, y, t });
  }

  /** Marks an activation; the first one ends the approach that the movement summary describes. */
  activation(x: number, y: number, t: number): void {
    if (this.#cueTime === null || this.#first) return;
    this.#first = { x, y, t };
  }

  /** Number of samples held (for tests and diagnostics). */
  get sampleCount(): number {
    return this.#samples.length;
  }

  /** Movement summary of the first approach (section 12.7), or null without a cued target or a movement. */
  summary(): MovementSummaryV1 | null {
    const target = this.#target;
    const cueTime = this.#cueTime;
    if (!target || cueTime === null) return null;
    const end = this.#first;
    const samples = this.#samples.filter((s) => s.t >= cueTime && (!end || s.t <= end.t));
    if (end) samples.push(end);
    if (samples.length < 2) return null;

    const origin = samples[0];
    let moveIndex = -1;
    for (let i = 1; i < samples.length; i += 1) {
      if (Math.hypot(samples[i].x - origin.x, samples[i].y - origin.y) > MOVE_THRESHOLD_PX) {
        moveIndex = i;
        break;
      }
    }
    if (moveIndex < 0) return null;

    const startSample = samples[moveIndex - 1];
    const path = samples.slice(moveIndex - 1);
    const last = path[path.length - 1];
    let length = 0;
    const speeds: number[] = [];
    let entries = 0;
    let wasInside = inside(target, startSample.x, startSample.y);
    if (wasInside) entries = 1;
    for (let i = 1; i < path.length; i += 1) {
      const step = Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
      length += step;
      const dt = path[i].t - path[i - 1].t;
      if (dt > 0) speeds.push((step / dt) * 1000);
      const isInside = inside(target, path[i].x, path[i].y);
      if (isInside && !wasInside) entries += 1;
      wasInside = isInside;
    }
    const straight = Math.hypot(last.x - startSample.x, last.y - startSample.y);
    const peak = speeds.length > 0 ? Math.max(...speeds) : 0;
    let minima = 0;
    for (let i = 1; i < speeds.length - 1; i += 1) {
      if (speeds[i] < speeds[i - 1] && speeds[i] <= speeds[i + 1] && speeds[i] < 0.2 * peak) minima += 1;
    }
    const endpoint = end ?? last;
    return {
      rt_ms: clamp(samples[moveIndex].t - cueTime, 0, 60000),
      mt_ms: clamp(endpoint.t - samples[moveIndex].t, 0, 60000),
      path_ratio: clamp(straight >= 1 ? length / straight : 1, 1, 50),
      submovements: Math.round(clamp(minima, 0, 50)),
      peak_speed_px_s: clamp(peak, 0, 50000),
      reentries: Math.round(clamp(entries - 1, 0, 50)),
      endpoint_dx_norm: clamp((endpoint.x - target.cx) / target.w, -20, 20),
      endpoint_dy_norm: clamp((endpoint.y - target.cy) / target.h, -20, 20),
    };
  }

  /** The TrialV1 payload of section 14.4. Drops every sample; the result holds no raw pointer path. */
  buildTrial(input: BuildTrialInput): TrialV1 {
    const movement = input.task === "idle" ? null : this.summary();
    this.discard();
    const state = input.state;
    const interrupted = input.interrupted ?? state.phase === "interrupted";
    const committed = state.phase === "committed" ? state.committedIndex : null;
    const veto =
      input.veto ??
      (state.veto && state.veto.decisionId && state.veto.level
        ? { decision_id: state.veto.decisionId, level: state.veto.level }
        : null);
    return {
      trial_id: input.trialId,
      block_index: input.blockIndex,
      trial_index: input.trialIndex,
      task: input.task,
      regime: input.regime,
      regime_decision_id: input.regimeDecisionId,
      table_decision_ids: input.tableDecisionIds.slice(0, 4),
      applied_regime: input.appliedRegime,
      l1_fallback_reason: input.l1FallbackReason,
      fallbacks_issued_at: state.cueFallbacksIssuedAt,
      spread: input.spread,
      layout: input.layout,
      candidates: input.candidates.map((rect) => ({ ...rect })),
      surface_w: input.surface.w,
      surface_h: input.surface.h,
      dpr: input.dpr,
      reversible: [...input.reversible],
      cue_at: isoString(input.cueAt),
      intended_index: input.task === "idle" ? null : input.intendedIndex,
      modality: input.modality,
      // All of them: the reducer ends a trial at MAX_ACTIVATIONS, so the commit and revert below match this list.
      activations: state.activations.map(toActivationV1),
      revert: state.revert
        ? {
            reason: state.revert.reason,
            from_activation: state.revert.fromActivation,
            regime: state.revert.regime,
            fallbacks_issued_at: state.revert.fallbacksIssuedAt,
          }
        : null,
      committed_index: committed,
      commit_t_ms: committed === null || state.commitTMs === null ? null : clamp(state.commitTMs, 0, 60000),
      commit_via: committed === null ? "none" : state.commitVia,
      corrections: Math.round(clamp(state.corrections, 0, 50)),
      time_ms: clamp(input.timeMs, 0, 60000),
      limit_ms: input.limitMs,
      duration_ms: clamp(input.durationMs, 0, 60000),
      movement,
      veto,
      offers: (input.offers ?? []).slice(0, 2),
      hand: input.hand ?? null,
      gaze: input.gaze ?? null,
      interrupted,
      interruption_reason: interrupted ? (input.interruptionReason ?? state.interruptionReason ?? "paused") : null,
      explanation_views: Math.round(clamp(input.explanationViews ?? 0, 0, 20)),
    };
  }

  /** Drops the samples without building a record (interrupted practice, exit). */
  discard(): void {
    this.#samples = [];
    this.#first = null;
    this.#cueTime = null;
    this.#target = null;
  }
}

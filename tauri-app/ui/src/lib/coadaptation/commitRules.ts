import { isActGuarded } from "./presentation";
import { ambiguityBin, cellId } from "./targetPosterior";
import type {
  ActivationGazeV1,
  ActivationOutcome,
  ActivationResult,
  CommitVia,
  InterruptionReason,
  L1Variant,
  L2FallbackReason,
  L2Source,
  L2Variant,
  Modality,
  RectV1,
} from "./types";

/**
 * Trial flow and commit rules (spec section 12.5) as a pure reducer. Only the person's own explicit input commits:
 * an activation that hits a candidate, a fresh second input on a confirm step, a completed hold, or a choice in a
 * clarification prompt. A limit, an interrupt, a hold abort or silence never commits; `wait_for_evidence` and every
 * L2 response outside a `pin_act` collapse never commit the activation that triggered them; rule C9 gives an
 * irreversible candidate a confirm step whatever the regime. Pure: no DOM access.
 */

export type TrialPhase =
  "ready" | "cued" | "pressed" | "confirming" | "holding" | "clarifying" | "committed" | "abandoned" | "interrupted";

/** Minimum delay between a confirm chip appearing and the fresh press that confirms it. */
export const CONFIRM_MIN_MS = 250;
/**
 * Timer and clock jitter a hold completion may show against the recorded hold duration (a browser timer can fire a
 * millisecond early against performance.now(), and both times are rounded to whole ms). Far below any real release.
 */
export const HOLD_COMPLETE_SLACK_MS = 5;
export const MAX_TABLE_DRAWS = 4;
/**
 * Activations one trial may hold: TrialV1 carries at most 32 (ActivationV1.index 0..31, RevertV1.from_activation
 * 0..32, section 14.4), so every activation of a trial is sent and its commit and revert stay consistent with them.
 */
export const MAX_ACTIVATIONS = 32;

/** The L2 response the study proposes for an ambiguous activation, before the cap rule is applied. */
export interface L2Proposal {
  variant: L2Variant;
  source: Exclude<L2Source, "cap">;
  /** The table decision the response belongs to or replaced, else the block's L1 decision. */
  decisionId: string;
  /** Table draws: the draw index used (4 after the maximum). Defaults to the draws already used in the trial. */
  drawIndex?: number;
  fallbackReason?: L2FallbackReason | null;
  /** A `pin_act` collapse response (`act`, or `wait_for_evidence` without a hit): exempt from the cap rule. */
  pinAct?: boolean;
}

export interface ActivationEvent {
  type: "activation";
  x: number;
  y: number;
  modality: Modality;
  hitIndex: number | null;
  /** Candidate under the press point; -1 when the press hit nothing. Defaults to hitIndex, else -1. */
  pressIndex?: number;
  aGate: number;
  nPlausible: number;
  /** Candidates by with-gaze posterior, most probable first (the clarification prompt shows the top k). */
  plausible?: number[];
  expandedIndex?: number | null;
  hitRect?: RectV1 | null;
  gaze?: ActivationGazeV1 | null;
  /** The literal hit's `reversible` flag; defaults to the trial's per-candidate list. */
  reversible?: boolean;
  /** Shorthand for a table response; `l2` carries the full proposal. */
  l2Variant?: L2Variant;
  l2?: L2Proposal;
  /** Milliseconds since the cue. */
  tMs?: number;
}

export type TrialEvent =
  | { type: "cue"; tMs?: number }
  | { type: "press"; x: number; y: number; hitIndex: number | null; expandedIndex?: number | null; tMs?: number }
  | { type: "pressCancel" }
  | ActivationEvent
  | {
      type: "confirmInput";
      kind: "click" | "key" | "pinch";
      onIndex: number | null;
      sinceChipMs: number;
      /** False for a key repeat or an input that did not start with a new press. */
      fresh?: boolean;
      tMs?: number;
    }
  | { type: "holdRelease"; elapsedMs: number }
  | { type: "holdAbort"; reason: string }
  | { type: "holdComplete"; tMs: number }
  | { type: "cancel" }
  | { type: "choose"; index: number | null; tMs?: number }
  | { type: "undo" }
  | { type: "veto"; decisionId?: string; level?: "block_regime" | "ambiguity_response" }
  | { type: "refreshPending" }
  | { type: "refresh"; regime: L1Variant | null; interruptionCap: number | null; fallbacksIssuedAt: string }
  | { type: "limit" }
  | { type: "interrupt"; reason?: InterruptionReason };

export interface TrialL2 {
  decisionId: string;
  cellId: string;
  drawIndex: number;
  variant: L2Variant;
  source: L2Source;
  fallbackReason: L2FallbackReason | null;
}

export interface TrialActivation {
  index: number;
  tMs: number;
  modality: Modality;
  x: number;
  y: number;
  hitIndex: number | null;
  pressIndex: number;
  aGate: number;
  nPlausible: number;
  expandedIndex: number | null;
  hitRect: RectV1 | null;
  gaze: ActivationGazeV1 | null;
  l2: TrialL2 | null;
  irreversibleRule: boolean;
  /** Immediate response, set once when the activation is applied. */
  result: ActivationResult;
  /** What finally happened to that response; `open_at_end` while a step is still open. */
  outcome: ActivationOutcome;
  committedIndex: number | null;
  /** Opened by the trial's L1 confirm or hold regime (counts once per trial toward the cap). */
  l1Step: boolean;
}

export interface TrialRevert {
  reason: "veto" | "refresh";
  fromActivation: number;
  regime: L1Variant;
  fallbacksIssuedAt: string;
}

export interface TrialState {
  phase: TrialPhase;
  regime: L1Variant;
  /** The regime a "Not this time" revert applies (fallbacks.plain held at cue); null when there is none. */
  plainRegime: L1Variant | null;
  /** Issue time of the fallbacks in force; a refresh replaces it. */
  fallbacksIssuedAt: string;
  /** Issue time of the fallbacks held at cue (TrialV1.fallbacks_issued_at). */
  cueFallbacksIssuedAt: string;
  interruptionCap: number | null;
  /** The block's count k for the cap rule (section 9.3), carried from trial to trial. */
  interruptionsUsed: number;
  l1StepCounted: boolean;
  inputPaused: boolean;
  tau: number;
  reversible: boolean[];
  pressIndex: number | null;
  pendingIndex: number | null;
  /** Index in `activations` of the activation whose step is open. */
  pendingActivation: number | null;
  confirmStartedAt: number | null;
  holdStartedAt: number | null;
  /** Hold duration of the regime that opened the hold step, recorded when the phase became `holding`. */
  holdDurationMs: number | null;
  askCandidates: number[];
  /** Table draws used in this trial (0..4). */
  l2Used: number;
  /** The latest L2 response applied, for the chip or hint. */
  lastL2: TrialL2 | null;
  committedIndex: number | null;
  commitVia: CommitVia;
  commitTMs: number | null;
  corrections: number;
  revert: TrialRevert | null;
  veto: { decisionId: string | null; level: "block_regime" | "ambiguity_response" | null } | null;
  /** Set by a refresh: the block ends after this trial. */
  endBlockAfterTrial: boolean;
  interruptionReason: InterruptionReason | null;
  activations: TrialActivation[];
  events: TrialEvent[];
}

export interface TrialStateOptions {
  plainRegime: L1Variant | null;
  fallbacksIssuedAt: string;
  interruptionCap: number | null;
  interruptionsUsed: number;
  tau: number;
  reversible: boolean[];
}

export function initialTrialState(regime: L1Variant, options: TrialStateOptions): TrialState {
  return {
    phase: "ready",
    regime,
    plainRegime: options.plainRegime,
    fallbacksIssuedAt: options.fallbacksIssuedAt,
    cueFallbacksIssuedAt: options.fallbacksIssuedAt,
    interruptionCap: options.interruptionCap,
    interruptionsUsed: options.interruptionsUsed,
    l1StepCounted: false,
    inputPaused: false,
    tau: options.tau,
    reversible: [...options.reversible],
    pressIndex: null,
    pendingIndex: null,
    pendingActivation: null,
    confirmStartedAt: null,
    holdStartedAt: null,
    holdDurationMs: null,
    askCandidates: [],
    l2Used: 0,
    lastL2: null,
    committedIndex: null,
    commitVia: "none",
    commitTMs: null,
    corrections: 0,
    revert: null,
    veto: null,
    endBlockAfterTrial: false,
    interruptionReason: null,
    activations: [],
    events: [],
  };
}

/** Committed, abandoned or interrupted. A wrong commit can still be undone (`undo` returns to `cued`). */
export function isTerminal(state: TrialState): boolean {
  return state.phase === "committed" || state.phase === "abandoned" || state.phase === "interrupted";
}

/** The regime that applies to the next activation: the revert regime once the trial is reverted. */
export function effectiveRegime(state: TrialState): L1Variant {
  return state.revert?.regime ?? state.regime;
}

/** Whether the cap rule turns the next L2 response outside a `pin_act` collapse into `wait_for_evidence`. */
export function capReached(state: TrialState): boolean {
  return state.interruptionCap !== null && state.interruptionsUsed >= state.interruptionCap;
}

/** Whether an activation would be routed to an L2 response (act-guarded regime and A_gate > tau or no hit). */
export function needsL2Response(state: TrialState, activation: { aGate: number; hitIndex: number | null }): boolean {
  if (!acceptsActivation(state)) return false;
  if (!isActGuarded(effectiveRegime(state))) return false;
  return activation.hitIndex === null || activation.aGate > state.tau;
}

export function acceptsActivation(state: TrialState): boolean {
  return (
    !state.inputPaused &&
    (state.phase === "cued" || state.phase === "pressed") &&
    state.activations.length < MAX_ACTIVATIONS
  );
}

/** Every activation slot is used and no step is open or commit made: no further input can complete the trial. */
function activationsExhausted(state: TrialState): boolean {
  return state.activations.length >= MAX_ACTIVATIONS && (state.phase === "cued" || state.phase === "pressed");
}

function holdMsOf(regime: L1Variant): number | null {
  if (regime === "confirm_selection:hold_400") return 400;
  if (regime === "confirm_selection:hold_700") return 700;
  return null;
}

/** Hold duration of the regime in force, or null outside hold regimes. */
export function holdMs(state: TrialState): number | null {
  return holdMsOf(effectiveRegime(state));
}

function withEvent(state: TrialState, event: TrialEvent): TrialState {
  return { ...state, events: [...state.events, event] };
}

function updateActivation(state: TrialState, index: number | null, patch: Partial<TrialActivation>): TrialActivation[] {
  if (index === null) return state.activations;
  return state.activations.map((activation, i) => (i === index ? { ...activation, ...patch } : activation));
}

function closeStep(
  state: TrialState,
): Pick<
  TrialState,
  "pendingIndex" | "pendingActivation" | "confirmStartedAt" | "holdStartedAt" | "holdDurationMs" | "askCandidates"
> {
  return {
    pendingIndex: null,
    pendingActivation: null,
    confirmStartedAt: null,
    holdStartedAt: null,
    holdDurationMs: null,
    askCandidates: [],
  };
}

/**
 * Whether a hold completion at `tMs` comes after the hold duration recorded when the step opened. A completion
 * without a time, or with no recorded start or duration, is refused: an uncertain hold never commits.
 */
function holdElapsed(state: TrialState, tMs: number | undefined): boolean {
  if (state.holdStartedAt === null || state.holdDurationMs === null) return false;
  if (typeof tMs !== "number" || !Number.isFinite(tMs)) return false;
  return tMs - state.holdStartedAt >= state.holdDurationMs - HOLD_COMPLETE_SLACK_MS;
}

function reversibleOf(state: TrialState, index: number, flag: boolean | undefined): boolean {
  if (flag !== undefined) return flag;
  // An unknown flag is treated as irreversible: C9 then asks for a second input.
  return state.reversible[index] ?? false;
}

function proposalOf(event: ActivationEvent): L2Proposal | null {
  if (event.l2) return event.l2;
  if (event.l2Variant) {
    return { variant: event.l2Variant, source: "table", decisionId: "", pinAct: event.l2Variant === "act" };
  }
  return null;
}

function terminate(
  state: TrialState,
  phase: "abandoned" | "interrupted",
  reason: InterruptionReason | null,
): TrialState {
  return {
    ...state,
    ...closeStep(state),
    phase,
    interruptionReason: phase === "interrupted" ? reason : state.interruptionReason,
  };
}

function applyActivation(state: TrialState, event: ActivationEvent): TrialState {
  const regime = effectiveRegime(state);
  const hit = event.hitIndex;
  const tMs = event.tMs ?? 0;
  const index = state.activations.length;
  const base: TrialActivation = {
    index,
    tMs,
    modality: event.modality,
    x: Math.round(event.x),
    y: Math.round(event.y),
    hitIndex: hit,
    pressIndex: event.pressIndex ?? state.pressIndex ?? hit ?? -1,
    aGate: event.aGate,
    nPlausible: event.nPlausible,
    expandedIndex: event.expandedIndex ?? null,
    hitRect: hit === null ? null : (event.hitRect ?? null),
    gaze: event.gaze ?? null,
    l2: null,
    irreversibleRule: false,
    result: "none",
    outcome: "none",
    committedIndex: null,
    l1Step: false,
  };
  const next: TrialState = { ...state, pressIndex: null, phase: "cued" };

  const record = (activation: TrialActivation, patch: Partial<TrialState> = {}): TrialState => ({
    ...next,
    ...patch,
    activations: [...state.activations, activation],
  });

  const commitOrC9 = (activation: TrialActivation, candidate: number): TrialState => {
    if (reversibleOf(state, candidate, event.reversible)) {
      return record(
        { ...activation, result: "committed", outcome: "committed", committedIndex: candidate },
        { phase: "committed", committedIndex: candidate, commitVia: "direct", commitTMs: tMs },
      );
    }
    // Rule C9: an irreversible candidate never commits on a single activation.
    return record(
      { ...activation, irreversibleRule: true, result: "confirming", outcome: "open_at_end" },
      {
        phase: "confirming",
        pendingIndex: candidate,
        pendingActivation: index,
        confirmStartedAt: tMs,
        interruptionsUsed: next.interruptionsUsed + 1,
      },
    );
  };

  const l1Step = (phase: "confirming" | "holding"): TrialState => {
    const counted = state.l1StepCounted;
    return record(
      { ...base, result: phase === "confirming" ? "confirming" : "held", outcome: "open_at_end", l1Step: true },
      {
        phase,
        pendingIndex: hit,
        pendingActivation: index,
        confirmStartedAt: phase === "confirming" ? tMs : null,
        holdStartedAt: phase === "holding" ? tMs : null,
        holdDurationMs: phase === "holding" ? holdMsOf(regime) : null,
        interruptionsUsed: counted ? next.interruptionsUsed : next.interruptionsUsed + 1,
        l1StepCounted: true,
      },
    );
  };

  if (regime === "assistance_off") {
    return hit === null ? record(base) : commitOrC9(base, hit);
  }
  if (regime === "confirm_selection:second_click") {
    return hit === null ? record(base) : l1Step("confirming");
  }
  if (regime === "confirm_selection:hold_400" || regime === "confirm_selection:hold_700") {
    return hit === null ? record(base) : l1Step("holding");
  }
  if (!isActGuarded(regime)) return record(base);
  if (hit !== null && event.aGate <= state.tau) return commitOrC9(base, hit);

  // Ambiguous activation under act or enlarge_target: an L2 response, after the cap rule.
  const bin = ambiguityBin(hit === null ? 1 : event.aGate, state.tau) ?? "b3";
  const cell = cellId(bin, event.nPlausible, hit !== null);
  const proposal = proposalOf(event);
  const pinAct = proposal?.pinAct ?? proposal?.variant === "act";
  if (!pinAct && capReached(state)) {
    const l2: TrialL2 = {
      decisionId: proposal?.decisionId ?? "",
      cellId: cell,
      drawIndex: state.l2Used,
      variant: "wait_for_evidence",
      source: "cap",
      fallbackReason: null,
    };
    return record({ ...base, l2 }, { lastL2: l2 });
  }
  if (!proposal) {
    const l2: TrialL2 = {
      decisionId: "",
      cellId: cell,
      drawIndex: state.l2Used,
      variant: "wait_for_evidence",
      source: "fixed_rule",
      fallbackReason: null,
    };
    return record({ ...base, l2 }, { lastL2: l2 });
  }

  let l2Used = state.l2Used;
  let drawIndex = state.l2Used;
  if (proposal.source === "table") {
    drawIndex = Math.min(MAX_TABLE_DRAWS, proposal.drawIndex ?? state.l2Used);
    if (drawIndex < MAX_TABLE_DRAWS) l2Used = Math.max(l2Used, drawIndex + 1);
  }
  const l2: TrialL2 = {
    decisionId: proposal.decisionId,
    cellId: cell,
    drawIndex,
    variant: proposal.variant,
    source: proposal.source,
    fallbackReason: proposal.source === "fallback" ? (proposal.fallbackReason ?? null) : null,
  };
  const withL2 = { ...base, l2 };
  const patch = { l2Used, lastL2: l2 };

  switch (proposal.variant) {
    case "act":
      // Only the person's own pin_act commits an ambiguous literal hit; anything else commits nothing.
      if (pinAct && hit !== null) {
        const committed = commitOrC9(withL2, hit);
        return { ...committed, l2Used, lastL2: l2 };
      }
      return record(withL2, patch);
    case "confirm_selection:second_click":
      if (hit === null) return record(withL2, patch);
      return record(
        { ...withL2, result: "confirming", outcome: "open_at_end" },
        {
          ...patch,
          phase: "confirming",
          pendingIndex: hit,
          pendingActivation: index,
          confirmStartedAt: tMs,
          interruptionsUsed: next.interruptionsUsed + 1,
        },
      );
    case "ask_clarification:2":
    case "ask_clarification:3": {
      const k = proposal.variant === "ask_clarification:3" ? 3 : 2;
      return record(
        { ...withL2, result: "asked", outcome: "open_at_end" },
        {
          ...patch,
          phase: "clarifying",
          pendingIndex: null,
          pendingActivation: index,
          askCandidates: (event.plausible ?? []).slice(0, k),
          interruptionsUsed: next.interruptionsUsed + 1,
        },
      );
    }
    default:
      return record(withL2, patch);
  }
}

function commitPending(state: TrialState, candidate: number, via: CommitVia, tMs: number | undefined): TrialState {
  return {
    ...state,
    ...closeStep(state),
    phase: "committed",
    committedIndex: candidate,
    commitVia: via,
    commitTMs: tMs ?? state.commitTMs,
    activations: updateActivation(state, state.pendingActivation, { outcome: "committed", committedIndex: candidate }),
  };
}

/**
 * Applies one event. A trial whose activations reach MAX_ACTIVATIONS with nothing open or committed (after a miss,
 * a wait, a cancel or an undo) ends as abandoned at once, never with a commit.
 */
export function reduceTrial(state: TrialState, event: TrialEvent): TrialState {
  const next = reduceTrialStep(state, event);
  if (next === state || !activationsExhausted(next)) return next;
  return terminate(next, "abandoned", null);
}

function reduceTrialStep(state: TrialState, event: TrialEvent): TrialState {
  const ended = state.phase === "abandoned" || state.phase === "interrupted";
  switch (event.type) {
    case "cue":
      return state.phase === "ready" ? withEvent({ ...state, phase: "cued" }, event) : state;

    case "press":
      if (!acceptsActivation(state)) return state;
      return withEvent({ ...state, phase: "pressed", pressIndex: event.hitIndex ?? -1 }, event);

    case "pressCancel":
      return state.phase === "pressed" ? withEvent({ ...state, phase: "cued", pressIndex: null }, event) : state;

    case "activation":
      if (!acceptsActivation(state)) return state;
      return withEvent(applyActivation(state, event), event);

    case "confirmInput": {
      if (state.phase !== "confirming" || state.pendingIndex === null) return state;
      if (event.fresh === false || event.sinceChipMs < CONFIRM_MIN_MS) return state;
      if (event.onIndex !== null && event.onIndex !== state.pendingIndex) return state;
      return withEvent(commitPending(state, state.pendingIndex, "confirm", event.tMs), event);
    }

    case "cancel":
      if (state.phase !== "confirming") return state;
      return withEvent(
        {
          ...state,
          ...closeStep(state),
          phase: "cued",
          corrections: state.corrections + 1,
          activations: updateActivation(state, state.pendingActivation, { outcome: "cancelled" }),
        },
        event,
      );

    case "holdComplete":
      if (state.phase !== "holding" || state.pendingIndex === null) return state;
      // Second guard behind the input adapter's timer: no commit before the recorded hold duration has elapsed.
      if (!holdElapsed(state, event.tMs)) return state;
      return withEvent(commitPending(state, state.pendingIndex, "hold", event.tMs), event);

    case "holdRelease":
      if (state.phase !== "holding") return state;
      return withEvent(
        {
          ...state,
          ...closeStep(state),
          phase: "cued",
          corrections: state.corrections + 1,
          activations: updateActivation(state, state.pendingActivation, { outcome: "cancelled" }),
        },
        event,
      );

    case "holdAbort":
      if (state.phase !== "holding") return state;
      return withEvent(
        {
          ...state,
          ...closeStep(state),
          phase: "cued",
          activations: updateActivation(state, state.pendingActivation, { outcome: "aborted" }),
        },
        event,
      );

    case "choose": {
      if (state.phase !== "clarifying") return state;
      if (event.index === null) {
        return withEvent(
          {
            ...state,
            ...closeStep(state),
            phase: "cued",
            corrections: state.corrections + 1,
            activations: updateActivation(state, state.pendingActivation, { outcome: "cancelled" }),
          },
          event,
        );
      }
      if (!state.askCandidates.includes(event.index)) return state;
      if (reversibleOf(state, event.index, undefined)) {
        return withEvent(commitPending(state, event.index, "ask", event.tMs), event);
      }
      // Rule C9 on the chosen candidate: a confirm step before it commits.
      return withEvent(
        {
          ...state,
          phase: "confirming",
          pendingIndex: event.index,
          confirmStartedAt: event.tMs ?? state.confirmStartedAt,
          askCandidates: [],
          activations: updateActivation(state, state.pendingActivation, { irreversibleRule: true }),
        },
        event,
      );
    }

    case "undo":
      if (state.phase !== "committed") return state;
      return withEvent(
        {
          ...state,
          phase: "cued",
          committedIndex: null,
          commitVia: "none",
          commitTMs: null,
          corrections: state.corrections + 1,
        },
        event,
      );

    case "veto": {
      if (ended) return state;
      const veto = { decisionId: event.decisionId ?? null, level: event.level ?? null };
      if (state.revert !== null) return withEvent({ ...state, veto }, event);
      if (state.plainRegime === null) {
        // Nothing to revert to: the trial stops, never with a commit.
        return withEvent({ ...terminate(state, "interrupted", "no_allowed_regime"), veto }, event);
      }
      return withEvent(
        {
          ...state,
          veto,
          revert: {
            reason: "veto",
            fromActivation: state.activations.length,
            regime: state.plainRegime,
            fallbacksIssuedAt: state.fallbacksIssuedAt,
          },
        },
        event,
      );
    }

    case "refreshPending":
      return withEvent({ ...state, inputPaused: true }, event);

    case "refresh": {
      const refreshed: TrialState = {
        ...state,
        inputPaused: false,
        interruptionCap: event.interruptionCap,
        fallbacksIssuedAt: event.fallbacksIssuedAt,
        endBlockAfterTrial: true,
      };
      if (ended) return withEvent(refreshed, event);
      if (event.regime === null || state.revert !== null) {
        return withEvent(terminate(refreshed, "interrupted", "refresh"), event);
      }
      return withEvent(
        {
          ...refreshed,
          revert: {
            reason: "refresh",
            fromActivation: state.activations.length,
            regime: event.regime,
            fallbacksIssuedAt: event.fallbacksIssuedAt,
          },
        },
        event,
      );
    }

    case "limit":
      if (isTerminal(state)) return state;
      return withEvent(terminate(state, "abandoned", null), event);

    case "interrupt":
      if (ended) return state;
      return withEvent(terminate(state, "interrupted", event.reason ?? "paused"), event);
  }
}

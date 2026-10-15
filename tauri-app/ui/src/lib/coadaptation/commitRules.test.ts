import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  acceptsActivation,
  capReached,
  effectiveRegime,
  HOLD_COMPLETE_SLACK_MS,
  initialTrialState,
  isTerminal,
  MAX_ACTIVATIONS,
  needsL2Response,
  reduceTrial,
  type ActivationEvent,
  type L2Proposal,
  type TrialEvent,
  type TrialState,
} from "./commitRules";
import type { L1Variant, L2Variant } from "./types";

const ISSUED = "2026-10-15T10:00:00+00:00";
const FRESH = "2026-10-15T10:05:00+00:00";

function start(
  regime: L1Variant,
  options: Partial<{
    plainRegime: L1Variant | null;
    interruptionCap: number | null;
    interruptionsUsed: number;
    reversible: boolean[];
  }> = {},
): TrialState {
  const state = initialTrialState(regime, {
    plainRegime: options.plainRegime === undefined ? "assistance_off" : options.plainRegime,
    fallbacksIssuedAt: ISSUED,
    interruptionCap: options.interruptionCap ?? null,
    interruptionsUsed: options.interruptionsUsed ?? 0,
    tau: 0.25,
    reversible: options.reversible ?? [true, true, true],
  });
  return reduceTrial(state, { type: "cue" });
}

function activation(hitIndex: number | null, aGate: number, extra: Partial<ActivationEvent> = {}): ActivationEvent {
  return {
    type: "activation",
    x: 100,
    y: 100,
    modality: "mouse",
    hitIndex,
    pressIndex: hitIndex ?? -1,
    aGate,
    nPlausible: 2,
    plausible: [hitIndex ?? 0, 2, 1],
    hitRect: hitIndex === null ? null : { cx: 100, cy: 100, w: 28, h: 28 },
    ...extra,
  };
}

function l2(variant: L2Proposal["variant"], extra: Partial<L2Proposal> = {}): L2Proposal {
  return { variant, source: "table", decisionId: "table-1", ...extra };
}

function run(state: TrialState, events: TrialEvent[]): TrialState {
  return events.reduce(reduceTrial, state);
}

/** A hold completion well after any hold duration (activations default to 0 ms after the cue). */
const holdComplete = (tMs = 1000): TrialEvent => ({ type: "holdComplete", tMs });

const confirmInput = (sinceChipMs = 400, onIndex: number | null = null): TrialEvent => ({
  type: "confirmInput",
  kind: "click",
  onIndex,
  sinceChipMs,
});

describe("transition table (section 12.5)", () => {
  it("assistance_off commits the hit whatever the ambiguity and does nothing without one", () => {
    expect(run(start("assistance_off"), [activation(1, 0.1)]).committedIndex).toBe(1);
    const ambiguous = run(start("assistance_off"), [activation(1, 0.9)]);
    expect(ambiguous.phase).toBe("committed");
    expect(ambiguous.activations[0]).toMatchObject({ result: "committed", outcome: "committed", l2: null });
    const none = run(start("assistance_off"), [activation(null, 1)]);
    expect(none.phase).toBe("cued");
    expect(none.activations[0]).toMatchObject({ result: "none", outcome: "none" });
  });

  it("act and enlarge commit a clear hit and send ambiguity and no-hit to the L2 table", () => {
    for (const regime of ["act", "enlarge_target:1.5", "enlarge_target:2.0"] as const) {
      expect(run(start(regime), [activation(0, 0.2)]).phase).toBe("committed");
      const ambiguous = run(start(regime), [activation(0, 0.5, { l2: l2("ask_clarification:2") })]);
      expect(ambiguous.phase).toBe("clarifying");
      expect(ambiguous.activations[0].l2).toMatchObject({ source: "table", cellId: "b2:p2:h1", drawIndex: 0 });
      const noHit = run(start(regime), [activation(null, 1, { nPlausible: 1, l2: l2("wait_for_evidence") })]);
      expect(noHit.phase).toBe("cued");
      expect(noHit.activations[0].l2?.cellId).toBe("b3:p1:h0");
    }
  });

  it("confirm second_click opens a confirm step on a hit, clear or ambiguous; nothing without a hit", () => {
    expect(run(start("confirm_selection:second_click"), [activation(1, 0.1)]).phase).toBe("confirming");
    expect(run(start("confirm_selection:second_click"), [activation(1, 0.9)]).phase).toBe("confirming");
    const none = run(start("confirm_selection:second_click"), [activation(null, 1)]);
    expect(none.phase).toBe("cued");
    expect(none.activations[0].result).toBe("none");
  });

  it("hold regimes start holding on a hit (the press is the activation)", () => {
    const held = run(start("confirm_selection:hold_700"), [
      { type: "press", x: 100, y: 100, hitIndex: 2 },
      activation(2, 0.6, { pressIndex: 2 }),
    ]);
    expect(held.phase).toBe("holding");
    expect(held.activations[0]).toMatchObject({ result: "held", outcome: "open_at_end", pressIndex: 2 });
    expect(run(start("confirm_selection:hold_400"), [activation(null, 1)]).phase).toBe("cued");
  });

  it("a reverted trial follows the row of revert.regime", () => {
    const vetoed = run(start("act", { plainRegime: "confirm_selection:second_click" }), [
      { type: "veto", decisionId: "d-1", level: "block_regime" },
      activation(1, 0.1),
    ]);
    expect(vetoed.revert).toMatchObject({
      reason: "veto",
      fromActivation: 0,
      regime: "confirm_selection:second_click",
    });
    expect(effectiveRegime(vetoed)).toBe("confirm_selection:second_click");
    expect(vetoed.phase).toBe("confirming");

    const refreshed = run(start("act"), [
      { type: "refreshPending" },
      { type: "refresh", regime: "assistance_off", interruptionCap: null, fallbacksIssuedAt: FRESH },
      activation(1, 0.9),
    ]);
    expect(refreshed.revert).toMatchObject({ reason: "refresh", regime: "assistance_off", fallbacksIssuedAt: FRESH });
    expect(refreshed.phase).toBe("committed");
  });
});

describe("never commits without the person's input", () => {
  it("limit, interrupt and holdAbort never commit", () => {
    for (const regime of ["act", "confirm_selection:second_click", "confirm_selection:hold_700"] as const) {
      const open = run(start(regime), [activation(0, 0.9, { l2: l2("confirm_selection:second_click") })]);
      const limited = reduceTrial(open, { type: "limit" });
      expect(limited.phase).toBe("abandoned");
      expect(limited.committedIndex).toBeNull();
      expect(limited.activations[0].outcome).toBe(open.activations[0].result === "none" ? "none" : "open_at_end");
      const interrupted = reduceTrial(open, { type: "interrupt", reason: "blur" });
      expect(interrupted.phase).toBe("interrupted");
      expect(interrupted.interruptionReason).toBe("blur");
      expect(interrupted.committedIndex).toBeNull();
    }
    const holding = run(start("confirm_selection:hold_700"), [activation(0, 0.1)]);
    const aborted = reduceTrial(holding, { type: "holdAbort", reason: "pointercancel" });
    expect(aborted.phase).toBe("cued");
    expect(aborted.committedIndex).toBeNull();
    expect(aborted.corrections).toBe(0);
    expect(aborted.activations[0]).toMatchObject({ result: "held", outcome: "aborted" });
  });

  it("wait_for_evidence never commits its activation, from any source", () => {
    for (const source of ["table", "fixed_rule", "fallback"] as const) {
      const state = run(start("act"), [activation(1, 0.7, { l2: l2("wait_for_evidence", { source }) })]);
      expect(state.phase).toBe("cued");
      expect(state.committedIndex).toBeNull();
      expect(state.activations[0]).toMatchObject({ result: "none", outcome: "none" });
    }
  });

  it("no L2 response outside a pin_act collapse commits the triggering activation", () => {
    for (const variant of [
      "confirm_selection:second_click",
      "wait_for_evidence",
      "ask_clarification:2",
      "ask_clarification:3",
    ] as const) {
      const state = run(start("enlarge_target:1.5"), [activation(1, 0.8, { nPlausible: 3, l2: l2(variant) })]);
      expect(state.phase).not.toBe("committed");
    }
    const notPinned = run(start("act"), [activation(1, 0.8, { l2: l2("act", { pinAct: false }) })]);
    expect(notPinned.phase).toBe("cued");
    const pinned = run(start("act"), [activation(1, 0.8, { l2: l2("act", { pinAct: true, source: "fixed_rule" }) })]);
    expect(pinned.phase).toBe("committed");
    expect(pinned.activations[0]).toMatchObject({ result: "committed", outcome: "committed" });
  });

  it("only activation, confirmInput, holdComplete and choose reach committed", () => {
    const committedBy = new Set<string>();
    const sequences: TrialEvent[][] = [
      [activation(0, 0.1)],
      [activation(0, 0.9, { l2: l2("confirm_selection:second_click") }), confirmInput()],
      [activation(0, 0.9, { nPlausible: 2, l2: l2("ask_clarification:2") }), { type: "choose", index: 2 }],
    ];
    for (const events of sequences) {
      let state = start("act");
      for (const event of events) {
        const next = reduceTrial(state, event);
        if (next.phase === "committed" && state.phase !== "committed") committedBy.add(event.type);
        state = next;
      }
    }
    let hold = start("confirm_selection:hold_400");
    hold = reduceTrial(hold, activation(1, 0.2));
    const done = reduceTrial(hold, holdComplete());
    if (done.phase === "committed") committedBy.add("holdComplete");
    expect([...committedBy].sort()).toEqual(["activation", "choose", "confirmInput", "holdComplete"]);
  });
});

describe("rule C9 (irreversible candidates)", () => {
  const irreversible = { reversible: [true, false, true] };

  it("turns every single-activation commit into a confirm step and records irreversible_rule", () => {
    const cases: Array<[L1Variant, TrialEvent[]]> = [
      ["assistance_off", [activation(1, 0.1)]],
      ["act", [activation(1, 0.1)]],
      ["enlarge_target:2.0", [activation(1, 0.1)]],
      ["act", [activation(1, 0.8, { l2: l2("act", { pinAct: true, source: "fixed_rule" }) })]],
      ["act", [{ type: "veto" }, activation(1, 0.9)]],
    ];
    for (const [regime, events] of cases) {
      const state = run(start(regime, irreversible), events);
      expect(state.phase, regime).toBe("confirming");
      expect(state.pendingIndex).toBe(1);
      expect(state.activations.at(-1)).toMatchObject({ irreversibleRule: true, result: "confirming" });
      expect(state.committedIndex).toBeNull();
      const confirmed = reduceTrial(state, confirmInput());
      expect(confirmed.phase).toBe("committed");
      expect(confirmed.commitVia).toBe("confirm");
    }
  });

  it("leaves confirm, hold and ask responses unchanged", () => {
    const confirm = run(start("confirm_selection:second_click", irreversible), [activation(1, 0.1)]);
    expect(confirm.activations[0].irreversibleRule).toBe(false);
    const hold = run(start("confirm_selection:hold_700", irreversible), [activation(1, 0.1)]);
    expect(hold.phase).toBe("holding");
    expect(hold.activations[0].irreversibleRule).toBe(false);
  });
});

describe("confirm, hold and clarification steps", () => {
  it("confirm needs a fresh input at least 250 ms after the chip; double-clicks and key repeats are ignored", () => {
    const open = run(start("confirm_selection:second_click"), [activation(1, 0.1)]);
    expect(reduceTrial(open, confirmInput(120)).phase).toBe("confirming");
    expect(reduceTrial(open, confirmInput(249)).phase).toBe("confirming");
    expect(reduceTrial(open, { ...confirmInput(600), fresh: false } as TrialEvent).phase).toBe("confirming");
    expect(reduceTrial(open, { type: "confirmInput", kind: "key", onIndex: 1, sinceChipMs: 30 }).phase).toBe(
      "confirming",
    );
    expect(reduceTrial(open, confirmInput(250, 1)).phase).toBe("committed");
    expect(reduceTrial(open, confirmInput(400, 2)).phase).toBe("confirming");
  });

  it("cancel returns to cued with a correction and keeps result = confirming", () => {
    const cancelled = run(start("act"), [
      activation(1, 0.7, { l2: l2("confirm_selection:second_click") }),
      { type: "cancel" },
    ]);
    expect(cancelled.phase).toBe("cued");
    expect(cancelled.corrections).toBe(1);
    expect(cancelled.activations[0]).toMatchObject({ result: "confirming", outcome: "cancelled" });
  });

  it("hold early release cancels with a correction; holdAbort cancels without one", () => {
    const released = run(start("confirm_selection:hold_700"), [
      activation(0, 0.1),
      { type: "holdRelease", elapsedMs: 300 },
    ]);
    expect(released.phase).toBe("cued");
    expect(released.corrections).toBe(1);
    expect(released.activations[0]).toMatchObject({ result: "held", outcome: "cancelled" });
    const aborted = run(start("confirm_selection:hold_700"), [
      activation(0, 0.1),
      { type: "holdAbort", reason: "stale" },
    ]);
    expect(aborted.corrections).toBe(0);
    expect(aborted.activations[0].outcome).toBe("aborted");
    const done = run(start("confirm_selection:hold_700"), [activation(0, 0.1), holdComplete()]);
    expect(done).toMatchObject({ phase: "committed", committedIndex: 0, commitVia: "hold" });
  });

  it("refuses a hold completion before the hold duration recorded at the press has elapsed", () => {
    const held = run(start("confirm_selection:hold_700"), [activation(0, 0.1, { tMs: 1000 })]);
    expect(held).toMatchObject({ phase: "holding", holdStartedAt: 1000, holdDurationMs: 700 });
    // A completion 400 ms in (a stale or wrong timer) commits nothing, and the hold step stays open.
    expect(reduceTrial(held, holdComplete(1400))).toBe(held);
    expect(reduceTrial(held, holdComplete(1000 + 700 - HOLD_COMPLETE_SLACK_MS - 1))).toBe(held);
    // A completion without a time cannot be checked, so it commits nothing either.
    expect(reduceTrial(held, { type: "holdComplete" } as unknown as TrialEvent)).toBe(held);
    const done = reduceTrial(held, holdComplete(1700));
    expect(done).toMatchObject({ phase: "committed", committedIndex: 0, commitVia: "hold", commitTMs: 1700 });
    expect(done.holdDurationMs).toBeNull();
    // The duration is the one of the regime that opened the step.
    const short = run(start("confirm_selection:hold_400"), [activation(1, 0.1, { tMs: 50 })]);
    expect(short.holdDurationMs).toBe(400);
    expect(reduceTrial(short, holdComplete(350)).phase).toBe("holding");
    expect(reduceTrial(short, holdComplete(450)).phase).toBe("committed");
    // A released hold leaves no duration behind for a later completion.
    const released = reduceTrial(short, { type: "holdRelease", elapsedMs: 100 });
    expect(released.holdDurationMs).toBeNull();
    expect(reduceTrial(released, holdComplete(5000))).toBe(released);
  });

  it("ask commits only on an explicit choice; None of these is a correction", () => {
    const asked = run(start("act"), [
      activation(0, 0.7, { nPlausible: 3, plausible: [2, 0, 1], l2: l2("ask_clarification:3") }),
    ]);
    expect(asked.phase).toBe("clarifying");
    expect(asked.askCandidates).toEqual([2, 0, 1]);
    expect(reduceTrial(asked, { type: "limit" }).committedIndex).toBeNull();
    const chosen = reduceTrial(asked, { type: "choose", index: 2 });
    expect(chosen).toMatchObject({ phase: "committed", committedIndex: 2, commitVia: "ask" });
    expect(chosen.activations[0]).toMatchObject({ result: "asked", outcome: "committed", committedIndex: 2 });
    const none = reduceTrial(asked, { type: "choose", index: null });
    expect(none).toMatchObject({ phase: "cued", corrections: 1 });
    expect(none.activations[0].outcome).toBe("cancelled");
  });

  it("undo after a wrong commit returns to cued with a correction and keeps the outcome", () => {
    const undone = run(start("assistance_off"), [activation(0, 0.1), { type: "undo" }]);
    expect(undone).toMatchObject({ phase: "cued", committedIndex: null, corrections: 1 });
    expect(undone.activations[0].outcome).toBe("committed");
    expect(isTerminal(undone)).toBe(false);
  });
});

describe("veto and refresh", () => {
  it("a veto reverts to plainRegime; a second veto keeps the revert", () => {
    const vetoed = run(start("enlarge_target:2.0", { plainRegime: "assistance_off" }), [
      activation(0, 0.7, { l2: l2("wait_for_evidence") }),
      { type: "veto", decisionId: "t-1", level: "ambiguity_response" },
    ]);
    expect(vetoed.revert).toEqual({
      reason: "veto",
      fromActivation: 1,
      regime: "assistance_off",
      fallbacksIssuedAt: ISSUED,
    });
    expect(vetoed.veto).toEqual({ decisionId: "t-1", level: "ambiguity_response" });
    const again = reduceTrial(vetoed, { type: "veto" });
    expect(again.revert).toEqual(vetoed.revert);
  });

  it("a veto while plainRegime is null records the veto, sets no revert and interrupts with no_allowed_regime", () => {
    const state = run(start("act", { plainRegime: null }), [
      { type: "veto", decisionId: "d-9", level: "block_regime" },
      activation(0, 0.1),
    ]);
    expect(state.phase).toBe("interrupted");
    expect(state.interruptionReason).toBe("no_allowed_regime");
    expect(state.revert).toBeNull();
    expect(state.veto).toEqual({ decisionId: "d-9", level: "block_regime" });
    expect(state.committedIndex).toBeNull();
    expect(state.activations).toHaveLength(0);
  });

  it("refreshPending ignores presses and activations, but an open confirm step can still be completed", () => {
    const open = run(start("confirm_selection:second_click"), [activation(1, 0.1)]);
    const paused = reduceTrial(open, { type: "refreshPending" });
    expect(paused.inputPaused).toBe(true);
    expect(reduceTrial(paused, confirmInput()).phase).toBe("committed");

    const cued = reduceTrial(start("assistance_off"), { type: "refreshPending" });
    expect(reduceTrial(cued, { type: "press", x: 1, y: 1, hitIndex: 0 })).toBe(cued);
    expect(reduceTrial(cued, activation(0, 0.1))).toBe(cued);
    expect(needsL2Response(cued, { aGate: 0.9, hitIndex: 0 })).toBe(false);
  });

  it("refresh applies its fresh regime and cap: hold_700 at cue, second_click after the revert", () => {
    const state = run(start("confirm_selection:hold_700", { plainRegime: "confirm_selection:hold_700" }), [
      { type: "refreshPending" },
      { type: "refresh", regime: "confirm_selection:second_click", interruptionCap: 3, fallbacksIssuedAt: FRESH },
      activation(1, 0.1),
    ]);
    expect(state.inputPaused).toBe(false);
    expect(state.interruptionCap).toBe(3);
    expect(state.fallbacksIssuedAt).toBe(FRESH);
    expect(state.cueFallbacksIssuedAt).toBe(ISSUED);
    expect(state.endBlockAfterTrial).toBe(true);
    expect(state.phase).toBe("confirming");
    expect(state.activations[0].result).toBe("confirming");
  });

  it("a refresh naming the current (frozen) regime keeps it and records the revert", () => {
    const state = run(start("confirm_selection:hold_700"), [
      { type: "refreshPending" },
      { type: "refresh", regime: "confirm_selection:hold_700", interruptionCap: null, fallbacksIssuedAt: FRESH },
      activation(1, 0.1),
    ]);
    expect(state.revert).toMatchObject({ reason: "refresh", regime: "confirm_selection:hold_700" });
    expect(state.phase).toBe("holding");
  });

  it("a refresh with a null regime, or after a veto or an earlier refresh, interrupts the trial with refresh", () => {
    const nullRegime = run(start("act"), [
      { type: "refreshPending" },
      { type: "refresh", regime: null, interruptionCap: null, fallbacksIssuedAt: FRESH },
    ]);
    expect(nullRegime).toMatchObject({ phase: "interrupted", interruptionReason: "refresh", inputPaused: false });
    const afterVeto = run(start("act"), [
      { type: "veto" },
      { type: "refresh", regime: "assistance_off", interruptionCap: null, fallbacksIssuedAt: FRESH },
    ]);
    expect(afterVeto).toMatchObject({ phase: "interrupted", interruptionReason: "refresh" });
    expect(afterVeto.revert?.reason).toBe("veto");
    const twice = run(start("act"), [
      { type: "refresh", regime: "assistance_off", interruptionCap: null, fallbacksIssuedAt: FRESH },
      { type: "refresh", regime: "confirm_selection:hold_700", interruptionCap: null, fallbacksIssuedAt: FRESH },
    ]);
    expect(twice.phase).toBe("interrupted");
    expect(twice.revert?.regime).toBe("assistance_off");
  });
});

describe("cap rule and interruption count (section 9.3)", () => {
  it("at the cap an ambiguous activation under act waits (source cap) and uses no table draw", () => {
    const state = run(start("act", { interruptionCap: 2, interruptionsUsed: 2 }), [
      activation(0, 0.7, { l2: l2("ask_clarification:2", { drawIndex: 0 }) }),
    ]);
    expect(capReached(state)).toBe(true);
    expect(state.phase).toBe("cued");
    expect(state.l2Used).toBe(0);
    expect(state.activations[0].l2).toMatchObject({ variant: "wait_for_evidence", source: "cap", drawIndex: 0 });
    expect(state.activations[0].result).toBe("none");
    expect(state.interruptionsUsed).toBe(2);
  });

  it("a pin_act collapse still commits at the cap", () => {
    const state = run(start("act", { interruptionCap: 2, interruptionsUsed: 2 }), [
      activation(0, 0.7, { l2: l2("act", { pinAct: true, source: "fixed_rule" }) }),
    ]);
    expect(state.phase).toBe("committed");
    expect(state.activations[0].l2?.source).toBe("fixed_rule");
  });

  it("below the cap a table draw is used and counted", () => {
    const state = run(start("act", { interruptionCap: 3, interruptionsUsed: 2 }), [
      activation(0, 0.7, { l2: l2("confirm_selection:second_click", { drawIndex: 0 }) }),
    ]);
    expect(state.l2Used).toBe(1);
    expect(state.interruptionsUsed).toBe(3);
    expect(state.activations[0].l2?.source).toBe("table");
  });

  it("every clarification and every L2 or C9 confirm step counts, also after a cancel", () => {
    const state = run(start("act", { reversible: [false, true, true] }), [
      activation(1, 0.7, { l2: l2("confirm_selection:second_click", { drawIndex: 0 }) }),
      { type: "cancel" },
      activation(1, 0.7, { l2: l2("confirm_selection:second_click", { drawIndex: 1 }) }),
      { type: "cancel" },
      activation(1, 0.7, { l2: l2("ask_clarification:2", { drawIndex: 2 }) }),
      { type: "choose", index: null },
      activation(1, 0.7, { l2: l2("ask_clarification:2", { drawIndex: 3 }) }),
      { type: "choose", index: null },
      activation(0, 0.1),
    ]);
    expect(state.interruptionsUsed).toBe(5);
    expect(state.activations.map((a) => a.result)).toEqual([
      "confirming",
      "confirming",
      "asked",
      "asked",
      "confirming",
    ]);
    expect(state.activations[4].irreversibleRule).toBe(true);
  });

  it("confirm, cancel and re-confirm under second_click add 1 in a trial", () => {
    const state = run(start("confirm_selection:second_click"), [
      activation(1, 0.1),
      { type: "cancel" },
      activation(1, 0.1),
      confirmInput(),
    ]);
    expect(state.interruptionsUsed).toBe(1);
    expect(state.l1StepCounted).toBe(true);
    expect(state.activations.map((a) => [a.result, a.outcome])).toEqual([
      ["confirming", "cancelled"],
      ["confirming", "committed"],
    ]);
  });

  it("hold, early release and re-hold under hold_700 add 1 in a trial", () => {
    const state = run(start("confirm_selection:hold_700"), [
      activation(1, 0.1),
      { type: "holdRelease", elapsedMs: 200 },
      activation(1, 0.1),
      { type: "holdAbort", reason: "lostpointercapture" },
      activation(1, 0.1),
      holdComplete(),
    ]);
    expect(state.interruptionsUsed).toBe(1);
    expect(state.activations.map((a) => a.outcome)).toEqual(["cancelled", "aborted", "committed"]);
  });

  it("a trial reverted from one confirm regime to another adds 1 in all", () => {
    const state = run(start("confirm_selection:hold_700", { plainRegime: "confirm_selection:second_click" }), [
      activation(1, 0.1),
      { type: "holdRelease", elapsedMs: 100 },
      { type: "veto" },
      activation(1, 0.1),
      { type: "cancel" },
    ]);
    expect(effectiveRegime(state)).toBe("confirm_selection:second_click");
    expect(state.interruptionsUsed).toBe(1);
  });

  it("carries k from trial to trial: a cap of 8 under hold_700 is reached only after the eighth trial", () => {
    let k = 0;
    for (let trial = 0; trial < 8; trial += 1) {
      const state = run(start("confirm_selection:hold_700", { interruptionCap: 8, interruptionsUsed: k }), [
        activation(1, 0.1),
        { type: "holdRelease", elapsedMs: 150 },
        activation(1, 0.1),
        holdComplete(),
      ]);
      expect(state.phase).toBe("committed");
      k = state.interruptionsUsed;
      expect(k).toBe(trial + 1);
    }
  });
});

describe("activation limit (TrialV1 holds at most 32 activations)", () => {
  const misses = (count: number): TrialEvent[] => Array.from({ length: count }, () => activation(null, 1));

  it("the 32nd activation that leaves nothing open ends the trial as abandoned; later input is ignored", () => {
    const state = run(start("assistance_off"), misses(31));
    expect(state.phase).toBe("cued");
    expect(acceptsActivation(state)).toBe(true);
    const capped = reduceTrial(state, activation(null, 1));
    expect(capped.activations).toHaveLength(MAX_ACTIVATIONS);
    expect(capped.phase).toBe("abandoned");
    expect(capped.interruptionReason).toBeNull();
    expect(acceptsActivation(capped)).toBe(false);
    expect(run(capped, [activation(0, 0.1), { type: "press", x: 1, y: 1, hitIndex: 0 }])).toBe(capped);
  });

  it("a commit on the 32nd activation stands; undoing it then ends the trial with the wrong commit recorded", () => {
    const wrong = run(start("assistance_off"), [...misses(31), activation(2, 0.1)]);
    expect(wrong).toMatchObject({ phase: "committed", committedIndex: 2 });
    expect(wrong.activations[31]).toMatchObject({ index: 31, outcome: "committed", committedIndex: 2 });
    const undone = reduceTrial(wrong, { type: "undo" });
    expect(undone.phase).toBe("abandoned");
    expect(undone.committedIndex).toBeNull();
    expect(undone.activations).toHaveLength(MAX_ACTIVATIONS);
    expect(undone.activations[31].outcome).toBe("committed");
  });

  it("a confirm step opened by the 32nd activation can still be confirmed, and a cancel ends the trial", () => {
    const opened = run(start("confirm_selection:second_click"), [...misses(31), activation(1, 0.1)]);
    expect(opened.phase).toBe("confirming");
    expect(reduceTrial(opened, confirmInput(400, 1))).toMatchObject({ phase: "committed", committedIndex: 1 });
    expect(reduceTrial(opened, { type: "cancel" }).phase).toBe("abandoned");
    // A veto at the limit reverts from activation 32, the last value TrialV1 allows.
    expect(reduceTrial(opened, { type: "veto" }).revert?.fromActivation).toBe(MAX_ACTIVATIONS);
  });

  it("wait_for_evidence responses count toward the limit", () => {
    const state = run(
      start("act"),
      Array.from({ length: 40 }, () => activation(1, 0.9)),
    );
    expect(state.activations).toHaveLength(MAX_ACTIVATIONS);
    expect(state.phase).toBe("abandoned");
    expect(state.activations.every((a) => a.l2?.variant === "wait_for_evidence")).toBe(true);
  });
});

describe("isTerminal and helpers", () => {
  it("reports committed, abandoned and interrupted as terminal", () => {
    expect(isTerminal(start("act"))).toBe(false);
    expect(isTerminal(run(start("act"), [activation(0, 0.1)]))).toBe(true);
    expect(isTerminal(run(start("act"), [{ type: "limit" }]))).toBe(true);
    expect(isTerminal(run(start("act"), [{ type: "interrupt" }]))).toBe(true);
  });

  it("routes only ambiguous or no-hit activations of act-guarded regimes to L2", () => {
    expect(needsL2Response(start("act"), { aGate: 0.3, hitIndex: 1 })).toBe(true);
    expect(needsL2Response(start("act"), { aGate: 0.25, hitIndex: 1 })).toBe(false);
    expect(needsL2Response(start("act"), { aGate: 1, hitIndex: null })).toBe(true);
    expect(needsL2Response(start("assistance_off"), { aGate: 1, hitIndex: null })).toBe(false);
  });
});

// ---- Shared recount fixture (daemon/tests/fixtures/coadaptation_cap_recount.json, spec 9.3) ------------------------

interface RecountActivation {
  type: "activation";
  regime: L1Variant;
  ambiguous: boolean;
  hit: boolean;
  n_plausible: number;
  irreversible: boolean;
  pin_act: boolean;
  l2: { variant: L2Variant; source: "table" | "fixed_rule" | "fallback" } | null;
  then: string | null;
  expected: {
    k_before: number;
    variant: L2Variant | null;
    source: string | null;
    result: string;
    outcome: string;
  };
}

type RecountItem =
  | RecountActivation
  | { type: "veto"; level: "block_regime" | "ambiguity_response" }
  | { type: "refresh"; regime: L1Variant | null; interruption_cap: number | null; freeze?: string | null };

interface RecountTrial {
  kind: "active" | "idle";
  regime: L1Variant;
  plain_regime: L1Variant | null;
  items: RecountItem[];
  end: "committed" | "limit" | "interrupt";
  expected: { verification: string; k_after: number };
}

interface RecountBlock {
  id: string;
  description: string;
  interruption_cap: number | null;
  pin_act: boolean;
  trials: RecountTrial[];
}

const recountPath = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../../../daemon/tests/fixtures/coadaptation_cap_recount.json",
);

function loadRecount(): { tau: number; blocks: RecountBlock[] } {
  return JSON.parse(readFileSync(recountPath, "utf8")) as { tau: number; blocks: RecountBlock[] };
}

/** Drives one fixture trial through reduceTrial; returns the final state and the recorded activations' checks. */
function driveTrial(trialCase: RecountTrial, block: RecountBlock, tau: number, kBefore: number) {
  let state = reduceTrial(
    initialTrialState(trialCase.regime, {
      plainRegime: trialCase.plain_regime,
      fallbacksIssuedAt: ISSUED,
      interruptionCap: block.interruption_cap,
      interruptionsUsed: kBefore,
      tau,
      reversible: [true, true, true],
    }),
    { type: "cue" },
  );
  const checks: Array<{ index: number; item: RecountActivation; kBefore: number }> = [];
  for (const item of trialCase.items) {
    if (item.type === "veto") {
      state = reduceTrial(state, { type: "veto", decisionId: "d", level: item.level });
      continue;
    }
    if (item.type === "refresh") {
      state = reduceTrial(state, { type: "refreshPending" });
      state = reduceTrial(state, {
        type: "refresh",
        regime: item.regime,
        interruptionCap: item.interruption_cap,
        fallbacksIssuedAt: FRESH,
      });
      continue;
    }
    expect(effectiveRegime(state), `${block.id}: governing regime`).toBe(item.regime);
    const hitIndex = item.hit ? 0 : null;
    const aGate = item.hit ? (item.ambiguous ? 0.5 : 0.1) : 1;
    let proposal: L2Proposal | undefined;
    if (item.pin_act) {
      proposal = { variant: item.hit ? "act" : "wait_for_evidence", source: "table", decisionId: "t", pinAct: true };
    } else if (item.l2) {
      proposal = { variant: item.l2.variant, source: item.l2.source, decisionId: "t", pinAct: false };
    }
    const kNow = state.interruptionsUsed;
    state = reduceTrial(state, {
      type: "activation",
      x: 10,
      y: 10,
      modality: "mouse",
      hitIndex,
      pressIndex: hitIndex ?? -1,
      aGate,
      nPlausible: item.n_plausible,
      plausible: [0, 1, 2],
      reversible: item.hit ? !item.irreversible : undefined,
      l2: proposal,
    });
    checks.push({ index: state.activations.length - 1, item, kBefore: kNow });
    const next: TrialEvent | null = (() => {
      switch (item.then) {
        case "confirmed":
          return { type: "confirmInput", kind: "click", onIndex: null, sinceChipMs: 400 };
        case "cancelled":
          return { type: "cancel" };
        case "hold_completed":
          return holdComplete();
        case "released":
          return { type: "holdRelease", elapsedMs: 200 };
        case "aborted":
          return { type: "holdAbort", reason: "lostpointercapture" };
        case "choice":
          return { type: "choose", index: state.askCandidates[0] ?? null };
        case "none_of_these":
          return { type: "choose", index: null };
        case "limit":
          return { type: "limit" };
        case "interrupt":
          return { type: "interrupt", reason: "blur" };
        default:
          return null;
      }
    })();
    if (next) state = reduceTrial(state, next);
  }
  if (trialCase.end === "limit") state = reduceTrial(state, { type: "limit" });
  if (trialCase.end === "interrupt") state = reduceTrial(state, { type: "interrupt", reason: "blur" });
  return { state, checks };
}

describe("cap recount parity fixture", () => {
  const fixture = loadRecount();

  it("has the 20 blocks of section 9.3", () => {
    expect(fixture.blocks).toHaveLength(20);
  });

  it.each(fixture.blocks.map((block) => [block.id, block] as const))("%s matches", (_, block) => {
    let k = 0;
    for (const [t, trialCase] of block.trials.entries()) {
      // An idle epoch counts from 0 on its own and leaves the block's count unchanged.
      const kStart = trialCase.kind === "idle" ? 0 : k;
      const { state, checks } = driveTrial(trialCase, block, fixture.tau, kStart);
      for (const { index, item, kBefore } of checks) {
        const label = `${block.id} trial ${t} activation ${index}`;
        const recorded = state.activations[index];
        expect(kBefore, `${label} k_before`).toBe(item.expected.k_before);
        expect(recorded.result, `${label} result`).toBe(item.expected.result);
        expect(recorded.outcome, `${label} outcome`).toBe(item.expected.outcome);
        expect(recorded.l2?.variant ?? null, `${label} variant`).toBe(item.expected.variant);
        expect(recorded.l2?.source ?? null, `${label} source`).toBe(item.expected.source);
      }
      expect(state.interruptionsUsed, `${block.id} trial ${t} k_after`).toBe(trialCase.expected.k_after);
      expect(state.phase === "interrupted", `${block.id} trial ${t} interrupted`).toBe(
        trialCase.expected.verification === "interrupted",
      );
      if (trialCase.kind !== "idle") k = state.interruptionsUsed;
    }
  });
});

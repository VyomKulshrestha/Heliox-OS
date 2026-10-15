import { describe, expect, it } from "vitest";
import { initialTrialState, reduceTrial, type TrialEvent } from "./commitRules";
import { TrialRecorder, type BuildTrialInput } from "./trialRecorder";
import type { RectV1 } from "./types";

const TARGET: RectV1 = { cx: 300, cy: 100, w: 20, h: 20 };
const ISSUED = "2026-10-15T10:00:00+00:00";

function stateAfter(events: TrialEvent[]) {
  const initial = initialTrialState("act", {
    plainRegime: "assistance_off",
    fallbacksIssuedAt: ISSUED,
    interruptionCap: null,
    interruptionsUsed: 0,
    tau: 0.25,
    reversible: [true, true],
  });
  return [{ type: "cue" } as TrialEvent, ...events].reduce(reduceTrial, initial);
}

function input(overrides: Partial<BuildTrialInput> = {}): BuildTrialInput {
  return {
    trialId: "33333333-3333-4333-8333-333333333333",
    blockIndex: 2,
    trialIndex: 5,
    task: "pointing",
    regime: "act",
    regimeDecisionId: "44444444-4444-4444-8444-444444444444",
    tableDecisionIds: ["55555555-5555-4555-8555-555555555555"],
    appliedRegime: "act",
    l1FallbackReason: null,
    spread: { r: 1.02, floor: 1 },
    layout: {
      task: "pointing",
      n_candidates: 2,
      w_px: 20,
      h_px: 20,
      gap_px: 8,
      id_bar: 3.5,
      menu_n: 0,
      origin_x: 320,
      origin_y: 396,
    },
    candidates: [TARGET, { cx: 328, cy: 100, w: 20, h: 20 }],
    surface: { w: 700, h: 500 },
    dpr: 1.25,
    reversible: [true, true],
    cueAt: new Date("2026-10-15T10:01:00.000Z"),
    intendedIndex: 0,
    modality: "mouse",
    state: stateAfter([
      {
        type: "activation",
        x: 302.4,
        y: 101.6,
        modality: "mouse",
        hitIndex: 0,
        pressIndex: 0,
        aGate: 0.1,
        nPlausible: 1,
        hitRect: TARGET,
        tMs: 900,
      },
    ]),
    timeMs: 900,
    limitMs: 10000,
    durationMs: 900,
    ...overrides,
  };
}

/** A straight approach from (0, 400) to the target with one pause in the middle. */
function feedApproach(recorder: TrialRecorder) {
  recorder.sample(0, 400, 0);
  recorder.sample(0, 400, 150);
  let t = 150;
  for (let i = 1; i <= 10; i += 1) {
    t += 20;
    recorder.sample(i * 15, 400 - i * 15, t);
  }
  t += 60;
  recorder.sample(151, 249, t);
  for (let i = 1; i <= 10; i += 1) {
    t += 20;
    recorder.sample(150 + i * 15, 250 - i * 15, t);
  }
  return t;
}

describe("TrialRecorder", () => {
  it("summarises the first approach", () => {
    const recorder = new TrialRecorder();
    recorder.start(0, TARGET);
    const end = feedApproach(recorder);
    recorder.activation(302, 102, end + 10);
    const summary = recorder.summary()!;
    expect(summary.rt_ms).toBe(170);
    expect(summary.mt_ms).toBeGreaterThan(400);
    expect(summary.path_ratio).toBeGreaterThanOrEqual(1);
    expect(summary.path_ratio).toBeLessThan(1.1);
    expect(summary.peak_speed_px_s).toBeGreaterThan(1000);
    expect(summary.submovements).toBe(1);
    expect(summary.reentries).toBe(0);
    expect(summary.endpoint_dx_norm).toBeCloseTo(0.1, 12);
    expect(summary.endpoint_dy_norm).toBeCloseTo(0.1, 12);
  });

  it("counts re-entries into the target", () => {
    const recorder = new TrialRecorder();
    recorder.start(0, TARGET);
    const points = [
      [250, 100],
      [270, 100],
      [300, 100],
      [330, 100],
      [300, 100],
      [330, 100],
      [300, 101],
    ];
    points.forEach(([x, y], i) => recorder.sample(x, y, 100 + i * 16));
    recorder.activation(300, 101, 100 + points.length * 16);
    expect(recorder.summary()!.reentries).toBe(2);
  });

  it("builds a TrialV1 with integer endpoints and no raw point arrays, and clears the samples", () => {
    const recorder = new TrialRecorder();
    recorder.start(0, TARGET);
    feedApproach(recorder);
    recorder.activation(302, 102, 700);
    expect(recorder.sampleCount).toBeGreaterThan(10);
    const trial = recorder.buildTrial(input());
    expect(recorder.sampleCount).toBe(0);
    expect(trial.activations[0]).toMatchObject({
      x: 302,
      y: 102,
      hit_index: 0,
      press_index: 0,
      result: "committed",
      outcome: "committed",
      committed_index: 0,
      l2: null,
      irreversible_rule: false,
    });
    expect(Number.isInteger(trial.activations[0].x)).toBe(true);
    expect(trial).toMatchObject({
      committed_index: 0,
      commit_via: "direct",
      commit_t_ms: 900,
      fallbacks_issued_at: ISSUED,
      cue_at: "2026-10-15T10:01:00.000Z",
      interrupted: false,
      interruption_reason: null,
      spread: { r: 1.02, floor: 1 },
      l1_fallback_reason: null,
    });
    expect(trial.movement).not.toBeNull();
    const serialised = JSON.stringify(trial);
    expect(serialised).not.toMatch(/samples|path"|points/);
    const arrays = Object.entries(trial).filter(([, value]) => Array.isArray(value));
    expect(arrays.map(([key]) => key).sort()).toEqual(
      ["activations", "candidates", "offers", "reversible", "table_decision_ids"].sort(),
    );
  });

  it("records press_index -1 for a press on a gap and per-activation L2 fallback reasons", () => {
    const state = stateAfter([
      {
        type: "activation",
        x: 314,
        y: 100,
        modality: "mouse",
        hitIndex: null,
        pressIndex: -1,
        aGate: 1,
        nPlausible: 1,
        l2: { variant: "wait_for_evidence", source: "fallback", decisionId: "t-1", fallbackReason: "late" },
      },
    ]);
    const trial = new TrialRecorder().buildTrial(input({ state }));
    expect(trial.activations[0]).toMatchObject({
      press_index: -1,
      hit_index: null,
      hit_rect: null,
      result: "none",
      outcome: "none",
      l2: { source: "fallback", fallback_reason: "late", variant: "wait_for_evidence", cell_id: "b3:p1:h0" },
    });
    expect(trial.l1_fallback_reason).toBeNull();
  });

  it("marks interrupted trials and the revert", () => {
    const state = stateAfter([
      { type: "veto", decisionId: "d", level: "block_regime" },
      { type: "interrupt", reason: "blur" },
    ]);
    const trial = new TrialRecorder().buildTrial(input({ state }));
    expect(trial.interrupted).toBe(true);
    expect(trial.interruption_reason).toBe("blur");
    expect(trial.revert).toEqual({
      reason: "veto",
      from_activation: 0,
      regime: "assistance_off",
      fallbacks_issued_at: ISSUED,
    });
    expect(trial.veto).toEqual({ decision_id: "d", level: "block_regime" });
    expect(trial.committed_index).toBeNull();
    expect(trial.commit_via).toBe("none");
  });

  it("sets no intended index and no movement for an idle epoch", () => {
    const recorder = new TrialRecorder();
    recorder.start(0, null);
    recorder.sample(1, 1, 10);
    const trial = recorder.buildTrial(input({ task: "idle", intendedIndex: 3, limitMs: 30000, durationMs: 30000 }));
    expect(trial.intended_index).toBeNull();
    expect(trial.movement).toBeNull();
  });

  it("sends every activation of a long trial, consistent with its commit and revert (TrialV1 limits)", () => {
    const miss: TrialEvent = {
      type: "activation",
      x: 200,
      y: 300,
      modality: "mouse",
      hitIndex: null,
      aGate: 1,
      nPlausible: 1,
    };
    const hit = (index: number, aGate: number): TrialEvent => ({
      type: "activation",
      x: 302,
      y: 101,
      modality: "mouse",
      hitIndex: index,
      aGate,
      nPlausible: 2,
      hitRect: TARGET,
    });
    const run = (regime: "assistance_off" | "act", events: TrialEvent[]) =>
      [{ type: "cue" } as TrialEvent, ...events, { type: "limit" } as TrialEvent].reduce(
        reduceTrial,
        initialTrialState(regime, {
          plainRegime: "assistance_off",
          fallbacksIssuedAt: ISSUED,
          interruptionCap: null,
          interruptionsUsed: 0,
          tau: 0.25,
          reversible: [true, true],
        }),
      );
    const misses = (count: number) => Array.from({ length: count }, () => miss);
    const states = {
      // Frustrated gap clicks, then a wrong commit that is undone and a correct one.
      wrongThenRight: run("assistance_off", [...misses(34), hit(1, 0.1), { type: "undo" }, hit(0, 0.1)]),
      // A wrong commit on the last activation, then undone.
      wrongAtLimit: run("assistance_off", [...misses(31), hit(1, 0.1), { type: "undo" }]),
      // Many ambiguous activations answered with wait_for_evidence, then a veto.
      vetoAfterMany: run("act", [...Array.from({ length: 33 }, () => hit(0, 0.9)), { type: "veto" }]),
      // A veto after 31 ambiguous activations, then a commit under the plain regime on the last one.
      vetoThenCommitAtLimit: run("act", [
        ...Array.from({ length: 31 }, () => hit(0, 0.9)),
        { type: "veto" },
        hit(1, 0.9),
      ]),
    };
    for (const [name, state] of Object.entries(states)) {
      const trial = new TrialRecorder().buildTrial(input({ state, intendedIndex: 0 }));
      expect(trial.activations.length, name).toBeLessThanOrEqual(32);
      expect(trial.activations.length, name).toBe(state.activations.length);
      trial.activations.forEach((activation, i) => expect(activation.index, name).toBe(i));
      if (trial.revert) expect(trial.revert.from_activation, name).toBeLessThanOrEqual(trial.activations.length);
      const commits = trial.activations.filter((a) => a.outcome === "committed").map((a) => a.committed_index);
      if (trial.committed_index !== null) expect(commits, name).toContain(trial.committed_index);
    }
    // The undone wrong commit stays in the record, so the unintended activation is counted.
    const wrong = new TrialRecorder().buildTrial(input({ state: states.wrongAtLimit, intendedIndex: 0 }));
    expect(wrong.activations[31]).toMatchObject({ outcome: "committed", committed_index: 1 });
    expect(wrong.committed_index).toBeNull();
  });

  it("discard clears the samples", () => {
    const recorder = new TrialRecorder();
    recorder.start(0, TARGET);
    feedApproach(recorder);
    recorder.discard();
    expect(recorder.sampleCount).toBe(0);
    expect(recorder.summary()).toBeNull();
    recorder.sample(1, 2, 3);
    expect(recorder.sampleCount).toBe(0);
  });
});

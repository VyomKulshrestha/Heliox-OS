import { describe, expect, it } from "vitest";
import { generateMenuLayout, generatePointingLayout, layoutSummary, mulberry32 } from "./geometry";
import {
  buildBlockTrials,
  burdenDue,
  idleDue,
  idleEpoch,
  idleEpochSeed,
  mainBlocks,
  menuLabelOffset,
  nextBlockIndex,
  TRIAL_LIMIT_MS,
} from "./studySchedule";
import type { SessionPlanV1 } from "./types";

const SURFACE = { w: 700, h: 500 };
const PLAN: SessionPlanV1 = {
  session_number: 1,
  end_questionnaire: true,
  blocks: [
    { index: 0, phase: "practice", task: "pointing", trials: 8, idle_after: false, burden_after: false },
    { index: 1, phase: "A", task: "menu", trials: 8, idle_after: false, burden_after: true },
    { index: 2, phase: "A", task: "pointing", trials: 8, idle_after: false, burden_after: true },
    { index: 3, phase: "A", task: "menu", trials: 8, idle_after: false, burden_after: false },
    { index: 4, phase: "A", task: "pointing", trials: 8, idle_after: true, burden_after: true },
  ],
};

describe("buildBlockTrials", () => {
  it("builds the planned number of trials, seeded per block and trial", () => {
    const block = buildBlockTrials(PLAN, 2, 1234567, SURFACE, "mouse");
    expect(block.trials).toHaveLength(8);
    expect(block.layouts).toHaveLength(8);
    block.trials.forEach((trial, t) => {
      const expected = generatePointingLayout(mulberry32(1234567 + 2 * 1000 + t), SURFACE);
      expect(trial.layout).toEqual(expected);
      expect(trial.rects).toEqual(expected.rects);
      expect(trial.intendedIndex).toBe(expected.intendedIndex);
      expect(trial.labelWord).toBe(expected.labels[expected.intendedIndex]);
      expect(block.layouts[t]).toEqual(layoutSummary(expected));
    });
    expect(buildBlockTrials(PLAN, 2, 1234567, SURFACE, "mouse")).toEqual(block);
  });

  it("builds menu blocks with the session's menu words and never names the target in the summaries", () => {
    const block = buildBlockTrials(PLAN, 1, 99, SURFACE, "keyboard");
    const offset = menuLabelOffset(99);
    block.trials.forEach((trial, t) => {
      expect(trial.layout.task).toBe("menu");
      expect(trial.layout).toEqual(generateMenuLayout(mulberry32(99 + 1000 + t), SURFACE, { labelOffset: offset }));
    });
    for (const summary of block.layouts) {
      expect(summary).not.toHaveProperty("intended_index");
      expect(summary).not.toHaveProperty("target");
      expect(summary.task).toBe("menu");
    }
  });

  it("interleaves 4 pointing and 4 menu trials in the practice block", () => {
    const block = buildBlockTrials(PLAN, 0, 5, SURFACE, "mouse");
    expect(block.trials.map((t) => t.layout.task)).toEqual([
      "pointing",
      "menu",
      "pointing",
      "menu",
      "pointing",
      "menu",
      "pointing",
      "menu",
    ]);
  });

  it("returns no trials for a block outside the plan", () => {
    expect(buildBlockTrials(PLAN, 42, 5, SURFACE, "mouse")).toEqual({ trials: [], layouts: [] });
  });
});

describe("idle epochs", () => {
  it("shows a pointing layout without a cue for 30 s", () => {
    const epoch = idleEpoch(idleEpochSeed(1234567, 4), SURFACE);
    expect(epoch.durationMs).toBe(30000);
    expect(epoch.limitMs).toBe(30000);
    expect(epoch.layout.task).toBe("idle");
    expect(epoch.summary.task).toBe("idle");
    expect(epoch.rects).toHaveLength(12);
    expect(idleEpoch(idleEpochSeed(1234567, 4), SURFACE)).toEqual(epoch);
    expect(idleEpochSeed(10, 4)).toBe(10 + 4000 + 999);
  });

  it("follows the plan's idle_after flags", () => {
    expect(idleDue(PLAN, 4)).toBe(true);
    expect(idleDue(PLAN, 3)).toBe(false);
  });
});

describe("plan helpers", () => {
  it("asks burden only after main blocks that plan it", () => {
    expect(burdenDue(PLAN, 0)).toBe(false);
    expect(burdenDue(PLAN, 1)).toBe(true);
    expect(burdenDue(PLAN, 3)).toBe(false);
    expect(burdenDue(PLAN, 9)).toBe(false);
  });

  it("walks the blocks in order", () => {
    expect(nextBlockIndex(PLAN, 0)).toBe(1);
    expect(nextBlockIndex(PLAN, 3)).toBe(4);
    expect(nextBlockIndex(PLAN, 4)).toBeNull();
    expect(mainBlocks(PLAN).map((b) => b.index)).toEqual([1, 2, 3, 4]);
  });

  it("uses the trial limits of section 12.5", () => {
    expect(TRIAL_LIMIT_MS).toEqual({ pointing: 10000, menu: 15000, idle: 30000 });
  });
});

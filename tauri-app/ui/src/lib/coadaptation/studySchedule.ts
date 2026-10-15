import {
  generateMenuLayout,
  generatePointingLayout,
  layoutSummary,
  mulberry32,
  STUDY_WORDS,
  type StudyLayout,
  type Surface,
} from "./geometry";
import type { BlockPlanV1, LayoutSummaryV1, Modality, RectV1, SessionPlanV1, TaskKind } from "./types";

/**
 * Builds the trials of a planned block and the idle epochs (spec sections 12.2-12.4). Every layout is seeded with
 * mulberry32(layout_seed + block_index * 1000 + trial_index), so the same session reproduces the same layouts. The
 * cued target stays in the UI until the trial record; the block's `layouts` (sent with `recommend`) never name it.
 * Pure: no DOM access.
 */

export const IDLE_EPOCH_MS = 30000;
export const TRIAL_LIMIT_MS: Readonly<Record<TaskKind, 10000 | 15000 | 30000>> = {
  pointing: 10000,
  menu: 15000,
  idle: 30000,
};
/** Seed offset of a block's idle epoch, above any trial index. */
export const IDLE_SEED_OFFSET = 999;

export interface ScheduledTrial {
  layout: StudyLayout;
  rects: RectV1[];
  intendedIndex: number;
  labelWord: string;
}

export interface BlockTrials {
  trials: ScheduledTrial[];
  /** One LayoutSummaryV1 per planned trial, for the `block_regime` and `ambiguity_table` contexts. */
  layouts: LayoutSummaryV1[];
}

export interface IdleEpochLayout {
  layout: StudyLayout;
  rects: RectV1[];
  summary: LayoutSummaryV1;
  durationMs: number;
  limitMs: 30000;
}

export function blockPlan(plan: SessionPlanV1, blockIndex: number): BlockPlanV1 | null {
  return plan.blocks.find((block) => block.index === blockIndex) ?? null;
}

/** Index of the session's first menu word, so a menu item keeps its word for the whole session. */
export function menuLabelOffset(seed: number): number {
  return Math.floor(mulberry32(seed)() * STUDY_WORDS.length);
}

/**
 * The block's trials. The practice block (phase "practice") interleaves pointing and menu trials (4 of each); other
 * blocks use the planned task. Layouts do not depend on the modality: keyboard sessions navigate the same targets.
 */
export function buildBlockTrials(
  plan: SessionPlanV1,
  blockIndex: number,
  seed: number,
  surface: Surface,
  modality: Modality,
): BlockTrials {
  const block = blockPlan(plan, blockIndex);
  if (!block) return { trials: [], layouts: [] };
  const labelOffset = menuLabelOffset(seed);
  const trials: ScheduledTrial[] = [];
  for (let t = 0; t < block.trials; t += 1) {
    const rng = mulberry32(seed + blockIndex * 1000 + t);
    const task: TaskKind = block.phase === "practice" ? (t % 2 === 0 ? "pointing" : "menu") : block.task;
    const layout =
      task === "menu" ? generateMenuLayout(rng, surface, { labelOffset }) : generatePointingLayout(rng, surface);
    trials.push({
      layout,
      rects: layout.rects,
      intendedIndex: layout.intendedIndex,
      labelWord: layout.labels[layout.intendedIndex],
    });
  }
  return { trials, layouts: trials.map((trial) => layoutSummary(trial.layout)) };
}

/** A 30 s rest epoch with visible targets and no cue; any commit during it is an unintended activation. */
export function idleEpoch(seed: number, surface: Surface): IdleEpochLayout {
  const pointing = generatePointingLayout(mulberry32(seed), surface);
  const layout: StudyLayout = { ...pointing, task: "idle" };
  return {
    layout,
    rects: layout.rects,
    summary: layoutSummary(layout),
    durationMs: IDLE_EPOCH_MS,
    limitMs: 30000,
  };
}

/** Seed of the idle epoch that follows `blockIndex`. */
export function idleEpochSeed(layoutSeed: number, blockIndex: number): number {
  return layoutSeed + blockIndex * 1000 + IDLE_SEED_OFFSET;
}

/** Whether a burden prompt follows the block (never after practice). */
export function burdenDue(plan: SessionPlanV1, blockIndex: number): boolean {
  const block = blockPlan(plan, blockIndex);
  return block !== null && block.phase !== "practice" && block.burden_after;
}

/** Whether an idle epoch is planned after the block. */
export function idleDue(plan: SessionPlanV1, blockIndex: number): boolean {
  const block = blockPlan(plan, blockIndex);
  return block !== null && block.phase !== "practice" && block.idle_after;
}

/** The index of the main block after `blockIndex`, or null at the end of the plan. */
export function nextBlockIndex(plan: SessionPlanV1, blockIndex: number): number | null {
  const later = plan.blocks.filter((block) => block.index > blockIndex).map((block) => block.index);
  return later.length > 0 ? Math.min(...later) : null;
}

/** Main blocks (everything but practice), for "Block n of N". */
export function mainBlocks(plan: SessionPlanV1): BlockPlanV1[] {
  return plan.blocks.filter((block) => block.phase !== "practice");
}

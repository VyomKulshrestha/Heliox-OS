import type { AmbiguityTableV1, FallbacksV1, L2Variant, TableCellV1 } from "./types";

/**
 * L2 table lookups (spec section 9.3). The daemon pre-draws the uniform numbers; the UI maps each one through the
 * logged cell propensities with `selectFromCell`, which must match the daemon's `select_from_cell` exactly (the
 * daemon re-derives every draw when the trial is recorded). Pure: no DOM access.
 */

export const MAX_DRAWS_PER_TRIAL = 4;
export const AFTER_MAX: L2Variant = "wait_for_evidence";

/**
 * The first variant in `order` whose cumulative propensity exceeds `u`; if rounding leaves none, the last variant in
 * `order` with positive propensity.
 */
export function selectFromCell(
  order: readonly string[],
  propensities: Readonly<Record<string, number>>,
  u: number,
): string {
  let cumulative = 0;
  for (const variant of order) {
    cumulative += propensities[variant] ?? 0;
    if (cumulative > u) return variant;
  }
  for (let i = order.length - 1; i >= 0; i -= 1) {
    if ((propensities[order[i]] ?? 0) > 0) return order[i];
  }
  throw new Error("The table cell has no variant with positive propensity.");
}

export function findCell(table: AmbiguityTableV1, cellId: string): TableCellV1 | null {
  return table.cells.find((cell) => cell.cell_id === cellId) ?? null;
}

export interface TableDraw {
  variant: L2Variant;
  /** The draw index used (0..3), or 4 for the deterministic after-max response. */
  drawIndex: number;
  cellId: string;
}

/**
 * The table response for an ambiguous activation in block trial `trialIndex` (0-based within the block) when
 * `drawIndex` table draws were already used in that trial. Row `trialIndex - start_trial_index` of the table is read
 * (a table requested after a veto starts at the next trial). From the fifth draw on the response is
 * `wait_for_evidence` with draw index 4. Returns null when the table has no such row or cell; the caller then applies
 * a fallback response.
 */
export function drawForActivation(
  table: AmbiguityTableV1,
  trialIndex: number,
  drawIndex: number,
  cellId: string,
): TableDraw | null {
  const cell = findCell(table, cellId);
  if (!cell) return null;
  const maxDraws = Math.min(table.max_draws_per_trial ?? MAX_DRAWS_PER_TRIAL, MAX_DRAWS_PER_TRIAL);
  const row = table.draws[trialIndex - table.start_trial_index];
  if (!row) return null;
  if (drawIndex >= maxDraws) {
    return { variant: (table.after_max ?? AFTER_MAX) as L2Variant, drawIndex: MAX_DRAWS_PER_TRIAL, cellId };
  }
  const u = row[drawIndex];
  if (typeof u !== "number") return null;
  const variant = selectFromCell(cell.order, cell.propensities, u) as L2Variant;
  return { variant, drawIndex, cellId };
}

export interface L2Applicability {
  hasHit: boolean;
  nPlausible: number;
}

/** Applicability of an L2 response to an activation (section 4.5). */
export function l2Applicable(variant: string, activation: L2Applicability): boolean {
  switch (variant) {
    case "act":
    case "confirm_selection:second_click":
      return activation.hasHit;
    case "ask_clarification:2":
      return activation.nPlausible >= 2;
    case "ask_clarification:3":
      return activation.nPlausible >= 3;
    case "wait_for_evidence":
      return true;
    default:
      return false;
  }
}

/**
 * The first entry of `fallbacks.l2` applicable to the activation: the fixed L2 rule used for UI fallbacks, the
 * practice block, idle epochs and reverted trials. `wait_for_evidence` is always applicable and closes every list.
 */
export function fixedL2Response(fallbacks: Pick<FallbacksV1, "l2"> | null, activation: L2Applicability): L2Variant {
  for (const variant of fallbacks?.l2 ?? []) {
    if (l2Applicable(variant, activation)) return variant;
  }
  return "wait_for_evidence";
}

/** True when the fallbacks hold a `pin_act` collapse (`["act", "wait_for_evidence"]`, section 9.2 step 4). */
export function isPinActFallback(fallbacks: Pick<FallbacksV1, "l2"> | null): boolean {
  return (fallbacks?.l2 ?? [])[0] === "act";
}

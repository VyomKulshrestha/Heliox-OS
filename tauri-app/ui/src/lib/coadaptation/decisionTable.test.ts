import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { drawForActivation, findCell, fixedL2Response, selectFromCell } from "./decisionTable";
import type { AmbiguityTableV1, FallbacksV1, TableCellV1 } from "./types";

interface SelectCase {
  order: string[];
  propensities: Record<string, number>;
  u: number;
  expected: string;
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../../daemon/tests/fixtures/coadaptation_decision_table.json");

function loadCases(): SelectCase[] {
  const raw = JSON.parse(readFileSync(fixturePath, "utf8")) as { cases?: SelectCase[] } | SelectCase[];
  return Array.isArray(raw) ? raw : (raw.cases ?? []);
}

const ORDER = [
  "act",
  "confirm_selection:second_click",
  "wait_for_evidence",
  "ask_clarification:2",
  "ask_clarification:3",
] as const;

function cell(cellId: string, propensities: Record<string, number>): TableCellV1 {
  const [bin, p, h] = cellId.split(":");
  return {
    cell_id: cellId,
    bin: bin as TableCellV1["bin"],
    n_plausible: Number(p.slice(1)),
    has_hit: h === "h1",
    order: [...ORDER],
    propensities,
    eligible: Object.keys(propensities).filter((key) => propensities[key] > 0),
    excluded: {},
  };
}

function table(startTrialIndex: number, draws: number[][]): AmbiguityTableV1 {
  return {
    decision_id: "11111111-1111-4111-8111-111111111111",
    level: "ambiguity_table",
    block_decision_id: "22222222-2222-4222-8222-222222222222",
    mode: "randomized",
    ambiguity_threshold: 0.25,
    bins: [
      { id: "b1", lo: 0.25, hi: 0.4 },
      { id: "b2", lo: 0.4, hi: 0.6 },
      { id: "b3", lo: 0.6, hi: 1 },
    ],
    cells: [
      cell("b1:p2:h1", {
        act: 0,
        "confirm_selection:second_click": 0.5,
        wait_for_evidence: 0.25,
        "ask_clarification:2": 0.25,
        "ask_clarification:3": 0,
      }),
      cell("b3:p1:h0", { act: 0, wait_for_evidence: 1 }),
    ],
    start_trial_index: startTrialIndex,
    draws,
    max_draws_per_trial: 4,
    after_max: "wait_for_evidence",
    explanation: null,
    seed: "0123456789abcdef",
    versions: {},
    authority: "presentation_only",
    execution_authority: false,
  };
}

describe("selectFromCell", () => {
  it("returns the first variant whose cumulative propensity exceeds u", () => {
    const propensities = { act: 0, "confirm_selection:second_click": 0.5, wait_for_evidence: 0.5 };
    expect(selectFromCell(ORDER, propensities, 0)).toBe("confirm_selection:second_click");
    expect(selectFromCell(ORDER, propensities, 0.4999)).toBe("confirm_selection:second_click");
    expect(selectFromCell(ORDER, propensities, 0.5)).toBe("wait_for_evidence");
  });

  it("falls back to the last positive variant when rounding leaves none", () => {
    const propensities = { "confirm_selection:second_click": 0.3, wait_for_evidence: 0.6999999 };
    expect(selectFromCell(ORDER, propensities, 0.99999999)).toBe("wait_for_evidence");
  });

  it("matches every case of the daemon fixture", () => {
    const cases = loadCases();
    expect(cases.length).toBe(200);
    for (const c of cases) expect(selectFromCell(c.order, c.propensities, c.u)).toBe(c.expected);
  });
});

describe("drawForActivation", () => {
  it("reads the trial's row and the draw index, then waits after the maximum", () => {
    const t = table(0, [
      [0.1, 0.6, 0.9, 0.2],
      [0.8, 0.3, 0.55, 0.7],
    ]);
    expect(drawForActivation(t, 0, 0, "b1:p2:h1")).toEqual({
      variant: "confirm_selection:second_click",
      drawIndex: 0,
      cellId: "b1:p2:h1",
    });
    expect(drawForActivation(t, 0, 1, "b1:p2:h1")?.variant).toBe("wait_for_evidence");
    expect(drawForActivation(t, 0, 2, "b1:p2:h1")?.variant).toBe("ask_clarification:2");
    expect(drawForActivation(t, 1, 0, "b1:p2:h1")?.variant).toBe("ask_clarification:2");
    expect(drawForActivation(t, 0, 4, "b1:p2:h1")).toEqual({
      variant: "wait_for_evidence",
      drawIndex: 4,
      cellId: "b1:p2:h1",
    });
    expect(drawForActivation(t, 0, 7, "b1:p2:h1")?.drawIndex).toBe(4);
    expect(drawForActivation(t, 0, 0, "b3:p1:h0")?.variant).toBe("wait_for_evidence");
  });

  it("reads row trialIndex - start_trial_index of a table requested after a veto", () => {
    const t = table(5, [
      [0.9, 0.9, 0.9, 0.9],
      [0.1, 0.1, 0.1, 0.1],
    ]);
    expect(drawForActivation(t, 5, 0, "b1:p2:h1")?.variant).toBe("ask_clarification:2");
    expect(drawForActivation(t, 6, 0, "b1:p2:h1")?.variant).toBe("confirm_selection:second_click");
    expect(drawForActivation(t, 4, 0, "b1:p2:h1")).toBeNull();
    expect(drawForActivation(t, 7, 0, "b1:p2:h1")).toBeNull();
  });

  it("returns null for a cell the table lacks", () => {
    const t = table(0, [[0.5, 0.5, 0.5, 0.5]]);
    expect(findCell(t, "b2:p3:h1")).toBeNull();
    expect(drawForActivation(t, 0, 0, "b2:p3:h1")).toBeNull();
  });
});

describe("fixedL2Response", () => {
  const fallbacks = (l2: FallbacksV1["l2"]) => ({ l2 });

  it("picks the first applicable entry: asking needs plausible candidates", () => {
    const list = fallbacks(["ask_clarification:3", "ask_clarification:2", "wait_for_evidence"]);
    expect(fixedL2Response(list, { hasHit: true, nPlausible: 3 })).toBe("ask_clarification:3");
    expect(fixedL2Response(list, { hasHit: true, nPlausible: 2 })).toBe("ask_clarification:2");
    expect(fixedL2Response(list, { hasHit: false, nPlausible: 1 })).toBe("wait_for_evidence");
  });

  it("gives act only to an activation with a hit (pin_act collapse)", () => {
    const list = fallbacks(["act", "wait_for_evidence"]);
    expect(fixedL2Response(list, { hasHit: true, nPlausible: 1 })).toBe("act");
    expect(fixedL2Response(list, { hasHit: false, nPlausible: 2 })).toBe("wait_for_evidence");
  });

  it("waits when there is no list at all", () => {
    expect(fixedL2Response(null, { hasHit: true, nPlausible: 3 })).toBe("wait_for_evidence");
  });
});

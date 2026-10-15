import { cleanup, render } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import StudyPointingField, { type FieldTrialEnd } from "./StudyPointingField.svelte";
import { generatePointingLayout, mulberry32 } from "../../coadaptation/geometry";
import type { FallbacksV1, L1Variant } from "../../coadaptation/types";

const SURFACE = { w: 700, h: 500 };
const ISSUED = "2026-10-15T10:00:00.000000+00:00";
const FRESH = "2026-10-15T10:05:00.000000+00:00";

function fallbacks(overrides: Partial<FallbacksV1> = {}): FallbacksV1 {
  return {
    l1: "confirm_selection:hold_700",
    plain: "confirm_selection:hold_700",
    refresh: "confirm_selection:hold_700",
    practice: "confirm_selection:hold_700",
    freezable: [],
    freeze_conflict: null,
    l2: ["wait_for_evidence"],
    interruption_cap: null,
    issued_at: ISSUED,
    ...overrides,
  };
}

function pointer(type: string, x: number, y: number): Event {
  // jsdom has no PointerEvent: a MouseEvent with the pointer fields stands in for it.
  const event = new MouseEvent(type, { clientX: x, clientY: y, bubbles: true, button: 0 });
  for (const [key, value] of Object.entries({ pointerId: 1, isPrimary: true, pointerType: "mouse" })) {
    Object.defineProperty(event, key, { value });
  }
  return event;
}

/** A pointing trial in a hand-free mouse session, cued (the home button pressed). */
async function cuedTrial(applied: L1Variant) {
  const layout = generatePointingLayout(mulberry32(42), SURFACE, { intendedIndex: 0, wPx: 40, gapPx: 40 });
  const ends: FieldTrialEnd[] = [];
  const view = render(StudyPointingField, {
    props: {
      trial: {
        trialId: "trial-1",
        blockIndex: 1,
        trialIndex: 0,
        task: "pointing",
        layout,
        intendedIndex: layout.intendedIndex,
        labelWord: null,
        limitMs: 10000,
        record: false,
      },
      regime: {
        regime: applied,
        applied,
        decisionId: "decision-1",
        mode: "adaptive",
        l1FallbackReason: null,
        explanation: null,
        pinnedRule: null,
        pinnedVariant: null,
        fallbacks: fallbacks(),
        interruptionsUsed: 0,
        tau: 0.25,
        chips: false,
      },
      table: { table: null, decisionId: null, fallbackReason: null, fixedRule: true },
      posteriorParams: {
        modality: "mouse",
        spread: { r: 1.02, floor: 1 },
        surface: SURFACE,
        dpr: 1,
        gazeAccuracy: null,
        handPointer: false,
      },
      inputMode: "pointer",
      ontrialend: (end: FieldTrialEnd) => ends.push(end),
    },
  });
  (view.container.querySelector("button.home") as HTMLButtonElement).click();
  await vi.advanceTimersByTimeAsync(0);
  const surface = view.container.querySelector(".surface") as HTMLElement;
  const target = layout.rects[layout.intendedIndex];
  return {
    field: view.component,
    ends,
    press: () => surface.dispatchEvent(pointer("pointerdown", target.cx, target.cy)),
    release: () => surface.dispatchEvent(pointer("pointerup", target.cx, target.cy)),
    committed: () => surface.querySelector(".candidate.committed") !== null,
    holding: () => surface.querySelector(".candidate.pending") !== null,
  };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("a revert between hold regimes (section 12.5)", () => {
  it("a revert from hold_400 to hold_700 makes every later hold last 700 ms", async () => {
    vi.useFakeTimers();
    const trial = await cuedTrial("confirm_selection:hold_400");
    trial.field.refresh(fallbacks({ refresh: "confirm_selection:hold_700", issued_at: FRESH }));
    await vi.advanceTimersByTimeAsync(10);
    trial.press();
    await vi.advanceTimersByTimeAsync(450);
    // The 400 ms rule no longer applies: the hold is still open, nothing is selected.
    expect(trial.committed()).toBe(false);
    expect(trial.holding()).toBe(true);
    await vi.advanceTimersByTimeAsync(300);
    expect(trial.committed()).toBe(true);
  });

  it("a hold open during the revert keeps its own duration, and its ring shows that duration", async () => {
    vi.useFakeTimers();
    const trial = await cuedTrial("confirm_selection:hold_400");
    trial.press();
    await vi.advanceTimersByTimeAsync(100);
    trial.field.refresh(fallbacks({ refresh: "confirm_selection:hold_700", issued_at: FRESH }));
    await vi.advanceTimersByTimeAsync(200);
    // The hold began under the 400 ms rule and finishes under it: 300 ms is three quarters of its ring.
    const ring = document.querySelector('[role="progressbar"]');
    expect(Number(ring?.getAttribute("aria-valuenow"))).toBeGreaterThanOrEqual(70);
    await vi.advanceTimersByTimeAsync(150);
    expect(trial.committed()).toBe(true);
  });

  it("a hold that is open when the trial reverts to a click regime can still be completed", async () => {
    vi.useFakeTimers();
    const trial = await cuedTrial("confirm_selection:hold_400");
    trial.press();
    await vi.advanceTimersByTimeAsync(100);
    expect(trial.holding()).toBe(true);
    trial.field.dispatch({ type: "refreshPending" });
    trial.field.refresh(fallbacks({ refresh: "assistance_off", plain: "assistance_off", issued_at: FRESH }));
    await vi.advanceTimersByTimeAsync(400);
    expect(trial.committed()).toBe(true);
  });

  it("a hold that is open when the trial reverts to a click regime can still be cancelled, then a click selects", async () => {
    vi.useFakeTimers();
    const trial = await cuedTrial("confirm_selection:hold_400");
    trial.press();
    await vi.advanceTimersByTimeAsync(100);
    trial.field.dispatch({ type: "refreshPending" });
    trial.field.refresh(fallbacks({ refresh: "assistance_off", plain: "assistance_off", issued_at: FRESH }));
    await vi.advanceTimersByTimeAsync(50);
    // Releasing early cancels the open hold; the trial is not stranded in "holding".
    trial.release();
    await vi.advanceTimersByTimeAsync(10);
    expect(trial.holding()).toBe(false);
    expect(trial.committed()).toBe(false);
    // The next selection follows the revert regime: a plain click.
    trial.press();
    trial.release();
    await vi.advanceTimersByTimeAsync(10);
    expect(trial.committed()).toBe(true);
    await vi.advanceTimersByTimeAsync(2000);
    expect(trial.ends).toHaveLength(1);
    expect(trial.ends[0].state.phase).toBe("committed");
  });
});

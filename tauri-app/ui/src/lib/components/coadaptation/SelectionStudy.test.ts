import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { get, type Writable } from "svelte/store";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import SelectionStudy from "./SelectionStudy.svelte";
import { coadaptation } from "../../stores/coadaptation";
import { session } from "../../stores/session";
import { buildBlockTrials } from "../../coadaptation/studySchedule";
import { keyboardEndpoint } from "../../coadaptation/targetPosterior";
import type { FallbacksV1, Modality, RectV1, SessionPlanV1, TrialV1 } from "../../coadaptation/types";

type NotificationHandler = (method: string, params: unknown) => void;
type Params = Record<string, unknown>;

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Array<(method: string, params: unknown) => void>,
}));

vi.mock("../../api/daemon", () => ({
  call: daemonMocks.call,
  onNotification: (handler: NotificationHandler) => {
    daemonMocks.handlers.push(handler);
  },
  offNotification: vi.fn(),
  onConnectionState: vi.fn(),
}));

vi.mock("../../stores/session", async () => {
  const { writable } = await import("svelte/store");
  return { session: writable({ loading: false, confirmRequired: false }) };
});

vi.mock("../../stores/neural", async () => {
  const { writable } = await import("svelte/store");
  return { neural: writable({ state: "disconnected" }) };
});

const PK = "p-3fa29c1e";
const SEED = 1234567;
const SURFACE = { w: 700, h: 500 };
const ISSUED = "2026-10-15T10:00:00.000000+00:00";
const FRESH = "2026-10-15T10:05:00.000000+00:00";

function fallbacks(overrides: Partial<FallbacksV1> = {}): FallbacksV1 {
  return {
    l1: "confirm_selection:hold_700",
    plain: "assistance_off",
    refresh: "assistance_off",
    practice: "assistance_off",
    freezable: [
      "assistance_off",
      "act",
      "enlarge_target:1.5",
      "enlarge_target:2.0",
      "confirm_selection:second_click",
      "confirm_selection:hold_700",
    ],
    freeze_conflict: null,
    l2: ["ask_clarification:3", "ask_clarification:2", "wait_for_evidence"],
    interruption_cap: null,
    issued_at: ISSUED,
    ...overrides,
  };
}

function plan(blocks = 2, trials = 2, task: "pointing" | "menu" = "pointing"): SessionPlanV1 {
  return {
    session_number: 1,
    end_questionnaire: false,
    blocks: Array.from({ length: blocks }, (_, i) => ({
      index: i + 1,
      phase: "A" as const,
      task,
      trials,
      idle_after: false,
      burden_after: false,
    })),
  };
}

let world: {
  plan: SessionPlanV1;
  fallbacks: FallbacksV1;
  statusFallbacks: FallbacksV1;
  preferences: Array<Record<string, unknown>>;
  refreshSeq: number;
  regimes: Array<{ variant: string; mode: string }>;
  spread: { r: number; floor: number } | null;
};
let replies: Record<string, (params: Params) => unknown> = {};

function statusReply() {
  return {
    status: "ok",
    available: true,
    config: { enabled: true, learn_across_sessions: false },
    keyring_available: true,
    participants: [
      {
        participant_key: PK,
        label: "Lab A",
        consent: {
          consent_id: "c-1",
          purposes: ["selection_study"],
          granted_at: "2026-10-01T00:00:00+00:00",
          expires_at: "2099-01-01T00:00:00+00:00",
          retention_days: 90,
          consent_text_version: "coadapt-consent-v1",
          consent_text_sha256: "0".repeat(64),
          protocol_id: "coadapt-study-v1",
          active: true,
        },
        fallbacks: { mouse: world.statusFallbacks, keyboard: world.statusFallbacks },
        sessions: 0,
        last_session_at: null,
      },
    ],
    preferences: world.preferences,
    proposals: [],
    protocol: { protocol_id: "coadapt-study-v1", sha256: "a".repeat(64) },
    authority: "presentation_only",
    execution_authority: false,
  };
}

function beginReply(modality: Modality) {
  return {
    status: "ok",
    session_id: "s-1",
    participant_key: PK,
    day_index: 740000,
    session_number: 1,
    plan: world.plan,
    layout_seed: SEED,
    spread: modality === "keyboard" ? {} : { [modality]: 1.5 },
    floors: modality === "keyboard" ? {} : { [modality]: 1 },
    thresholds: { ambiguity: 0.25, bins: { b1: [0.25, 0.4], b2: [0.4, 0.6], b3: [0.6, 1] } },
    gaze_prior: false,
    hand_pointer: false,
    burden_prompt_every_blocks: 1,
    evidence_kind: "live_human",
    fallbacks: world.fallbacks,
    refresh_seq: world.refreshSeq,
  };
}

function recommendation(variant: string, mode: string, spread: { r: number; floor: number } | null) {
  return {
    decision_id: "ignored",
    level: "block_regime",
    mode,
    variant,
    params: {},
    eligible: [variant],
    excluded: {},
    propensities: { [variant]: 1 },
    p_thompson: null,
    exploration: 0,
    fallback: false,
    protocol_deviation: false,
    round_index: 0,
    spread,
    priorities: { fewer_mistakes: 0.5, faster: 0.5, fewer_interruptions: 0.5 },
    predictions: {},
    explanation: null,
    seed: "0f",
    versions: {},
    authority: "presentation_only",
    execution_authority: false,
  };
}

function defaultReplies(): Record<string, (params: Params) => unknown> {
  return {
    co_adaptation_status: () => statusReply(),
    co_adaptation_session_begin: (params) => beginReply((params.device as { modality: Modality }).modality),
    co_adaptation_recommend: (params) => {
      if (params.level === "block_regime") {
        const next = world.regimes.shift() ?? { variant: "assistance_off", mode: "randomized" };
        return {
          status: "ok",
          logged: true,
          recommendation: { ...recommendation(next.variant, next.mode, world.spread), decision_id: params.decision_id },
        };
      }
      if (params.level === "offer") {
        return { status: "ok", logged: true, recommendation: { ...recommendation("no_offer", "randomized", null) } };
      }
      return { status: "error", message: "no table in this test" };
    },
    co_adaptation_record_trial: () => ({
      status: "recorded",
      verification: "ok",
      labels: null,
      preference_violation: false,
      refresh_required: false,
      refresh_seq: world.refreshSeq,
      fallbacks: null,
    }),
    co_adaptation_report_burden: () => ({ status: "recorded" }),
    co_adaptation_session_resume: () => ({
      status: "ok",
      refresh_required: false,
      refresh_seq: world.refreshSeq,
      fallbacks: world.fallbacks,
    }),
    co_adaptation_session_end: () => ({
      status: "ok",
      summary: {
        trials: 2,
        completion: 1,
        ua_rate: 0,
        idle_ua_per_min: null,
        median_time_ms: 900,
        burden_mean: {},
        overrides: 0,
        regime_entropy_bits: 0,
        switch_rate: 0,
        allowed_now: [],
        excluded_now: {},
        predictions: {},
      },
    }),
    co_adaptation_preference_set: () => ({
      status: "ok",
      preference: null,
      notes: [],
      refresh_required: false,
      refresh_seq: null,
      fallbacks: null,
    }),
    co_adaptation_preference_retract: () => ({
      status: "ok",
      refresh_required: false,
      refresh_seq: null,
      fallbacks: null,
    }),
  };
}

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => SURFACE.w });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => SURFACE.h });
});

afterAll(() => {
  delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
  delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientHeight;
});

beforeEach(async () => {
  world = {
    plan: plan(),
    fallbacks: fallbacks(),
    statusFallbacks: fallbacks(),
    preferences: [],
    refreshSeq: 4,
    regimes: [],
    spread: { r: 1.02, floor: 1 },
  };
  replies = defaultReplies();
  daemonMocks.call.mockReset();
  daemonMocks.call.mockImplementation(async (method: string, params: Params) => {
    const reply = replies[method];
    if (!reply) throw new Error(`unexpected call ${method}`);
    return reply(params ?? {});
  });
  (session as unknown as Writable<Record<string, unknown>>).set({ loading: false, confirmRequired: false });
  await coadaptation.load();
});

afterEach(async () => {
  cleanup();
  await coadaptation.exitStudy();
  vi.useRealTimers();
});

// ---- Helpers -----------------------------------------------------------------------------------------------------

function calls(method: string): Params[] {
  return daemonMocks.call.mock.calls.filter(([name]) => name === method).map(([, params]) => params as Params);
}

function recommendCalls(level: string): Params[] {
  return calls("co_adaptation_recommend").filter((params) => params.level === level);
}

function recordedTrials(): TrialV1[] {
  return calls("co_adaptation_record_trial").map((params) => params.trial as TrialV1);
}

function broadcast(method: string, params: unknown) {
  for (const handler of daemonMocks.handlers) handler(method, params);
}

function pointer(type: string, x: number, y: number): Event {
  // jsdom has no PointerEvent: a MouseEvent with the pointer fields stands in for it.
  const event = new MouseEvent(type, { clientX: x, clientY: y, bubbles: true, button: 0 });
  for (const [key, value] of Object.entries({ pointerId: 1, isPrimary: true, pointerType: "mouse" })) {
    Object.defineProperty(event, key, { value });
  }
  return event;
}

function layoutOf(blockIndex: number, trialIndex: number, modality: Modality = "mouse") {
  return buildBlockTrials(world.plan, blockIndex, SEED, SURFACE, modality).trials[trialIndex];
}

function surfaceEl(): HTMLElement {
  const el = document.querySelector(".surface");
  if (!el) throw new Error("no study surface");
  return el as HTMLElement;
}

/** A press and release at the centre of a rect (or at a point). */
function click(target: RectV1 | { x: number; y: number }) {
  const x = "cx" in target ? target.cx : target.x;
  const y = "cy" in target ? target.cy : target.y;
  surfaceEl().dispatchEvent(pointer("pointerdown", x, y));
  surfaceEl().dispatchEvent(pointer("pointerup", x, y));
}

async function openStudy(modality: "pointer" | "keyboard" = "pointer") {
  coadaptation.openStudy();
  const onexit = vi.fn();
  const view = render(SelectionStudy, { props: { paused: false, onexit } });
  await screen.findByText("Who is selecting?");
  if (modality === "keyboard") await fireEvent.click(screen.getByLabelText("Keyboard (arrows and Enter)"));
  return { ...view, onexit };
}

async function startSession(modality: "pointer" | "keyboard" = "pointer") {
  const view = await openStudy(modality);
  await fireEvent.click(screen.getByRole("button", { name: "Start" }));
  await screen.findByRole("button", { name: "Start block" });
  return view;
}

async function startBlock() {
  const start = (await screen.findByRole("button", { name: "Start block" })) as HTMLButtonElement;
  await vi.waitFor(() => expect(start.disabled).toBe(false));
  await fireEvent.click(start);
}

/** Presses the home button so the trial shows its cue. */
async function cue() {
  await fireEvent.click(await screen.findByRole("button", { name: "Start" }));
  await screen.findByText(/^Select: /);
}

// ---- Tests -------------------------------------------------------------------------------------------------------

describe("before the session", () => {
  it("sends no study RPC without an active consent", async () => {
    replies.co_adaptation_status = () => ({ ...statusReply(), participants: [] });
    await coadaptation.load();
    await openStudy();
    const start = screen.getByRole("button", { name: "Start" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    await fireEvent.click(start);
    for (const method of ["co_adaptation_session_begin", "co_adaptation_recommend", "co_adaptation_record_trial"]) {
      expect(calls(method)).toHaveLength(0);
    }
  });

  it("returns to Setup with the message of a session_begin refusal", async () => {
    replies.co_adaptation_session_begin = () => ({ status: "refused", reason: "recording_active" });
    await openStudy();
    await fireEvent.click(screen.getByRole("button", { name: "Start" }));
    expect(
      await screen.findByText("Stop the workflow recording first: it could record your study clicks."),
    ).toBeTruthy();
    expect(calls("co_adaptation_recommend")).toHaveLength(0);
  });

  it("shows the start refusals it can see (a running task)", async () => {
    (session as unknown as Writable<Record<string, unknown>>).set({ loading: true, confirmRequired: false });
    await openStudy();
    expect(screen.getByText("Finish or stop the running task first.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("block start", () => {
  it("requests the block's decision without the cued target and shows the intro for its mode", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await startSession();
    expect(await screen.findByText("Block 1 of 2")).toBeTruthy();
    expect(await screen.findByText("This block follows the study schedule you agreed to.")).toBeTruthy();
    const [request] = recommendCalls("block_regime");
    const context = request.context as Record<string, unknown>;
    expect(Object.keys(context).sort()).toEqual(
      ["block_index", "hand_quality_recent", "layouts", "previous_regime", "session_minute", "trials_today"].sort(),
    );
    const text = JSON.stringify(request);
    expect(text).not.toContain("intended");
    expect(text).not.toContain("target");
    expect((context.layouts as unknown[]).length).toBe(2);
  });

  const MODES = [
    ["probe", "assistance_off", "This block uses standard selection without assistance."],
    ["comparator", "act", "This block uses a fixed kind of help, for comparison."],
    ["adaptive", "enlarge_target:1.5", "Heliox chose this help for you."],
    ["pinned", "confirm_selection:hold_700", "You paused adapting, so this block keeps the help you chose."],
  ] as const;
  for (const [mode, variant, text] of MODES) {
    it(`shows the ${mode} intro`, async () => {
      world.regimes = [{ variant, mode }];
      await startSession();
      expect(await screen.findByText(text)).toBeTruthy();
    });
  }

  const NO_BLOCK_CASES: Array<[string, () => Promise<void>]> = [
    [
      "fallbacks.l1 is null",
      async () => {
        world.fallbacks = fallbacks({ l1: null });
      },
    ],
    [
      "the session holds no fallbacks",
      async () => {
        // Neither the session_begin reply nor the status carries fallbacks for the session's modality.
        const begin = defaultReplies().co_adaptation_session_begin;
        replies.co_adaptation_session_begin = (params) => ({ ...(begin(params) as Params), fallbacks: null });
        replies.co_adaptation_status = () => {
          const status = statusReply();
          status.participants[0].fallbacks = {} as (typeof status.participants)[0]["fallbacks"];
          return status;
        };
        await coadaptation.load();
      },
    ],
  ];
  for (const [name, arrange] of NO_BLOCK_CASES) {
    it(`does not start a block when ${name}`, async () => {
      await arrange();
      await startSession();
      expect(
        await screen.findByText(
          "Your rules and Heliox's current caution leave no kind of help to use in this block. Choose Pause adapting to continue without changes, change a rule, or stop the session.",
        ),
      ).toBeTruthy();
      expect((screen.getByRole("button", { name: "Start block" }) as HTMLButtonElement).disabled).toBe(true);
      expect(recommendCalls("block_regime")).toHaveLength(0);
    });
  }

  it("pauses on a refused recommend and asks again after session_resume", async () => {
    let first = true;
    replies.co_adaptation_recommend = () => {
      if (first) {
        first = false;
        return { status: "refused", reason: "task_running" };
      }
      return {
        status: "ok",
        logged: true,
        recommendation: recommendation("assistance_off", "randomized", world.spread),
      };
    };
    await startSession();
    expect(await screen.findByText("Paused")).toBeTruthy();
    expect(screen.getByText("Finish or stop the running task first.")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
    const order = daemonMocks.call.mock.calls.map(([name]) => name);
    expect(order.indexOf("co_adaptation_session_resume")).toBeLessThan(order.lastIndexOf("co_adaptation_recommend"));
  });

  it("applies fallbacks.l1 when the block decision fails, and records the L1 fallback", async () => {
    replies.co_adaptation_recommend = () => {
      throw new Error("connection lost");
    };
    world.fallbacks = fallbacks({ l1: "assistance_off" });
    await startSession();
    expect(await screen.findByText("Using the most cautious option your rules allow.")).toBeTruthy();
    await startBlock();
    await cue();
    click(layoutOf(1, 0).rects[layoutOf(1, 0).intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.applied_regime).toBe("assistance_off");
    expect(trial.l1_fallback_reason).toBe("rpc_error");
    expect(trial.spread).toEqual({ r: 1.5, floor: 1 });
  });
});

describe("trials", () => {
  it("records a committed selection with integer endpoints and the block's spread pair", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    world.spread = { r: 1.02, floor: 1 };
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    click({ x: target.cx + 0.4, y: target.cy - 0.4 });
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.spread).toEqual({ r: 1.02, floor: 1 });
    expect(trial.intended_index).toBe(scheduled.intendedIndex);
    expect(trial.committed_index).toBe(scheduled.intendedIndex);
    expect(trial.activations).toHaveLength(1);
    const [activation] = trial.activations;
    expect(Number.isInteger(activation.x) && Number.isInteger(activation.y)).toBe(true);
    expect(activation.press_index).toBe(scheduled.intendedIndex);
    expect(activation.hit_rect).toEqual(target);
    expect(activation.expanded_index).toBeNull();
    expect(activation.result).toBe("committed");
    expect(trial.interrupted).toBe(false);
  });

  it("records a press on a gap as press_index -1 with no literal hit", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    // Press on empty space well above the grid, release on the target.
    surfaceEl().dispatchEvent(pointer("pointerdown", 4, 4));
    surfaceEl().dispatchEvent(pointer("pointerup", target.cx, target.cy));
    await fireEvent.click(surfaceEl());
    // Then select the target so the trial ends.
    click(target);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [first, second] = recordedTrials()[0].activations;
    expect(first.press_index).toBe(-1);
    expect(first.hit_index).toBeNull();
    expect(first.result).toBe("none");
    expect(second.result).toBe("committed");
  });

  it("excludes the current trial as interrupted when the paused prop turns on", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    const view = await startSession();
    await startBlock();
    await cue();
    await view.rerender({ paused: true, onexit: view.onexit });
    expect(await screen.findByText("Paused")).toBeTruthy();
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    expect(recordedTrials()[0].interrupted).toBe(true);
    expect(recordedTrials()[0].interruption_reason).toBe("modal");
    expect((screen.getByRole("button", { name: "Resume" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("a keyboard session sends and echoes a null spread and records keyboard activations", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    world.spread = null;
    await startSession("keyboard");
    expect(calls("co_adaptation_session_begin")[0].device).toMatchObject({ modality: "keyboard" });
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0, "keyboard");
    await fireEvent.keyDown(surfaceEl(), { key: "Enter" });
    // The focused candidate may not be the cued one: a wrong selection ends after its Undo window.
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1), { timeout: 3000 });
    const [trial] = recordedTrials();
    expect(trial.spread).toBeNull();
    expect(trial.modality).toBe("keyboard");
    const [activation] = trial.activations;
    expect(activation.modality).toBe("keyboard");
    expect(activation.a_gate).toBe(0);
    expect(activation.n_plausible).toBe(1);
    expect(activation.press_index).toBe(activation.hit_index);
    const focused = scheduled.rects[activation.hit_index!];
    expect({ x: activation.x, y: activation.y }).toEqual(keyboardEndpoint(focused));
    expect(activation.hit_rect).toEqual(focused);
    expect(activation.gaze).toBeNull();
  });
});

describe("session end", () => {
  it("Stop session ends the open session as stopped", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    const { onexit } = await startSession();
    await fireEvent.click((await screen.findAllByRole("button", { name: "Pause" }))[0]);
    await fireEvent.click(await screen.findByRole("button", { name: "Stop session" }));
    await vi.waitFor(() => expect(calls("co_adaptation_session_end")).toHaveLength(1));
    expect(calls("co_adaptation_session_end")[0]).toEqual({ session_id: "s-1", reason: "stopped" });
    expect(onexit).toHaveBeenCalled();
  });

  for (const reason of ["reset", "erased"] as const) {
    it(`a ${reason} notification for the running participant ends the session`, async () => {
      world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
      await startSession();
      broadcast("co_adaptation_updated", { reason, participant_key: PK, refresh_seq: null });
      const message =
        reason === "reset"
          ? "This session ended because learning was reset."
          : "This session ended because today's data was erased.";
      expect(await screen.findByText(message)).toBeTruthy();
      expect(get(coadaptation).study.sessionId).toBe("");
    });
  }
});

// ---- L2 responses, vetoes and fallbacks ----------------------------------------------------------------------------

function table(cells: Array<{ cell: string; variant: string }>, startTrialIndex = 0, rows = 2) {
  const order = [
    "act",
    "confirm_selection:second_click",
    "wait_for_evidence",
    "ask_clarification:2",
    "ask_clarification:3",
  ];
  return {
    decision_id: "ignored",
    level: "ambiguity_table",
    block_decision_id: "ignored",
    mode: "adaptive",
    ambiguity_threshold: 0.25,
    bins: [
      { id: "b1", lo: 0.25, hi: 0.4 },
      { id: "b2", lo: 0.4, hi: 0.6 },
      { id: "b3", lo: 0.6, hi: 1 },
    ],
    cells: cells.map(({ cell, variant }) => {
      const [bin, p, h] = cell.split(":");
      return {
        cell_id: cell,
        bin,
        n_plausible: Number(p.slice(1)),
        has_hit: h === "h1",
        order,
        propensities: Object.fromEntries(order.map((name) => [name, name === variant ? 1 : 0])),
        eligible: [variant],
        excluded: { act: "ambiguous_activation" },
      };
    }),
    start_trial_index: startTrialIndex,
    draws: Array.from({ length: rows }, () => [0.5, 0.5, 0.5, 0.5]),
    max_draws_per_trial: 4,
    after_max: "wait_for_evidence",
    explanation: null,
    seed: "0f",
    versions: {},
    authority: "presentation_only",
    execution_authority: false,
  };
}

const ALL_NO_HIT_CELLS = ["b3:p1:h0", "b3:p2:h0", "b3:p3:h0"];

describe("L2 responses", () => {
  it("applies the table's draw to an ambiguous activation and records its source", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    const base = defaultReplies().co_adaptation_recommend;
    replies.co_adaptation_recommend = (params) =>
      params.level === "ambiguity_table"
        ? {
            status: "ok",
            logged: true,
            recommendation: {
              ...table(ALL_NO_HIT_CELLS.map((cell) => ({ cell, variant: "wait_for_evidence" }))),
              decision_id: params.decision_id,
            },
          }
        : base(params);
    await startSession();
    await startBlock();
    const tableRequest = recommendCalls("ambiguity_table")[0];
    expect((tableRequest.context as Params).start_trial_index).toBe(0);
    await cue();
    click({ x: 4, y: 4 });
    expect(await screen.findByText("Not selected — aim again")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Not this time" })).toBeTruthy();
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.activations[0].l2).toEqual({
      decision_id: tableRequest.decision_id,
      cell_id: "b3:p1:h0",
      draw_index: 0,
      variant: "wait_for_evidence",
      source: "table",
      fallback_reason: null,
    });
    expect(trial.activations[0].result).toBe("none");
    expect(trial.table_decision_ids).toContain(tableRequest.decision_id);
  });

  it("a failed table records the L2 fallback on the activation only", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    const base = defaultReplies().co_adaptation_recommend;
    replies.co_adaptation_recommend = (params) => {
      if (params.level === "ambiguity_table") throw new Error("connection lost");
      return base(params);
    };
    await startSession();
    await startBlock();
    await cue();
    click({ x: 4, y: 4 });
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    const tableRequest = recommendCalls("ambiguity_table")[0];
    expect(trial.l1_fallback_reason).toBeNull();
    expect(trial.activations[0].l2).toMatchObject({
      decision_id: tableRequest.decision_id,
      source: "fallback",
      fallback_reason: "rpc_error",
      variant: "wait_for_evidence",
    });
  });
});

describe("Not this time", () => {
  it("reverts the trial to fallbacks.plain and records the veto", async () => {
    world.regimes = [{ variant: "confirm_selection:second_click", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    const l1Request = recommendCalls("block_regime")[0];
    expect(trial.veto).toEqual({ decision_id: l1Request.decision_id, level: "block_regime" });
    expect(trial.revert).toEqual({
      reason: "veto",
      from_activation: 0,
      regime: "assistance_off",
      fallbacks_issued_at: ISSUED,
    });
    expect(trial.activations[0].result).toBe("committed");
  });

  it("with C7 fallbacks (no freeze or pin) reverts to the non-acting plain regime, never assistance_off", async () => {
    world.fallbacks = fallbacks({
      l1: "confirm_selection:hold_700",
      plain: "confirm_selection:hold_700",
      refresh: "confirm_selection:hold_700",
      practice: "confirm_selection:hold_700",
    });
    world.regimes = [{ variant: "confirm_selection:second_click", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    surfaceEl().dispatchEvent(pointer("pointerdown", target.cx, target.cy));
    expect(await screen.findByText(/^Keep holding to select/)).toBeTruthy();
    surfaceEl().dispatchEvent(pointer("pointerup", target.cx, target.cy));
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.revert?.regime).toBe("confirm_selection:hold_700");
    expect(trial.activations[0].result).toBe("held");
    expect(trial.activations[0].outcome).toBe("cancelled");
    expect(trial.committed_index).toBeNull();
  });

  it("the practice block under C7 fallbacks uses the non-acting practice regime and no RPC", async () => {
    world.fallbacks = fallbacks({ practice: "confirm_selection:hold_700", plain: "confirm_selection:hold_700" });
    world.plan = {
      ...plan(1, 2),
      blocks: [
        { index: 0, phase: "practice", task: "pointing", trials: 2, idle_after: false, burden_after: false },
        ...plan(1, 2).blocks,
      ],
    };
    await openStudy();
    await fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Start practice" }));
    await cue();
    const scheduled = layoutOf(0, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    surfaceEl().dispatchEvent(pointer("pointerdown", target.cx, target.cy));
    expect(await screen.findByText(/^Keep holding to select/)).toBeTruthy();
    expect(calls("co_adaptation_recommend")).toHaveLength(0);
    expect(calls("co_adaptation_record_trial")).toHaveLength(0);
  });

  it("with no plain regime interrupts the trial with no_allowed_regime and ends the block", async () => {
    world.fallbacks = fallbacks({ plain: null });
    world.regimes = [
      { variant: "confirm_selection:second_click", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.interrupted).toBe(true);
    expect(trial.interruption_reason).toBe("no_allowed_regime");
    expect(trial.veto).toMatchObject({ level: "block_regime" });
    expect(trial.revert).toBeNull();
    expect(trial.committed_index).toBeNull();
    expect(
      await screen.findByText(
        "Your rules and Heliox's current caution leave nothing else to use for this selection, so this block ends.",
      ),
    ).toBeTruthy();
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
  });

  it("in a frozen block reverts this trial only and offers to resume adapting", async () => {
    world.preferences = [
      {
        pref_id: "pref-freeze",
        kind: "freeze_adaptation",
        scope: { context: "study", modality: "mouse", participant_key: PK },
        value: { regime: "confirm_selection:hold_700" },
        origin: "explicit_user",
        created_at: "2026-10-15T09:00:00+00:00",
        retracted_at: null,
      },
    ];
    await coadaptation.load();
    world.regimes = [{ variant: "confirm_selection:hold_700", mode: "pinned" }];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(
      screen.getByText("You paused adapting, so Heliox keeps this help in later blocks. Resume adapting?"),
    ).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Keep paused" }));
    expect(calls("co_adaptation_preference_retract")).toHaveLength(0);
    expect(calls("co_adaptation_preference_set")).toHaveLength(0);
    expect(recommendCalls("ambiguity_table")).toHaveLength(0);
  });

  it("Resume adapting retracts the freeze", async () => {
    world.preferences = [
      {
        pref_id: "pref-freeze",
        kind: "freeze_adaptation",
        scope: { context: "study", modality: "mouse", participant_key: PK },
        value: { regime: "confirm_selection:hold_700" },
        origin: "explicit_user",
        created_at: "2026-10-15T09:00:00+00:00",
        retracted_at: null,
      },
    ];
    await coadaptation.load();
    world.regimes = [{ variant: "confirm_selection:hold_700", mode: "pinned" }];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await fireEvent.click(screen.getByRole("button", { name: "Resume adapting" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_retract")).toHaveLength(1));
    expect(calls("co_adaptation_preference_retract")[0]).toEqual({ pref_id: "pref-freeze" });
  });

  it("with pin_act, Not this time on an adaptive block's regime chip is an ordinary veto with no rule removal", async () => {
    world.preferences = [
      {
        pref_id: "pref-pin-act",
        kind: "pin_act",
        scope: { context: "study", modality: null, participant_key: PK },
        value: true,
        origin: "explicit_user",
        created_at: "2026-10-15T09:00:00+00:00",
        retracted_at: null,
      },
    ];
    world.fallbacks = fallbacks({ l2: ["act", "wait_for_evidence"] });
    world.statusFallbacks = world.fallbacks;
    await coadaptation.load();
    // pin_act collapses only L2 responses: enlarged targets stay an adaptive L1 regime with a normal chip.
    world.regimes = [{ variant: "enlarge_target:2.0", mode: "adaptive" }];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(screen.queryByText("You chose to select exactly what you click. Remove that rule?")).toBeNull();
    expect(screen.queryByRole("button", { name: "Remove rule" })).toBeNull();
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.veto).toMatchObject({ level: "block_regime" });
    expect(trial.revert).toMatchObject({ reason: "veto", regime: "assistance_off" });
    expect(calls("co_adaptation_preference_retract")).toHaveLength(0);
  });
});

describe("refresh signals", () => {
  const freshFallbacks = () => fallbacks({ issued_at: FRESH, refresh: "assistance_off" });

  it("Never pauses input during its step, saves the rule, and one change reverts the trial once", async () => {
    world.regimes = [
      { variant: "confirm_selection:second_click", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    replies.co_adaptation_preference_set = () => {
      world.refreshSeq = 5;
      world.statusFallbacks = freshFallbacks();
      return {
        status: "ok",
        preference: null,
        notes: [],
        refresh_required: true,
        refresh_seq: 5,
        fallbacks: freshFallbacks(),
      };
    };
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];

    await fireEvent.click(screen.getByRole("button", { name: "Never" }));
    click(target);
    await fireEvent.click(screen.getByRole("button", { name: "Yes" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0]).toEqual({
      kind: "forbid_variant",
      scope: { context: "study", modality: null, participant_key: PK },
      value: "confirm_selection:second_click",
    });
    await vi.waitFor(() => expect(get(coadaptation).study.lastRefreshSeq).toBe(5));
    // The broadcast of the same change arrives afterwards: a duplicate.
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));

    click(target);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.interrupted).toBe(false);
    expect(trial.revert).toEqual({
      reason: "refresh",
      from_activation: 0,
      regime: "assistance_off",
      fallbacks_issued_at: FRESH,
    });
    expect(trial.activations).toHaveLength(1);
    expect(trial.activations[0].result).toBe("committed");
    // The block ends after this trial, and the next one starts with fresh decisions.
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
  });

  it("a drift broadcast in a block pinned to enlarge_target:1.5 keeps that regime", async () => {
    world.preferences = [
      {
        pref_id: "pref-pin",
        kind: "pin_regime",
        scope: { context: "study", modality: null, participant_key: PK },
        value: "enlarge_target:1.5",
        origin: "explicit_user",
        created_at: "2026-10-15T09:00:00+00:00",
        retracted_at: null,
      },
    ];
    world.fallbacks = fallbacks({
      l1: "enlarge_target:1.5",
      refresh: "enlarge_target:1.5",
      practice: "enlarge_target:1.5",
    });
    world.statusFallbacks = world.fallbacks;
    await coadaptation.load();
    world.regimes = [{ variant: "enlarge_target:1.5", mode: "pinned" }];
    const base = defaultReplies().co_adaptation_recommend;
    replies.co_adaptation_recommend = (params) =>
      params.level === "ambiguity_table"
        ? { status: "ok", logged: true, recommendation: { ...table([]), mode: "pinned" } }
        : base(params);
    await startSession();
    await startBlock();
    await cue();
    world.statusFallbacks = fallbacks({
      l1: "enlarge_target:1.5",
      refresh: "enlarge_target:1.5",
      practice: "enlarge_target:1.5",
      issued_at: FRESH,
    });
    broadcast("co_adaptation_updated", { reason: "drift", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.fallbacks?.issued_at).toBe(FRESH));
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.revert).toMatchObject({ reason: "refresh", regime: "enlarge_target:1.5" });
    expect(trial.revert?.regime).not.toBe("confirm_selection:hold_700");
  });

  it("a rule saved in Settings while paused: the next trial waits for a fresh block decision", async () => {
    world.regimes = [
      { variant: "assistance_off", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    world.refreshSeq = 5;
    world.statusFallbacks = fallbacks({ issued_at: FRESH });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.lastRefreshSeq).toBe(5));
    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: fallbacks({ issued_at: FRESH }),
    });
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
    expect(screen.queryByText(/^Select: /)).toBeNull();
  });

  it("the same holds when only the session_resume reply carries the new refresh_seq", async () => {
    world.regimes = [
      { variant: "assistance_off", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: fallbacks({ issued_at: FRESH }),
    });
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
  });
});

describe("refresh signals with no open decision", () => {
  it("a rule change during practice switches the running practice trial to the fresh fallbacks", async () => {
    world.plan = {
      ...plan(1, 2),
      blocks: [
        { index: 0, phase: "practice", task: "pointing", trials: 2, idle_after: false, burden_after: false },
        ...plan(1, 2).blocks,
      ],
    };
    await openStudy();
    await fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Start practice" }));
    await cue();
    // Practice requests no decision; the daemon still sends a new refresh_seq for the change.
    world.statusFallbacks = fallbacks({
      issued_at: FRESH,
      practice: "confirm_selection:hold_700",
      refresh: "confirm_selection:hold_700",
    });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.fallbacks?.issued_at).toBe(FRESH));
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    // The trial began under assistance_off (a press would select at once); it now needs a hold.
    const scheduled = layoutOf(0, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    surfaceEl().dispatchEvent(pointer("pointerdown", target.cx, target.cy));
    expect(await screen.findByText(/^Keep holding to select/)).toBeTruthy();
    expect(calls("co_adaptation_recommend")).toHaveLength(0);
    expect(calls("co_adaptation_record_trial")).toHaveLength(0);
  });

  it("a rule change in a block whose L1 request failed ends the block after the running trial", async () => {
    const base = defaultReplies().co_adaptation_recommend;
    let failL1 = true;
    replies.co_adaptation_recommend = (params) =>
      params.level === "block_regime" && failL1 ? { status: "error", message: "engine failed" } : base(params);
    world.fallbacks = fallbacks({ l1: "confirm_selection:second_click" });
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    failL1 = false;
    world.statusFallbacks = fallbacks({ issued_at: FRESH, l1: "assistance_off", refresh: "assistance_off" });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.fallbacks?.issued_at).toBe(FRESH));
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.applied_regime).toBe("confirm_selection:second_click");
    expect(trial.l1_fallback_reason).not.toBeNull();
    expect(trial.revert).toMatchObject({ reason: "refresh", regime: "assistance_off", fallbacks_issued_at: FRESH });
    expect(trial.committed_index).toBe(scheduled.intendedIndex);
    // The block's second trial never runs on the old fallbacks: block 2 starts with a fresh decision.
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
  });
});

describe("Pause adapting", () => {
  it("mid-trial freezes the current hold regime, and the next activation still starts a hold", async () => {
    world.regimes = [{ variant: "confirm_selection:hold_700", mode: "randomized" }];
    const fresh = fallbacks({
      issued_at: FRESH,
      l1: "confirm_selection:hold_700",
      refresh: "confirm_selection:hold_700",
      practice: "confirm_selection:hold_700",
    });
    replies.co_adaptation_preference_set = () => {
      world.refreshSeq = 5;
      world.statusFallbacks = fresh;
      return { status: "ok", preference: null, notes: [], refresh_required: true, refresh_seq: 5, fallbacks: fresh };
    };
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Pause adapting" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0]).toEqual({
      kind: "freeze_adaptation",
      scope: { context: "study", modality: "mouse", participant_key: PK },
      value: { regime: "confirm_selection:hold_700" },
    });
    expect(await screen.findByText("Adapting paused. Heliox keeps Hold to select until you resume.")).toBeTruthy();
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    surfaceEl().dispatchEvent(pointer("pointerdown", target.cx, target.cy));
    expect(await screen.findByText(/^Keep holding to select/)).toBeTruthy();
    surfaceEl().dispatchEvent(pointer("pointerup", target.cx, target.cy));
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.revert).toMatchObject({ reason: "refresh", regime: "confirm_selection:hold_700" });
    expect(trial.activations[0].result).toBe("held");
    expect(trial.committed_index).toBeNull();
  });

  it("freezes at the most cautious allowed regime when the current one is not freezable", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    world.fallbacks = fallbacks({ freezable: ["confirm_selection:second_click", "confirm_selection:hold_700"] });
    await startSession();
    await startBlock();
    await fireEvent.click(screen.getByRole("button", { name: "Pause adapting" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0].value).toEqual({ regime: "confirm_selection:hold_700" });
    expect(
      await screen.findByText(
        "Adapting paused. Your rules do not allow keeping Select when the click is clear, so Heliox uses Hold to select, the most cautious option they allow.",
      ),
    ).toBeTruthy();
  });

  it("with nothing freezable sends nothing and offers to remove the blocking rule", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    world.fallbacks = fallbacks({
      freezable: [],
      freeze_conflict: {
        message:
          "Your rule 'At most 2 extra steps per block' (Only this participant) blocks a pause. Remove that rule?",
        conflicting: ["pref-cap", "pref-min"],
      },
    });
    replies.co_adaptation_preference_retract = (params) => ({
      status: "ok",
      refresh_required: params.pref_id === "pref-min",
      refresh_seq: params.pref_id === "pref-min" ? 5 : null,
      fallbacks: params.pref_id === "pref-min" ? fallbacks({ issued_at: FRESH }) : null,
    });
    await startSession();
    await startBlock();
    await fireEvent.click(screen.getByRole("button", { name: "Pause adapting" }));
    expect(
      await screen.findByText(
        "Your rule 'At most 2 extra steps per block' (Only this participant) blocks a pause. Remove that rule?",
      ),
    ).toBeTruthy();
    expect(calls("co_adaptation_preference_set")).toHaveLength(0);
    await fireEvent.click(screen.getByRole("button", { name: "Remove that rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_retract")).toEqual([{ pref_id: "pref-cap" }, { pref_id: "pref-min" }]);
    expect(calls("co_adaptation_preference_set")[0].kind).toBe("freeze_adaptation");
  });

  it("a conflict reply to Pause with standard selection: Remove that rule retries standard selection", async () => {
    world.regimes = [{ variant: "confirm_selection:second_click", mode: "adaptive" }];
    world.preferences = [
      {
        pref_id: "pref-mc",
        kind: "min_caution",
        scope: { context: "study", modality: null, participant_key: PK },
        value: "confirm_selection:second_click",
        origin: "explicit_user",
        created_at: "2026-10-15T08:00:00+00:00",
        retracted_at: null,
      },
    ];
    await coadaptation.load();
    let sets = 0;
    replies.co_adaptation_preference_set = () => {
      sets += 1;
      if (sets === 1) {
        return {
          status: "error",
          reason: "preference_conflict",
          message: "Your minimum caution is Confirm each selection; Standard selection is below it.",
          conflicting: [{ pref_id: "pref-mc", summary: "Minimum caution: Confirm each selection" }],
        };
      }
      return { status: "ok", preference: null, notes: [], refresh_required: false, refresh_seq: null, fallbacks: null };
    };
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Remove that rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(2));
    expect(calls("co_adaptation_preference_retract")).toEqual([{ pref_id: "pref-mc" }]);
    // The retry is the person's own choice again (standard selection), never the current confirm regime.
    expect(calls("co_adaptation_preference_set").map((params) => params.value)).toEqual([
      { regime: "assistance_off" },
      { regime: "assistance_off" },
    ]);
  });

  describe("replacing an earlier pause of the same scope", () => {
    const earlier = {
      pref_id: "pref-freeze",
      kind: "freeze_adaptation",
      scope: { context: "study", modality: "mouse", participant_key: PK },
      value: { regime: "confirm_selection:hold_700" },
      origin: "explicit_user",
      created_at: "2026-10-15T09:00:00+00:00",
      retracted_at: null,
    };

    beforeEach(async () => {
      world.preferences = [earlier];
      world.regimes = [{ variant: "confirm_selection:hold_700", mode: "pinned" }];
      await coadaptation.load();
    });

    it("retracts the earlier freeze first, then sends the new one, and says so", async () => {
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
      const order = daemonMocks.call.mock.calls
        .map(([name]) => name)
        .filter((name) => name === "co_adaptation_preference_set" || name === "co_adaptation_preference_retract");
      expect(order).toEqual(["co_adaptation_preference_retract", "co_adaptation_preference_set"]);
      expect(calls("co_adaptation_preference_retract")[0]).toEqual({ pref_id: "pref-freeze" });
      expect(calls("co_adaptation_preference_set")[0].value).toEqual({ regime: "assistance_off" });
      expect(
        await screen.findByText(
          "Adapting paused. Heliox keeps Standard selection until you resume. This replaces your earlier pause at Hold to select.",
        ),
      ).toBeTruthy();
    });

    const minCaution = {
      pref_id: "pref-min",
      kind: "min_caution",
      scope: { context: "study", modality: null, participant_key: PK },
      value: "confirm_selection:second_click",
      origin: "explicit_user",
      created_at: "2026-10-15T08:00:00+00:00",
      retracted_at: null,
    };

    it("sends nothing, keeps the earlier pause and names the rule that blocks standard selection", async () => {
      world.preferences = [earlier, minCaution];
      // As the daemon sends them: freeze_conflict is computed for the current regime (hold_700), which is allowed.
      world.fallbacks = fallbacks({
        l1: "confirm_selection:hold_700",
        plain: "confirm_selection:second_click",
        freezable: ["confirm_selection:second_click", "confirm_selection:hold_700"],
        freeze_conflict: null,
      });
      await coadaptation.load();
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      expect(
        await screen.findByText(
          "Your rules do not allow standard selection: Minimum caution: Confirm each selection. Remove that rule to pause with standard selection, or keep it.",
        ),
      ).toBeTruthy();
      expect(screen.getByRole("button", { name: "Remove that rule" })).toBeTruthy();
      expect(calls("co_adaptation_preference_set")).toHaveLength(0);
      expect(calls("co_adaptation_preference_retract")).toHaveLength(0);
    });

    it("Remove that rule after a standard-selection conflict retries standard selection, not another regime", async () => {
      world.preferences = [earlier, minCaution];
      world.fallbacks = fallbacks({
        l1: "confirm_selection:hold_700",
        plain: "confirm_selection:second_click",
        freezable: ["confirm_selection:second_click", "confirm_selection:hold_700"],
        freeze_conflict: null,
      });
      replies.co_adaptation_preference_retract = (params) => {
        world.preferences = world.preferences.filter((pref) => pref.pref_id !== params.pref_id);
        if (params.pref_id !== "pref-min") {
          return { status: "ok", refresh_required: false, refresh_seq: null, fallbacks: null };
        }
        world.refreshSeq = 5;
        return { status: "ok", refresh_required: true, refresh_seq: 5, fallbacks: fallbacks({ issued_at: FRESH }) };
      };
      await coadaptation.load();
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await fireEvent.click(await screen.findByRole("button", { name: "Remove that rule" }));
      await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
      expect(calls("co_adaptation_preference_retract")).toEqual([{ pref_id: "pref-min" }, { pref_id: "pref-freeze" }]);
      expect(calls("co_adaptation_preference_set")[0]).toEqual({
        kind: "freeze_adaptation",
        scope: { context: "study", modality: "mouse", participant_key: PK },
        value: { regime: "assistance_off" },
      });
    });

    it("re-sends the earlier freeze unchanged when the daemon rejects the new one", async () => {
      let sets = 0;
      replies.co_adaptation_preference_set = () => {
        sets += 1;
        if (sets === 1) {
          return {
            status: "error",
            reason: "preference_conflict",
            message: "Your rules changed: Standard selection is no longer allowed.",
            conflicting: [{ pref_id: "pref-min", summary: "Minimum caution: Confirm each selection" }],
          };
        }
        return {
          status: "ok",
          preference: null,
          notes: [],
          refresh_required: false,
          refresh_seq: null,
          fallbacks: null,
        };
      };
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(2));
      expect(calls("co_adaptation_preference_set")[1]).toEqual({
        kind: "freeze_adaptation",
        scope: earlier.scope,
        value: earlier.value,
      });
      expect(await screen.findByText("Your earlier pause at Hold to select is kept.")).toBeTruthy();
      expect(screen.getByText("Your rules changed: Standard selection is no longer allowed.")).toBeTruthy();
    });

    it("never says the earlier pause is kept when its restore is rejected too", async () => {
      const conflictReply = {
        status: "error",
        reason: "preference_conflict",
        message: "You chose to select exactly what you click; pausing would contradict it.",
        conflicting: [{ pref_id: "pref-pin-act", summary: "Always select exactly what I click" }],
      };
      replies.co_adaptation_preference_retract = (params) => {
        world.preferences = world.preferences.filter((pref) => pref.pref_id !== params.pref_id);
        return { status: "ok", refresh_required: false, refresh_seq: null, fallbacks: null };
      };
      // A rule saved elsewhere between the retract and the set blocks both the new freeze and the restore.
      replies.co_adaptation_preference_set = () => conflictReply;
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(2));
      expect(
        await screen.findByText(
          "Your earlier pause at Hold to select could not be restored (You chose to select exactly what you click; pausing would contradict it.), so adapting is not paused now.",
        ),
      ).toBeTruthy();
      expect(screen.queryByText("Your earlier pause at Hold to select is kept.")).toBeNull();
      // The status was reloaded to say what is in force.
      expect(calls("co_adaptation_status").length).toBeGreaterThan(2);
    });

    it("when the new freeze was saved although its reply was lost, says that pause is in force", async () => {
      replies.co_adaptation_preference_retract = (params) => {
        world.preferences = world.preferences.filter((pref) => pref.pref_id !== params.pref_id);
        return { status: "ok", refresh_required: false, refresh_seq: null, fallbacks: null };
      };
      let sets = 0;
      replies.co_adaptation_preference_set = (params) => {
        sets += 1;
        if (sets === 1) {
          // The daemon saves the freeze, then the connection drops before the reply arrives.
          world.preferences = [
            { ...earlier, pref_id: "pref-standard", value: params.value, created_at: "2026-10-15T10:01:00+00:00" },
          ];
          throw new Error("Daemon connection was interrupted");
        }
        return {
          status: "error",
          reason: "preference_conflict",
          message: "You already paused adapting at Standard selection; pausing at Hold to select would contradict it.",
          conflicting: [{ pref_id: "pref-standard", summary: "Adapting paused at: Standard selection" }],
        };
      };
      await startSession();
      await startBlock();
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(2));
      expect(
        await screen.findByText(
          "Adapting paused. Heliox keeps Standard selection until you resume. This replaces your earlier pause at Hold to select.",
        ),
      ).toBeTruthy();
      expect(screen.queryByText("Your earlier pause at Hold to select is kept.")).toBeNull();
      expect(screen.queryByRole("button", { name: "Remove that rule" })).toBeNull();
    });
  });
});

describe("other recording", () => {
  it("sends skipped burden items as skipped", async () => {
    world.plan = {
      ...plan(1, 1),
      blocks: [{ index: 1, phase: "A", task: "pointing", trials: 1, idle_after: false, burden_after: true }],
    };
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await fireEvent.click(await screen.findByRole("button", { name: "Continue" }));
    await vi.waitFor(() => expect(calls("co_adaptation_report_burden")).toHaveLength(1));
    expect(calls("co_adaptation_report_burden")[0].report).toEqual({
      block_index: 1,
      scope: "block",
      effort: null,
      intrusiveness: null,
      control: null,
      tlx: null,
      predictability: null,
      preferred_regime: null,
      skipped: ["effort", "control"],
    });
  });

  it("records an Enter in an open menu of a pointer session as a keyboard activation", async () => {
    world.plan = plan(1, 1, "menu");
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await startSession();
    await startBlock();
    await fireEvent.click(await screen.findByRole("button", { name: "Menu" }));
    await fireEvent.keyDown(surfaceEl(), { key: "ArrowDown" });
    await fireEvent.keyDown(surfaceEl(), { key: "Enter" });
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1), { timeout: 3000 });
    const [trial] = recordedTrials();
    expect(trial.modality).toBe("mouse");
    expect(trial.spread).toEqual(world.spread);
    expect(trial.activations[0].modality).toBe("keyboard");
    expect(trial.activations[0].a_gate).toBe(0);
  });

  it("sends gaze_check null to session_begin without a gaze check", async () => {
    await startSession();
    expect(calls("co_adaptation_session_begin")[0].gaze_check).toBeNull();
  });
});

// ---- More of section 22.5 ----------------------------------------------------------------------------------------

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const HIT_CELLS = ["b1", "b2", "b3"].flatMap((bin) => [1, 2, 3].map((p) => `${bin}:p${p}:h1`));

describe("refreshed tables and decisions", () => {
  it("in an adaptive block Not this time on an L2 response sends the veto, then requests a table from the next trial", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    // A wide spread makes a click on the target ambiguous, so it gets the table's confirm step.
    world.spread = { r: 6, floor: 1 };
    const base = defaultReplies().co_adaptation_recommend;
    replies.co_adaptation_recommend = (params) =>
      params.level === "ambiguity_table"
        ? {
            status: "ok",
            logged: true,
            recommendation: {
              ...table([
                ...HIT_CELLS.map((cell) => ({ cell, variant: "confirm_selection:second_click" })),
                ...ALL_NO_HIT_CELLS.map((cell) => ({ cell, variant: "wait_for_evidence" })),
              ]),
              decision_id: params.decision_id,
            },
          }
        : base(params);
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    click(target);
    expect(await screen.findByText(/^Select .*\?$/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await fireEvent.click(screen.getByRole("button", { name: "Not this" }));
    click(target);
    await vi.waitFor(() => expect(recommendCalls("ambiguity_table")).toHaveLength(2));
    const [trial] = recordedTrials();
    expect(trial.veto).toMatchObject({ level: "ambiguity_response" });
    expect(trial.revert).toMatchObject({ reason: "veto", regime: "assistance_off" });
    const refreshed = recommendCalls("ambiguity_table")[1].context as Params;
    expect(refreshed.start_trial_index).toBe(1);
    expect(refreshed.trials_in_block).toBe(1);
    const order = daemonMocks.call.mock.calls.map(([name, params]) =>
      name === "co_adaptation_recommend" ? `recommend:${(params as Params).level}` : name,
    );
    expect(order.indexOf("co_adaptation_record_trial")).toBeLessThan(order.lastIndexOf("recommend:ambiguity_table"));
  });

  it("after an L1 fallback, trials echo the last applied spread pair", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    world.spread = { r: 1.3, floor: 1 };
    world.fallbacks = fallbacks({ l1: "assistance_off" });
    world.plan = plan(2, 1);
    await startSession();
    await startBlock();
    // The next block's decision fails: that block falls back to fallbacks.l1.
    replies.co_adaptation_recommend = () => {
      throw new Error("connection lost");
    };
    await cue();
    click(layoutOf(1, 0).rects[layoutOf(1, 0).intendedIndex]);
    await screen.findByText("Block 2 of 2");
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    await startBlock();
    await cue();
    click(layoutOf(2, 0).rects[layoutOf(2, 0).intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
    const [first, second] = recordedTrials();
    expect(first.spread).toEqual({ r: 1.3, floor: 1 });
    expect(second.l1_fallback_reason).not.toBeNull();
    expect(second.spread).toEqual({ r: 1.3, floor: 1 });
  });

  it("the broadcast applied before the pause, repeated by session_resume, still ends the block", async () => {
    world.regimes = [
      { variant: "assistance_off", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    world.refreshSeq = 5;
    world.statusFallbacks = fallbacks({ issued_at: FRESH });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    expect(recordedTrials()[0].revert).toMatchObject({ reason: "refresh" });
    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: fallbacks({ issued_at: FRESH }),
    });
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
  });
});

describe("refusals during the session", () => {
  it("a refused record_trial pauses with its message; the trial is sent again, marked interrupted, after the resume", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    let refuse = true;
    const base = defaultReplies().co_adaptation_record_trial;
    replies.co_adaptation_record_trial = (params) =>
      refuse ? { status: "refused", reason: "recording_active" } : base(params);
    await startSession();
    await startBlock();
    await cue();
    click(layoutOf(1, 0).rects[layoutOf(1, 0).intendedIndex]);
    expect(
      await screen.findByText("Stop the workflow recording first: it could record your study clicks."),
    ).toBeTruthy();
    expect(screen.getByText("Paused")).toBeTruthy();
    refuse = false;
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
    const [first, again] = recordedTrials();
    expect(again.trial_id).toBe(first.trial_id);
    expect(again.interrupted).toBe(true);
    expect(again.interruption_reason).toBe("refused");
    const order = daemonMocks.call.mock.calls.map(([name]) => name);
    expect(order.indexOf("co_adaptation_session_resume")).toBeLessThan(order.lastIndexOf("co_adaptation_record_trial"));
  });

  it("a refused session_resume keeps the study paused with its message", async () => {
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    replies.co_adaptation_session_resume = () => ({ status: "refused", reason: "neural_armed" });
    await startSession();
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    expect(await screen.findByText("Disarm neural control first.")).toBeTruthy();
    expect(screen.getByText("Paused")).toBeTruthy();
    expect(get(coadaptation).study.phase).toBe("paused");
  });

  it("a pause that arrives while session_resume is in flight keeps the study paused, and nothing runs hidden", async () => {
    world.plan = plan(1, 4, "menu");
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    const view = await startSession();
    await startBlock();
    // A menu trial shows its cue as soon as it mounts.
    await vi.waitFor(() => expect(document.querySelector(".surface")).toBeTruthy());
    const pauseButtons = screen.getAllByRole("button", { name: "Pause" });
    await fireEvent.click(pauseButtons[pauseButtons.length - 1]);
    expect(await screen.findByText("Paused")).toBeTruthy();
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    expect(document.querySelector(".surface")).toBeNull();

    let release: (() => void) | null = null;
    replies.co_adaptation_session_resume = () =>
      new Promise((resolve) => {
        release = () =>
          resolve({ status: "ok", refresh_required: false, refresh_seq: world.refreshSeq, fallbacks: world.fallbacks });
      });
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await vi.waitFor(() => expect(release).not.toBeNull());
    // An approval opens (App shows the Command tab) while the reply is outstanding.
    (session as unknown as Writable<Record<string, unknown>>).set({ loading: false, confirmRequired: true });
    await view.rerender({ paused: true, onexit: view.onexit });
    release!();
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Resume" })).toBeTruthy());
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(get(coadaptation).study.phase).toBe("paused");
    expect(screen.getByText("Paused")).toBeTruthy();
    // The next menu trial did not mount (and cue) behind the approval.
    expect(document.querySelector(".surface")).toBeNull();
    expect(recordedTrials()).toHaveLength(1);

    // Once the approval is gone, Resume continues with that trial.
    (session as unknown as Writable<Record<string, unknown>>).set({ loading: false, confirmRequired: false });
    await view.rerender({ paused: false, onexit: view.onexit });
    replies.co_adaptation_session_resume = defaultReplies().co_adaptation_session_resume;
    await fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await vi.waitFor(() => expect(get(coadaptation).study.phase).toBe("running"));
    await vi.waitFor(() => expect(document.querySelector(".surface")).toBeTruthy());
  });
});

describe("rest epochs", () => {
  it("an epoch whose block ended for a refresh runs after the next block, under that block's regime", async () => {
    world.plan = {
      session_number: 1,
      end_questionnaire: false,
      blocks: [
        { index: 1, phase: "A", task: "pointing", trials: 1, idle_after: true, burden_after: false },
        { index: 2, phase: "A", task: "pointing", trials: 1, idle_after: false, burden_after: false },
      ],
    };
    world.regimes = [
      { variant: "assistance_off", mode: "randomized" },
      { variant: "act", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    world.refreshSeq = 5;
    world.statusFallbacks = fallbacks({ issued_at: FRESH });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(get(coadaptation).study.inputPaused).toBe(false));
    click(layoutOf(1, 0).rects[layoutOf(1, 0).intendedIndex]);
    // The block ended for the refresh: no rest yet, the next block comes first.
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    expect(screen.queryByText("Rest — please don't select anything.")).toBeNull();
    await startBlock();
    await cue();
    click(layoutOf(2, 0).rects[layoutOf(2, 0).intendedIndex]);
    expect(await screen.findByText("Rest — please don't select anything.")).toBeTruthy();
    await fireEvent.keyDown(window, { key: "Escape" });
    await vi.waitFor(() => expect(recordedTrials().filter((trial) => trial.task === "idle")).toHaveLength(1));
    const idle = recordedTrials().find((trial) => trial.task === "idle")!;
    expect(idle.block_index).toBe(2);
    expect(idle.regime).toBe("act");
    expect(idle.intended_index).toBeNull();
    expect(idle.limit_ms).toBe(30000);
  });

  it("an epoch in progress when a refresh arrives stops as interrupted with refresh", async () => {
    world.plan = {
      session_number: 1,
      end_questionnaire: false,
      blocks: [
        { index: 1, phase: "A", task: "pointing", trials: 1, idle_after: true, burden_after: false },
        { index: 2, phase: "A", task: "pointing", trials: 1, idle_after: false, burden_after: false },
      ],
    };
    world.regimes = [
      { variant: "assistance_off", mode: "randomized" },
      { variant: "assistance_off", mode: "randomized" },
    ];
    await startSession();
    await startBlock();
    await cue();
    click(layoutOf(1, 0).rects[layoutOf(1, 0).intendedIndex]);
    expect(await screen.findByText("Rest — please don't select anything.")).toBeTruthy();
    world.refreshSeq = 5;
    world.statusFallbacks = fallbacks({ issued_at: FRESH });
    broadcast("co_adaptation_updated", { reason: "drift", participant_key: PK, refresh_seq: 5 });
    await vi.waitFor(() => expect(recordedTrials().filter((trial) => trial.task === "idle")).toHaveLength(1));
    const idle = recordedTrials().find((trial) => trial.task === "idle")!;
    expect(idle.interrupted).toBe(true);
    expect(idle.interruption_reason).toBe("refresh");
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
  });
});

describe("presentation and verification inputs", () => {
  it("records the frozen expansion and the hit's rendered rect under enlarge_target", async () => {
    world.regimes = [{ variant: "enlarge_target:2.0", mode: "randomized" }];
    await startSession();
    await startBlock();
    await cue();
    const scheduled = layoutOf(1, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    surfaceEl().dispatchEvent(pointer("pointermove", target.cx, target.cy));
    click(target);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [activation] = recordedTrials()[0].activations;
    expect(activation.expanded_index).toBe(scheduled.intendedIndex);
    expect(activation.hit_rect).toEqual({ cx: target.cx, cy: target.cy, w: target.w * 2, h: target.h * 2 });
    expect(activation.result).toBe("committed");
  });

  it("runs the gaze check before session_begin and sends its accuracy there", async () => {
    const { updateGazeRuntime, resetGazeRuntime } = await import("../../stores/gazeRuntime");
    replies.co_adaptation_status = () => {
      const status = statusReply();
      status.config = { ...status.config, gaze_prior_enabled: true } as typeof status.config;
      status.participants[0].consent.purposes = ["selection_study", "gaze_prior"];
      return status;
    };
    await coadaptation.load();
    updateGazeRuntime({ cameraActive: true, phase: "active", region: "center", confidence: 0.9 });
    try {
      await openStudy();
      await fireEvent.click(screen.getByLabelText("Use gaze hint"));
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      expect(await screen.findByText("Gaze check")).toBeTruthy();
      expect(calls("co_adaptation_session_begin")).toHaveLength(0);
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      await vi.advanceTimersByTimeAsync(13000);
      vi.useRealTimers();
      await vi.waitFor(() => expect(calls("co_adaptation_session_begin")).toHaveLength(1));
      const params = calls("co_adaptation_session_begin")[0];
      // Only the centre region matched, so one of five marked regions: an accuracy near 0.2.
      const accuracy = (params.gaze_check as { accuracy: number }).accuracy;
      expect(accuracy).toBeGreaterThan(0.15);
      expect(accuracy).toBeLessThan(0.25);
      expect(params.device).toMatchObject({ gaze: false });
    } finally {
      vi.useRealTimers();
      resetGazeRuntime();
    }
  });

  it("reuses a gaze check only for the participant it measured; another participant takes their own", async () => {
    const { updateGazeRuntime, resetGazeRuntime } = await import("../../stores/gazeRuntime");
    const { consentTextSha256, englishConsentText } = await import("../../coadaptation/consentText");
    // The real consent hash, so Setup accepts a second Start once it has hashed the consent text.
    const consentHash = await consentTextSha256(englishConsentText());
    const PK_B = "p-bbbbbbbb";
    replies.co_adaptation_status = () => {
      const status = statusReply();
      status.config = { ...status.config, gaze_prior_enabled: true } as typeof status.config;
      const first = status.participants[0];
      first.consent.purposes = ["selection_study", "gaze_prior"];
      first.consent.consent_text_sha256 = consentHash;
      status.participants.push({
        ...first,
        participant_key: PK_B,
        label: "Lab B",
        consent: { ...first.consent, consent_id: "c-2" },
      });
      return status;
    };
    // Every session_begin is refused, so the person stays on Setup and can retry or switch participant.
    replies.co_adaptation_session_begin = () => ({ status: "refused", reason: "task_running" });
    await coadaptation.load();
    updateGazeRuntime({ cameraActive: true, phase: "active", region: "center", confidence: 0.9 });
    try {
      await openStudy();
      expect((screen.getByLabelText("Lab A") as HTMLInputElement).checked).toBe(true);
      await fireEvent.click(screen.getByLabelText("Use gaze hint"));
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      expect(await screen.findByText("Gaze check")).toBeTruthy();
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      await vi.advanceTimersByTimeAsync(13000);
      vi.useRealTimers();
      await vi.waitFor(() => expect(calls("co_adaptation_session_begin")).toHaveLength(1));
      const measured = calls("co_adaptation_session_begin")[0];
      expect(measured.participant_key).toBe(PK);
      expect(measured.gaze_check).not.toBeNull();

      // The same person retries from Setup: their own measurement is sent again, with no second check.
      await screen.findByText("Who is selecting?");
      const retryBox = screen.getByLabelText("Use gaze hint") as HTMLInputElement;
      if (!retryBox.checked) await fireEvent.click(retryBox);
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      await vi.waitFor(() => expect(calls("co_adaptation_session_begin")).toHaveLength(2));
      expect(calls("co_adaptation_session_begin")[1]).toMatchObject({
        participant_key: PK,
        gaze_check: measured.gaze_check,
      });

      // Another participant with gaze: they take their own gaze check, and nothing is sent under their key first.
      await screen.findByText("Who is selecting?");
      await fireEvent.click(screen.getByLabelText("Lab B"));
      const gazeBox = screen.getByLabelText("Use gaze hint") as HTMLInputElement;
      if (!gazeBox.checked) await fireEvent.click(gazeBox);
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      expect(await screen.findByText("Gaze check")).toBeTruthy();
      expect(calls("co_adaptation_session_begin")).toHaveLength(2);
      expect(calls("co_adaptation_session_begin").some((params) => params.participant_key === PK_B)).toBe(false);
    } finally {
      vi.useRealTimers();
      resetGazeRuntime();
    }
  });
});

describe("Pause adapting replacing a pause in the middle of a trial", () => {
  it("accepts no press until both replies returned, then interrupts the trial with refresh and ends the block", async () => {
    world.preferences = [
      {
        pref_id: "pref-freeze",
        kind: "freeze_adaptation",
        scope: { context: "study", modality: "mouse", participant_key: PK },
        value: { regime: "assistance_off" },
        origin: "explicit_user",
        created_at: "2026-10-15T09:00:00+00:00",
        retracted_at: null,
      },
    ];
    await coadaptation.load();
    world.regimes = [
      { variant: "assistance_off", mode: "pinned" },
      { variant: "assistance_off", mode: "pinned" },
    ];
    const retract = deferred<unknown>();
    const set = deferred<unknown>();
    replies.co_adaptation_preference_retract = () => retract.promise;
    replies.co_adaptation_preference_set = () => set.promise;
    await startSession();
    await startBlock();
    await cue();
    await fireEvent.click(screen.getByRole("button", { name: "Pause adapting" }));
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    retract.resolve({
      status: "ok",
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: fallbacks({ issued_at: FRESH }),
    });
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    click(scheduled.rects[scheduled.intendedIndex]);
    set.resolve({
      status: "ok",
      preference: null,
      notes: [],
      refresh_required: true,
      refresh_seq: 6,
      fallbacks: fallbacks({ issued_at: "2026-10-15T10:06:00.000000+00:00" }),
    });
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.activations).toHaveLength(0);
    expect(trial.interrupted).toBe(true);
    expect(trial.interruption_reason).toBe("refresh");
    expect(await screen.findByText(/This replaces your earlier pause at Standard selection\./)).toBeTruthy();
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
  });

  it("shows a conflict reply to a freeze with its retract offer", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    replies.co_adaptation_preference_set = () => ({
      status: "error",
      reason: "preference_conflict",
      message: "You already pinned Larger targets; pausing at Select when the click is clear would contradict it.",
      conflicting: [{ pref_id: "pref-pin", summary: "Always use: Larger targets" }],
    });
    await startSession();
    await startBlock();
    await fireEvent.click(screen.getByRole("button", { name: "Pause adapting" }));
    expect(
      await screen.findByText(
        "You already pinned Larger targets; pausing at Select when the click is clear would contradict it.",
      ),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Remove that rule" })).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Keep it" }));
    expect(calls("co_adaptation_preference_retract")).toHaveLength(0);
  });
});

describe("offers and gaze", () => {
  /** A confirm step cancelled once (a correction), then confirmed with a fresh press after 250 ms. */
  async function selectWithOneCorrection(blockIndex: number, trialIndex: number) {
    const scheduled = layoutOf(blockIndex, trialIndex);
    const target = scheduled.rects[scheduled.intendedIndex];
    await cue();
    click(target);
    await fireEvent.click(await screen.findByRole("button", { name: "Not this" }));
    click(target);
    const select = await screen.findByRole("button", { name: "Select" });
    await new Promise((resolve) => setTimeout(resolve, 280));
    await fireEvent.pointerDown(select, { button: 0 });
    await fireEvent.click(select, { detail: 1 });
  }

  for (const failure of ["rpc_error", "late"] as const) {
    it(`a ${failure === "late" ? "late" : "failed"} offer is recorded on the offer only, as no_offer with reason ${failure}`, async () => {
      world.plan = plan(1, 3);
      world.regimes = [{ variant: "confirm_selection:second_click", mode: "randomized" }];
      const base = defaultReplies().co_adaptation_recommend;
      replies.co_adaptation_recommend = (params) => {
        if (params.level === "offer") {
          // A late offer never arrives within its 150 ms (spec 9.7).
          if (failure === "late") return new Promise(() => undefined);
          throw new Error("connection lost");
        }
        return base(params);
      };
      await startSession();
      await startBlock();
      await selectWithOneCorrection(1, 0);
      await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
      await selectWithOneCorrection(1, 1);
      await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
      // Two of the last three selections needed a correction: the third trial asks for an offer.
      await vi.waitFor(() => expect(recommendCalls("offer")).toHaveLength(1));
      const offerRequest = recommendCalls("offer")[0];
      expect(offerRequest.context).toMatchObject({ block_index: 1, trial_index: 2, struggle_kind: "corrections" });
      const scheduled = layoutOf(1, 2);
      await cue();
      click(scheduled.rects[scheduled.intendedIndex]);
      const select = await screen.findByRole("button", { name: "Select" });
      await new Promise((resolve) => setTimeout(resolve, 280));
      await fireEvent.pointerDown(select, { button: 0 });
      await fireEvent.click(select, { detail: 1 });
      await vi.waitFor(() => expect(recordedTrials()).toHaveLength(3));
      const third = recordedTrials()[2];
      expect(third.offers).toEqual([
        { decision_id: offerRequest.decision_id, variant: "no_offer", response: "ignored", fallback_reason: failure },
      ]);
      expect(third.l1_fallback_reason).toBeNull();
      expect(recordedTrials()[0].corrections).toBe(1);
    });
  }

  it("with gaze in use, each activation records the gaze region and confidence it used", async () => {
    const { updateGazeRuntime, resetGazeRuntime } = await import("../../stores/gazeRuntime");
    replies.co_adaptation_status = () => {
      const status = statusReply();
      status.config = { ...status.config, gaze_prior_enabled: true } as typeof status.config;
      status.participants[0].consent.purposes = ["selection_study", "gaze_prior"];
      return status;
    };
    replies.co_adaptation_session_begin = (params) => ({
      ...beginReply((params.device as { modality: Modality }).modality),
      gaze_prior: true,
    });
    world.regimes = [{ variant: "assistance_off", mode: "randomized" }];
    await coadaptation.load();
    updateGazeRuntime({ cameraActive: true, phase: "active", region: "center", confidence: 0.8 });
    try {
      await openStudy();
      await fireEvent.click(screen.getByLabelText("Use gaze hint"));
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      await screen.findByText("Gaze check");
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      // The person looks where each dot is: every stabilized region matches.
      for (const region of ["center", "left", "right", "up", "down"] as const) {
        updateGazeRuntime({ region });
        await vi.advanceTimersByTimeAsync(2500);
      }
      vi.useRealTimers();
      await vi.waitFor(() => expect(calls("co_adaptation_session_begin")).toHaveLength(1));
      expect(calls("co_adaptation_session_begin")[0].gaze_check).toEqual({ accuracy: 1 });
      expect(calls("co_adaptation_session_begin")[0].device).toMatchObject({ gaze: true });

      updateGazeRuntime({ region: "up", confidence: 0.8 });
      await startBlock();
      await cue();
      const scheduled = layoutOf(1, 0);
      click(scheduled.rects[scheduled.intendedIndex]);
      await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
      const [trial] = recordedTrials();
      expect(trial.activations[0].gaze).toEqual({ region: "up", conf: 0.8 });
      expect(trial.gaze).not.toBeNull();
    } finally {
      vi.useRealTimers();
      resetGazeRuntime();
    }
  });
});

describe("late decisions and stale tables", () => {
  /** Fake timers for the 2000 ms decision fallback; promises and Svelte's microtasks keep running. */
  function fakeTimers() {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
  }

  it("during Pause adapting with assistance_off a late block decision applies assistance_off, the frozen regime", async () => {
    world.regimes = [{ variant: "confirm_selection:hold_700", mode: "randomized" }];
    const frozen = fallbacks({
      issued_at: FRESH,
      l1: "assistance_off",
      refresh: "assistance_off",
      practice: "assistance_off",
    });
    replies.co_adaptation_preference_set = () => {
      world.refreshSeq = 5;
      world.statusFallbacks = frozen;
      return { status: "ok", preference: null, notes: [], refresh_required: true, refresh_seq: 5, fallbacks: frozen };
    };
    const base = defaultReplies().co_adaptation_recommend;
    let blockRequests = 0;
    replies.co_adaptation_recommend = (params) => {
      // The second block's decision never arrives within 2000 ms.
      if (params.level === "block_regime" && ++blockRequests > 1) return new Promise(() => undefined);
      return base(params);
    };
    await startSession();
    expect(await screen.findByText("Block 1 of 2")).toBeTruthy();
    fakeTimers();
    try {
      await fireEvent.click(screen.getByRole("button", { name: "Pause with standard selection" }));
      await vi.waitFor(() => expect(recommendCalls("block_regime")).toHaveLength(2));
      await vi.advanceTimersByTimeAsync(2100);
    } finally {
      vi.useRealTimers();
    }
    expect(calls("co_adaptation_preference_set")[0].value).toEqual({ regime: "assistance_off" });
    expect(await screen.findByText("Block 2 of 2")).toBeTruthy();
    expect(await screen.findByText("Using the most cautious option your rules allow.")).toBeTruthy();
    await startBlock();
    await cue();
    const scheduled = layoutOf(2, 0);
    const target = scheduled.rects[scheduled.intendedIndex];
    click(target);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    expect(trial.block_index).toBe(2);
    expect(trial.applied_regime).toBe("assistance_off");
    expect(trial.l1_fallback_reason).toBe("late");
    expect(trial.fallbacks_issued_at).toBe(FRESH);
    expect(trial.activations).toHaveLength(1);
    expect(trial.activations[0].result).toBe("committed");
    expect(trial.committed_index).toBe(scheduled.intendedIndex);
  });

  it("a late table records the L2 fallback with reason late on the activation and leaves the L1 decision alone", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    const base = defaultReplies().co_adaptation_recommend;
    replies.co_adaptation_recommend = (params) =>
      params.level === "ambiguity_table" ? new Promise(() => undefined) : base(params);
    await openStudy();
    fakeTimers();
    try {
      await fireEvent.click(screen.getByRole("button", { name: "Start" }));
      await vi.waitFor(() => expect(recommendCalls("ambiguity_table")).toHaveLength(1));
      await vi.advanceTimersByTimeAsync(2100);
    } finally {
      vi.useRealTimers();
    }
    await startBlock();
    await cue();
    click({ x: 4, y: 4 });
    const scheduled = layoutOf(1, 0);
    click(scheduled.rects[scheduled.intendedIndex]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(1));
    const [trial] = recordedTrials();
    const tableRequest = recommendCalls("ambiguity_table")[0];
    expect(trial.applied_regime).toBe("act");
    expect(trial.l1_fallback_reason).toBeNull();
    expect(trial.activations[0].l2).toMatchObject({
      decision_id: tableRequest.decision_id,
      source: "fallback",
      fallback_reason: "late",
      variant: "wait_for_evidence",
    });
  });

  it("while a table requested after a veto has not arrived, activations use the fixed rule as a stale table", async () => {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    // A wide spread makes a click on the target ambiguous, so it gets the table's confirm step.
    world.spread = { r: 6, floor: 1 };
    const base = defaultReplies().co_adaptation_recommend;
    let tables = 0;
    replies.co_adaptation_recommend = (params) => {
      if (params.level !== "ambiguity_table") return base(params);
      tables += 1;
      // The refreshed table requested after the veto has not arrived yet.
      if (tables > 1) return new Promise(() => undefined);
      return {
        status: "ok",
        logged: true,
        recommendation: {
          ...table([
            ...HIT_CELLS.map((cell) => ({ cell, variant: "confirm_selection:second_click" })),
            ...ALL_NO_HIT_CELLS.map((cell) => ({ cell, variant: "wait_for_evidence" })),
          ]),
          decision_id: params.decision_id,
        },
      };
    };
    await startSession();
    await startBlock();
    await cue();
    const first = layoutOf(1, 0);
    click(first.rects[first.intendedIndex]);
    expect(await screen.findByText(/^Select .*\?$/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await fireEvent.click(screen.getByRole("button", { name: "Not this" }));
    click(first.rects[first.intendedIndex]);
    await vi.waitFor(() => expect(recommendCalls("ambiguity_table")).toHaveLength(2));

    await cue();
    click({ x: 4, y: 4 });
    // Pausing ends the trial as interrupted, which records its activation.
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
    const trial = recordedTrials()[1];
    expect(trial.trial_index).toBe(1);
    expect(trial.l1_fallback_reason).toBeNull();
    expect(trial.applied_regime).toBe("act");
    const [activation] = trial.activations;
    expect(activation.l2).toMatchObject({ source: "fallback", fallback_reason: "stale_table" });
    // The fixed rule of the cached fallbacks: never a variant outside fallbacks.l2.
    expect(fallbacks().l2).toContain(activation.l2?.variant);
    // It names the table it stands in for (the one requested after the veto), never the block's first table, whose
    // draws in trial 0 were applied as recommended and must not be relabelled.
    const [firstTable, refreshed] = recommendCalls("ambiguity_table");
    expect(activation.l2?.decision_id).toBe(refreshed.decision_id);
    expect(trial.table_decision_ids).not.toContain(firstTable.decision_id);
  });

  /** An adaptive act block whose first table asks to confirm a clear-ish hit; the refreshed one waits everywhere. */
  function vetoTables(refreshed: () => unknown) {
    world.regimes = [{ variant: "act", mode: "adaptive" }];
    // A wide spread makes a click on the target ambiguous, so it gets the table's confirm step.
    world.spread = { r: 6, floor: 1 };
    const base = defaultReplies().co_adaptation_recommend;
    let tables = 0;
    replies.co_adaptation_recommend = (params) => {
      if (params.level !== "ambiguity_table") return base(params);
      tables += 1;
      if (tables > 1) return refreshed();
      return {
        status: "ok",
        logged: true,
        recommendation: {
          ...table([
            ...HIT_CELLS.map((cell) => ({ cell, variant: "confirm_selection:second_click" })),
            ...ALL_NO_HIT_CELLS.map((cell) => ({ cell, variant: "wait_for_evidence" })),
          ]),
          decision_id: params.decision_id,
        },
      };
    };
  }

  function waitEverywhere(decisionId: unknown) {
    return {
      status: "ok",
      logged: true,
      recommendation: {
        ...table(
          [...HIT_CELLS, ...ALL_NO_HIT_CELLS].map((cell) => ({ cell, variant: "wait_for_evidence" })),
          1,
          1,
        ),
        decision_id: decisionId,
      },
    };
  }

  /** Trial 0: the table's confirm step, "Not this time" on it, then a plain selection that ends the trial. */
  async function vetoInFirstTrial() {
    await startSession();
    await startBlock();
    await cue();
    const first = layoutOf(1, 0);
    click(first.rects[first.intendedIndex]);
    expect(await screen.findByText(/^Select .*\?$/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await fireEvent.click(screen.getByRole("button", { name: "Not this" }));
    click(first.rects[first.intendedIndex]);
    await vi.waitFor(() => expect(recommendCalls("ambiguity_table")).toHaveLength(2));
  }

  it("a table requested after a veto that arrives before the next trial's activation is used for it", async () => {
    vetoTables(() => waitEverywhere(recommendCalls("ambiguity_table")[1].decision_id));
    await vetoInFirstTrial();
    await new Promise((resolve) => setTimeout(resolve, 20));
    await cue();
    click({ x: 4, y: 4 });
    expect(await screen.findByText("Not selected — aim again")).toBeTruthy();
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
    const trial = recordedTrials()[1];
    const refreshed = recommendCalls("ambiguity_table")[1];
    expect(trial.activations[0].l2).toMatchObject({
      decision_id: refreshed.decision_id,
      source: "table",
      fallback_reason: null,
      variant: "wait_for_evidence",
      draw_index: 0,
    });
  });

  it("a table that arrives while the next trial runs replaces the stale-table fallback for its later activations", async () => {
    // Fallbacks whose fixed rule only waits, so the stale-table activation opens no prompt over the surface.
    world.fallbacks = fallbacks({ l2: ["wait_for_evidence"] });
    world.statusFallbacks = world.fallbacks;
    let release: (() => void) | null = null;
    vetoTables(
      () =>
        new Promise((resolve) => {
          release = () => resolve(waitEverywhere(recommendCalls("ambiguity_table")[1].decision_id));
        }),
    );
    await vetoInFirstTrial();
    await cue();
    click({ x: 4, y: 4 });
    await new Promise((resolve) => setTimeout(resolve, 20));
    release!();
    await vi.waitFor(() => expect(get(coadaptation).study.table?.start_trial_index).toBe(1));
    click({ x: 4, y: 4 });
    await fireEvent.click(screen.getAllByRole("button", { name: "Pause" })[0]);
    await vi.waitFor(() => expect(recordedTrials()).toHaveLength(2));
    const trial = recordedTrials()[1];
    const [firstTable, refreshed] = recommendCalls("ambiguity_table");
    const [stale, drawn] = trial.activations;
    // Before the table arrived: the fixed rule, naming the table it stands in for.
    expect(stale.l2).toMatchObject({ source: "fallback", fallback_reason: "stale_table" });
    expect(stale.l2?.decision_id).toBe(refreshed.decision_id);
    // After it arrived: the table's own draw.
    expect(drawn.l2).toMatchObject({ source: "table", decision_id: refreshed.decision_id, draw_index: 0 });
    expect(trial.table_decision_ids).toEqual([refreshed.decision_id]);
    expect(JSON.stringify(trial.activations)).not.toContain(String(firstTable.decision_id));
  });
});

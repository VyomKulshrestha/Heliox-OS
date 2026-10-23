import { get } from "svelte/store";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialTrialState, reduceTrial, type TrialEvent, type TrialState } from "../coadaptation/commitRules";
import { fixedL2Response, isPinActFallback } from "../coadaptation/decisionTable";
import type { DeviceV1, FallbacksV1, TrialV1 } from "../coadaptation/types";

type NotificationHandler = (method: string, params: unknown) => void;

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Array<(method: string, params: unknown) => void>,
}));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  onNotification: (handler: NotificationHandler) => {
    daemonMocks.handlers.push(handler);
  },
  offNotification: vi.fn(),
  onConnectionState: vi.fn(),
}));

const PK = "p-3fa29c1e";
const OTHER = "p-0badc0de";
const ISSUED = "2026-10-15T10:00:00.000000+00:00";
const DEVICE: DeviceV1 = {
  modality: "mouse",
  pointer_types: ["mouse"],
  dpr: 1,
  surface_w: 700,
  surface_h: 500,
  gaze: false,
  hand_pointer: false,
};

function fallbacks(overrides: Partial<FallbacksV1> = {}): FallbacksV1 {
  return {
    l1: "confirm_selection:hold_700",
    plain: "assistance_off",
    refresh: "assistance_off",
    practice: "assistance_off",
    freezable: ["assistance_off", "act", "confirm_selection:hold_700"],
    freeze_conflict: null,
    l2: ["ask_clarification:3", "ask_clarification:2", "wait_for_evidence"],
    interruption_cap: null,
    issued_at: ISSUED,
    ...overrides,
  };
}

let statusFallbacks: FallbacksV1 = fallbacks();
let enabled = true;
let participants = [PK];
let replies: Record<string, (params: Record<string, unknown>) => unknown> = {};

function statusReply() {
  return {
    status: "ok",
    available: true,
    config: { enabled },
    participants: participants.map((key) => ({
      participant_key: key,
      label: null,
      consent: {
        consent_id: "c",
        purposes: ["selection_study", "research_export"],
        granted_at: "2026-10-01T00:00:00+00:00",
        expires_at: "2099-01-01T00:00:00+00:00",
        retention_days: 90,
        consent_text_version: "coadapt-consent-v1",
        consent_text_sha256: "0".repeat(64),
        protocol_id: "coadapt-study-v1",
        active: true,
      },
      fallbacks: { mouse: statusFallbacks },
      sessions: 0,
      last_session_at: null,
    })),
    preferences: [],
    proposals: [],
    authority: "presentation_only",
    execution_authority: false,
  };
}

function beginReply(seq: number) {
  return {
    status: "ok",
    session_id: "s-1",
    participant_key: PK,
    day_index: 740000,
    session_number: 1,
    plan: { session_number: 1, end_questionnaire: true, blocks: [] },
    layout_seed: 1234567,
    spread: { mouse: 1.02 },
    floors: { mouse: 1 },
    thresholds: { ambiguity: 0.25, bins: { b1: [0.25, 0.4], b2: [0.4, 0.6], b3: [0.6, 1] } },
    gaze_prior: false,
    hand_pointer: false,
    burden_prompt_every_blocks: 1,
    evidence_kind: "live_human",
    fallbacks: fallbacks(),
    refresh_seq: seq,
  };
}

function broadcast(method: string, params: unknown) {
  for (const handler of daemonMocks.handlers) handler(method, params);
}

async function flush(times = 4) {
  for (let i = 0; i < times; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

function methodCalls(method: string) {
  return daemonMocks.call.mock.calls.filter(([name]) => name === method);
}

function trial(index: number): TrialV1 {
  return { trial_id: `t-${index}`, trial_index: index, interrupted: false, interruption_reason: null } as TrialV1;
}

async function loadStore() {
  const module = await import("./coadaptation");
  return module;
}

async function startSession(seq = 4) {
  const module = await loadStore();
  replies.co_adaptation_session_begin = () => beginReply(seq);
  await module.coadaptation.load();
  module.coadaptation.openStudy();
  const outcome = await module.coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
  expect(outcome.ok).toBe(true);
  return module;
}

/** A trial whose reducer receives the store's dispatches. */
function attachTrial(module: Awaited<ReturnType<typeof loadStore>>) {
  const box: { state: TrialState; events: TrialEvent[] } = {
    state: reduceTrial(
      initialTrialState("act", {
        plainRegime: "assistance_off",
        fallbacksIssuedAt: ISSUED,
        interruptionCap: null,
        interruptionsUsed: 0,
        tau: 0.25,
        reversible: [true, true, true],
      }),
      { type: "cue" },
    ),
    events: [],
  };
  const detach = module.coadaptation.attachTrial((event) => {
    box.events.push(event);
    box.state = reduceTrial(box.state, event);
  });
  return { box, detach };
}

beforeEach(() => {
  vi.resetModules();
  daemonMocks.handlers.length = 0;
  daemonMocks.call.mockReset();
  statusFallbacks = fallbacks();
  enabled = true;
  participants = [PK];
  replies = {
    co_adaptation_status: () => statusReply(),
    co_adaptation_session_end: () => ({ status: "ok", summary: {} }),
  };
  daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    const handler = replies[method];
    if (!handler) throw new Error(`unexpected call ${method}`);
    return handler(params);
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("loading and availability", () => {
  it("maps Method not found to available false with the older-daemon message", async () => {
    replies.co_adaptation_status = () => {
      throw new Error("Method not found: co_adaptation_status");
    };
    const { coadaptation } = await loadStore();
    await coadaptation.load();
    expect(get(coadaptation).available).toBe(false);
    expect(get(coadaptation).error).toBe(
      "This app is connected to an older Heliox daemon. Restart the daemon, then retry.",
    );
  });

  it("reloads status on co_adaptation_updated and on a co-adaptation config change", async () => {
    const { coadaptation } = await loadStore();
    await coadaptation.load();
    expect(methodCalls("co_adaptation_status")).toHaveLength(1);
    broadcast("co_adaptation_updated", { reason: "consent", participant_key: PK, refresh_seq: null });
    await flush();
    expect(methodCalls("co_adaptation_status")).toHaveLength(2);
    broadcast("config_updated", { method: "co_adaptation_consent_grant" });
    await flush();
    expect(methodCalls("co_adaptation_status")).toHaveLength(3);
  });
});

describe("recording needs consent", () => {
  it("sends no recording call without an active consent, or while the feature is off", async () => {
    participants = [];
    const { coadaptation } = await loadStore();
    await coadaptation.load();
    coadaptation.openStudy();
    const begun = await coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
    expect(begun.ok).toBe(false);
    expect(coadaptation.recordTrial(trial(0))).toBe(false);
    expect((await coadaptation.recommendBlock({} as never)).status).toBe("ignored");
    expect(await coadaptation.reportBurden({} as never)).toBe(false);
    await expect(coadaptation.exportData(PK)).rejects.toThrow();

    participants = [PK];
    enabled = false;
    await coadaptation.load();
    expect((await coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null })).ok).toBe(false);
    for (const method of [
      "co_adaptation_session_begin",
      "co_adaptation_recommend",
      "co_adaptation_record_trial",
      "co_adaptation_report_burden",
      "co_adaptation_export",
    ]) {
      expect(methodCalls(method)).toHaveLength(0);
    }
  });

  it("starts a session with the consent hash and the session_begin refresh_seq", async () => {
    const { coadaptation } = await startSession(4);
    const [, params] = methodCalls("co_adaptation_session_begin")[0];
    expect(params).toMatchObject({
      participant_key: PK,
      context: "study",
      protocol_id: "coadapt-study-v1",
      gaze_check: null,
    });
    expect(params.consent_text_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(get(coadaptation).study).toMatchObject({
      sessionId: "s-1",
      lastRefreshSeq: 4,
      blockRefreshSeq: 4,
      spread: { r: 1.02, floor: 1 },
      phase: "practice",
    });
  });
});

describe("cancel while a session starts", () => {
  /** Holds the session_begin reply until the test releases it. */
  function holdBegin() {
    const gate: { release: (reply: unknown) => void } = { release: () => undefined };
    replies.co_adaptation_session_begin = () =>
      new Promise((resolve) => {
        gate.release = resolve;
      });
    return gate;
  }

  it("a begin reply after Cancel ends the daemon session and leaves the study closed", async () => {
    const { coadaptation, studyActive } = await loadStore();
    await coadaptation.load();
    coadaptation.openStudy();
    const gate = holdBegin();
    const pending = coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
    // Wait for the call itself: a fixed number of event-loop turns is too few on a busy CI runner.
    await vi.waitFor(() => expect(methodCalls("co_adaptation_session_begin")).toHaveLength(1));

    await coadaptation.exitStudy();
    expect(get(studyActive)).toBe(false);
    gate.release(beginReply(4));
    const outcome = await pending;

    expect(outcome.ok).toBe(false);
    expect(get(studyActive)).toBe(false);
    expect(get(coadaptation).study).toMatchObject({ active: false, sessionId: "", phase: "setup", message: "" });
    expect(methodCalls("co_adaptation_session_end").map(([, params]) => params)).toEqual([
      { session_id: "s-1", reason: "abandoned" },
    ]);
  });

  it("a begin reply after Cancel and a reopen leaves the new study in setup", async () => {
    const { coadaptation } = await loadStore();
    await coadaptation.load();
    coadaptation.openStudy();
    const gate = holdBegin();
    const pending = coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
    await flush();
    await coadaptation.exitStudy();
    coadaptation.openStudy();
    gate.release(beginReply(4));

    expect((await pending).ok).toBe(false);
    expect(get(coadaptation).study).toMatchObject({ active: true, phase: "setup", sessionId: "", message: "" });
    expect(methodCalls("co_adaptation_session_end")).toHaveLength(1);
    // Nothing is recorded for the abandoned session.
    expect(coadaptation.recordTrial(trial(0))).toBe(false);
  });

  it("sends no session_begin when Cancel lands before the request leaves", async () => {
    const { coadaptation } = await loadStore();
    coadaptation.openStudy();
    // The status load runs first because nothing is loaded yet.
    const pending = coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
    await coadaptation.exitStudy();
    expect((await pending).ok).toBe(false);
    expect(methodCalls("co_adaptation_session_begin")).toHaveLength(0);
    expect(get(coadaptation).study.active).toBe(false);
  });
});

describe("refresh signals during a study", () => {
  it("a preferences broadcast pauses input, reloads status and dispatches the reloaded refresh regime", async () => {
    const module = await startSession(4);
    const { box } = attachTrial(module);
    statusFallbacks = fallbacks({ refresh: "confirm_selection:second_click", issued_at: "2026-10-15T10:05:00+00:00" });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    expect(box.events).toEqual([{ type: "refreshPending" }]);
    expect(get(module.coadaptation).study).toMatchObject({
      inputPaused: true,
      endBlockAfterTrial: true,
      lastRefreshSeq: 5,
    });
    await flush();
    expect(methodCalls("co_adaptation_status").length).toBeGreaterThanOrEqual(2);
    expect(box.events[1]).toEqual({
      type: "refresh",
      regime: "confirm_selection:second_click",
      interruptionCap: null,
      fallbacksIssuedAt: "2026-10-15T10:05:00+00:00",
    });
    expect(get(module.coadaptation).study.inputPaused).toBe(false);
    expect(get(module.coadaptation).study.fallbacks?.refresh).toBe("confirm_selection:second_click");
    // The trial never applies the old revert regime after the broadcast: a veto keeps the refresh revert.
    box.state = reduceTrial(box.state, { type: "veto" });
    expect(box.state.revert).toMatchObject({ reason: "refresh", regime: "confirm_selection:second_click" });
    box.state = reduceTrial(box.state, {
      type: "activation",
      x: 1,
      y: 1,
      modality: "mouse",
      hitIndex: 0,
      aGate: 0.1,
      nPlausible: 1,
    });
    expect(box.state.phase).toBe("confirming");
  });

  it("a drift broadcast while a pin_regime is in force dispatches the pinned regime", async () => {
    const module = await startSession(0);
    const { box } = attachTrial(module);
    statusFallbacks = fallbacks({
      l1: "enlarge_target:1.5",
      refresh: "enlarge_target:1.5",
      practice: "enlarge_target:1.5",
    });
    broadcast("co_adaptation_updated", { reason: "drift", participant_key: PK, refresh_seq: 1 });
    await flush();
    expect(box.events.map((e) => e.type)).toEqual(["refreshPending", "refresh"]);
    expect(box.events[1]).toMatchObject({ regime: "enlarge_target:1.5" });
    expect(box.state.revert?.regime).toBe("enlarge_target:1.5");
  });

  it("applies a reply and its broadcast (either order) once, ignores repeats and applies a new value", async () => {
    for (const order of ["reply-first", "broadcast-first"] as const) {
      vi.resetModules();
      daemonMocks.handlers.length = 0;
      const module = await startSession(4);
      const { box } = attachTrial(module);
      replies.co_adaptation_preference_set = () => ({
        status: "ok",
        preference: { pref_id: "r1" },
        notes: [],
        refresh_required: true,
        refresh_seq: 5,
        fallbacks: fallbacks({ refresh: "act", issued_at: "2026-10-15T10:06:00+00:00" }),
      });
      const scope = { context: "study" as const, modality: null, participant_key: PK };
      if (order === "reply-first") {
        await module.coadaptation.setPreference("forbid_variant", scope, "ask_clarification:3");
        broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
      } else {
        broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
        await module.coadaptation.setPreference("forbid_variant", scope, "ask_clarification:3");
      }
      await flush();
      expect(
        box.events.map((e) => e.type),
        order,
      ).toEqual(["refreshPending", "refresh"]);
      expect(box.state.phase).toBe("cued");
      expect(box.state.revert?.reason).toBe("refresh");

      // A record_trial or session_resume reply repeating 5 dispatches nothing.
      replies.co_adaptation_record_trial = () => ({
        status: "recorded",
        verification: "ok",
        labels: null,
        preference_violation: false,
        refresh_required: true,
        refresh_seq: 5,
        fallbacks: fallbacks(),
      });
      module.coadaptation.recordTrial(trial(0));
      await flush();
      expect(box.events).toHaveLength(2);
      replies.co_adaptation_session_resume = () => ({
        status: "ok",
        refresh_required: true,
        refresh_seq: 5,
        fallbacks: fallbacks(),
      });
      expect(await module.coadaptation.resumeStudy()).toBe(true);
      await flush();
      expect(box.events).toHaveLength(2);
      expect(box.state.phase).toBe("cued");

      // A broadcast with 6 is a second change and is applied.
      broadcast("co_adaptation_updated", { reason: "calibration_suspended", participant_key: PK, refresh_seq: 6 });
      await flush();
      expect(box.events.map((e) => e.type)).toEqual(["refreshPending", "refresh", "refreshPending", "refresh"]);
      expect(box.state.phase).toBe("interrupted");
    }
  });

  it("only reloads status for another participant or a null refresh_seq", async () => {
    const module = await startSession(4);
    const { box } = attachTrial(module);
    const before = methodCalls("co_adaptation_status").length;
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: OTHER, refresh_seq: 9 });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: null });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: null, refresh_seq: null });
    await flush();
    expect(box.events).toEqual([]);
    expect(methodCalls("co_adaptation_status").length).toBe(before + 3);
    expect(get(module.coadaptation).study.lastRefreshSeq).toBe(4);
  });

  it("with no trial in progress a new value dispatches nothing and ends the block", async () => {
    const module = await startSession(4);
    module.coadaptation.setStudy({ phase: "burden" });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await flush();
    const study = get(module.coadaptation).study;
    expect(study).toMatchObject({ endBlockAfterTrial: true, inputPaused: false, lastRefreshSeq: 5 });
    expect(study.message).toBe("Your rules changed, so a new block starts with them.");
    module.coadaptation.blockEnded();
    expect(get(module.coadaptation).study.endBlockAfterTrial).toBe(false);
  });

  it("a repeated value ends a stale block before the next trial, but not a block requested after the refresh", async () => {
    const module = await startSession(4);
    replies.co_adaptation_recommend = () => ({ status: "ok", logged: true, recommendation: { variant: "act" } });
    await module.coadaptation.recommendBlock({} as never);
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 5 });
    await flush();
    module.coadaptation.blockEnded();
    // The study has not requested a new block yet: a repeating reply still ends this (stale) block.
    replies.co_adaptation_record_trial = () => ({
      status: "recorded",
      verification: "ok",
      labels: null,
      preference_violation: false,
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: fallbacks(),
    });
    module.coadaptation.recordTrial(trial(0));
    await flush();
    expect(get(module.coadaptation).study.endBlockAfterTrial).toBe(true);
    module.coadaptation.blockEnded();
    // A block requested after the refresh is not stale: the same repeat changes nothing.
    await module.coadaptation.recommendBlock({} as never);
    expect(get(module.coadaptation).study.blockRefreshSeq).toBe(5);
    module.coadaptation.recordTrial(trial(1));
    await flush();
    expect(get(module.coadaptation).study.endBlockAfterTrial).toBe(false);
  });

  it("a rule change with no open decision (practice, or a block whose L1 request failed) refreshes the trial", async () => {
    // Practice: the study requested no decision, so blockRefreshSeq is still session_begin's value.
    const module = await startSession(4);
    module.coadaptation.setStudy({ phase: "practice" });
    const practice = attachTrial(module);
    const pinAct = fallbacks({
      l2: ["act", "wait_for_evidence"],
      refresh: "act",
      practice: "act",
      issued_at: "2026-10-15T10:06:00.000000+00:00",
    });
    replies.co_adaptation_preference_set = () => ({
      status: "ok",
      preference: { pref_id: "r1" },
      notes: [],
      refresh_required: true,
      refresh_seq: 5,
      fallbacks: pinAct,
    });
    const scope = { context: "study" as const, modality: null, participant_key: PK };
    await module.coadaptation.setPreference("pin_act", scope, true);
    await flush();
    expect(practice.box.events.map((e) => e.type)).toEqual(["refreshPending", "refresh"]);
    expect(practice.box.events[1]).toMatchObject({ regime: "act", fallbacksIssuedAt: pinAct.issued_at });
    let study = get(module.coadaptation).study;
    // endBlockAfterTrial drives SelectionStudy's onBlockStale; the held fallbacks are the ones the trial switches to.
    expect(study).toMatchObject({
      endBlockAfterTrial: true,
      inputPaused: false,
      lastRefreshSeq: 5,
      blockRefreshSeq: 4,
    });
    expect(study.fallbacks).toEqual(pinAct);
    expect(isPinActFallback(study.fallbacks)).toBe(true);
    expect(fixedL2Response(study.fallbacks, { hasHit: true, nPlausible: 2 })).toBe("act");
    practice.detach();
    module.coadaptation.blockEnded();

    // A main block whose L1 request failed holds no open decision either; the next change ends it after the trial.
    replies.co_adaptation_recommend = () => ({ status: "error", message: "engine failed" });
    expect((await module.coadaptation.recommendBlock({} as never)).status).toBe("rpc_error");
    expect(get(module.coadaptation).study.blockRefreshSeq).toBe(5);
    module.coadaptation.setStudy({ phase: "running" });
    const main = attachTrial(module);
    statusFallbacks = fallbacks({ issued_at: "2026-10-15T10:07:00.000000+00:00" });
    broadcast("co_adaptation_updated", { reason: "preferences", participant_key: PK, refresh_seq: 6 });
    await flush();
    expect(main.box.events.map((e) => e.type)).toEqual(["refreshPending", "refresh"]);
    expect(main.box.state.revert).toMatchObject({ reason: "refresh", regime: "assistance_off" });
    study = get(module.coadaptation).study;
    expect(study).toMatchObject({ endBlockAfterTrial: true, lastRefreshSeq: 6 });
    expect(study.fallbacks?.issued_at).toBe("2026-10-15T10:07:00.000000+00:00");
  });

  it("ends the session for revoked, expired, reset and erased notifications of the running participant", async () => {
    for (const reason of ["revoked", "expired", "reset", "erased"]) {
      vi.resetModules();
      daemonMocks.handlers.length = 0;
      const module = await startSession(1);
      broadcast("co_adaptation_updated", { reason, participant_key: OTHER, refresh_seq: null });
      expect(get(module.coadaptation).study.sessionId).toBe("s-1");
      broadcast("co_adaptation_updated", { reason, participant_key: PK, refresh_seq: null });
      const study = get(module.coadaptation).study;
      expect(study.sessionId).toBe("");
      expect(study.phase).toBe("error");
      expect(study.message).toMatch(/session ended/);
    }
  });
});

describe("pause and resume", () => {
  it("pauses on recorded_workflow_status_changed", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    broadcast("recorded_workflow_status_changed", { active: false });
    expect(get(module.coadaptation).study.phase).toBe("running");
    broadcast("recorded_workflow_status_changed", { active: true });
    expect(get(module.coadaptation).study.phase).toBe("paused");
    expect(get(module.coadaptation).study.message).toBe(
      "Stop the workflow recording first: it could record your study clicks.",
    );
  });

  it("resumes only on an ok session_resume reply", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    module.coadaptation.pauseStudy("modal");
    replies.co_adaptation_session_resume = () => ({ status: "refused", reason: "task_running" });
    expect(await module.coadaptation.resumeStudy()).toBe(false);
    expect(get(module.coadaptation).study.phase).toBe("paused");
    expect(get(module.coadaptation).study.message).toBe("Finish or stop the running task first.");
    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: false,
      refresh_seq: 1,
      fallbacks: fallbacks(),
    });
    expect(await module.coadaptation.resumeStudy()).toBe(true);
    expect(get(module.coadaptation).study.phase).toBe("running");
  });

  it("keeps a pause applied while session_resume is in flight instead of resuming", async () => {
    const module = await startSession(1);
    const okResume = { status: "ok", refresh_required: false, refresh_seq: 1, fallbacks: fallbacks() };
    const daemon = await import("../api/daemon");
    const connectionCalls = vi.mocked(daemon.onConnectionState).mock.calls;
    const connection = connectionCalls[connectionCalls.length - 1][0] as (connected: boolean) => void;
    replies.co_adaptation_record_trial = () => ({ status: "refused", reason: "task_running" });
    const pauses: Array<[string, () => Promise<void> | void, string]> = [
      [
        "recording",
        () => broadcast("recorded_workflow_status_changed", { active: true }),
        "Stop the workflow recording first: it could record your study clicks.",
      ],
      ["daemon", () => connection(false), module.refusalMessage("daemon")],
      [
        "refused call",
        async () => {
          module.coadaptation.recordTrial(trial(0));
          await flush();
        },
        "Finish or stop the running task first.",
      ],
    ];
    module.coadaptation.setStudy({ phase: "running" });
    module.coadaptation.pauseStudy("user");
    for (const [name, pause, message] of pauses) {
      let answer: (value: unknown) => void = () => undefined;
      replies.co_adaptation_session_resume = () => new Promise((resolve) => (answer = resolve));
      const resumed = module.coadaptation.resumeStudy();
      await flush(1);
      await pause();
      answer(okResume);
      expect(await resumed, name).toBe(false);
      expect(get(module.coadaptation).study.phase, name).toBe("paused");
      expect(get(module.coadaptation).study.message, name).toBe(message);
    }
    // The refused trial waits at the head: nothing more was sent while the study stayed paused.
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(1);
    expect(get(module.coadaptation).study.pendingTrials).toHaveLength(1);
    // A resume with nothing in between continues the phase the study was paused in, and sends the trial again.
    replies.co_adaptation_session_resume = () => okResume;
    replies.co_adaptation_record_trial = () => ({
      status: "recorded",
      verification: "ok",
      labels: null,
      preference_violation: false,
      refresh_required: false,
      refresh_seq: 1,
      fallbacks: null,
    });
    expect(await module.coadaptation.resumeStudy()).toBe(true);
    await flush();
    expect(get(module.coadaptation).study.phase).toBe("running");
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(2);
    expect(get(module.coadaptation).study.pendingTrials).toHaveLength(0);
  });

  it("a resume reply with a new refresh_seq ends the block before the next trial", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    module.coadaptation.pauseStudy("user");
    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: true,
      refresh_seq: 2,
      fallbacks: fallbacks({ refresh: "act", issued_at: "2026-10-15T11:00:00+00:00" }),
    });
    expect(await module.coadaptation.resumeStudy()).toBe(true);
    await flush();
    expect(get(module.coadaptation).study).toMatchObject({ endBlockAfterTrial: true, lastRefreshSeq: 2 });
    expect(get(module.coadaptation).study.fallbacks?.refresh).toBe("act");
  });
});

describe("refusal messages and erasure", () => {
  it("explains an unreadable rules file", async () => {
    const module = await loadStore();
    expect(module.refusalMessage("preferences_unreadable")).toBe(
      "Heliox cannot read its assistance rules file. Fix or remove co_adaptation_preferences.json, then restart Heliox.",
    );
    replies.co_adaptation_session_begin = () => ({ status: "refused", reason: "preferences_unreadable" });
    await module.coadaptation.load();
    module.coadaptation.openStudy();
    const outcome = await module.coadaptation.beginSession({ participantKey: PK, device: DEVICE, gazeCheck: null });
    expect(outcome).toMatchObject({ ok: false, reason: "preferences_unreadable" });
    expect(get(module.coadaptation).study.message).toMatch(/^Heliox cannot read its assistance rules file\./);
  });

  it("reports what a withdrawal could not delete yet", async () => {
    const module = await loadStore();
    await module.coadaptation.load();
    const erased = { rows: 3, export_files: 1, rules_deleted: 0 };
    const cases: Array<[Record<string, unknown>, { keyPending: boolean; filesPending: boolean }]> = [
      [
        { ...erased, key_destroyed: true, export_files_left: 0, pending: false },
        { keyPending: false, filesPending: false },
      ],
      [
        { ...erased, key_destroyed: false, export_files_left: 0, pending: true },
        { keyPending: true, filesPending: false },
      ],
      [
        { ...erased, key_destroyed: true, export_files_left: 1, pending: true },
        { keyPending: false, filesPending: true },
      ],
      // An older daemon without the pending field.
      [
        { ...erased, key_destroyed: true },
        { keyPending: false, filesPending: false },
      ],
    ];
    for (const [fields, expected] of cases) {
      replies.co_adaptation_consent_revoke = () => ({ status: "ok", erased: fields });
      expect(await module.coadaptation.revokeConsent(PK)).toEqual(expected);
    }
  });
});

describe("trial queue", () => {
  it("is bounded, sends one trial at a time in order and keeps a refused trial at the head until resume", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    const pending: Array<(value: unknown) => void> = [];
    replies.co_adaptation_record_trial = () => new Promise((resolve) => pending.push(resolve));
    for (let i = 0; i < 25; i += 1) module.coadaptation.recordTrial(trial(i));
    expect(get(module.coadaptation).study.pendingTrials).toHaveLength(20);
    await flush();
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(1);
    expect((methodCalls("co_adaptation_record_trial")[0][1] as { trial: TrialV1 }).trial.trial_id).toBe("t-0");

    pending.shift()!({
      status: "recorded",
      verification: "ok",
      refresh_required: false,
      refresh_seq: 1,
      fallbacks: null,
    });
    await flush();
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(2);
    expect((methodCalls("co_adaptation_record_trial")[1][1] as { trial: TrialV1 }).trial.trial_id).toBe("t-1");

    pending.shift()!({ status: "refused", reason: "neural_armed" });
    await flush();
    const study = get(module.coadaptation).study;
    expect(study.phase).toBe("paused");
    expect(study.pendingTrials[0]).toMatchObject({
      trial_id: "t-1",
      interrupted: true,
      interruption_reason: "refused",
    });
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(2);

    replies.co_adaptation_session_resume = () => ({
      status: "ok",
      refresh_required: false,
      refresh_seq: 1,
      fallbacks: fallbacks(),
    });
    expect(await module.coadaptation.resumeStudy()).toBe(true);
    await flush();
    const resent = methodCalls("co_adaptation_record_trial")[2][1] as { trial: TrialV1 };
    expect(resent.trial).toMatchObject({ trial_id: "t-1", interrupted: true, interruption_reason: "refused" });
  });

  it("ends the session when the daemon no longer holds it, instead of dropping trial after trial", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    const answers: unknown[] = [
      { status: "error", message: "Co-adaptation could not complete record_trial." },
      { status: "ignored", reason: "session_closed" },
    ];
    replies.co_adaptation_record_trial = () => answers.shift() ?? { status: "ignored", reason: "session_closed" };
    for (let i = 0; i < 5; i += 1) module.coadaptation.recordTrial(trial(i));
    await flush(8);

    // The failed trial is dropped; the first session_closed reply ends the session and nothing more is sent.
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(2);
    const study = get(module.coadaptation).study;
    expect(study).toMatchObject({ sessionId: "", phase: "error", pendingTrials: [] });
    expect(study.message).toBe("This session is no longer open.");
    expect(module.coadaptation.recordTrial(trial(9))).toBe(false);
  });

  it("maps each session-level ignored reason to its ended message", async () => {
    const cases: Array<[string, RegExp]> = [
      ["unknown_session", /no longer open/],
      ["no_consent", /no longer open/],
      ["purpose_missing", /no longer open/],
      ["consent_expired", /consent expired/],
      ["disabled", /turned off/],
    ];
    for (const [reason, message] of cases) {
      vi.resetModules();
      daemonMocks.handlers.length = 0;
      const module = await startSession(1);
      module.coadaptation.setStudy({ phase: "running" });
      replies.co_adaptation_record_trial = () => ({ status: "ignored", reason });
      module.coadaptation.recordTrial(trial(0));
      module.coadaptation.recordTrial(trial(1));
      await flush(8);
      const study = get(module.coadaptation).study;
      expect(study.phase, reason).toBe("error");
      expect(study.sessionId, reason).toBe("");
      expect(study.message, reason).toMatch(message);
    }
  });

  it("a burden report or a decision the daemon ignores for a closed session ends the session", async () => {
    for (const call of ["burden", "block"] as const) {
      vi.resetModules();
      daemonMocks.handlers.length = 0;
      const module = await startSession(1);
      module.coadaptation.setStudy({ phase: "running" });
      replies.co_adaptation_report_burden = () => ({ status: "ignored", reason: "session_closed" });
      replies.co_adaptation_recommend = () => ({
        status: "ignored",
        reason: "session_closed",
        logged: false,
        recommendation: { variant: "assistance_off" },
      });
      if (call === "burden") {
        expect(await module.coadaptation.reportBurden({} as never)).toBe(false);
      } else {
        expect((await module.coadaptation.recommendBlock({} as never)).status).toBe("ignored");
      }
      expect(get(module.coadaptation).study, call).toMatchObject({
        sessionId: "",
        phase: "error",
        message: "This session is no longer open.",
      });
    }
  });

  it("drops a refused trial when the session is stopped instead", async () => {
    const module = await startSession(1);
    module.coadaptation.setStudy({ phase: "running" });
    replies.co_adaptation_record_trial = () => ({ status: "refused", reason: "task_running" });
    module.coadaptation.recordTrial(trial(0));
    await flush();
    expect(get(module.coadaptation).study.pendingTrials).toHaveLength(1);
    await module.coadaptation.exitStudy();
    expect(methodCalls("co_adaptation_record_trial")).toHaveLength(1);
    expect(methodCalls("co_adaptation_session_end")[0][1]).toEqual({ session_id: "s-1", reason: "stopped" });
    expect(get(module.coadaptation).study).toMatchObject({ active: false, pendingTrials: [] });
  });
});

describe("browser storage", () => {
  it("writes nothing to localStorage or sessionStorage during a study, a revoke and an expired notification", async () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const removeItem = vi.spyOn(Storage.prototype, "removeItem");
    const clear = vi.spyOn(Storage.prototype, "clear");
    const module = await startSession(1);
    const { box } = attachTrial(module);
    replies.co_adaptation_recommend = () => ({ status: "ok", logged: true, recommendation: { variant: "act" } });
    replies.co_adaptation_record_trial = () => ({
      status: "recorded",
      verification: "ok",
      refresh_required: false,
      refresh_seq: 1,
      fallbacks: null,
    });
    replies.co_adaptation_report_burden = () => ({ status: "recorded" });
    await module.coadaptation.recommendBlock({} as never);
    broadcast("co_adaptation_updated", { reason: "drift", participant_key: PK, refresh_seq: 2 });
    await flush();
    expect(box.events).toHaveLength(2);
    module.coadaptation.recordTrial(trial(0));
    await module.coadaptation.reportBurden({} as never);
    await module.coadaptation.endSession("completed");
    replies.co_adaptation_consent_revoke = () => ({ status: "ok", erased: {}, config: {} });
    await module.coadaptation.revokeConsent(PK);
    broadcast("co_adaptation_updated", { reason: "expired", participant_key: PK, refresh_seq: null });
    await module.coadaptation.exitStudy();
    await flush();
    expect(setItem).not.toHaveBeenCalled();
    expect(removeItem).not.toHaveBeenCalled();
    expect(clear).not.toHaveBeenCalled();
    setItem.mockRestore();
    removeItem.mockRestore();
    clear.mockRestore();
  });
});

describe("studyActive", () => {
  it("covers openStudy until exitStudy", async () => {
    const { coadaptation, studyActive } = await loadStore();
    expect(get(studyActive)).toBe(false);
    coadaptation.openStudy();
    expect(get(studyActive)).toBe(true);
    await coadaptation.exitStudy();
    expect(get(studyActive)).toBe(false);
  });
});

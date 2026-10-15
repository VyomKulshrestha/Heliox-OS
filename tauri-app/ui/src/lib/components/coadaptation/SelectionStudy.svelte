<script lang="ts">
  /**
   * The Selection Study host (spec 12.2): setup, gaze check, session begin, practice, blocks with their decisions,
   * burden prompts, rest epochs, the end questionnaire and the summary. It owns the session's sequencing; the
   * co-adaptation store owns the RPCs, the refresh sequence and the trial queue.
   *
   * Authority: the study changes how its own inert targets are shown and what a selection needs. It never approves,
   * clicks or runs anything, and it sends nothing about approvals; it reads the session and neural stores only to
   * pause while a task runs or neural control is armed, and it reads gaze locally (no gaze_event is sent).
   */
  import { onDestroy, onMount, untrack } from "svelte";
  import { get } from "svelte/store";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { coadaptation, PreferenceConflictError, refusalMessage } from "../../stores/coadaptation";
  import { requestStudyPointer, studyPointer } from "../../stores/studyPointer";
  import { session } from "../../stores/session";
  import { neural } from "../../stores/neural";
  import {
    blockPlan,
    buildBlockTrials,
    burdenDue,
    idleDue,
    idleEpoch,
    idleEpochSeed,
    mainBlocks,
    nextBlockIndex,
    TRIAL_LIMIT_MS,
    IDLE_EPOCH_MS,
    type ScheduledTrial,
  } from "../../coadaptation/studySchedule";
  import { FIT_H, FIT_W, type Surface } from "../../coadaptation/geometry";
  import { isActGuarded } from "../../coadaptation/presentation";
  import { isPinActFallback } from "../../coadaptation/decisionTable";
  import type { TrialEvent } from "../../coadaptation/commitRules";
  import {
    CAUTION_ORDER,
    isL1Variant,
    isPresentationOnly,
    L1_ARMS,
    type AmbiguityTableV1,
    type BlockPlanV1,
    type BurdenV1,
    type ExplanationV1,
    type FallbacksV1,
    type InterruptionReason,
    type L1Variant,
    type L2FallbackReason,
    type L3Variant,
    type LayoutSummaryV1,
    type Modality,
    type OfferResponseV1,
    type PreferenceV1,
    type PreferenceScopeV1,
    type SessionPlanV1,
    type SessionSummaryV1,
    type TaskKind,
  } from "../../coadaptation/types";
  import { variantLabel, explanationText, type Translate } from "./AssistChip.svelte";
  import { ruleSummary } from "./AssistanceRulesEditor.svelte";
  import StudySetup, { type StudySetupChoice } from "./StudySetup.svelte";
  import StudyGazeCheck from "./StudyGazeCheck.svelte";
  import StudyPointingField, {
    type FieldRegime,
    type FieldTable,
    type FieldTrial,
    type FieldTrialEnd,
    type FieldVeto,
    type InputMode,
    type PinnedRule,
    type PosteriorParams,
  } from "./StudyPointingField.svelte";
  import StudyMenuTask from "./StudyMenuTask.svelte";
  import StudyIdleEpoch from "./StudyIdleEpoch.svelte";
  import BurdenPrompt, { type BurdenAnswers } from "./BurdenPrompt.svelte";
  import EndQuestionnaire, { type EndAnswers } from "./EndQuestionnaire.svelte";
  import StudySummary from "./StudySummary.svelte";

  interface Props {
    /** Set by App while a modal is open or another tab is shown: the study pauses and the current trial is excluded. */
    paused: boolean;
    onexit: () => void;
  }

  let { paused, onexit }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const ARMED_NEURAL_STATES = new Set([
    "armed_safe_ui",
    "armed_safe_desktop",
    "candidate_intent",
    "previewed",
    "committed",
    "cooldown",
  ]);
  const PAUSE_REASONS: Record<string, InterruptionReason> = {
    user: "paused",
    modal: "modal",
    tab: "modal",
    hidden: "hidden",
    blur: "blur",
    resize: "resize",
    palm_exit: "palm_exit",
    daemon: "daemon",
  };

  type Screen =
    | "setup"
    | "gaze_check"
    | "practice_intro"
    | "block_intro"
    | "trial"
    | "burden"
    | "idle"
    | "end_questionnaire"
    | "summary";

  /** What the host needs from the trial surface it shows (pointing field, menu task or rest epoch). */
  interface TrialSurface {
    dispatch(event: TrialEvent): void;
    refresh(fresh: FallbacksV1): void;
    interrupt(reason: InterruptionReason): void;
    discard(): void;
    inProgress(): boolean;
    /** A table requested after a veto arrived while this trial runs (pointing and menu trials). */
    adoptTable?(table: FieldTable): void;
  }

  type Step =
    | { kind: "block_intro"; blockIndex: number }
    | { kind: "trial"; t: number }
    | { kind: "burden" }
    | { kind: "idle"; blockIndex: number }
    | { kind: "after_block"; byRefresh: boolean }
    | { kind: "end" };

  /** The current block: its plan, trials and the decisions that govern it. */
  interface BlockRun {
    index: number;
    plan: BlockPlanV1;
    practice: boolean;
    /** 1-based position among the main blocks (0 for practice). */
    position: number;
    trials: ScheduledTrial[];
    layouts: LayoutSummaryV1[];
    /** True once the L1 decision was requested (a refresh then ends the block, section 9.7). */
    requested: boolean;
    /** No block starts without fallbacks or with fallbacks.l1 null (section 9.7). */
    blocked: boolean;
    ready: boolean;
    regime: L1Variant;
    applied: L1Variant;
    decisionId: string;
    mode: string;
    l1FallbackReason: "late" | "rpc_error" | null;
    explanation: ExplanationV1 | null;
    table: AmbiguityTableV1 | null;
    tableDecisionId: string | null;
    tableFallback: L2FallbackReason | null;
    /** A refreshed table requested after a veto has not arrived yet (stale_table until it does). */
    tablePending: boolean;
    /**
     * The decision id of that refreshed table, chosen before it is requested: stale_table fallbacks name it, the table
     * they stand in for, so the daemon relabels only that late decision and never the earlier table whose draws were
     * applied (spec 9.7, L2AppliedV1).
     */
    pendingTableDecisionId: string | null;
  }

  // ---- Session-wide state ----------------------------------------------------------------------------------------

  let screen = $state<Screen>("setup");
  let starting = $state(false);
  let setupMessage = $state("");
  let reviewConsent = $state(false);
  let gazeMessage = $state("");
  let pendingSetup: StudySetupChoice | null = null;
  /**
   * The gaze-check accuracy, in memory only, with the participant and input it was measured for: kept for a retry of
   * that same choice from the Setup screen and sent by session_begin (spec 12.2 step 3). It is one person's
   * measurement, so it is never sent with another participant's key or for another input.
   */
  let heldGaze: { participantKey: string; modality: Modality; accuracy: number } | null = null;

  let modality = $state<Modality>("mouse");
  let inputMode = $state<InputMode>("pointer");
  let plan: SessionPlanV1 | null = null;
  let layoutSeed = 0;
  let tau = 0.25;
  let sessionSurface = $state<Surface>({ w: FIT_W, h: FIT_H });
  let sessionDpr = 1;
  let handPointerEffective = false;
  let sessionStartedAt = 0;
  let trialsRecorded = 0;

  let block = $state.raw<BlockRun | null>(null);
  let trialStep = $state(0);
  let fieldTrial = $state.raw<FieldTrial | null>(null);
  let fieldRegime = $state.raw<FieldRegime | null>(null);
  let fieldTable = $state.raw<FieldTable | null>(null);
  let idleRun = $state.raw<{ trial: FieldTrial; regime: FieldRegime; afterBlock: number } | null>(null);
  let field = $state<TrialSurface | null>(null);
  let detachTrial: (() => void) | null = null;
  let interruptionsUsed = 0;
  let previousRegime: string | null = null;
  /** An idle epoch moved behind the next block because its own block ended for a refresh (section 9.7). */
  let movedIdleFrom: number | null = null;
  let afterBurden: Step | null = null;
  let pendingStep: Step | null = null;
  let pauseReason: string | null = null;
  let summary = $state.raw<SessionSummaryV1 | null>(null);
  let summaryLoaded = $state(false);

  let recentCorrections: boolean[] = [];
  let regimeTrials = new Map<string, number>();
  let handQualityBlock: number[] = [];
  let lastBlockHandQuality: number | null = null;
  let vetoRefreshTable = false;

  let offers = $state<OfferResponseV1[]>([]);
  let offerChip = $state<{ decisionId: string; variant: L3Variant } | null>(null);
  let neverHold = $state(false);
  let freezeHold = $state(false);
  /** A rule conflict that blocks a pause, with the toolbar choice it blocked ("Remove that rule" retries that one). */
  let freezeConflict = $state<{ message: string; conflicting: string[]; standard: boolean } | null>(null);
  let adaptMessage = $state("");
  let blockMessage = $state("");
  /** Why the previous block ended early; shown on the next block's intro. */
  let endedMessage = $state("");
  let resuming = $state(false);
  /** A pause asked for while session_resume was in flight: applied as soon as that resume returns (spec 12.2). */
  let pauseDuringResume: string | null = null;

  let stageEl = $state<HTMLDivElement | null>(null);
  let stageSize = $state<Surface>({ w: 0, h: 0 });
  let regimesUsed = $state<string[]>([]);
  let lastPalmExits = 0;
  let trackingLossesAtCue = 0;
  let trackingOfferRequested = false;

  const study = $derived($coadaptation.study);
  /** Read only: whether a task runs or an approval waits (the study pauses for both). */
  const sessionState = $derived($session);
  const sessionOpen = $derived(study.active && study.sessionId !== "");
  const storePaused = $derived(study.phase === "paused");
  // A boolean, so the effect below runs when the flag turns on, not on every other change of the study state.
  const staleBlock = $derived(study.endBlockAfterTrial);
  const endedByDaemon = $derived(study.phase === "error");
  const participantKey = $derived(study.participantKey);
  const inputPaused = $derived(study.inputPaused || neverHold || freezeHold);
  const posteriorParams = $derived<PosteriorParams>({
    modality,
    spread: modality === "keyboard" ? null : study.spread,
    surface: sessionSurface,
    dpr: sessionDpr,
    gazeAccuracy: study.gazeCheckAccuracy !== null && study.gazeCheckAccuracy >= 0.6 ? study.gazeCheckAccuracy : null,
    handPointer: handPointerEffective,
  });
  const totalBlocks = $derived(plan ? mainBlocks(plan).length : 0);
  const runningScreen = $derived(
    sessionOpen && (screen === "block_intro" || screen === "trial" || screen === "burden" || screen === "idle"),
  );
  const toolbarVisible = $derived(
    sessionOpen &&
      !storePaused &&
      (screen === "block_intro" || screen === "trial") &&
      block !== null &&
      !block.practice,
  );
  const setupRefusals = $derived.by(() => {
    const list: string[] = [];
    if (sessionState.loading) list.push(refusalMessage("task_running"));
    if (ARMED_NEURAL_STATES.has(String($neural.state))) list.push(refusalMessage("neural_armed"));
    if (stageSize.w < FIT_W || stageSize.h < FIT_H) list.push(refusalMessage("surface_too_small"));
    return list;
  });

  // ---- Lifecycle ---------------------------------------------------------------------------------------------------

  onMount(() => {
    window.addEventListener("keydown", onWindowKey);
    window.addEventListener("resize", onResize);
    measureStage();
  });

  onDestroy(() => {
    window.removeEventListener("keydown", onWindowKey);
    window.removeEventListener("resize", onResize);
    detachTrial?.();
    detachTrial = null;
    if (get(studyPointer).requested) requestStudyPointer(false);
  });

  // App's `paused` prop (a modal or another tab): pause the running session.
  $effect(() => {
    if (!paused || !runningOrIntro()) return;
    // With an approval waiting App shows it in the Command tab; otherwise another tab or modal is open.
    untrack(() => pauseStudy(sessionState.confirmRequired ? "modal" : "tab"));
  });

  // A task that starts, or neural control that becomes armed, pauses the study (spec 12.2).
  $effect(() => {
    const busy = sessionState.loading;
    const armed = ARMED_NEURAL_STATES.has(String($neural.state));
    if (!runningOrIntro()) return;
    untrack(() => {
      if (busy) pauseStudy("task_running");
      else if (armed) pauseStudy("neural_armed");
    });
  });

  // A palm exit during a hand-pointer session pauses the study.
  $effect(() => {
    const exits = $studyPointer.palmExits;
    untrack(() => {
      if (exits > lastPalmExits && inputMode === "hand_pointer" && runningScreen) pauseStudy("palm_exit");
      lastPalmExits = exits;
    });
  });

  // The second tracking loss in a hand-pointer trial is a struggle signal for an offer (spec 9.4).
  $effect(() => {
    const losses = $studyPointer.trackingLosses;
    untrack(() => {
      if (screen === "trial" && inputMode === "hand_pointer" && !trackingOfferRequested) {
        if (losses - trackingLossesAtCue >= 2) {
          trackingOfferRequested = true;
          void requestOffer("tracking_loss");
        }
      }
    });
  });

  // Whatever paused the session (this component, App, a refusal or a notification), the current trial is excluded.
  $effect(() => {
    if (!storePaused) return;
    untrack(onPaused);
  });

  // The daemon ended the session (revoked, expired, reset, erased or turned off): nothing more is recorded.
  $effect(() => {
    if (!endedByDaemon) return;
    untrack(() => {
      field?.discard();
      clearField();
      idleRun = null;
      pendingStep = null;
    });
  });

  // A refresh signal with no trial in progress ends the block at once (spec 9.7, 16.2).
  $effect(() => {
    if (!staleBlock) return;
    untrack(onBlockStale);
  });

  // ---- Helpers -----------------------------------------------------------------------------------------------------

  /** A session is open and a study screen (not setup, gaze check or summary) is shown. */
  function runningOrIntro(): boolean {
    return sessionOpen && screen !== "setup" && screen !== "gaze_check" && screen !== "summary";
  }

  function newId(): string {
    return globalThis.crypto.randomUUID();
  }

  function measureStage(): Surface {
    const size = { w: Math.floor(stageEl?.clientWidth ?? 0), h: Math.floor(stageEl?.clientHeight ?? 0) };
    stageSize = size;
    return size;
  }

  function surfaceLargeEnough(): boolean {
    const size = measureStage();
    return size.w >= FIT_W && size.h >= FIT_H;
  }

  function onResize() {
    const size = measureStage();
    if (!sessionOpen || !(screen === "trial" || screen === "idle")) return;
    if (size.w < sessionSurface.w || size.h < sessionSurface.h || (window.devicePixelRatio || 1) !== sessionDpr) {
      pauseStudy("resize");
    }
  }

  function onWindowKey(event: KeyboardEvent) {
    // The trial surfaces handle Escape themselves while a trial or rest epoch is shown.
    if (event.key !== "Escape" || event.repeat || screen === "trial" || screen === "idle") return;
    if (runningScreen && !storePaused) pauseStudy("user");
  }

  function fallbacksNow(): FallbacksV1 | null {
    return get(coadaptation).study.fallbacks;
  }

  function activePreferences(): PreferenceV1[] {
    return (get(coadaptation).status?.preferences ?? []).filter((pref) => !pref.retracted_at);
  }

  /** Whether a rule's scope covers this study session (context, modality and participant). */
  function coversSession(scope: PreferenceScopeV1): boolean {
    return (
      (scope.context === "study" || scope.context === "all") &&
      (scope.modality === null || scope.modality === modality) &&
      (scope.participant_key === null || scope.participant_key === participantKey)
    );
  }

  function studyFreezeScope(): PreferenceScopeV1 {
    return { context: "study", modality, participant_key: participantKey };
  }

  /** The active freeze with exactly the scope a new Pause adapting would have (it is replaced, spec 9.7). */
  function sameScopeFreeze(): PreferenceV1 | null {
    return (
      activePreferences().find(
        (pref) =>
          pref.kind === "freeze_adaptation" &&
          pref.scope.context === "study" &&
          pref.scope.modality === modality &&
          pref.scope.participant_key === participantKey,
      ) ?? null
    );
  }

  function frozenRegimeOf(pref: PreferenceV1 | null): string {
    return (pref?.value as { regime?: string } | null)?.regime ?? "";
  }

  /** Which rule put the block in pinned mode, for the retract offer of spec 16.6. */
  function pinnedRuleFor(
    mode: string,
    fallbacks: FallbacksV1 | null,
  ): { rule: PinnedRule | null; variant: string | null } {
    const rules = activePreferences().filter((pref) => coversSession(pref.scope));
    if (mode === "pinned") {
      if (rules.some((pref) => pref.kind === "freeze_adaptation")) return { rule: "freeze", variant: null };
      const pin = rules.find((pref) => pref.kind === "pin_regime");
      if (pin) return { rule: "pin", variant: String(pin.value) };
      return { rule: "freeze", variant: null };
    }
    if (isPinActFallback(fallbacks)) return { rule: "pin_act", variant: null };
    return { rule: null, variant: null };
  }

  function sessionMinute(): number {
    return Math.max(0, (Date.now() - sessionStartedAt) / 60000);
  }

  function mean(values: number[]): number | null {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  // ---- Pause and resume --------------------------------------------------------------------------------------------

  function pauseStudy(reason: string) {
    if (!sessionOpen) return;
    if (screen === "setup" || screen === "gaze_check" || screen === "summary") return;
    if (storePaused) {
      // Already paused, but a resume in flight would end this pause: remember it, so resume() pauses again.
      if (resuming) pauseDuringResume = pauseDuringResume ?? reason;
      return;
    }
    pauseReason = reason;
    coadaptation.pauseStudy(reason);
  }

  /** Why the study must stay paused now (App's paused prop, a running task, armed neural control), or null. */
  function pauseNeeded(): string | null {
    if (paused) return sessionState.confirmRequired ? "modal" : "tab";
    if (sessionState.loading) return "task_running";
    if (ARMED_NEURAL_STATES.has(String(get(neural).state))) return "neural_armed";
    return null;
  }

  /** The session is paused: the current trial or rest epoch ends as interrupted (spec 12.2). */
  function onPaused() {
    if (inputMode === "hand_pointer" && get(studyPointer).requested) requestStudyPointer(false);
    const reason = PAUSE_REASONS[pauseReason ?? ""] ?? "paused";
    pauseReason = null;
    if (!field) return;
    if (field.inProgress()) {
      field.interrupt(reason);
    } else if (screen === "trial") {
      // A trial that never reached its cue is not recorded; it runs again after the resume.
      field.discard();
      pendingStep = { kind: "trial", t: trialStep };
      clearField();
    }
  }

  async function resume() {
    if (paused || resuming) return;
    resuming = true;
    pauseDuringResume = null;
    try {
      // co_adaptation_session_resume first; continue only on "ok" (spec 12.2).
      const ok = await coadaptation.resumeStudy();
      if (!ok) return;
      // A modal, another tab, a task or armed neural control that arrived while the reply was in flight pauses the
      // study again before anything runs: no trial may run hidden or behind a modal, and the study pointer is not
      // requested. The pending step stays for the next resume.
      const again = pauseNeeded() ?? pauseDuringResume;
      pauseDuringResume = null;
      if (again !== null) {
        pauseStudy(again);
        return;
      }
      if (inputMode === "hand_pointer") requestStudyPointer(true);
      const step = pendingStep;
      pendingStep = null;
      continueAfterResume(step);
    } finally {
      resuming = false;
      pauseDuringResume = null;
    }
  }

  function continueAfterResume(step: Step | null) {
    const stale = get(coadaptation).study.endBlockAfterTrial;
    if (!step) {
      // Paused on an intro or a prompt: a refresh applied meanwhile ends the block now (spec 9.7).
      if (stale) onBlockStale();
      return;
    }
    if (stale && block) {
      if (step.kind === "trial" || (step.kind === "block_intro" && block.requested)) {
        // The block's decisions predate a refresh: it ends, and the next trial waits for fresh decisions.
        go({ kind: "after_block", byRefresh: true });
        return;
      }
      if (step.kind === "idle") {
        moveIdle(step.blockIndex);
        coadaptation.blockEnded();
        go(afterBurdenOrNext());
        return;
      }
      coadaptation.blockEnded();
      if (step.kind === "burden" && idleDue(plan!, block.index)) moveIdle(block.index);
    }
    go(step);
  }

  async function stopSession() {
    await exit();
  }

  async function exit() {
    heldGaze = null;
    detachTrial?.();
    detachTrial = null;
    field?.discard();
    clearField();
    await coadaptation.exitStudy();
    onexit();
  }

  // ---- Flow --------------------------------------------------------------------------------------------------------

  function go(step: Step) {
    if (get(coadaptation).study.phase === "paused" || endedByDaemon) {
      pendingStep = step;
      return;
    }
    switch (step.kind) {
      case "block_intro":
        void prepareBlock(step.blockIndex);
        break;
      case "trial":
        startTrial(step.t);
        break;
      case "burden":
        screen = "burden";
        coadaptation.setStudy({ phase: "burden" });
        break;
      case "idle":
        startIdle(step.blockIndex);
        break;
      case "after_block":
        endBlock(step.byRefresh);
        break;
      case "end":
        void finishSession();
        break;
    }
  }

  async function onSetupStart(choice: StudySetupChoice) {
    pendingSetup = choice;
    setupMessage = "";
    reviewConsent = false;
    gazeMessage = "";
    const held = heldGaze;
    if (
      choice.gaze &&
      held !== null &&
      held.participantKey === choice.participantKey &&
      held.modality === choice.modality
    ) {
      await beginSession(held.accuracy);
      return;
    }
    // Another participant or input (or no gaze now): the held measurement is not theirs, so it is dropped.
    heldGaze = null;
    if (choice.gaze) {
      screen = "gaze_check";
      coadaptation.setStudy({ phase: "gaze_check" });
      return;
    }
    await beginSession(null);
  }

  async function onGazeDone(accuracy: number | null) {
    // The accuracy stays in memory until session_begin sends it (spec 12.2 step 3), tied to whom it measured.
    const choice = pendingSetup;
    heldGaze =
      accuracy !== null && choice
        ? { participantKey: choice.participantKey, modality: choice.modality, accuracy }
        : null;
    if (accuracy !== null) {
      const pct = Math.round(accuracy * 100);
      gazeMessage = t(accuracy >= 0.6 ? "study.gaze_check.passed" : "study.gaze_check.low", { values: { pct } });
    }
    await beginSession(accuracy);
  }

  async function beginSession(accuracy: number | null) {
    const choice = pendingSetup;
    if (!choice) return;
    screen = "setup";
    if (!surfaceLargeEnough()) {
      setupMessage = refusalMessage("surface_too_small");
      return;
    }
    starting = true;
    const size = measureStage();
    const dpr = window.devicePixelRatio || 1;
    try {
      const outcome = await coadaptation.beginSession({
        participantKey: choice.participantKey,
        device: {
          modality: choice.modality,
          pointer_types: choice.pointerTypes,
          dpr,
          surface_w: size.w,
          surface_h: size.h,
          gaze: choice.gaze && accuracy !== null && accuracy >= 0.6,
          hand_pointer: choice.modality === "hand_pointer",
        },
        gazeCheck: accuracy === null ? null : { accuracy },
      });
      if (!outcome.ok) {
        setupMessage = outcome.message;
        reviewConsent = outcome.reason === "consent_text_changed" || outcome.reason === "protocol_changed";
        return;
      }
      const reply = outcome.reply;
      heldGaze = null;
      modality = choice.modality;
      inputMode = modality === "keyboard" ? "keyboard" : modality === "hand_pointer" ? "hand_pointer" : "pointer";
      plan = reply.plan;
      layoutSeed = reply.layout_seed;
      tau = reply.thresholds?.ambiguity ?? 0.25;
      sessionSurface = size;
      sessionDpr = dpr;
      handPointerEffective = Boolean(reply.hand_pointer) && modality === "hand_pointer";
      sessionStartedAt = Date.now();
      trialsRecorded = 0;
      previousRegime = null;
      movedIdleFrom = null;
      regimeTrials = new Map();
      recentCorrections = [];
      if (inputMode === "hand_pointer") requestStudyPointer(true);
      const practice = plan.blocks.find((entry) => entry.phase === "practice");
      if (practice) {
        startPractice(practice);
      } else {
        const first = mainBlocks(plan)[0];
        go(first ? { kind: "block_intro", blockIndex: first.index } : { kind: "end" });
      }
    } finally {
      starting = false;
    }
  }

  function startPractice(practice: BlockPlanV1) {
    const built = buildBlockTrials(plan!, practice.index, layoutSeed, sessionSurface, modality);
    const fallbacks = fallbacksNow();
    const regime = fallbacks?.practice ?? null;
    block = {
      index: practice.index,
      plan: practice,
      practice: true,
      position: 0,
      trials: built.trials,
      layouts: built.layouts,
      requested: false,
      blocked: regime === null,
      ready: regime !== null,
      regime: regime ?? "assistance_off",
      applied: regime ?? "assistance_off",
      decisionId: "",
      mode: "baseline",
      l1FallbackReason: null,
      explanation: null,
      table: null,
      tableDecisionId: null,
      tableFallback: null,
      tablePending: false,
      pendingTableDecisionId: null,
    };
    interruptionsUsed = 0;
    screen = "practice_intro";
    coadaptation.setStudy({ phase: "practice", blockIndex: practice.index, trialIndex: 0, interruptionsUsed: 0 });
  }

  function startPracticeTrials() {
    if (!block?.practice) return;
    go({ kind: "trial", t: 0 });
  }

  /** Block start (spec 12.2 step 4a): build the trials, request the L1 decision and, for act-guarded regimes, the table. */
  async function prepareBlock(blockIndex: number) {
    const planned = plan ? blockPlan(plan, blockIndex) : null;
    if (!plan || !planned) {
      go({ kind: "end" });
      return;
    }
    clearField();
    blockMessage = "";
    const built = buildBlockTrials(plan, blockIndex, layoutSeed, sessionSurface, modality);
    const position = mainBlocks(plan).findIndex((entry) => entry.index === blockIndex) + 1;
    const run: BlockRun = {
      index: blockIndex,
      plan: planned,
      practice: false,
      position,
      trials: built.trials,
      layouts: built.layouts,
      requested: false,
      blocked: false,
      ready: false,
      regime: "assistance_off",
      applied: "assistance_off",
      decisionId: "",
      mode: "",
      l1FallbackReason: null,
      explanation: null,
      table: null,
      tableDecisionId: null,
      tableFallback: null,
      tablePending: false,
      pendingTableDecisionId: null,
    };
    block = run;
    interruptionsUsed = 0;
    handQualityBlock = [];
    screen = "block_intro";
    coadaptation.setStudy({ phase: "running", blockIndex, trialIndex: 0, interruptionsUsed: 0 });

    const fallbacks = fallbacksNow();
    if (!fallbacks || fallbacks.l1 === null) {
      // No block starts without fallbacks or when nothing is allowed (spec 9.7); Pause adapting can unblock it.
      block = { ...run, blocked: true };
      blockMessage = refusalMessage("no_allowed_regime");
      return;
    }

    const context = {
      block_index: blockIndex,
      layouts: built.layouts,
      previous_regime: previousRegime,
      session_minute: sessionMinute(),
      trials_today: trialsRecorded,
      hand_quality_recent: inputMode === "hand_pointer" ? lastBlockHandQuality : null,
    };
    const introAt = Date.now();
    block = { ...run, requested: true };
    const outcome = await coadaptation.recommendBlock(context);
    if (block?.index !== blockIndex || screen !== "block_intro") return;
    if (outcome.status === "refused") {
      // The store paused the study with the refusal's message; the block is requested again after the resume.
      pendingStep = { kind: "block_intro", blockIndex };
      block = { ...block, requested: false };
      return;
    }
    if (outcome.status === "ignored") {
      blockMessage = outcome.message;
      block = { ...block, blocked: true, requested: false };
      return;
    }
    const held = fallbacksNow() ?? fallbacks;
    let next: BlockRun = { ...block };
    const recommendation = outcome.status === "ok" ? outcome.recommendation : null;
    if (recommendation && isPresentationOnly(recommendation) && isL1Variant(recommendation.variant)) {
      next = {
        ...next,
        regime: recommendation.variant,
        applied: recommendation.variant,
        decisionId: outcome.decisionId,
        mode: recommendation.mode,
        l1FallbackReason: null,
        explanation: recommendation.explanation ?? null,
      };
    } else if (held.l1 === null) {
      block = { ...next, blocked: true };
      blockMessage = refusalMessage("no_allowed_regime");
      return;
    } else {
      // L1 fallback (late or failed): the most cautious allowed regime, recorded as a fallback (spec 9.7).
      next = {
        ...next,
        regime: held.l1,
        applied: held.l1,
        decisionId: outcome.decisionId,
        mode: "fallback",
        l1FallbackReason: outcome.status === "late" ? "late" : "rpc_error",
        explanation: null,
      };
    }

    if (isActGuarded(next.applied)) {
      if (next.l1FallbackReason !== null) {
        next = { ...next, tableFallback: next.l1FallbackReason };
      } else {
        const remaining = Math.max(1, 2000 - (Date.now() - introAt));
        const tableOutcome = await coadaptation.recommendTable(
          {
            ...context,
            block_decision_id: next.decisionId,
            start_trial_index: 0,
            trials_in_block: built.trials.length,
          },
          { timeoutMs: remaining },
        );
        if (block?.index !== blockIndex || screen !== "block_intro") return;
        if (tableOutcome.status === "refused") {
          pendingStep = { kind: "block_intro", blockIndex };
          block = { ...block, requested: false };
          return;
        }
        next = { ...next, tableDecisionId: tableOutcome.decisionId };
        if (tableOutcome.status === "ok" && isPresentationOnly(tableOutcome.recommendation)) {
          next = { ...next, table: tableOutcome.recommendation, tableFallback: null };
        } else {
          next = { ...next, table: null, tableFallback: tableOutcome.status === "late" ? "late" : "rpc_error" };
        }
      }
    }
    block = { ...next, ready: true };
  }

  function startBlock() {
    if (!block?.ready || block.blocked) return;
    endedMessage = "";
    go({ kind: "trial", t: 0 });
  }

  function blockIntroText(run: BlockRun): string {
    if (run.l1FallbackReason !== null) return t("study.block.fallback");
    switch (run.mode) {
      case "randomized":
        return t("study.block.randomized");
      case "probe":
        return t("study.block.probe");
      case "comparator":
        return t("study.block.comparator");
      case "adaptive":
        return t("study.block.adaptive");
      case "pinned":
        return t("study.block.pinned");
      default:
        return "";
    }
  }

  function clearField() {
    detachTrial?.();
    detachTrial = null;
    fieldTrial = null;
    fieldRegime = null;
    fieldTable = null;
  }

  function startTrial(index: number) {
    const run = block;
    if (!run || !plan) return;
    if (index >= run.trials.length) {
      go({ kind: "after_block", byRefresh: false });
      return;
    }
    if (!run.practice && get(coadaptation).study.endBlockAfterTrial) {
      go({ kind: "after_block", byRefresh: true });
      return;
    }
    const fallbacks = fallbacksNow();
    if (!fallbacks) {
      blockMessage = refusalMessage("no_allowed_regime");
      return;
    }
    let applied = run.applied;
    if (run.practice) {
      // The practice regime always comes from the newest fallbacks (never a stale one).
      if (fallbacks.practice === null) {
        go({ kind: "after_block", byRefresh: false });
        return;
      }
      applied = fallbacks.practice;
    }
    const scheduled = run.trials[index];
    const task: TaskKind = scheduled.layout.task === "menu" ? "menu" : "pointing";
    const pinned = pinnedRuleFor(run.mode, fallbacks);
    trialStep = index;
    offers = [];
    offerChip = null;
    trackingOfferRequested = false;
    trackingLossesAtCue = get(studyPointer).trackingLosses;
    fieldTrial = {
      trialId: newId(),
      blockIndex: run.index,
      trialIndex: index,
      task,
      layout: scheduled.layout,
      intendedIndex: scheduled.intendedIndex,
      labelWord: scheduled.labelWord,
      limitMs: TRIAL_LIMIT_MS[task],
      record: !run.practice,
    };
    fieldRegime = {
      regime: run.practice ? applied : run.regime,
      applied,
      decisionId: run.decisionId,
      mode: run.mode,
      l1FallbackReason: run.l1FallbackReason,
      explanation: run.explanation,
      pinnedRule: pinned.rule,
      pinnedVariant: pinned.variant,
      fallbacks,
      interruptionsUsed,
      tau,
      chips: !run.practice,
    };
    fieldTable = fieldTableOf(run);
    screen = "trial";
    coadaptation.setStudy({
      phase: run.practice ? "practice" : "running",
      trialIndex: index,
      interruptionsUsed,
    });
    if (!run.practice) void requestOffer("corrections");
  }

  /** The L2 side a trial of this block starts with. */
  function fieldTableOf(run: BlockRun): FieldTable {
    return {
      table: run.tablePending ? null : run.table,
      decisionId: run.tablePending ? (run.pendingTableDecisionId ?? run.tableDecisionId) : run.tableDecisionId,
      fallbackReason: run.tablePending ? "stale_table" : run.tableFallback,
      fixedRule: run.practice,
    };
  }

  /** The trial reached its cue: the store may now dispatch refreshPending and refresh to it (spec 16.2). */
  function onCue() {
    detachTrial?.();
    detachTrial = coadaptation.attachTrial(routeToTrial);
  }

  function routeToTrial(event: TrialEvent) {
    if (!field) return;
    if (event.type === "refresh") {
      // The store adopts the fresh fallbacks before it dispatches: hand them over for the fixed L2 rule too.
      const fresh = fallbacksNow();
      if (fresh && fresh.issued_at === event.fallbacksIssuedAt) {
        field.refresh(fresh);
        return;
      }
    }
    field.dispatch(event);
  }

  function onTrialEnd(end: FieldTrialEnd) {
    detachTrial?.();
    detachTrial = null;
    const run = block;
    if (!run) return;
    interruptionsUsed = end.state.interruptionsUsed;
    coadaptation.setStudy({ interruptionsUsed });
    if (end.trial) {
      if (coadaptation.recordTrial(end.trial)) trialsRecorded += 1;
      recentCorrections = [...recentCorrections, end.state.corrections > 0].slice(-3);
      regimeTrials.set(run.applied, (regimeTrials.get(run.applied) ?? 0) + 1);
    }
    if (handPointerEffective && end.trial?.hand) handQualityBlock.push(end.trial.hand.quality_mean);
    const index = trialStep;
    clearField();

    if (end.state.interruptionReason === "no_allowed_regime") {
      // A "Not this time" with nothing allowed to revert to: the block ends with the message of spec 16.6.
      endedMessage = t("study.chip.nothing_to_revert");
      go({ kind: "after_block", byRefresh: false });
      return;
    }
    if (end.state.interruptionReason === "refresh" && fallbacksNow()?.refresh === null) {
      // The new rules leave no regime to revert to: the selection stopped and the block ends (spec 16.6).
      endedMessage = t("study.refresh.nothing_left");
    }
    if (!run.practice && vetoRefreshTable) {
      vetoRefreshTable = false;
      void refreshTableAfterVeto(index + 1);
    }
    if (run.practice && (end.state.endBlockAfterTrial || get(coadaptation).study.endBlockAfterTrial)) {
      coadaptation.blockEnded();
    } else if (!run.practice && (end.state.endBlockAfterTrial || get(coadaptation).study.endBlockAfterTrial)) {
      go({ kind: "after_block", byRefresh: true });
      return;
    }
    go({ kind: "trial", t: index + 1 });
  }

  /** After a veto outside pinned mode the remaining trials get a refreshed table (spec 9.3); the veto is sent first. */
  async function refreshTableAfterVeto(startIndex: number) {
    const run = block;
    if (!run || run.practice || !isActGuarded(run.applied) || run.l1FallbackReason !== null) return;
    if (startIndex >= run.trials.length) return;
    const decisionId = newId();
    block = { ...run, tablePending: true, pendingTableDecisionId: decisionId };
    await coadaptation.flushTrials();
    const current = block;
    if (!current || current.index !== run.index) return;
    const outcome = await coadaptation.recommendTable(
      {
        block_index: run.index,
        layouts: run.layouts,
        previous_regime: previousRegime,
        session_minute: sessionMinute(),
        trials_today: trialsRecorded,
        hand_quality_recent: inputMode === "hand_pointer" ? lastBlockHandQuality : null,
        block_decision_id: run.decisionId,
        start_trial_index: startIndex,
        trials_in_block: run.trials.length - startIndex,
      },
      { decisionId },
    );
    if (!block || block.index !== run.index) return;
    if (outcome.status === "ok" && isPresentationOnly(outcome.recommendation)) {
      block = {
        ...block,
        table: outcome.recommendation,
        tableDecisionId: outcome.decisionId,
        tableFallback: null,
        tablePending: false,
        pendingTableDecisionId: null,
      };
    } else {
      block = {
        ...block,
        table: null,
        tableDecisionId: outcome.decisionId,
        tableFallback: outcome.status === "late" ? "late" : "rpc_error",
        tablePending: false,
        pendingTableDecisionId: null,
      };
    }
    // The trial already running (usually the one right after the veto) switches at once: the stale-table fallback
    // lasts only while the requested table has not arrived (spec 9.7).
    const running = fieldTrial;
    if (running && running.blockIndex === block.index && running.trialIndex >= startIndex && fieldTable) {
      fieldTable = fieldTableOf(block);
      field?.adoptTable?.(fieldTable);
    }
  }

  function onVeto(veto: FieldVeto) {
    // In pinned mode the veto reverts this trial only; the chip then offers to retract the rule (spec 9.7).
    if (veto.pinned) return;
    if (veto.level === "ambiguity_response") vetoRefreshTable = true;
  }

  async function onNever(variant: string) {
    neverHold = true;
    try {
      await coadaptation.setPreference(
        "forbid_variant",
        { context: "study", modality: null, participant_key: participantKey },
        variant,
      );
    } catch (cause) {
      adaptMessage = cause instanceof Error ? cause.message : String(cause);
    } finally {
      neverHold = false;
    }
  }

  async function onRetract() {
    const rule = fieldRegime?.pinnedRule ?? null;
    const prefs = activePreferences().filter((pref) => coversSession(pref.scope));
    const match =
      rule === "pin"
        ? prefs.find((pref) => pref.kind === "pin_regime")
        : rule === "pin_act"
          ? prefs.find((pref) => pref.kind === "pin_act")
          : prefs.find((pref) => pref.kind === "freeze_adaptation");
    if (!match) return;
    try {
      await coadaptation.retractPreference(match.pref_id);
    } catch (cause) {
      adaptMessage = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function endBlock(byRefresh: boolean) {
    const run = block;
    clearField();
    if (byRefresh) {
      coadaptation.blockEnded();
      endedMessage = endedMessage || t("study.refresh.new_block");
    }
    if (!run || !plan) {
      go({ kind: "end" });
      return;
    }
    if (run.practice) {
      const first = mainBlocks(plan)[0];
      go(first ? { kind: "block_intro", blockIndex: first.index } : { kind: "end" });
      return;
    }
    previousRegime = run.applied;
    lastBlockHandQuality = mean(handQualityBlock);
    const next = nextBlockIndex(plan, run.index);
    const nextStep: Step = next === null ? { kind: "end" } : { kind: "block_intro", blockIndex: next };
    if (byRefresh) {
      // No burden prompt for a block ended by a refresh; its idle epoch follows the next block instead (spec 9.7).
      if (idleDue(plan, run.index)) moveIdle(run.index);
      go(nextStep);
      return;
    }
    const idleNow = movedIdleFrom !== null || idleDue(plan, run.index);
    afterBurden = idleNow ? { kind: "idle", blockIndex: run.index } : nextStep;
    if (burdenDue(plan, run.index)) {
      go({ kind: "burden" });
      return;
    }
    go(afterBurdenOrNext());
  }

  function afterBurdenOrNext(): Step {
    const step = afterBurden;
    afterBurden = null;
    if (step) return step;
    const run = block;
    const next = run && plan ? nextBlockIndex(plan, run.index) : null;
    return next === null ? { kind: "end" } : { kind: "block_intro", blockIndex: next };
  }

  function moveIdle(fromBlock: number) {
    movedIdleFrom = movedIdleFrom ?? fromBlock;
    if (afterBurden?.kind === "idle") {
      const run = block;
      const next = run && plan ? nextBlockIndex(plan, run.index) : null;
      afterBurden = next === null ? { kind: "end" } : { kind: "block_intro", blockIndex: next };
    }
  }

  async function submitBurden(answers: BurdenAnswers) {
    const run = block;
    if (run) {
      const report: BurdenV1 = {
        block_index: run.index,
        scope: "block",
        effort: answers.effort,
        intrusiveness: answers.intrusiveness,
        control: answers.control,
        tlx: null,
        predictability: null,
        preferred_regime: null,
        skipped: answers.skipped,
      };
      await coadaptation.reportBurden(report);
    }
    go(afterBurdenOrNext());
  }

  /** A 30 s rest epoch under the regime of the block it follows (spec 12.3). */
  function startIdle(afterBlock: number) {
    const run = block;
    if (!run || !plan) {
      go(afterBurdenOrNext());
      return;
    }
    movedIdleFrom = null;
    const epoch = idleEpoch(idleEpochSeed(layoutSeed, afterBlock), sessionSurface);
    const fallbacks = fallbacksNow() ?? fieldRegime?.fallbacks ?? null;
    if (!fallbacks) {
      go(afterBurdenOrNext());
      return;
    }
    idleRun = {
      afterBlock,
      trial: {
        trialId: newId(),
        blockIndex: run.index,
        trialIndex: run.trials.length,
        task: "idle",
        layout: epoch.layout,
        intendedIndex: null,
        labelWord: null,
        limitMs: 30000,
        record: true,
      },
      regime: {
        regime: run.regime,
        applied: run.applied,
        decisionId: run.decisionId,
        mode: run.mode,
        l1FallbackReason: run.l1FallbackReason,
        explanation: null,
        pinnedRule: null,
        pinnedVariant: null,
        fallbacks,
        interruptionsUsed: 0,
        tau,
        chips: false,
      },
    };
    screen = "idle";
    coadaptation.setStudy({ phase: "idle_epoch", interruptionsUsed: 0 });
  }

  function onIdleEnd(end: FieldTrialEnd) {
    const run = idleRun;
    idleRun = null;
    if (end.trial) {
      if (coadaptation.recordTrial(end.trial)) trialsRecorded += 1;
    }
    if (end.state.interruptionReason === "refresh" && run) {
      // An epoch stopped by a refresh moves behind the next block (dropped when none follows).
      moveIdle(run.afterBlock);
      coadaptation.blockEnded();
    }
    const next = block && plan ? nextBlockIndex(plan, block.index) : null;
    if (get(coadaptation).study.phase === "paused") {
      pendingStep = next === null ? { kind: "end" } : { kind: "block_intro", blockIndex: next };
      return;
    }
    go(next === null ? { kind: "end" } : { kind: "block_intro", blockIndex: next });
  }

  /** A refresh signal applied while no trial is in progress (spec 9.7). */
  function onBlockStale() {
    const current = get(coadaptation).study;
    // While paused, or while a resume is being handled, continueAfterResume decides what ends.
    if (current.phase === "paused" || !sessionOpen || resuming) return;
    if (screen === "trial") {
      if (field?.inProgress()) return;
      if (block?.practice) {
        coadaptation.blockEnded();
        return;
      }
      field?.discard();
      go({ kind: "after_block", byRefresh: true });
      return;
    }
    if (screen === "block_intro" && block) {
      if (!block.requested || block.blocked) {
        // Nothing was decided under the old rules: try this block again with the fresh fallbacks.
        coadaptation.blockEnded();
        if (block.blocked) {
          blockMessage = "";
          go({ kind: "block_intro", blockIndex: block.index });
        }
        return;
      }
      go({ kind: "after_block", byRefresh: true });
      return;
    }
    if (screen === "idle") {
      field?.interrupt("refresh");
      return;
    }
    if (screen === "burden" && block && plan) {
      if (afterBurden?.kind === "idle") moveIdle(block.index);
      coadaptation.blockEnded();
      return;
    }
    coadaptation.blockEnded();
  }

  async function finishSession() {
    clearField();
    regimesUsed = [...regimeTrials.entries()].sort((a, b) => b[1] - a[1]).map(([regime]) => regime);
    if (plan?.end_questionnaire) {
      screen = "end_questionnaire";
      coadaptation.setStudy({ phase: "end_questionnaire" });
      return;
    }
    await completeSession();
  }

  async function submitEnd(answers: EndAnswers) {
    const report: BurdenV1 = {
      block_index: null,
      scope: "session",
      effort: null,
      intrusiveness: null,
      control: null,
      tlx: answers.tlx,
      predictability: answers.predictability,
      preferred_regime:
        answers.preferred_regime && isL1Variant(answers.preferred_regime) ? answers.preferred_regime : null,
      skipped: answers.skipped,
    };
    await coadaptation.reportBurden(report);
    await completeSession();
  }

  async function completeSession() {
    summary = await coadaptation.endSession("completed");
    summaryLoaded = true;
    screen = "summary";
    coadaptation.setStudy({ phase: "summary" });
    if (get(studyPointer).requested) requestStudyPointer(false);
  }

  async function neverFromSummary(variant: string) {
    try {
      await coadaptation.setPreference(
        "forbid_variant",
        { context: "study", modality: null, participant_key: participantKey },
        variant,
      );
    } catch (cause) {
      blockMessage = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function eraseFromSummary() {
    try {
      await coadaptation.eraseToday(participantKey);
    } catch (cause) {
      blockMessage = cause instanceof Error ? cause.message : String(cause);
    }
  }

  // ---- Offers (L3, spec 9.4) ---------------------------------------------------------------------------------------

  async function requestOffer(kind: "corrections" | "tracking_loss") {
    const run = block;
    const trial = fieldTrial;
    if (!run || run.practice || !trial || run.mode === "probe" || run.mode === "pinned" || offers.length >= 2) return;
    if (kind === "corrections" && recentCorrections.filter(Boolean).length < 2) return;
    const outcome = await coadaptation.recommendOffer({
      block_index: run.index,
      trial_index: trial.trialIndex,
      task: trial.task,
      struggle_kind: kind,
      keyboard_available: modality !== "keyboard",
    });
    if (fieldTrial?.trialId !== trial.trialId) return;
    if (outcome.status === "ok") {
      const variant = outcome.recommendation.variant as L3Variant;
      offers = [...offers, { decision_id: outcome.decisionId, variant, response: "ignored", fallback_reason: null }];
      if (variant !== "no_offer") offerChip = { decisionId: outcome.decisionId, variant };
    } else if (outcome.status === "late" || outcome.status === "rpc_error") {
      // No offer was shown: recorded as no_offer with the fallback reason (the trial is unaffected).
      offers = [
        ...offers,
        { decision_id: outcome.decisionId, variant: "no_offer", response: "ignored", fallback_reason: outcome.status },
      ];
    }
  }

  function respondOffer(response: "accepted" | "dismissed") {
    const chip = offerChip;
    if (!chip) return;
    offers = offers.map((offer) => (offer.decision_id === chip.decisionId ? { ...offer, response } : offer));
    offerChip = null;
  }

  // ---- Pause adapting (spec 9.7, 16.6) -----------------------------------------------------------------------------

  /** The regime the current selection runs under (the revert regime after a revert). */
  function currentRegime(): string {
    if (fieldRegime) return fieldRegime.applied;
    if (block?.ready && !block.blocked) return block.applied;
    return previousRegime ?? fallbacksNow()?.l1 ?? "assistance_off";
  }

  async function pauseAdapting(standard: boolean) {
    const fallbacks = fallbacksNow();
    if (!fallbacks || freezeHold) return;
    freezeConflict = null;
    adaptMessage = "";
    const current = currentRegime();
    const freezable = fallbacks.freezable as string[];
    const earlier = sameScopeFreeze();
    let regime: string;
    if (standard) {
      regime = "assistance_off";
    } else if (freezable.length === 0) {
      showConflict(fallbacks.freeze_conflict, false);
      return;
    } else {
      regime = freezable.includes(current) ? current : freezable[freezable.length - 1];
    }
    const keptMessage = standard
      ? t("study.pause_adapting.keeps", { values: { variant: variantLabel(t, regime) } })
      : regime === current
        ? t("study.pause_adapting.keeps", { values: { variant: variantLabel(t, regime) } })
        : t("study.pause_adapting.not_allowed", {
            values: { variant: variantLabel(t, current), fallback: variantLabel(t, regime) },
          });

    if (earlier) {
      // Replace the earlier pause only when the new freeze is known to be allowed (spec 9.7).
      if (!freezable.includes(regime)) {
        // fallbacks.freeze_conflict is the conflict of a freeze of the current regime; standard selection shows the
        // conflict of a freeze of assistance_off instead.
        showConflict(standard ? standardConflict(earlier) : fallbacks.freeze_conflict, standard);
        return;
      }
      const earlierRegime = frozenRegimeOf(earlier);
      freezeHold = true;
      try {
        try {
          await coadaptation.retractPreference(earlier.pref_id);
        } catch (cause) {
          adaptMessage = errorText(cause);
          return;
        }
        try {
          await coadaptation.setPreference("freeze_adaptation", studyFreezeScope(), { regime });
          adaptMessage = `${keptMessage} ${t("study.pause_adapting.replaces", {
            values: { variant: variantLabel(t, earlierRegime) },
          })}`;
        } catch (cause) {
          await restoreEarlierPause(earlier, cause, { standard, regime, keptMessage });
        }
      } finally {
        freezeHold = false;
      }
      return;
    }

    freezeHold = true;
    try {
      await coadaptation.setPreference("freeze_adaptation", studyFreezeScope(), { regime });
      adaptMessage = keptMessage;
    } catch (cause) {
      showError(cause, standard);
    } finally {
      freezeHold = false;
    }
  }

  /**
   * The daemon rejected the new freeze after the earlier one was retracted (the rules changed in between): the
   * earlier freeze is sent again unchanged and the conflict shown (spec 9.7). "Kept" is said only when that restore
   * succeeded or a fresh status shows the earlier pause in force; otherwise the study says the pause could not be
   * restored, so a lost pause is never reported as kept.
   */
  async function restoreEarlierPause(
    earlier: PreferenceV1,
    cause: unknown,
    attempt: { standard: boolean; regime: string; keptMessage: string },
  ) {
    const earlierRegime = frozenRegimeOf(earlier);
    const earlierName = variantLabel(t, earlierRegime);
    let restoreFailure: unknown = null;
    try {
      await coadaptation.setPreference("freeze_adaptation", earlier.scope, earlier.value);
    } catch (failure) {
      restoreFailure = failure;
    }
    showError(cause, attempt.standard);
    if (restoreFailure === null) {
      adaptMessage = t("study.pause_adapting.kept", { values: { variant: earlierName } });
      return;
    }
    const before = get(coadaptation).status;
    await coadaptation.load();
    const reloaded = get(coadaptation).status !== before;
    const inForce = reloaded ? sameScopeFreeze() : null;
    const inForceRegime = frozenRegimeOf(inForce);
    if (inForce && inForceRegime === earlierRegime) {
      adaptMessage = t("study.pause_adapting.kept", { values: { variant: earlierName } });
    } else if (inForce && inForceRegime === attempt.regime) {
      // The new freeze had been saved after all (its reply was lost): that pause is in force, with no conflict.
      freezeConflict = null;
      adaptMessage = `${attempt.keptMessage} ${t("study.pause_adapting.replaces", {
        values: { variant: earlierName },
      })}`;
    } else {
      adaptMessage = t(reloaded ? "study.pause_adapting.restore_failed" : "study.pause_adapting.restore_unknown", {
        values: { variant: earlierName, reason: errorText(restoreFailure) },
      });
    }
  }

  /**
   * The conflict of a study freeze of assistance_off ("Pause with standard selection"), judged like the daemon judges
   * a freeze (spec 9.6) with the freeze it would replace left out: a forbid of assistance_off, a minimum caution above
   * it, or a pin or freeze of another regime of this input in an overlapping scope. The daemon validates the freeze
   * again when it is sent. Null when no such rule is found.
   */
  function standardConflict(replaced: PreferenceV1 | null): { message: string; conflicting: string[] } | null {
    const arms: string[] = L1_ARMS(modality);
    const offRank = CAUTION_ORDER.indexOf("assistance_off");
    const holdsOther = (held: string) => arms.includes(held) && held !== "assistance_off";
    const blocking = activePreferences().filter((pref) => {
      if (pref.pref_id === replaced?.pref_id || !coversSession(pref.scope)) return false;
      switch (pref.kind) {
        case "forbid_variant":
          return pref.value === "assistance_off";
        case "min_caution":
          return CAUTION_ORDER.indexOf(String(pref.value)) > offRank;
        case "pin_regime":
          return holdsOther(String(pref.value));
        case "freeze_adaptation":
          return holdsOther(frozenRegimeOf(pref));
        default:
          return false;
      }
    });
    if (blocking.length === 0) return null;
    const rules = blocking.map((pref) => ruleSummary(t, pref)).join("; ");
    return {
      message: t(
        blocking.length === 1
          ? "study.pause_adapting.standard_conflict"
          : "study.pause_adapting.standard_conflict_many",
        { values: { rules } },
      ),
      conflicting: blocking.map((pref) => pref.pref_id),
    };
  }

  function errorText(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  function showConflict(conflict: { message: string; conflicting: string[] } | null, standard: boolean) {
    freezeConflict = conflict
      ? { ...conflict, standard }
      : { message: refusalMessage("no_allowed_regime"), conflicting: [], standard };
  }

  function showError(cause: unknown, standard: boolean) {
    if (cause instanceof PreferenceConflictError) {
      freezeConflict = {
        message: cause.message,
        conflicting: cause.conflicting.map((entry) => entry.pref_id),
        standard,
      };
    } else {
      adaptMessage = errorText(cause);
    }
  }

  /** "Remove that rule": retract each conflicting rule, then try the same Pause adapting choice again. */
  async function removeConflicting() {
    const conflict = freezeConflict;
    if (!conflict) return;
    freezeHold = true;
    try {
      for (const prefId of conflict.conflicting) await coadaptation.retractPreference(prefId);
    } catch (cause) {
      adaptMessage = errorText(cause);
      return;
    } finally {
      freezeHold = false;
    }
    freezeConflict = null;
    await pauseAdapting(conflict.standard);
  }

  async function dismissStudy() {
    await exit();
  }

  function rememberAcrossSessions(): boolean {
    const status = get(coadaptation).status;
    const participant = status?.participants?.find((entry) => entry.participant_key === participantKey);
    return Boolean(
      status?.config.learn_across_sessions && participant?.consent.purposes.includes("cross_session_learning"),
    );
  }
</script>

<div class="study" aria-label={t("study.title")}>
  <div class="stage" bind:this={stageEl}>
    <div class="screen" inert={storePaused || endedByDaemon}>
      {#if screen === "setup"}
        <StudySetup
          onstart={onSetupStart}
          oncancel={dismissStudy}
          message={setupMessage}
          refusals={setupRefusals}
          {starting}
          {reviewConsent}
        />
      {:else if screen === "gaze_check"}
        <StudyGazeCheck ondone={onGazeDone} />
      {:else if screen === "practice_intro"}
        <div class="panel">
          <h3>{t("study.practice.title")}</h3>
          <p>{t("study.practice.intro")}</p>
          {#if gazeMessage}<p class="note">{gazeMessage}</p>{/if}
          {#if block?.blocked}
            <p class="error" role="alert">{refusalMessage("no_allowed_regime")}</p>
          {/if}
          <div class="actions">
            <button class="secondary" onclick={stopSession}>{t("study.pause.stop")}</button>
            <button class="primary" onclick={startPracticeTrials} disabled={!block || block.blocked}>
              {t("study.practice.start")}
            </button>
          </div>
        </div>
      {:else if screen === "block_intro" && block}
        <div class="panel">
          <h3>{t("study.block.intro", { values: { n: block.position, total: totalBlocks } })}</h3>
          {#if block.blocked}
            <p class="error" role="alert">{blockMessage}</p>
          {:else if !block.ready}
            <p>{t("study.block.preparing")}</p>
          {:else}
            <p>{blockIntroText(block)}</p>
            <p class="note">{t("study.block.help_label", { values: { variant: variantLabel(t, block.applied) } })}</p>
            {#if block.mode === "adaptive" && block.explanation}
              <details>
                <summary>{t("study.block.why")}</summary>
                <p class="note">{explanationText(t, block.explanation)}</p>
              </details>
            {/if}
          {/if}
          {#if blockMessage && !block.blocked}<p class="note" role="status">{blockMessage}</p>{/if}
          {#if endedMessage}<p class="note" role="status">{endedMessage}</p>{/if}
          <div class="actions">
            <button class="secondary" onclick={() => pauseStudy("user")}>{t("study.pause.pause")}</button>
            <button class="primary" onclick={startBlock} disabled={!block.ready || block.blocked}>
              {t("study.block.start")}
            </button>
          </div>
        </div>
      {:else if screen === "trial" && fieldTrial && fieldRegime && fieldTable}
        {#key fieldTrial.trialId}
          {#if fieldTrial.task === "menu"}
            <StudyMenuTask
              bind:this={field}
              trial={fieldTrial}
              regime={fieldRegime}
              table={fieldTable}
              {posteriorParams}
              {inputMode}
              {inputPaused}
              {offers}
              ontrialend={onTrialEnd}
              onveto={onVeto}
              onnever={onNever}
              onneverstep={(open) => (neverHold = open)}
              onretract={onRetract}
              onpause={(reason) => pauseStudy(reason === "paused" ? "user" : reason)}
              oncue={onCue}
            />
          {:else}
            <StudyPointingField
              bind:this={field}
              trial={fieldTrial}
              regime={fieldRegime}
              table={fieldTable}
              {posteriorParams}
              {inputMode}
              {inputPaused}
              {offers}
              ontrialend={onTrialEnd}
              onveto={onVeto}
              onnever={onNever}
              onneverstep={(open) => (neverHold = open)}
              onretract={onRetract}
              onpause={(reason) => pauseStudy(reason === "paused" ? "user" : reason)}
              oncue={onCue}
            />
          {/if}
        {/key}
        {#if offerChip}
          <div class="offer" role="status">
            <span>
              {t(offerChip.variant === "offer_modality:keyboard" ? "study.chip.try_keyboard" : "study.chip.try_mouse")}
            </span>
            <button onclick={() => respondOffer("accepted")}>{t("study.chip.try_it")}</button>
            <button aria-label={t("study.chip.dismiss")} onclick={() => respondOffer("dismissed")}>×</button>
          </div>
        {/if}
      {:else if screen === "idle" && idleRun}
        {#key idleRun.trial.trialId}
          <StudyIdleEpoch
            bind:this={field}
            trial={idleRun.trial}
            durationMs={IDLE_EPOCH_MS}
            regime={idleRun.regime}
            {posteriorParams}
            {inputMode}
            {inputPaused}
            onend={onIdleEnd}
            onpause={(reason) => pauseStudy(reason === "paused" ? "user" : reason)}
          />
        {/key}
      {:else if screen === "burden" && block}
        <BurdenPrompt showIntrusiveness={block.applied !== "assistance_off"} onsubmit={submitBurden} />
      {:else if screen === "end_questionnaire"}
        <EndQuestionnaire {regimesUsed} rememberAcrossSessions={rememberAcrossSessions()} onsubmit={submitEnd} />
      {:else if screen === "summary"}
        {#if summaryLoaded}
          <StudySummary
            {summary}
            onclose={dismissStudy}
            onnever={neverFromSummary}
            onrules={dismissStudy}
            onerase={eraseFromSummary}
          />
          {#if blockMessage}<p class="summary-error" role="alert">{blockMessage}</p>{/if}
        {/if}
      {/if}
    </div>

    <!-- Over the current screen, so a running trial is still there to be ended as interrupted. -->
    {#if endedByDaemon}
      <div class="cover">
        <div class="panel" role="alert">
          <p>{study.message}</p>
          <button class="primary" onclick={dismissStudy}>{t("study.summary.close")}</button>
        </div>
      </div>
    {:else if storePaused}
      <div class="cover">
        <div class="panel paused" role="dialog" aria-labelledby="study-paused-title">
          <h3 id="study-paused-title">{t("study.pause.paused")}</h3>
          {#if study.message}<p>{study.message}</p>{/if}
          <div class="actions">
            <button class="secondary" onclick={stopSession}>{t("study.pause.stop")}</button>
            <button class="primary" onclick={resume} disabled={paused || resuming}>
              {resuming ? t("study.pause.resuming") : t("study.pause.resume")}
            </button>
          </div>
        </div>
      </div>
    {/if}
  </div>

  {#if toolbarVisible}
    <div class="toolbar" role="toolbar" aria-label={t("study.title")}>
      <button onclick={() => pauseAdapting(false)} disabled={freezeHold}>{t("study.pause_adapting.button")}</button>
      <button onclick={() => pauseAdapting(true)} disabled={freezeHold}>{t("study.pause_adapting.standard")}</button>
      <button onclick={() => pauseStudy("user")}>{t("study.pause.pause")}</button>
      {#if study.message && (study.inputPaused || study.endBlockAfterTrial)}
        <span class="status" role="status">{study.message}</span>
      {/if}
      {#if adaptMessage}
        <span class="status" role="status">{adaptMessage}</span>
      {/if}
      {#if freezeConflict}
        <div class="conflict" role="alert">
          <span>{freezeConflict.message}</span>
          {#if freezeConflict.conflicting.length}
            <button onclick={removeConflicting} disabled={freezeHold}
              >{t("study.pause_adapting.remove_that_rule")}</button
            >
          {/if}
          <button onclick={() => (freezeConflict = null)}>{t("study.pause_adapting.keep_it")}</button>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .study {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    min-height: 0;
    color: var(--text-primary);
  }

  .stage {
    position: relative;
    display: flex;
    flex: 1;
    align-items: center;
    justify-content: center;
    min-height: 0;
    overflow: hidden;
  }

  .panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 520px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }

  .panel h3 {
    margin: 0;
    font-size: 15px;
  }

  .panel p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .panel .note {
    font-size: 12px;
    color: var(--text-muted);
  }

  .panel .error {
    color: var(--danger);
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  button {
    padding: 5px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button.primary {
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 8px;
    padding: 6px 10px;
    border-top: 1px solid var(--border);
  }

  .status {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .conflict {
    display: flex;
    flex-basis: 100%;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    font-size: 12px;
    background: var(--bg-tertiary);
    border-left: 2px solid var(--warning);
    border-radius: var(--radius-sm);
  }

  .screen {
    display: contents;
  }

  .summary-error {
    margin: 0;
    font-size: 12px;
    color: var(--danger);
  }

  .cover {
    position: absolute;
    inset: 0;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: center;
    background: color-mix(in srgb, var(--bg-primary) 85%, transparent);
  }

  .offer {
    position: absolute;
    right: 12px;
    bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    font-size: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }
</style>

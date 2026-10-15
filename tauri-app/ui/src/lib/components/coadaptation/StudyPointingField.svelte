<script module lang="ts">
  import type { StudyLayout, Surface } from "../../coadaptation/geometry";
  import type { TrialState } from "../../coadaptation/commitRules";
  import type {
    AmbiguityTableV1,
    ExplanationV1,
    FallbackReason,
    FallbacksV1,
    InterruptionReason,
    L1Variant,
    L2FallbackReason,
    Modality,
    OfferResponseV1,
    SpreadV1,
    TaskKind,
    TrialV1,
  } from "../../coadaptation/types";

  export type InputMode = "pointer" | "keyboard" | "hand_pointer";
  export type PinnedRule = "freeze" | "pin" | "pin_act";

  /** One scheduled trial (a pointing or menu trial, or an idle epoch). */
  export interface FieldTrial {
    trialId: string;
    blockIndex: number;
    /** Index within the block; selects the table row (section 9.3). */
    trialIndex: number;
    task: TaskKind;
    layout: StudyLayout;
    /** The cued candidate; null for an idle epoch. */
    intendedIndex: number | null;
    labelWord: string | null;
    limitMs: 10000 | 15000 | 30000;
    /** How long the trial may run; defaults to `limitMs` (an idle epoch runs for its rest duration). */
    runMs?: number;
    /** Practice trials run under the same rules but are never recorded. */
    record: boolean;
  }

  /** The L1 side of the trial: the regime held at cue and the fallbacks it came with. */
  export interface FieldRegime {
    /** The block's recommended regime (TrialV1.regime). */
    regime: L1Variant;
    /** The regime applied at cue: the recommendation, or fallbacks.l1 after an L1 fallback. */
    applied: L1Variant;
    decisionId: string;
    /** The decision's mode (adaptive, randomized, probe, comparator, pinned, baseline). */
    mode: string;
    l1FallbackReason: FallbackReason | null;
    explanation: ExplanationV1 | null;
    /** The rule behind pinned mode: a freeze, a pin, or pin_act ("Always select exactly what I click"). */
    pinnedRule: PinnedRule | null;
    pinnedVariant: string | null;
    /** The fallbacks held at cue (plain regime, cap, fixed L2 rule, issue time). */
    fallbacks: FallbacksV1;
    /** The block's interruption count so far (cap rule, section 9.3). */
    interruptionsUsed: number;
    tau: number;
    /** Show the AssistChip for the regime (not in the practice block). */
    chips: boolean;
  }

  /** The L2 side: the block's ambiguity table, or why a fallback response applies instead. */
  export interface FieldTable {
    table: AmbiguityTableV1 | null;
    /** The table decision requested for the block, named by fallback responses; null when none was requested. */
    decisionId: string | null;
    /** Why the table is not used (late, rpc_error, stale_table); null while it is used. */
    fallbackReason: L2FallbackReason | null;
    /** Fixed L2 rule of the fallbacks (practice block, idle epochs). */
    fixedRule: boolean;
  }

  export interface PosteriorParams {
    /** The session's modality. */
    modality: Modality;
    /** The pair for every posterior of the block; null in a keyboard session. */
    spread: SpreadV1 | null;
    surface: Surface;
    dpr: number;
    /** Gaze-check accuracy when gaze is used for the session, else null (no gaze is read). */
    gazeAccuracy: number | null;
    /** The session's hand pointer is effective: hand-quality summaries are recorded. */
    handPointer: boolean;
  }

  export interface FieldTrialEnd {
    /** The record to send; null for practice or a trial that never reached its cue. */
    trial: TrialV1 | null;
    state: TrialState;
    started: boolean;
    durationMs: number;
  }

  export interface FieldVeto {
    level: "block_regime" | "ambiguity_response";
    decisionId: string;
    variant: string;
    /** Pinned mode: the veto reverts this trial only and no table is refreshed (section 9.7). */
    pinned: boolean;
  }

  export interface FieldActivation {
    modality: Modality;
    result: string;
    l2Variant: string | null;
  }
</script>

<script lang="ts">
  /**
   * One study trial (spec 12.4-12.7): draws the cue-time candidates (with the enlarge presentation), turns input into
   * reducer events and records the trial. Only the person's own input commits, through `reduceTrial`; this component
   * never commits on a timer, on gaze or on silence. Raw pointer samples stay inside the TrialRecorder and are dropped
   * when the trial ends.
   */
  import { onDestroy, onMount, tick, untrack } from "svelte";
  import { get } from "svelte/store";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { layoutSummary, rectContains, hitTest, type Point } from "../../coadaptation/geometry";
  import { ambiguityBin, cellId, computePosterior, inGazeRegion } from "../../coadaptation/targetPosterior";
  import { expandedCandidate, regimePresentation, renderedRects } from "../../coadaptation/presentation";
  import {
    effectiveRegime,
    holdMs as regimeHoldMs,
    initialTrialState,
    isTerminal,
    needsL2Response,
    reduceTrial,
    type L2Proposal,
    type TrialEvent,
  } from "../../coadaptation/commitRules";
  import { drawForActivation, fixedL2Response, isPinActFallback } from "../../coadaptation/decisionTable";
  import { TrialRecorder } from "../../coadaptation/trialRecorder";
  import {
    attachHandPointerInput,
    attachKeyboardInput,
    attachPointerInput,
    initialFocus,
    type ActivationInput,
    type PressInput,
  } from "../../coadaptation/inputAdapter";
  import { studyPointer } from "../../stores/studyPointer";
  import { gazeRuntime } from "../../stores/gazeRuntime";
  import type { ActivationGazeV1, HandSummaryV1, RectV1 } from "../../coadaptation/types";
  import AssistChip, { chipLabel, type Translate } from "./AssistChip.svelte";
  import ConfirmChip, { type ConfirmInput } from "./ConfirmChip.svelte";
  import ClarifyChoice from "./ClarifyChoice.svelte";

  interface Props {
    trial: FieldTrial;
    regime: FieldRegime;
    table: FieldTable;
    posteriorParams: PosteriorParams;
    inputMode: InputMode;
    /** Study input paused by the host (a refresh in flight, a Never step, a pause). */
    inputPaused?: boolean;
    /** Offer responses of this trial, merged into the record at the end. */
    offers?: OfferResponseV1[];
    onactivation?: (activation: FieldActivation) => void;
    ontrialend: (end: FieldTrialEnd) => void;
    onveto?: (veto: FieldVeto) => void;
    onnever?: (variant: string) => void | Promise<void>;
    onneverstep?: (open: boolean) => void;
    onretract?: () => void;
    /** Escape outside a confirm step, a window blur or a hidden document: the host pauses the study. */
    onpause?: (reason: InterruptionReason) => void;
    /** The trial reached its cue: from now on a refresh signal reverts it (section 9.7). */
    oncue?: () => void;
  }

  let {
    trial: trialProp,
    regime: regimeProp,
    table: tableProp,
    posteriorParams: paramsProp,
    inputMode: inputModeProp,
    inputPaused = false,
    offers = [],
    onactivation,
    ontrialend,
    onveto,
    onnever,
    onneverstep,
    onretract,
    onpause,
    oncue,
  }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const CORRECT_FEEDBACK_MS = 300;
  const WRONG_FEEDBACK_MS = 1500;

  // Each trial is a fresh instance (the host keys it by trial id), so its inputs are read once, at creation: the
  // trial keeps them even while the host clears its own state as the trial ends. The one exception is the table: a
  // table requested after a veto replaces the stale-table fallback as soon as it arrives (`adoptTable`).
  const cue0 = untrack(() => ({
    trial: trialProp,
    regime: regimeProp,
    table: tableProp,
    posteriorParams: paramsProp,
    inputMode: inputModeProp,
  }));
  const trial = cue0.trial;
  const regime = cue0.regime;
  let table = $state.raw<FieldTable>(cue0.table);
  const posteriorParams = cue0.posteriorParams;
  const inputMode = cue0.inputMode;
  const cueRects: RectV1[] = trial.layout.rects.map((rect) => ({ ...rect }));
  const labels = trial.layout.labels;
  const surface = posteriorParams.surface;
  const recorder = new TrialRecorder();

  let trialState = $state.raw<TrialState>(
    initialTrialState(regime.applied, {
      plainRegime: regime.fallbacks.plain,
      fallbacksIssuedAt: regime.fallbacks.issued_at,
      interruptionCap: regime.fallbacks.interruption_cap,
      interruptionsUsed: regime.interruptionsUsed,
      tau: regime.tau,
      // Every Heliox study candidate is inert, so reversible (rule C9 never fires in the study, section 9.2).
      reversible: cueRects.map(() => true),
    }),
  );
  let started = $state(false);
  let ended = false;
  let menuOpen = $state(trial.task !== "menu");
  let expandedIndex = $state<number | null>(null);
  let frozen = false;
  let focusIndex = $state<number | null>(null);
  let pointer = $state<Point | null>(null);
  let chipShownAt: number | null = null;
  let confirmPressAt: number | null = null;
  let pressIgnored = false;
  let hintVisible = $state(false);
  let neverStepOpen = $state(false);
  let feedback = $state<"correct" | "wrong" | "rest" | null>(null);
  let explanationViews = 0;
  let refreshFallbacks: FallbacksV1 | null = null;
  /** The regime vetoed in pinned mode; its chip stays to show the retract offer (section 9.7). */
  let pinnedVetoVariant = $state<string | null>(null);

  let surfaceEl = $state<HTMLDivElement | null>(null);
  let startButton = $state<HTMLButtonElement | null>(null);
  let detachInput: (() => void) | null = null;
  let detachStart: (() => void) | null = null;
  let attachedHold: boolean | null = null;
  /** The hold duration the attached adapter completes a hold after (null outside hold regimes). */
  let attachedHoldMs: number | null = null;
  /** The duration of the hold that is open now: the one its adapter started it with. */
  let openHoldMs = $state<number | null>(null);
  let limitTimer: ReturnType<typeof setTimeout> | null = null;
  let endTimer: ReturnType<typeof setTimeout> | null = null;
  let cueT = 0;
  let cueAt = new Date();
  let qualitySum = 0;
  let qualityCount = 0;
  let qualityMin = 1;
  let lossesAtCue = 0;
  let unsubscribeHand: (() => void) | null = null;
  let restTimer: ReturnType<typeof setInterval> | null = null;
  let restSeconds = $state(0);

  const regimeNow = $derived(effectiveRegime(trialState));
  const presentation = $derived(regimePresentation(regimeNow));
  const holdRegime = $derived(presentation.confirm?.method === "hold");
  const regimeHold = $derived(regimeHoldMs(trialState));
  const holdOpen = $derived(trialState.phase === "holding");
  const rendered = $derived(
    presentation.scale > 1
      ? renderedRects(cueRects, expandedIndex, presentation.scale)
      : cueRects.map((r) => ({ ...r })),
  );
  // New presses and activations wait while input is paused; an open confirm step, hold or clarification can still be
  // finished during a refresh (section 12.5), but not while the Never confirm step is open.
  const blocked = $derived(inputPaused || neverStepOpen || trialState.inputPaused);
  const pending = $derived(trialState.pendingIndex);
  const lastL2 = $derived(trialState.lastL2);
  const pinActActive = $derived(isPinActFallback(regime.fallbacks));
  // The regime (L1) chip is in pinned mode only in a frozen or pinned block. pin_act collapses L2 responses alone
  // (section 9.7), so in an adaptive or randomized block its regime chip vetoes normally and never offers to remove
  // the unrelated pin_act rule; the L2 chip below handles a pin_act collapse.
  const chipMode = $derived(regime.mode === "pinned" ? "pinned" : regime.mode);
  const regimeChipVariant = $derived(pinnedVetoVariant ?? regimeNow);
  const showRegimeChip = $derived(
    regime.chips &&
      started &&
      (pinnedVetoVariant !== null || (regimePresentation(regimeNow).chip && trialState.phase !== "committed")),
  );
  const l2StepOpen = $derived(
    lastL2 !== null &&
      lastL2.variant !== "wait_for_evidence" &&
      (trialState.phase === "confirming" || trialState.phase === "clarifying") &&
      trialState.pendingActivation === trialState.activations.length - 1,
  );
  const holdStartedAt = $derived(trialState.holdStartedAt === null ? null : cueT + trialState.holdStartedAt);
  const clarifyCandidates = $derived(
    trialState.askCandidates.map((index) => ({
      index,
      label: labels[index] ?? `${index + 1}`,
      cx: rendered[index]?.cx,
      cy: rendered[index]?.cy,
      w: rendered[index]?.w,
      h: rendered[index]?.h,
    })),
  );
  const confirmAnchor = $derived(pending !== null ? (rendered[pending] ?? cueRects[pending] ?? null) : null);

  onMount(() => {
    window.addEventListener("keydown", onWindowKey);
    if (trial.task === "pointing") {
      // Pointing trials start from the home button; idle epochs and menus show their cue at once.
      if (inputMode === "hand_pointer") detachStart = attachButtonByHand(trial.layout.origin, cue);
      else void tick().then(() => startButton?.focus({ preventScroll: true }));
    } else {
      cue();
    }
  });

  onDestroy(() => {
    window.removeEventListener("keydown", onWindowKey);
    teardown();
    if (!ended) {
      ended = true;
      recorder.discard();
    }
  });

  // A revert can switch between a hold regime and a click regime, or between hold_400 and hold_700: re-attach the
  // adapter with the new regime's options, so every later press follows the revert regime (section 12.5). A hold that
  // is already open is first completed, released or aborted through the adapter that started it (a step already open
  // finishes under the rules it began with); dropping that adapter mid-hold would strand the trial in "holding".
  $effect(() => {
    const hold = holdRegime;
    const ms = regimeHold;
    if (!started || ended || !menuOpen || holdOpen) return;
    if (attachedHold === hold && attachedHoldMs === ms) return;
    untrack(() => {
      // A click press still down belongs to the old adapter: it ends without an activation.
      if (trialState.phase === "pressed") onPressCancel();
      attachInput();
    });
  });

  // ---- Lifecycle -----------------------------------------------------------------------------------------------

  function cue() {
    if (started || ended) return;
    detachStart?.();
    detachStart = null;
    started = true;
    cueT = performance.now();
    cueAt = new Date();
    const intended = trial.intendedIndex === null ? null : (cueRects[trial.intendedIndex] ?? null);
    recorder.start(cueT, intended);
    apply({ type: "cue", tMs: 0 });
    limitTimer = setTimeout(onLimit, trial.runMs ?? trial.limitMs);
    oncue?.();
    if (posteriorParams.handPointer) startHandSummary();
    if (trial.task === "idle") {
      feedback = null;
      // A countdown for the rest epoch; it only shows time left and never ends or commits anything early.
      const runMs = trial.runMs ?? trial.limitMs;
      restSeconds = Math.ceil(runMs / 1000);
      restTimer = setInterval(() => {
        restSeconds = Math.max(0, Math.ceil((runMs - (performance.now() - cueT)) / 1000));
      }, 1000);
    }
    if (menuOpen) {
      attachInput();
      if (inputMode === "keyboard") focusIndex = initialFocus(cueRects, originPoint(), trial.task);
      void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
    } else if (inputMode === "hand_pointer") {
      detachStart = attachButtonByHand(trial.layout.origin, openMenu);
    } else {
      void tick().then(() => startButton?.focus({ preventScroll: true }));
    }
  }

  function openMenu() {
    if (menuOpen || ended) return;
    detachStart?.();
    detachStart = null;
    menuOpen = true;
    if (inputMode === "keyboard") focusIndex = 0;
    attachInput();
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  function teardown() {
    detachInput?.();
    detachInput = null;
    detachStart?.();
    detachStart = null;
    attachedHold = null;
    attachedHoldMs = null;
    if (limitTimer !== null) clearTimeout(limitTimer);
    limitTimer = null;
    if (endTimer !== null) clearTimeout(endTimer);
    endTimer = null;
    unsubscribeHand?.();
    unsubscribeHand = null;
    if (restTimer !== null) clearInterval(restTimer);
    restTimer = null;
  }

  function onLimit() {
    limitTimer = null;
    if (ended) return;
    if (trialState.phase !== "committed") apply({ type: "limit" });
    finish();
  }

  function finish() {
    if (ended) return;
    ended = true;
    teardown();
    const durationMs = started ? Math.max(0, performance.now() - cueT) : 0;
    let record: TrialV1 | null = null;
    if (started && trial.record) {
      const state = trialState;
      const timeMs =
        state.phase === "committed" && state.commitTMs !== null ? state.commitTMs : Math.min(durationMs, trial.limitMs);
      record = recorder.buildTrial({
        trialId: trial.trialId,
        blockIndex: trial.blockIndex,
        trialIndex: trial.trialIndex,
        task: trial.task,
        regime: regime.regime,
        regimeDecisionId: regime.decisionId,
        tableDecisionIds: tableDecisionIds(),
        appliedRegime: regime.applied,
        l1FallbackReason: regime.l1FallbackReason,
        spread: posteriorParams.spread,
        layout: layoutSummary(trial.layout),
        candidates: cueRects,
        surface: { w: surface.w, h: surface.h },
        dpr: posteriorParams.dpr,
        reversible: cueRects.map(() => true),
        cueAt,
        intendedIndex: trial.intendedIndex,
        modality: posteriorParams.modality,
        state,
        timeMs,
        limitMs: trial.limitMs,
        durationMs,
        offers,
        hand: handSummary(),
        gaze: gazeSummary(),
        explanationViews,
      });
    } else {
      recorder.discard();
    }
    ontrialend({ trial: record, state: trialState, started, durationMs });
  }

  // ---- Host controls (bind:this) --------------------------------------------------------------------------------

  /** Feeds a reducer event from the host (refreshPending, a veto from elsewhere). */
  export function dispatch(event: TrialEvent) {
    apply(event);
  }

  /** Applies fresh fallbacks after a refresh signal: the trial reverts to their `refresh` regime (section 9.7). */
  export function refresh(fresh: FallbacksV1) {
    refreshFallbacks = fresh;
    apply({
      type: "refresh",
      regime: fresh.refresh,
      interruptionCap: fresh.interruption_cap,
      fallbacksIssuedAt: fresh.issued_at,
    });
  }

  /** Ends the trial as interrupted (pause, modal, daemon loss); a trial that never reached its cue is not recorded. */
  export function interrupt(reason: InterruptionReason) {
    if (ended) return;
    // A selection already made (its feedback still showing) is complete; anything else stops as interrupted.
    if (started && !(trialState.phase === "committed" && trial.task !== "idle")) apply({ type: "interrupt", reason });
    finish();
  }

  /** Ends without a record or a callback (the host dropped the trial before its cue). */
  export function discard() {
    if (ended) return;
    ended = true;
    teardown();
    recorder.discard();
  }

  /** True from the cue until the trial ends: a refresh signal then reverts this trial. */
  export function inProgress(): boolean {
    return started && !ended;
  }

  /**
   * The table requested after a veto arrived (or is known to be late or failed) while this trial runs: later
   * ambiguous activations use it instead of the stale-table fallback, which applies only while that table has not
   * arrived (section 9.7). The table starts at the trial after the veto, so a trial before its first row ignores it.
   */
  export function adoptTable(next: FieldTable) {
    if (ended || table.fixedRule || next.fixedRule) return;
    if (next.table && trial.trialIndex < next.table.start_trial_index) return;
    table = next;
  }

  // ---- Reducer -------------------------------------------------------------------------------------------------

  function apply(event: TrialEvent) {
    if (ended) return;
    const before = trialState;
    const next = reduceTrial(before, event);
    if (next === before) return;
    trialState = next;
    afterChange(before, next);
  }

  function afterChange(before: TrialState, next: TrialState) {
    if (next.phase === "confirming" && before.phase !== "confirming") {
      chipShownAt = performance.now();
      confirmPressAt = null;
    } else if (next.phase !== "confirming") {
      chipShownAt = null;
    }
    if (next.phase === "holding" && before.phase !== "holding") openHoldMs = attachedHoldMs;
    else if (next.phase !== "holding") openHoldMs = null;
    if (next.activations.length > before.activations.length) {
      const last = next.activations[next.activations.length - 1];
      hintVisible = last.l2?.variant === "wait_for_evidence";
      onactivation?.({ modality: last.modality, result: last.result, l2Variant: last.l2?.variant ?? null });
    }
    if (next.phase !== "cued" && next.phase !== "pressed") frozen = next.phase === "holding" ? frozen : false;
    if (next.phase === "committed" && before.phase !== "committed") onCommitted(next);
    else if (next.phase === "abandoned" || next.phase === "interrupted") finish();
  }

  function onCommitted(state: TrialState) {
    if (trial.task === "idle") {
      // Any commit during a rest epoch is an unintended activation; the epoch runs on to its end.
      feedback = "rest";
      detachInput?.();
      detachInput = null;
      return;
    }
    if (state.committedIndex === trial.intendedIndex) {
      feedback = "correct";
      endTimer = setTimeout(finish, CORRECT_FEEDBACK_MS);
    } else {
      feedback = "wrong";
      endTimer = setTimeout(finish, WRONG_FEEDBACK_MS);
    }
  }

  function undo() {
    if (feedback !== "wrong" || ended) return;
    if (endTimer !== null) clearTimeout(endTimer);
    endTimer = null;
    feedback = null;
    apply({ type: "undo" });
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  // ---- Input ---------------------------------------------------------------------------------------------------

  function originPoint(): Point {
    return { x: trial.layout.origin.cx, y: trial.layout.origin.cy };
  }

  function tMs(t: number): number {
    return Math.max(0, Math.round(t - cueT));
  }

  function candidateAt(point: Point): number | null {
    if (!menuOpen) return null;
    return hitTest(rendered, point, presentation.scale > 1 ? expandedIndex : null);
  }

  function attachInput() {
    detachInput?.();
    detachInput = null;
    if (!surfaceEl || ended) return;
    const hold = holdRegime;
    const holdMs = regimeHoldMs(trialState);
    attachedHold = hold;
    attachedHoldMs = holdMs;
    const handlers = {
      candidateAt,
      onMove,
      onPress,
      onActivation,
      onPressCancel,
      onHoldRelease: (elapsedMs: number) => {
        frozen = false;
        apply({ type: "holdRelease", elapsedMs });
      },
      onHoldAbort: (reason: string) => {
        frozen = false;
        apply({ type: "holdAbort", reason });
      },
      onHoldComplete: () => apply({ type: "holdComplete", tMs: tMs(performance.now()) }),
      onInterrupt: (reason: "blur" | "hidden") => onpause?.(reason),
    };
    const detachers: Array<() => void> = [];
    if (inputMode === "keyboard" || trial.task === "menu") {
      // Keyboard sessions use only the keyboard; pointer sessions also accept keys in an open menu (12.6).
      detachers.push(
        attachKeyboardInput(
          surfaceEl,
          {
            rects: () => cueRects,
            focused: () => focusIndex,
            setFocus: (index: number) => (focusIndex = index),
            labels: () => labels,
            onPress,
            onActivation,
            onHoldRelease: handlers.onHoldRelease,
            onHoldAbort: handlers.onHoldAbort,
            onHoldComplete: handlers.onHoldComplete,
            onInterrupt: handlers.onInterrupt,
          },
          { holdRegime: hold, holdMs, task: trial.task, origin: originPoint() },
        ),
      );
    }
    if (inputMode === "pointer") {
      detachers.push(
        attachPointerInput(surfaceEl, handlers, { holdRegime: hold, holdMs, modality: posteriorParams.modality }),
      );
    } else if (inputMode === "hand_pointer") {
      detachers.push(
        attachHandPointerInput({ width: surface.w, height: surface.h }, studyPointer, handlers, {
          holdRegime: hold,
          holdMs,
        }),
      );
    }
    detachInput = () => {
      for (const detach of detachers) detach();
    };
  }

  /** A hand-pointer pinch released on a button rect (home or menu button) before the trial's own input attaches. */
  function attachButtonByHand(button: RectV1, action: () => void): () => void {
    return attachHandPointerInput(
      { width: surface.w, height: surface.h },
      studyPointer,
      {
        candidateAt: (point: Point) => (rectContains(button, point) ? 0 : null),
        onMove: (point: Point) => (pointer = point),
        onActivation: (activation: ActivationInput) => {
          if (activation.hitIndex === 0 && !blocked) action();
        },
        onInterrupt: (reason: "blur" | "hidden") => onpause?.(reason),
      },
      { holdRegime: false },
    );
  }

  function onMove(point: Point, time: number) {
    pointer = point;
    if (!started || ended) return;
    if (inputMode !== "keyboard") recorder.sample(point.x, point.y, time);
    if (presentation.scale > 1 && !frozen && posteriorParams.spread) {
      expandedIndex = expandedCandidate(
        point,
        cueRects,
        expandedIndex,
        presentation.scale,
        posteriorParams.spread.r,
        posteriorParams.spread.floor,
        surface,
        false,
      );
    }
  }

  function onPress(press: PressInput) {
    if (!started || ended) return;
    if (trialState.phase === "confirming") {
      pressIgnored = false;
      confirmPressAt = neverStepOpen ? null : press.t;
      return;
    }
    if (blocked) {
      // The whole press is ignored, its release included.
      pressIgnored = true;
      return;
    }
    pressIgnored = false;
    if (trialState.phase !== "cued") return;
    frozen = true;
    apply({
      type: "press",
      x: press.x,
      y: press.y,
      hitIndex: press.index,
      expandedIndex: presentation.scale > 1 ? expandedIndex : null,
      tMs: tMs(press.t),
    });
  }

  function onPressCancel() {
    frozen = false;
    pressIgnored = false;
    if (trialState.phase === "pressed") apply({ type: "pressCancel" });
  }

  function onActivation(input: ActivationInput) {
    if (!started || ended) return;
    if (pressIgnored && input.kind !== "key") {
      pressIgnored = false;
      return;
    }
    if (trialState.phase === "confirming") {
      if (!neverStepOpen) confirmStep(input);
      return;
    }
    if (blocked) return;
    if (trialState.phase !== "cued" && trialState.phase !== "pressed") return;
    if (input.modality !== "keyboard") recorder.activation(input.x, input.y, input.t);
    activate(input);
    if (input.kind !== "press") frozen = false;
  }

  function confirmStep(input: ActivationInput) {
    const shownAt = chipShownAt ?? performance.now();
    if (input.kind === "key") {
      apply({
        type: "confirmInput",
        kind: "key",
        onIndex: input.hitIndex,
        sinceChipMs: Math.round(input.t - shownAt),
        fresh: true,
        tMs: tMs(input.t),
      });
      return;
    }
    const pressAt = confirmPressAt;
    confirmPressAt = null;
    if (input.hitIndex !== null && input.hitIndex === trialState.pendingIndex) {
      apply({
        type: "confirmInput",
        kind: inputMode === "hand_pointer" ? "pinch" : "click",
        onIndex: input.hitIndex,
        sinceChipMs: pressAt === null ? 0 : Math.round(pressAt - shownAt),
        fresh: pressAt !== null,
        tMs: tMs(input.t),
      });
    } else if (pressAt !== null) {
      // A click elsewhere cancels the confirm step (a correction).
      apply({ type: "cancel" });
    }
  }

  function readGaze(): (ActivationGazeV1 & { acc: number }) | null {
    const acc = posteriorParams.gazeAccuracy;
    if (acc === null) return null;
    const gaze = get(gazeRuntime);
    if (gaze.phase !== "active" || gaze.region === null || gaze.confidence === null) return null;
    return { region: gaze.region, conf: gaze.confidence, acc };
  }

  function activate(input: ActivationInput) {
    const keyboard = input.modality === "keyboard";
    const enlarge = presentation.scale > 1;
    const rects = keyboard ? cueRects : rendered;
    const gaze = keyboard ? null : readGaze();
    const spread = posteriorParams.spread;
    if (!keyboard && !spread) return;
    const result = computePosterior(
      rects,
      { x: input.x, y: input.y },
      {
        modality: input.modality,
        r: keyboard ? null : (spread?.r ?? null),
        floor: keyboard ? null : (spread?.floor ?? null),
        surface,
        gaze,
        pressIndex: input.pressIndex,
        topIndex: enlarge && !keyboard ? expandedIndex : null,
      },
    );
    const hit = result.hitIndex;
    apply({
      type: "activation",
      x: input.x,
      y: input.y,
      modality: input.modality,
      hitIndex: hit,
      pressIndex: input.pressIndex,
      aGate: result.ambiguityGate,
      nPlausible: result.nPlausible,
      plausible: result.topK(3),
      expandedIndex: enlarge && !keyboard ? expandedIndex : null,
      hitRect: hit === null ? null : { ...rects[hit] },
      gaze: gaze ? { region: gaze.region, conf: gaze.conf } : null,
      reversible: true,
      l2: l2Proposal(result.ambiguityGate, result.nPlausible, hit),
      tMs: tMs(input.t),
    });
  }

  /** The L2 response for an ambiguous activation before the reducer's cap rule (sections 9.3, 9.7). */
  function l2Proposal(aGate: number, nPlausible: number, hit: number | null): L2Proposal | undefined {
    const state = trialState;
    if (!needsL2Response(state, { aGate, hitIndex: hit })) return undefined;
    const applicability = { hasHit: hit !== null, nPlausible };
    const collapse = (fallbacks: FallbacksV1, variant: string) =>
      isPinActFallback(fallbacks) && (variant === "act" || (variant === "wait_for_evidence" && hit === null));
    if (state.revert) {
      // A reverted trial uses the fixed rule of the fallbacks whose regime it applies.
      const fallbacks = state.revert.reason === "refresh" ? (refreshFallbacks ?? regime.fallbacks) : regime.fallbacks;
      const variant = fixedL2Response(fallbacks, applicability);
      return { variant, source: "fixed_rule", decisionId: regime.decisionId, pinAct: collapse(fallbacks, variant) };
    }
    if (table.fixedRule) {
      const variant = fixedL2Response(regime.fallbacks, applicability);
      return {
        variant,
        source: "fixed_rule",
        decisionId: regime.decisionId,
        pinAct: collapse(regime.fallbacks, variant),
      };
    }
    if (table.table && table.fallbackReason === null) {
      const bin = ambiguityBin(hit === null ? 1 : aGate, state.tau) ?? "b3";
      const draw = drawForActivation(
        table.table,
        trial.trialIndex,
        state.l2Used,
        cellId(bin, nPlausible, hit !== null),
      );
      if (draw) {
        return {
          variant: draw.variant,
          source: "table",
          decisionId: table.table.decision_id,
          drawIndex: draw.drawIndex,
          pinAct: collapse(regime.fallbacks, draw.variant),
        };
      }
      const variant = fixedL2Response(regime.fallbacks, applicability);
      return {
        variant,
        source: "fallback",
        decisionId: table.table.decision_id,
        fallbackReason: "stale_table",
        pinAct: collapse(regime.fallbacks, variant),
      };
    }
    const variant = fixedL2Response(regime.fallbacks, applicability);
    return {
      variant,
      source: "fallback",
      decisionId: table.decisionId ?? regime.decisionId,
      fallbackReason: table.fallbackReason ?? "late",
      pinAct: collapse(regime.fallbacks, variant),
    };
  }

  function onWindowKey(event: KeyboardEvent) {
    if (ended || event.repeat) return;
    if (event.key === "Escape") {
      if (trialState.phase === "confirming") {
        event.preventDefault();
        apply({ type: "cancel" });
      } else {
        onpause?.("paused");
      }
    } else if (event.key === "Backspace" && feedback === "wrong") {
      event.preventDefault();
      undo();
    }
  }

  // ---- Chips ---------------------------------------------------------------------------------------------------

  function onConfirmChip(input: ConfirmInput) {
    if (neverStepOpen || trialState.pendingIndex === null) return;
    apply({
      type: "confirmInput",
      kind: input.kind,
      onIndex: trialState.pendingIndex,
      sinceChipMs: input.sinceChipMs,
      fresh: true,
      tMs: tMs(performance.now()),
    });
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  function onConfirmCancel() {
    apply({ type: "cancel" });
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  function onChoose(index: number | null) {
    apply({ type: "choose", index, tMs: tMs(performance.now()) });
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  function veto(level: "block_regime" | "ambiguity_response", decisionId: string, variant: string, pinned: boolean) {
    if (pinned && level === "block_regime") pinnedVetoVariant = variant;
    apply({ type: "veto", decisionId, level });
    onveto?.({ level, decisionId, variant, pinned });
    void tick().then(() => surfaceEl?.focus({ preventScroll: true }));
  }

  function neverStep(open: boolean) {
    neverStepOpen = open;
    onneverstep?.(open);
  }

  // ---- Summaries -----------------------------------------------------------------------------------------------

  function startHandSummary() {
    lossesAtCue = get(studyPointer).trackingLosses;
    unsubscribeHand = studyPointer.subscribe((state) => {
      if (!state.handPresent) return;
      qualitySum += state.quality;
      qualityCount += 1;
      qualityMin = Math.min(qualityMin, state.quality);
    });
  }

  function handSummary(): HandSummaryV1 | null {
    if (!posteriorParams.handPointer) return null;
    return {
      quality_mean: qualityCount ? qualitySum / qualityCount : 0,
      quality_min: qualityCount ? qualityMin : 0,
      tracking_losses: Math.max(0, Math.min(100, get(studyPointer).trackingLosses - lossesAtCue)),
    };
  }

  function gazeSummary() {
    if (posteriorParams.gazeAccuracy === null || trial.intendedIndex === null) return null;
    const first = trialState.activations[0];
    const intended = cueRects[trial.intendedIndex];
    if (!first?.gaze || !intended) return { agreed_with_intended: null };
    return { agreed_with_intended: inGazeRegion(intended, first.gaze.region, surface) };
  }

  function tableDecisionIds(): string[] {
    const ids: string[] = [];
    if (table.table) ids.push(table.table.decision_id);
    for (const activation of trialState.activations) {
      const id = activation.l2?.decisionId;
      if (id && id !== regime.decisionId && !ids.includes(id)) ids.push(id);
    }
    return ids.slice(0, 4);
  }

  function rectStyle(rect: RectV1): string {
    return `left: ${rect.cx - rect.w / 2}px; top: ${rect.cy - rect.h / 2}px; width: ${rect.w}px; height: ${rect.h}px;`;
  }

  function chipStyle(anchor: RectV1): string {
    const top = Math.min(surface.h - 48, anchor.cy + anchor.h / 2 + 8);
    const left = Math.max(8, Math.min(surface.w - 240, anchor.cx - 110));
    return `left: ${left}px; top: ${top}px;`;
  }
</script>

<div class="field-root" style={`width: ${surface.w}px; height: ${surface.h}px;`}>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div
    class="surface"
    class:hand={inputMode === "hand_pointer"}
    bind:this={surfaceEl}
    tabindex="0"
    role="application"
    aria-label={trial.task === "idle" ? t("study.rest.instruction") : t("study.title")}
  >
    {#if menuOpen}
      {#each rendered as rect, index (index)}
        <div
          class="candidate"
          class:menu-item={trial.task === "menu"}
          class:cued={started && trial.task !== "idle" && index === trial.intendedIndex && trial.task === "pointing"}
          class:expanded={presentation.scale > 1 && index === expandedIndex}
          class:pending={index === pending && (trialState.phase === "confirming" || trialState.phase === "holding")}
          class:focused={index === focusIndex}
          class:committed={trialState.phase === "committed" && index === trialState.committedIndex}
          style={rectStyle(rect)}
          aria-hidden="true"
        >
          {#if trial.task === "menu"}{labels[index]}{/if}
        </div>
      {/each}
    {/if}
    {#if inputMode === "hand_pointer" && pointer}
      <span class="hand-dot" style={`left: ${pointer.x}px; top: ${pointer.y}px;`} aria-hidden="true"></span>
    {/if}
  </div>

  <div class="overlay">
    <div class="top-bar">
      {#if trial.task === "idle"}
        <p class="cue">{t("study.rest.instruction")}</p>
        {#if started}
          <p class="status">{t("study.rest.remaining", { values: { seconds: restSeconds } })}</p>
        {/if}
      {:else if !started && trial.task === "pointing"}
        <p class="status">{t("study.trial.home_hint")}</p>
      {:else if started && trial.labelWord}
        <p class="cue">
          {t("study.trial.cue", { values: { word: trial.labelWord } })}
        </p>
      {/if}
      {#if hintVisible && trialState.phase === "cued"}
        <AssistChip variant="wait_for_evidence" mode={chipMode} ondismiss={() => (hintVisible = false)} />
      {/if}
      {#if l2StepOpen && lastL2 && regime.chips}
        <AssistChip
          variant={lastL2.variant}
          explanation={table.table?.explanation ?? null}
          mode={table.table?.mode === "pinned" || pinActActive ? "pinned" : (table.table?.mode ?? regime.mode)}
          pinnedRule={regime.pinnedRule}
          pinnedVariant={regime.pinnedVariant}
          onwhy={() => (explanationViews += 1)}
          onnotthistime={() =>
            veto(
              "ambiguity_response",
              lastL2?.decisionId ?? "",
              lastL2?.variant ?? "",
              table.table?.mode === "pinned" || pinActActive,
            )}
          onnever={() => onnever?.(lastL2?.variant ?? "")}
          onneverstep={neverStep}
          onretract={() => onretract?.()}
        />
      {:else if showRegimeChip}
        <AssistChip
          variant={regimeChipVariant}
          explanation={regime.explanation}
          mode={chipMode}
          pinnedRule={regime.pinnedRule}
          pinnedVariant={regime.pinnedVariant}
          onwhy={() => (explanationViews += 1)}
          onnotthistime={() => veto("block_regime", regime.decisionId, regimeChipVariant, chipMode === "pinned")}
          onnever={() => onnever?.(regimeChipVariant)}
          onneverstep={neverStep}
          onretract={() => onretract?.()}
          onkeep={() => (pinnedVetoVariant = null)}
        />
      {/if}
      {#if blocked && trialState.inputPaused}
        <p class="status" role="status">{t("study.refresh.updating")}</p>
      {:else if trialState.endBlockAfterTrial && !isTerminal(trialState)}
        <p class="status" role="status">
          {t("study.refresh.ends_after")}
        </p>
      {/if}
    </div>

    {#if trial.task === "pointing" && !started}
      <button
        class="home"
        bind:this={startButton}
        style={rectStyle(trial.layout.origin)}
        onclick={cue}
        disabled={blocked}
      >
        {t("study.trial.home")}
      </button>
    {:else if trial.task === "menu" && !menuOpen}
      <button
        class="menu-button"
        bind:this={startButton}
        style={rectStyle(trial.layout.origin)}
        onclick={openMenu}
        disabled={blocked || !started}
      >
        {t("study.trial.menu_button")}
      </button>
    {/if}

    {#if trialState.phase === "confirming" && confirmAnchor}
      <div class="anchored" style={chipStyle(confirmAnchor)}>
        <ConfirmChip
          label={labels[pending ?? 0] ?? ""}
          method="second_click"
          onconfirm={onConfirmChip}
          oncancel={onConfirmCancel}
        />
      </div>
    {:else if trialState.phase === "holding" && confirmAnchor}
      <div class="anchored" style={chipStyle(confirmAnchor)}>
        <ConfirmChip
          label={labels[pending ?? 0] ?? ""}
          method="hold"
          holdMs={openHoldMs ?? regimeHoldMs(trialState) ?? 0}
          {holdStartedAt}
          onconfirm={onConfirmChip}
          oncancel={onConfirmCancel}
        />
      </div>
    {:else if trialState.phase === "clarifying"}
      <ClarifyChoice
        candidates={trial.task === "menu"
          ? clarifyCandidates.map(({ index, label }) => ({ index, label }))
          : clarifyCandidates}
        onchoose={onChoose}
      />
    {/if}

    {#if feedback === "wrong"}
      <div class="feedback wrong" role="status">
        <span>
          {t("study.trial.wrong", { values: { word: labels[trialState.committedIndex ?? 0] ?? "" } })}
        </span>
        <button onclick={undo}>{t("study.trial.undo")}</button>
      </div>
    {:else if feedback === "correct"}
      <div class="feedback correct" role="status">{t("study.trial.correct")}</div>
    {:else if feedback === "rest"}
      <div class="feedback" role="status">
        {t("study.trial.selected", { values: { word: labels[trialState.committedIndex ?? 0] ?? "" } })}
      </div>
    {/if}
  </div>
</div>

<style>
  .field-root {
    position: relative;
    flex-shrink: 0;
    user-select: none;
  }

  .surface {
    position: absolute;
    inset: 0;
    overflow: hidden;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    outline: none;
    touch-action: none;
  }

  .surface:focus-visible {
    border-color: var(--accent);
  }

  .surface.hand {
    cursor: none;
  }

  .candidate {
    position: absolute;
    box-sizing: border-box;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 3px;
  }

  .candidate.menu-item {
    display: flex;
    align-items: center;
    padding: 0 10px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border-radius: 0;
  }

  .candidate.cued {
    background: var(--accent);
    border-color: var(--accent-hover);
  }

  .candidate.expanded {
    z-index: 2;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
  }

  .candidate.pending {
    z-index: 3;
    outline: 3px solid var(--warning);
  }

  .candidate.focused {
    z-index: 3;
    outline: 2px solid var(--accent-hover);
    outline-offset: 2px;
  }

  .candidate.committed {
    outline: 3px solid var(--success);
  }

  .hand-dot {
    position: absolute;
    z-index: 5;
    width: 12px;
    height: 12px;
    background: var(--warning);
    border: 2px solid white;
    border-radius: 50%;
    pointer-events: none;
    transform: translate(-50%, -50%);
  }

  .overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .overlay :global(button),
  .overlay :global(.assist-chip),
  .overlay :global(.confirm-chip),
  .overlay :global(.clarify) {
    pointer-events: auto;
  }

  .top-bar {
    position: absolute;
    top: 6px;
    left: 8px;
    right: 8px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 10px;
  }

  .cue {
    margin: 0;
    padding: 3px 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
  }

  .status {
    margin: 0;
    padding: 3px 8px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
  }

  .home,
  .menu-button {
    position: absolute;
    box-sizing: border-box;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .home:disabled,
  .menu-button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .anchored {
    position: absolute;
    z-index: 6;
  }

  .feedback {
    position: absolute;
    bottom: 8px;
    left: 50%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    transform: translateX(-50%);
  }

  .feedback.correct {
    border-color: var(--success);
  }

  .feedback.wrong {
    border-color: var(--warning);
  }

  .feedback button {
    padding: 2px 10px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }
</style>

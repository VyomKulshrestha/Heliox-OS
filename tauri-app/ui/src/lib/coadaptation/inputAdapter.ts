import type { Point } from "./geometry";
import { keyboardEndpoint } from "./targetPosterior";
import type { Modality, RectV1, TaskKind } from "./types";

/**
 * Input adapters of the Selection Study (spec section 12.6). They turn DOM pointer events, keys and the in-window
 * hand pointer into presses and activations in integer surface pixels. Listeners exist only between an attach call
 * and the detach function it returns; nothing runs at module load.
 *
 * Commit decisions are not made here: the study feeds these events to the commit reducer, which accepts a selection
 * only from the person's own explicit input.
 */

/** A hand-pointer frame older than this no longer counts as fresh evidence. */
export const FRESH_FRAME_MS = 100;

export interface PressInput {
  x: number;
  y: number;
  /** Candidate under the press point, or null. */
  index: number | null;
  t: number;
  modality: Modality;
}

export interface ActivationInput {
  x: number;
  y: number;
  /** Candidate under the press point, -1 when the press hit nothing (keyboard: the focused candidate). */
  pressIndex: number;
  /** Literal hit: the candidate under the release point only when it is the press candidate. */
  hitIndex: number | null;
  t: number;
  modality: Modality;
  /** "release" for press-and-release, "press" in hold regimes (the press is the activation), "key" for keyboard. */
  kind: "release" | "press" | "key";
}

export interface InputHandlers {
  /** Hit test on the rects drawn now (frozen at the press under enlarge_target). */
  candidateAt(point: Point): number | null;
  onMove?(point: Point, t: number): void;
  onPress?(press: PressInput): void;
  onActivation(activation: ActivationInput): void;
  /** A press ended without an activation (lost input in a non-hold regime). */
  onPressCancel?(reason: string): void;
  onHoldRelease?(elapsedMs: number): void;
  onHoldAbort?(reason: string): void;
  onHoldComplete?(): void;
  /** Window blur or hidden document: the study interrupts the trial. */
  onInterrupt?(reason: "blur" | "hidden"): void;
}

export interface InputOptions {
  holdRegime: boolean;
  /** When given in a hold regime, the adapter reports `onHoldComplete` after this long of continuous holding. */
  holdMs?: number | null;
  /** Session modality for pointer events (mouse, touchpad, pen, touch); defaults to the event's pointer type. */
  modality?: Modality;
  /** Clock (performance.now by default). */
  now?: () => number;
}

function defaultNow(): number {
  return performance.now();
}

function attachWindowInterrupts(handlers: { onInterrupt?(reason: "blur" | "hidden"): void }, onInterrupt: () => void) {
  const onBlur = () => {
    onInterrupt();
    handlers.onInterrupt?.("blur");
  };
  const onVisibility = () => {
    if (document.visibilityState === "hidden") {
      onInterrupt();
      handlers.onInterrupt?.("hidden");
    }
  };
  window.addEventListener("blur", onBlur);
  document.addEventListener("visibilitychange", onVisibility);
  return () => {
    window.removeEventListener("blur", onBlur);
    document.removeEventListener("visibilitychange", onVisibility);
  };
}

function pointerModality(event: PointerEvent, fallback: Modality | undefined): Modality {
  if (fallback && fallback !== "keyboard" && fallback !== "hand_pointer") return fallback;
  if (event.pointerType === "pen") return "pen";
  if (event.pointerType === "touch") return "touch";
  return "mouse";
}

/**
 * Mouse, touchpad, pen and touch on the study surface. pointerdown takes pointer capture and records the press; in a
 * non-hold regime the activation is the pointerup (its literal hit only when release and press are on the same
 * candidate), and pointercancel or lostpointercapture cancel the press without an activation. In a hold regime the
 * press is the activation; pointerup before completion is a hold release and lost input aborts the hold.
 */
export function attachPointerInput(surfaceEl: HTMLElement, handlers: InputHandlers, options: InputOptions): () => void {
  const now = options.now ?? defaultNow;
  let press: (PressInput & { pointerId: number; completed: boolean }) | null = null;
  let holdTimer: ReturnType<typeof setTimeout> | null = null;

  const local = (event: PointerEvent): Point => {
    const rect = surfaceEl.getBoundingClientRect();
    return { x: Math.round(event.clientX - rect.left), y: Math.round(event.clientY - rect.top) };
  };
  const clearHold = () => {
    if (holdTimer !== null) clearTimeout(holdTimer);
    holdTimer = null;
  };
  const endPress = () => {
    clearHold();
    press = null;
  };

  const onPointerDown = (event: PointerEvent) => {
    if (!event.isPrimary || event.button !== 0 || press) return;
    try {
      surfaceEl.setPointerCapture(event.pointerId);
    } catch {
      /* capture can fail for synthetic events; the press still counts */
    }
    const point = local(event);
    const t = now();
    const modality = pointerModality(event, options.modality);
    const index = handlers.candidateAt(point);
    press = { ...point, index, t, modality, pointerId: event.pointerId, completed: false };
    handlers.onPress?.({ ...point, index, t, modality });
    if (options.holdRegime) {
      handlers.onActivation({ ...point, pressIndex: index ?? -1, hitIndex: index, t, modality, kind: "press" });
      const holdMs = options.holdMs;
      if (holdMs !== null && holdMs !== undefined && holdMs > 0) {
        holdTimer = setTimeout(() => {
          holdTimer = null;
          if (press && !press.completed) {
            press.completed = true;
            handlers.onHoldComplete?.();
          }
        }, holdMs);
      }
    }
  };

  const onPointerMove = (event: PointerEvent) => {
    if (!event.isPrimary) return;
    handlers.onMove?.(local(event), now());
  };

  const onPointerUp = (event: PointerEvent) => {
    if (!press || event.pointerId !== press.pointerId) return;
    const current = press;
    endPress();
    try {
      surfaceEl.releasePointerCapture(event.pointerId);
    } catch {
      /* already released */
    }
    if (options.holdRegime) {
      if (!current.completed) handlers.onHoldRelease?.(now() - current.t);
      return;
    }
    const point = local(event);
    const releaseIndex = handlers.candidateAt(point);
    handlers.onActivation({
      ...point,
      pressIndex: current.index ?? -1,
      hitIndex: releaseIndex !== null && releaseIndex === current.index ? releaseIndex : null,
      t: now(),
      modality: current.modality,
      kind: "release",
    });
  };

  const onLost = (event: PointerEvent) => {
    if (!press || event.pointerId !== press.pointerId) return;
    const current = press;
    endPress();
    if (options.holdRegime) {
      if (!current.completed) handlers.onHoldAbort?.(event.type);
    } else {
      handlers.onPressCancel?.(event.type);
    }
  };

  surfaceEl.addEventListener("pointerdown", onPointerDown);
  surfaceEl.addEventListener("pointermove", onPointerMove);
  surfaceEl.addEventListener("pointerup", onPointerUp);
  surfaceEl.addEventListener("pointercancel", onLost);
  surfaceEl.addEventListener("lostpointercapture", onLost);
  const detachWindow = attachWindowInterrupts(handlers, endPress);

  return () => {
    endPress();
    surfaceEl.removeEventListener("pointerdown", onPointerDown);
    surfaceEl.removeEventListener("pointermove", onPointerMove);
    surfaceEl.removeEventListener("pointerup", onPointerUp);
    surfaceEl.removeEventListener("pointercancel", onLost);
    surfaceEl.removeEventListener("lostpointercapture", onLost);
    detachWindow();
  };
}

// ---- Keyboard ------------------------------------------------------------------------------------------------------

/** Focus at the cue: the candidate nearest the origin (lower index on ties); the first item in a menu. */
export function initialFocus(rects: readonly RectV1[], origin: Point | null, task: TaskKind): number | null {
  if (rects.length === 0) return null;
  if (task === "menu" || origin === null) return 0;
  let best = 0;
  let bestDistance = Infinity;
  rects.forEach((rect, i) => {
    const distance = Math.hypot(rect.cx - origin.x, rect.cy - origin.y);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  });
  return best;
}

export type ArrowKey = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";

/** The nearest candidate in the arrow's direction within the same row or column; the current one when none. */
export function moveFocus(rects: readonly RectV1[], current: number, key: ArrowKey): number {
  const from = rects[current];
  if (!from) return current;
  const horizontal = key === "ArrowLeft" || key === "ArrowRight";
  const sign = key === "ArrowRight" || key === "ArrowDown" ? 1 : -1;
  let best = current;
  let bestDistance = Infinity;
  rects.forEach((rect, i) => {
    if (i === current) return;
    const along = horizontal ? rect.cx - from.cx : rect.cy - from.cy;
    const across = horizontal ? rect.cy - from.cy : rect.cx - from.cx;
    const band = (horizontal ? from.h : from.w) / 2;
    if (along * sign <= 0 || Math.abs(across) > band) return;
    if (Math.abs(along) < bestDistance) {
      best = i;
      bestDistance = Math.abs(along);
    }
  });
  return best;
}

export interface KeyboardHandlers {
  /** Cue-time rects (keyboard activations never use an expanded rect). */
  rects(): readonly RectV1[];
  focused(): number | null;
  setFocus(index: number): void;
  /** Labels for type-ahead in an open menu (optional). */
  labels?(): readonly string[];
  onPress?(press: PressInput): void;
  onActivation(activation: ActivationInput): void;
  onHoldRelease?(elapsedMs: number): void;
  onHoldAbort?(reason: string): void;
  onHoldComplete?(): void;
  onInterrupt?(reason: "blur" | "hidden"): void;
  /** Escape (cancel or pause) and Backspace (undo) are passed through. */
  onKey?(key: "Escape" | "Backspace"): void;
}

export interface KeyboardOptions extends InputOptions {
  task?: TaskKind;
  origin?: Point | null;
}

/**
 * Keyboard input: arrows move focus, Enter or Space activates the focused candidate (key repeats ignored). The
 * activation's endpoint is the focused candidate's rounded cue-time centre and its press and hit index are the
 * focused candidate (section 6.1). In a hold regime the keydown is the activation and an early keyup a hold release.
 */
export function attachKeyboardInput(
  surfaceEl: HTMLElement,
  handlers: KeyboardHandlers,
  options: KeyboardOptions,
): () => void {
  const now = options.now ?? defaultNow;
  let held: { key: string; t: number; completed: boolean } | null = null;
  let holdTimer: ReturnType<typeof setTimeout> | null = null;

  const endHold = () => {
    if (holdTimer !== null) clearTimeout(holdTimer);
    holdTimer = null;
    held = null;
  };

  const onKeyDown = (event: KeyboardEvent) => {
    const rects = handlers.rects();
    if (
      event.key === "ArrowUp" ||
      event.key === "ArrowDown" ||
      event.key === "ArrowLeft" ||
      event.key === "ArrowRight"
    ) {
      event.preventDefault();
      if (held) return;
      const current = handlers.focused() ?? initialFocus(rects, options.origin ?? null, options.task ?? "pointing");
      if (current === null) return;
      handlers.setFocus(handlers.focused() === null ? current : moveFocus(rects, current, event.key));
      return;
    }
    if (event.key === "Enter" || event.key === " " || event.key === "Spacebar") {
      event.preventDefault();
      if (event.repeat || held) return;
      const focus = handlers.focused();
      if (focus === null || !rects[focus]) return;
      const point = keyboardEndpoint(rects[focus]);
      const t = now();
      handlers.onPress?.({ ...point, index: focus, t, modality: "keyboard" });
      handlers.onActivation({ ...point, pressIndex: focus, hitIndex: focus, t, modality: "keyboard", kind: "key" });
      if (options.holdRegime) {
        held = { key: event.key, t, completed: false };
        const holdMs = options.holdMs;
        if (holdMs !== null && holdMs !== undefined && holdMs > 0) {
          holdTimer = setTimeout(() => {
            holdTimer = null;
            if (held && !held.completed) {
              held.completed = true;
              handlers.onHoldComplete?.();
            }
          }, holdMs);
        }
      }
      return;
    }
    if (event.key === "Escape" || event.key === "Backspace") {
      if (event.repeat) return;
      handlers.onKey?.(event.key);
      return;
    }
    if (event.key.length === 1 && !event.repeat && handlers.labels && options.task === "menu") {
      const labels = handlers.labels();
      const start = (handlers.focused() ?? -1) + 1;
      const letter = event.key.toLowerCase();
      for (let k = 0; k < labels.length; k += 1) {
        const i = (start + k) % labels.length;
        if (labels[i]?.toLowerCase().startsWith(letter)) {
          handlers.setFocus(i);
          break;
        }
      }
    }
  };

  const onKeyUp = (event: KeyboardEvent) => {
    if (!held || event.key !== held.key) return;
    const current = held;
    endHold();
    if (!current.completed) handlers.onHoldRelease?.(now() - current.t);
  };

  surfaceEl.addEventListener("keydown", onKeyDown);
  surfaceEl.addEventListener("keyup", onKeyUp);
  const detachWindow = attachWindowInterrupts(handlers, endHold);

  return () => {
    endHold();
    surfaceEl.removeEventListener("keydown", onKeyDown);
    surfaceEl.removeEventListener("keyup", onKeyUp);
    detachWindow();
  };
}

// ---- Hand pointer --------------------------------------------------------------------------------------------------

/** The fields of the studyPointer store the adapter reads (structural, so this module imports no store). */
export interface StudyPointerSnapshot {
  handPresent: boolean;
  ready: boolean;
  pressArmed: boolean;
  nx: number;
  ny: number;
  pinchClosed: boolean;
  t: number;
  palmExits?: number;
}

export interface StudyPointerSource {
  subscribe(run: (value: StudyPointerSnapshot) => void): () => void;
}

export interface HandPointerHandlers extends InputHandlers {
  onPalmExit?(): void;
}

/**
 * The in-window hand pointer. A press is a published pinch edge (false to true) while `pressArmed`, so a pinch held
 * closed through a hand loss, a reacquisition or a study pause never becomes a press. In a non-hold regime the
 * activation is the pinch opening on the same candidate; a frame gap over 100 ms, a hand loss or a store reset during
 * the press cancels it without an activation (the forced pinch opening is never a release). In a hold regime the
 * pinch close is the activation; the hold completes only on fresh frames and aborts on a stale frame or a hand loss.
 */
export function attachHandPointerInput(
  surfaceRect: { width: number; height: number },
  studyPointerStore: StudyPointerSource,
  handlers: HandPointerHandlers,
  options: InputOptions,
): () => void {
  const now = options.now ?? defaultNow;
  let initialised = false;
  let previousPinch = false;
  let lastFrameT: number | null = null;
  let lastPalmExits: number | null = null;
  let press: (PressInput & { completed: boolean }) | null = null;
  let watchdog: ReturnType<typeof setInterval> | null = null;

  const stopWatchdog = () => {
    if (watchdog !== null) clearInterval(watchdog);
    watchdog = null;
  };
  const lose = (reason: string) => {
    if (!press) return;
    const current = press;
    press = null;
    stopWatchdog();
    if (options.holdRegime) {
      if (!current.completed) handlers.onHoldAbort?.(reason);
    } else {
      handlers.onPressCancel?.(reason);
    }
  };
  const maybeComplete = () => {
    if (!press || !options.holdRegime || press.completed) return;
    const holdMs = options.holdMs;
    if (holdMs === null || holdMs === undefined || holdMs <= 0) return;
    const t = now();
    if (lastFrameT === null || t - lastFrameT > FRESH_FRAME_MS) {
      lose("stale_frame");
      return;
    }
    if (t - press.t >= holdMs) {
      press.completed = true;
      stopWatchdog();
      handlers.onHoldComplete?.();
    }
  };
  const startWatchdog = () => {
    stopWatchdog();
    watchdog = setInterval(() => {
      if (!press) {
        stopWatchdog();
        return;
      }
      if (lastFrameT === null || now() - lastFrameT > FRESH_FRAME_MS) {
        lose("stale_frame");
        return;
      }
      maybeComplete();
    }, 20);
  };

  const unsubscribe = studyPointerStore.subscribe((state) => {
    if (lastPalmExits !== null && (state.palmExits ?? 0) > lastPalmExits) handlers.onPalmExit?.();
    lastPalmExits = state.palmExits ?? 0;

    if (!state.handPresent) {
      lose("hand_lost");
      previousPinch = false;
      lastFrameT = null;
      initialised = true;
      return;
    }
    if (press && lastFrameT !== null && state.t - lastFrameT > FRESH_FRAME_MS) lose("stale_frame");
    if (press && (!state.ready || !state.pressArmed)) lose("reset");
    lastFrameT = state.t;

    const point = {
      x: Math.round(Math.max(0, Math.min(1, state.nx)) * surfaceRect.width),
      y: Math.round(Math.max(0, Math.min(1, state.ny)) * surfaceRect.height),
    };
    handlers.onMove?.(point, state.t);

    const pinch = state.pinchClosed;
    if (!initialised) {
      // The first value is the baseline: a pinch already closed at attach is never a press edge.
      initialised = true;
      previousPinch = pinch;
      return;
    }
    if (!previousPinch && pinch && !press && state.pressArmed && state.ready) {
      const index = handlers.candidateAt(point);
      const t = now();
      press = { ...point, index, t, modality: "hand_pointer", completed: false };
      handlers.onPress?.({ ...point, index, t, modality: "hand_pointer" });
      if (options.holdRegime) {
        handlers.onActivation({
          ...point,
          pressIndex: index ?? -1,
          hitIndex: index,
          t,
          modality: "hand_pointer",
          kind: "press",
        });
      }
      startWatchdog();
    } else if (previousPinch && !pinch && press) {
      const current = press;
      press = null;
      stopWatchdog();
      if (options.holdRegime) {
        if (!current.completed) handlers.onHoldRelease?.(now() - current.t);
      } else {
        const releaseIndex = handlers.candidateAt(point);
        handlers.onActivation({
          ...point,
          pressIndex: current.index ?? -1,
          hitIndex: releaseIndex !== null && releaseIndex === current.index ? releaseIndex : null,
          t: now(),
          modality: "hand_pointer",
          kind: "release",
        });
      }
    }
    previousPinch = pinch;
    maybeComplete();
  });
  const detachWindow = attachWindowInterrupts(handlers, () => {
    press = null;
    stopWatchdog();
  });

  return () => {
    press = null;
    stopWatchdog();
    unsubscribe();
    detachWindow();
  };
}

/**
 * On-device continual-learning/personalization loop for gesture recognition.
 *
 * Personalizes exactly two of the ~20 hardcoded constants in
 * GestureControl.svelte/spatialModel.ts — PINCH_DISTANCE_THRESHOLD and
 * THUMB_EXTENDED_RATIO — chosen because they depend on individual hand
 * anatomy/pinch style, unlike e.g. swipe velocity thresholds which are a
 * motion-preference/timing concern left untouched (see GESTURES.md).
 *
 * This does NOT retrain MediaPipe or add a learned model — it nudges two
 * plain numeric thresholds using a bounded exponential moving average (EMA),
 * fed only by an implicit signal already present in normal usage: whether a
 * fired gesture was immediately contradicted by an opposite gesture (a
 * misfire signal) or not (an implicit confirmation). There is no new "was
 * this right?" UI.
 *
 * Storage is exclusively `localStorage` (same pattern as session.ts's
 * `heliox_session_history`) — gesture recognition never leaves the browser
 * today, so this personalization loop never needs to either. Nothing is
 * transmitted anywhere. The learned values are bounded to +/-40% of the
 * shipped default and only take effect after MIN_SAMPLES_TO_APPLY confirmed
 * observations, so a single unusual session can't swing recognition
 * behavior — and everything here is visible/resettable from Settings.
 */

export interface GestureEvent {
  name: string;
  timestamp: number;
  /** The raw measured value (pinch distance, or thumb-extension ratio) at
   * the moment this gesture fired — only meaningful for calibrated gestures. */
  metricValue: number;
}

export type GestureOutcome = "positive" | "negative" | "unknown";

/** Only these gestures feed calibration — they're the ones that use
 * PINCH_DISTANCE_THRESHOLD (pinch/ok) or THUMB_EXTENDED_RATIO (thumbs_up/down). */
const CALIBRATED_GESTURES = new Set(["thumbs_up", "thumbs_down", "ok", "pinch"]);

/** How long to wait after a calibrated gesture fires before treating it as
 * an implicit confirmation (no contradiction arrived in time). */
export const REVERSAL_WINDOW_MS = 2500;

/** Gestures that semantically contradict a given gesture — firing one of
 * these shortly after is treated as "the previous one was a misfire." */
const CONTRADICTORY: Record<string, string[]> = {
  thumbs_up: ["thumbs_down", "palm"],
  thumbs_down: ["thumbs_up", "palm"],
  ok: ["palm"],
  pinch: ["palm"],
};

/**
 * Decide the implicit outcome of `prev` (a previously-fired gesture) given
 * `next` (the gesture that fired after it, or `null` if `REVERSAL_WINDOW_MS`
 * elapsed with nothing following).
 *
 * - `"unknown"` if `prev` isn't a calibrated gesture, or if `next` fired but
 *   isn't semantically related to `prev` (a differently-purposed gesture
 *   shouldn't count as either confirmation or rejection).
 * - `"negative"` if `next` is one of `prev`'s CONTRADICTORY gestures and
 *   arrived within the window — an implicit misfire signal.
 * - `"positive"` otherwise: either the window elapsed with no contradiction
 *   (`next === null`), or `next` arrived but wasn't contradictory (e.g. a
 *   fresh, unrelated calibrated gesture — the outer caller decides how to
 *   re-pair from there; this function only judges `prev` against `next`).
 */
export function classifyOutcome(prev: GestureEvent, next: GestureEvent | null): GestureOutcome {
  if (!CALIBRATED_GESTURES.has(prev.name)) return "unknown";

  if (next === null) return "positive";

  const elapsed = next.timestamp - prev.timestamp;
  if (elapsed > REVERSAL_WINDOW_MS) return "positive";

  const contradictions = CONTRADICTORY[prev.name] ?? [];
  if (contradictions.includes(next.name)) return "negative";

  return "positive";
}

interface GestureCalibrationData {
  version: 1;
  pinchThresholdEma: number | null;
  pinchSampleCount: number;
  thumbRatioEma: number | null;
  thumbSampleCount: number;
  lastUpdated: number;
}

const STORAGE_KEY = "heliox_gesture_calibration";
const EMA_ALPHA = 0.08;
const MIN_SAMPLES_TO_APPLY = 8;
const CLAMP_MIN_FACTOR = 0.6;
const CLAMP_MAX_FACTOR = 1.4;

function defaultData(): GestureCalibrationData {
  return {
    version: 1,
    pinchThresholdEma: null,
    pinchSampleCount: 0,
    thumbRatioEma: null,
    thumbSampleCount: 0,
    lastUpdated: 0,
  };
}

function ema(current: number | null, sample: number): number {
  return current === null ? sample : EMA_ALPHA * sample + (1 - EMA_ALPHA) * current;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Persists learned per-user gesture calibration to localStorage. One
 * instance is created per component lifetime (see GestureControl.svelte);
 * construction loads any existing data, every update persists immediately. */
export class GestureCalibrationStore {
  private data: GestureCalibrationData;
  private listeners = new Set<(snapshot: Readonly<GestureCalibrationData>) => void>();

  constructor() {
    this.data = this.readFromStorage();
  }

  private readFromStorage(): GestureCalibrationData {
    if (typeof localStorage === "undefined") return defaultData();
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultData();
      const parsed = JSON.parse(raw);
      if (!parsed || parsed.version !== 1) return defaultData();
      return { ...defaultData(), ...parsed };
    } catch {
      return defaultData();
    }
  }

  private persist(): void {
    if (typeof localStorage === "undefined") return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.data));
    } catch {
      // best-effort — localStorage may be full or disabled; losing one
      // calibration update isn't worth surfacing to the user.
    }
  }

  private notify(): void {
    const snapshot = this.getSnapshot();
    for (const listener of this.listeners) listener(snapshot);
  }

  subscribe(listener: (snapshot: Readonly<GestureCalibrationData>) => void): () => void {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => this.listeners.delete(listener);
  }

  /** Clears all learned calibration, reverting to shipped defaults. */
  reset(): void {
    this.data = defaultData();
    if (typeof localStorage !== "undefined") {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore
      }
    }
    this.notify();
  }

  /** Feed a gesture's implicit outcome into calibration. Only "positive"
   * (confirmed, unreversed) observations update the EMA — a misfire
   * ("negative") is simply not reinforced, rather than actively pushing the
   * threshold away from the measured value (we don't know what the "right"
   * value was in that instance, only that this firing was wrong). */
  recordOutcome(event: GestureEvent, outcome: GestureOutcome): void {
    if (outcome !== "positive") return;

    if (event.name === "pinch" || event.name === "ok") {
      this.data.pinchThresholdEma = ema(this.data.pinchThresholdEma, event.metricValue);
      this.data.pinchSampleCount++;
      this.data.lastUpdated = event.timestamp;
      this.persist();
      this.notify();
    } else if (event.name === "thumbs_up" || event.name === "thumbs_down") {
      this.data.thumbRatioEma = ema(this.data.thumbRatioEma, event.metricValue);
      this.data.thumbSampleCount++;
      this.data.lastUpdated = event.timestamp;
      this.persist();
      this.notify();
    }
  }

  /** Returns the personalized pinch/OK-sign distance threshold, clamped to
   * +/-40% of `base`, or `base` unchanged if fewer than
   * MIN_SAMPLES_TO_APPLY confirmed observations exist yet. */
  getEffectivePinchThreshold(base: number): number {
    if (this.data.pinchThresholdEma === null || this.data.pinchSampleCount < MIN_SAMPLES_TO_APPLY) {
      return base;
    }
    return clamp(this.data.pinchThresholdEma, base * CLAMP_MIN_FACTOR, base * CLAMP_MAX_FACTOR);
  }

  /** Same as getEffectivePinchThreshold(), for the thumb-extension ratio. */
  getEffectiveThumbRatio(base: number): number {
    if (this.data.thumbRatioEma === null || this.data.thumbSampleCount < MIN_SAMPLES_TO_APPLY) {
      return base;
    }
    return clamp(this.data.thumbRatioEma, base * CLAMP_MIN_FACTOR, base * CLAMP_MAX_FACTOR);
  }

  /** Read-only snapshot for the Settings transparency panel. */
  getSnapshot(): Readonly<GestureCalibrationData> {
    return { ...this.data };
  }
}

let sharedStore: GestureCalibrationStore | null = null;

/** One live store shared by camera recognition and Settings.
 *
 * Keeping a single in-memory instance makes Reset immediately affect the
 * thresholds currently used by an active camera session, while subscribe()
 * keeps the learned sample counts live in Settings.
 */
export function getSharedGestureCalibrationStore(): GestureCalibrationStore {
  sharedStore ??= new GestureCalibrationStore();
  return sharedStore;
}

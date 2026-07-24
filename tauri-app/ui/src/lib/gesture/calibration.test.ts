import { describe, it, expect, beforeEach } from "vitest";
import { classifyOutcome, GestureCalibrationStore, REVERSAL_WINDOW_MS, type GestureEvent } from "./calibration";

function ev(name: string, timestamp: number, metricValue = 0): GestureEvent {
  return { name, timestamp, metricValue };
}

describe("classifyOutcome", () => {
  it("is unknown for a gesture that isn't calibrated", () => {
    expect(classifyOutcome(ev("palm", 0), null)).toBe("unknown");
    expect(classifyOutcome(ev("swipe_left", 0), ev("swipe_right", 100))).toBe("unknown");
  });

  it("is positive when nothing follows within the reversal window (timeout)", () => {
    expect(classifyOutcome(ev("pinch", 0), null)).toBe("positive");
    expect(classifyOutcome(ev("thumbs_up", 0), null)).toBe("positive");
  });

  it("is negative when a contradictory gesture follows within the window", () => {
    expect(classifyOutcome(ev("thumbs_up", 1000), ev("thumbs_down", 1000 + REVERSAL_WINDOW_MS / 2))).toBe("negative");
    expect(classifyOutcome(ev("pinch", 1000), ev("palm", 1000 + 100))).toBe("negative");
    expect(classifyOutcome(ev("ok", 1000), ev("palm", 1000 + 100))).toBe("negative");
  });

  it("is positive when a contradictory-named gesture follows but AFTER the window elapsed", () => {
    expect(classifyOutcome(ev("thumbs_up", 0), ev("thumbs_down", REVERSAL_WINDOW_MS + 1))).toBe("positive");
  });

  it("is positive when an unrelated (non-contradictory) gesture follows within the window", () => {
    expect(classifyOutcome(ev("pinch", 1000), ev("fist", 1100))).toBe("positive");
  });
});

describe("GestureCalibrationStore", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("returns the base value unchanged with no recorded samples", () => {
    const store = new GestureCalibrationStore();
    expect(store.getEffectivePinchThreshold(0.05)).toBe(0.05);
    expect(store.getEffectiveThumbRatio(0.55)).toBe(0.55);
  });

  it("does not update on a negative outcome", () => {
    const store = new GestureCalibrationStore();
    store.recordOutcome(ev("pinch", 1, 0.03), "negative");
    expect(store.getSnapshot().pinchSampleCount).toBe(0);
  });

  it("does not update on an unknown outcome", () => {
    const store = new GestureCalibrationStore();
    store.recordOutcome(ev("pinch", 1, 0.03), "unknown");
    expect(store.getSnapshot().pinchSampleCount).toBe(0);
  });

  it("accumulates positive samples but withholds the effective value until the sample floor is reached", () => {
    const store = new GestureCalibrationStore();
    for (let i = 0; i < 7; i++) {
      store.recordOutcome(ev("pinch", i, 0.03), "positive");
    }
    expect(store.getSnapshot().pinchSampleCount).toBe(7);
    // Below MIN_SAMPLES_TO_APPLY (8) — still returns the base, unpersonalized.
    expect(store.getEffectivePinchThreshold(0.05)).toBe(0.05);

    store.recordOutcome(ev("pinch", 8, 0.03), "positive");
    expect(store.getSnapshot().pinchSampleCount).toBe(8);
    // Now personalized — every sample was exactly 0.03, so the EMA (which
    // only ever averages observed samples, never the shipped base) settles
    // exactly there rather than asymptotically approaching it.
    const effective = store.getEffectivePinchThreshold(0.05);
    expect(effective).toBeCloseTo(0.03, 5);
    expect(effective).toBeLessThan(0.05);
  });

  it("clamps the effective value to +/-40% of the base even with extreme samples", () => {
    const store = new GestureCalibrationStore();
    for (let i = 0; i < 20; i++) {
      store.recordOutcome(ev("pinch", i, 10), "positive"); // wildly larger than any real distance
    }
    const effective = store.getEffectivePinchThreshold(0.05);
    expect(effective).toBeLessThanOrEqual(0.05 * 1.4);
  });

  it("tracks pinch and thumb-ratio calibration independently", () => {
    const store = new GestureCalibrationStore();
    for (let i = 0; i < 8; i++) {
      store.recordOutcome(ev("pinch", i, 0.03), "positive");
    }
    expect(store.getSnapshot().thumbSampleCount).toBe(0);
    expect(store.getEffectiveThumbRatio(0.55)).toBe(0.55);
  });

  it("persists across store instances (simulating a page reload)", () => {
    const store1 = new GestureCalibrationStore();
    for (let i = 0; i < 8; i++) {
      store1.recordOutcome(ev("thumbs_up", i, 0.7), "positive");
    }
    const store2 = new GestureCalibrationStore();
    expect(store2.getSnapshot().thumbSampleCount).toBe(8);
    expect(store2.getEffectiveThumbRatio(0.55)).not.toBe(0.55);
  });

  it("reset() clears learned calibration back to shipped defaults", () => {
    const store = new GestureCalibrationStore();
    for (let i = 0; i < 8; i++) {
      store.recordOutcome(ev("pinch", i, 0.03), "positive");
    }
    store.reset();
    expect(store.getSnapshot().pinchSampleCount).toBe(0);
    expect(store.getEffectivePinchThreshold(0.05)).toBe(0.05);

    const reloaded = new GestureCalibrationStore();
    expect(reloaded.getSnapshot().pinchSampleCount).toBe(0);
  });

  it("notifies subscribers after learning and reset", () => {
    const store = new GestureCalibrationStore();
    const sampleCounts: number[] = [];
    const unsubscribe = store.subscribe((snapshot) => {
      sampleCounts.push(snapshot.pinchSampleCount);
    });

    store.recordOutcome(ev("pinch", 1, 0.03), "positive");
    store.reset();
    unsubscribe();
    store.recordOutcome(ev("pinch", 2, 0.03), "positive");

    expect(sampleCounts).toEqual([0, 1, 0]);
  });

  it("ignores corrupted localStorage content and falls back to defaults", () => {
    localStorage.setItem("heliox_gesture_calibration", "{not valid json");
    const store = new GestureCalibrationStore();
    expect(store.getEffectivePinchThreshold(0.05)).toBe(0.05);
  });
});

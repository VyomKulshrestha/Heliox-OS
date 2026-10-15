import { get } from "svelte/store";
import { beforeEach, describe, expect, it } from "vitest";
import {
  HAND_ACQUIRE_FRAMES,
  publishStudyPointerFrame,
  reportStudyPointerHandLost,
  reportStudyPointerPalmExit,
  requestStudyPointer,
  resetStudyPointer,
  setStudyPointerAvailable,
  studyPointer,
} from "./studyPointer";

let t = 0;
/** Published pinch values, one per frame, to look for false -> true edges. */
let published: boolean[] = [];

function frame(pinchClosed: boolean, dt = 33) {
  t += dt;
  publishStudyPointerFrame({ nx: 0.4, ny: 0.6, pinchClosed, quality: 0.8, t });
  published.push(get(studyPointer).pinchClosed);
}

function frames(n: number, pinchClosed: boolean) {
  for (let i = 0; i < n; i += 1) frame(pinchClosed);
}

function edges(): number[] {
  const out: number[] = [];
  for (let i = 1; i < published.length; i += 1) if (!published[i - 1] && published[i]) out.push(i);
  return out;
}

beforeEach(() => {
  t = 0;
  published = [];
  resetStudyPointer();
  requestStudyPointer(true);
});

describe("studyPointer store", () => {
  it("counts losses and palm exits only while requested, and publishes frames", () => {
    setStudyPointerAvailable(true);
    frame(false);
    expect(get(studyPointer)).toMatchObject({ available: true, handPresent: true, nx: 0.4, ny: 0.6, quality: 0.8 });
    const losses = get(studyPointer).trackingLosses;
    reportStudyPointerHandLost();
    reportStudyPointerPalmExit();
    expect(get(studyPointer).trackingLosses).toBe(losses + 1);
    expect(get(studyPointer).palmExits).toBeGreaterThan(0);
    requestStudyPointer(false);
    const before = get(studyPointer);
    reportStudyPointerHandLost();
    reportStudyPointerPalmExit();
    expect(get(studyPointer).trackingLosses).toBe(before.trackingLosses);
    expect(get(studyPointer).palmExits).toBe(before.palmExits);
  });

  it("is ready only after 8 consecutive frames and armed after 8 further fresh open frames", () => {
    expect(HAND_ACQUIRE_FRAMES).toBe(8);
    frames(7, false);
    expect(get(studyPointer).ready).toBe(false);
    frame(false);
    expect(get(studyPointer)).toMatchObject({ ready: true, pressArmed: false });
    frames(7, false);
    expect(get(studyPointer).pressArmed).toBe(false);
    frame(false);
    expect(get(studyPointer)).toMatchObject({ pressArmed: true, pinchClosed: false });
    frame(true);
    expect(get(studyPointer).pinchClosed).toBe(true);
  });

  it("publishes pinchClosed false until pressArmed", () => {
    frames(8, true);
    expect(get(studyPointer).ready).toBe(true);
    frames(20, true);
    expect(published.every((value) => value === false)).toBe(true);
    expect(get(studyPointer).pressArmed).toBe(false);
  });

  it("restarts the open count on a closed frame or a frame more than 100 ms after the previous one", () => {
    frames(8, false);
    frames(7, false);
    frame(true);
    frames(7, false);
    expect(get(studyPointer).pressArmed).toBe(false);
    frame(false, 150);
    frames(7, false);
    expect(get(studyPointer).pressArmed).toBe(false);
    frame(false);
    expect(get(studyPointer).pressArmed).toBe(true);
  });

  it("forces pinchClosed, ready and pressArmed false on a hand loss", () => {
    frames(16, false);
    frame(true);
    expect(get(studyPointer)).toMatchObject({ ready: true, pressArmed: true, pinchClosed: true });
    reportStudyPointerHandLost();
    expect(get(studyPointer)).toMatchObject({
      handPresent: false,
      ready: false,
      pressArmed: false,
      pinchClosed: false,
    });
  });

  it("never yields a false -> true edge for a pinch held closed through a loss until it was open 8 fresh frames", () => {
    frames(16, false);
    frame(true);
    const firstEdge = edges();
    expect(firstEdge).toHaveLength(1);
    reportStudyPointerHandLost();
    published.push(get(studyPointer).pinchClosed);
    frames(40, true);
    expect(edges()).toEqual(firstEdge);
    frames(7, false);
    frame(true);
    expect(edges()).toEqual(firstEdge);
    frames(8, false);
    expect(edges()).toEqual(firstEdge);
    frame(true);
    expect(edges()).toHaveLength(2);
    expect(published.at(-1)).toBe(true);
  });

  it("resets both counters on reset", () => {
    frames(16, false);
    expect(get(studyPointer).pressArmed).toBe(true);
    resetStudyPointer();
    expect(get(studyPointer)).toMatchObject({ ready: false, pressArmed: false, pinchClosed: false });
    frames(7, false);
    expect(get(studyPointer).ready).toBe(false);
  });

  it("fully resets on requestStudyPointer(false) and requestStudyPointer(true)", () => {
    for (const on of [false, true]) {
      requestStudyPointer(true);
      frames(16, false);
      frame(true);
      expect(get(studyPointer).pinchClosed).toBe(true);
      requestStudyPointer(on);
      expect(get(studyPointer)).toMatchObject({ requested: on, ready: false, pressArmed: false, pinchClosed: false });
      frames(7, false);
      expect(get(studyPointer).ready).toBe(false);
    }
  });

  it("gives no edge after resume for a pinch closed during a pause until it was open 8 fresh frames after ready", () => {
    frames(16, false);
    expect(get(studyPointer).pressArmed).toBe(true);
    requestStudyPointer(false);
    published.push(get(studyPointer).pinchClosed);
    frames(6, true);
    requestStudyPointer(true);
    published.push(get(studyPointer).pinchClosed);
    frames(30, true);
    expect(edges()).toEqual([]);
    frames(8, false);
    expect(edges()).toEqual([]);
    frame(true);
    expect(edges()).toHaveLength(1);
  });

  it("does the same for a pinch held closed from before the pause through the resume", () => {
    frames(16, false);
    frame(true);
    const firstEdge = edges();
    requestStudyPointer(false);
    published.push(get(studyPointer).pinchClosed);
    frames(4, true);
    requestStudyPointer(true);
    published.push(get(studyPointer).pinchClosed);
    frames(25, true);
    expect(edges()).toEqual(firstEdge);
    frames(8, false);
    frame(true);
    expect(edges()).toHaveLength(2);
  });
});

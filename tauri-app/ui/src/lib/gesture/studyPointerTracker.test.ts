import { get } from "svelte/store";
import { afterEach, describe, expect, it } from "vitest";
import {
  createAcquireCounter,
  createGestureArm,
  createPinchHysteresis,
  gazeEventAllowed,
  normalizedPointer,
} from "./studyPointerTracker";
import {
  publishStudyPointerFrame,
  reportStudyPointerHandLost,
  requestStudyPointer,
  resetStudyPointer,
  studyPointer,
} from "../stores/studyPointer";

const THRESHOLD = 0.05;
const CLOSED = 0.02;
const OPEN = 0.1;

describe("createPinchHysteresis", () => {
  it("closes after two consecutive frames below the threshold", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    expect(pinch.update(CLOSED)).toBe(false);
    expect(pinch.update(CLOSED)).toBe(true);
    expect(pinch.closed).toBe(true);
  });

  it("restarts the close count when one frame is above the threshold", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    pinch.update(CLOSED);
    pinch.update(OPEN);
    expect(pinch.update(CLOSED)).toBe(false);
    expect(pinch.update(CLOSED)).toBe(true);
  });

  it("opens only above 1.3 times the threshold", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    pinch.update(CLOSED);
    pinch.update(CLOSED);
    expect(pinch.update(THRESHOLD * 1.2)).toBe(true);
    expect(pinch.update(THRESHOLD * 1.3)).toBe(true);
    expect(pinch.update(THRESHOLD * 1.31)).toBe(false);
  });

  it("uses a threshold passed with the frame", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    pinch.update(0.06, 0.07);
    expect(pinch.update(0.06, 0.07)).toBe(true);
  });

  it("reset() returns to open and needs two closed frames again", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    pinch.update(CLOSED);
    pinch.update(CLOSED);
    pinch.reset();
    expect(pinch.closed).toBe(false);
    expect(pinch.update(CLOSED)).toBe(false);
    expect(pinch.update(CLOSED)).toBe(true);
  });

  it("ignores a non-finite distance", () => {
    const pinch = createPinchHysteresis(THRESHOLD);
    expect(pinch.update(Number.NaN)).toBe(false);
  });
});

describe("createAcquireCounter", () => {
  it("needs 8 consecutive verified frames", () => {
    const counter = createAcquireCounter(8);
    for (let i = 0; i < 7; i++) expect(counter.frame(true)).toBe(false);
    expect(counter.frame(true)).toBe(true);
    expect(counter.acquired).toBe(true);
  });

  it("restarts on a lost frame", () => {
    const counter = createAcquireCounter(8);
    for (let i = 0; i < 7; i++) counter.frame(true);
    expect(counter.frame(false)).toBe(false);
    expect(counter.count).toBe(0);
    for (let i = 0; i < 7; i++) expect(counter.frame(true)).toBe(false);
    expect(counter.frame(true)).toBe(true);
  });
});

describe("normalizedPointer", () => {
  it("mirrors x like the gesture cursor and maps into [0, 1]", () => {
    const { nx, ny } = normalizedPointer({ x: 0.2, y: 0.3 }, 1);
    expect(nx).toBeCloseTo(800 / 999, 6);
    expect(ny).toBeCloseTo(300 / 999, 6);
  });

  it("reaches both edges and stays inside them at high sensitivity", () => {
    expect(normalizedPointer({ x: 1, y: 0 }, 1)).toEqual({ nx: 0, ny: 0 });
    expect(normalizedPointer({ x: 0, y: 1 }, 1)).toEqual({ nx: 1, ny: 1 });
    const far = normalizedPointer({ x: -0.5, y: 1.5 }, 3);
    expect(far.nx).toBe(1);
    expect(far.ny).toBe(1);
  });
});

describe("gazeEventAllowed", () => {
  it("is false while a study session is active and true otherwise", () => {
    expect(gazeEventAllowed(true)).toBe(false);
    expect(gazeEventAllowed(false)).toBe(true);
  });
});

/** Feeds frames every `step` ms from `from` (exclusive of nothing: the first frame is at `from`). */
function feed(arm: ReturnType<typeof createGestureArm>, verified: boolean, from: number, count: number, step = 33) {
  let t = from;
  for (let i = 0; i < count; i++) {
    arm.frame(verified, t);
    t += step;
  }
  return t; // time of the next frame
}

describe("createGestureArm", () => {
  it("starts armed and allows any gesture before the first disarm", () => {
    const arm = createGestureArm();
    expect(arm.armed).toBe(true);
    expect(arm.allows(0)).toBe(true);
  });

  it("stays disarmed while the hand is seen continuously", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    feed(arm, true, 0, 10);
    arm.disarm(330);
    feed(arm, true, 330, 120);
    expect(arm.armed).toBe(false);
    expect(arm.armedAt).toBeNull();
    expect(arm.allows(5000)).toBe(false);
  });

  it("does not re-arm after a 150 ms absence followed by 8 verified frames", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    // Frames with no verified hand at 30, 60 ... 120, then the hand is back at 150.
    feed(arm, false, 30, 4, 30);
    feed(arm, true, 150, 8);
    expect(arm.armed).toBe(false);
  });

  it("re-arms at the eighth verified frame after an absence of at least 200 ms", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    feed(arm, false, 50, 4, 50); // no hand at 50, 100, 150, 200
    let t = 250;
    for (let i = 1; i <= 7; i++) {
      expect(arm.frame(true, t)).toBe(false);
      t += 33;
    }
    expect(arm.frame(true, t)).toBe(true);
    expect(arm.armedAt).toBe(t);
  });

  it("also counts an absence that ends with a gap of at least 200 ms since the last verified frame", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    arm.frame(false, 100);
    let t = 210;
    for (let i = 0; i < 8; i++) {
      arm.frame(true, t);
      t += 33;
    }
    expect(arm.armed).toBe(true);
  });

  it("never treats a gap with no frames at all as the hand having left", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    // The window was hidden: no frames for two seconds, the hand is still there afterwards.
    feed(arm, true, 2000, 20);
    expect(arm.armed).toBe(false);
  });

  it("re-arms while the hand is away after 500 ms with no verified hand", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    expect(arm.frame(false, 250)).toBe(false);
    expect(arm.frame(false, 499)).toBe(false);
    expect(arm.frame(false, 500)).toBe(true);
    expect(arm.armedAt).toBe(500);
  });

  it("restarts the present count when the hand drops out again before it re-armed", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    arm.frame(false, 250);
    feed(arm, true, 280, 7);
    arm.frame(false, 520);
    expect(arm.armed).toBe(false);
    feed(arm, true, 550, 7);
    expect(arm.armed).toBe(false);
    arm.frame(true, 550 + 7 * 33);
    expect(arm.armed).toBe(true);
  });

  it("allows only gestures that began at or after the re-arm", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.frame(true, 0);
    arm.disarm(0);
    arm.frame(false, 600);
    expect(arm.armedAt).toBe(600);
    // A pose recognised before the re-arm stays dropped even when it completes later.
    expect(arm.allows(400)).toBe(false);
    expect(arm.allows(599.9)).toBe(false);
    expect(arm.allows(600)).toBe(true);
    expect(arm.allows(900)).toBe(true);
  });

  it("a new disarm resets armedAt", () => {
    const arm = createGestureArm({ absentMs: 200, idleMs: 500, acquireFrames: 8 });
    arm.disarm(0);
    arm.frame(false, 600);
    expect(arm.armedAt).toBe(600);
    arm.disarm(700);
    expect(arm.armed).toBe(false);
    expect(arm.armedAt).toBeNull();
    expect(arm.allows(800)).toBe(false);
    arm.frame(false, 1300);
    expect(arm.armedAt).toBe(1300);
  });
});

describe("study pointer tracker feeding the real study pointer store", () => {
  afterEach(() => {
    requestStudyPointer(false);
    resetStudyPointer();
  });

  /** Mirrors GestureControl: a verified frame goes through the hysteresis into publishStudyPointerFrame. */
  function createLoop() {
    const pinch = createPinchHysteresis(THRESHOLD);
    let t = 1000;
    const published: boolean[] = [];
    return {
      pinch,
      published,
      frame(distance: number) {
        t += 33;
        const pinchClosed = pinch.update(distance);
        publishStudyPointerFrame({ nx: 0.5, ny: 0.5, pinchClosed, quality: 0.9, t });
        published.push(get(studyPointer).pinchClosed);
      },
      frames(distance: number, count: number) {
        for (let i = 0; i < count; i++) this.frame(distance);
      },
      lose() {
        // No verified hand for more than 200 ms: GestureControl reports the loss once and resets the hysteresis.
        t += 250;
        reportStudyPointerHandLost();
        pinch.reset();
      },
      /** Index of the first published false -> true edge at or after `from`, or -1. */
      firstRise(from = 0) {
        for (let i = Math.max(1, from); i < published.length; i++) {
          if (published[i] && !published[i - 1]) return i;
        }
        if (from === 0 && published[0]) return 0;
        return -1;
      },
    };
  }

  it("a pinch held closed through a hand loss never becomes a press until it was open for 8 fresh frames", () => {
    requestStudyPointer(true);
    const loop = createLoop();
    loop.frames(OPEN, 24); // ready, then armed with the pinch open
    loop.frames(CLOSED, 4); // a real press before the loss
    expect(get(studyPointer).pinchClosed).toBe(true);

    loop.lose();
    expect(get(studyPointer).pinchClosed).toBe(false);
    const afterLoss = loop.published.length;

    loop.frames(CLOSED, 30); // reacquired with the pinch still closed: through ready and long after
    loop.frames(OPEN, 5); // open for fewer than 8 fresh frames
    loop.frames(CLOSED, 6);
    expect(loop.published.slice(afterLoss).every((closed) => !closed)).toBe(true);

    loop.frames(OPEN, 12); // open long enough to arm a press
    const beforeClose = loop.published.length;
    expect(loop.published.slice(afterLoss).every((closed) => !closed)).toBe(true);
    loop.frames(CLOSED, 4);
    const rise = loop.firstRise(afterLoss);
    expect(rise).toBeGreaterThanOrEqual(beforeClose);
  });

  it("a pinch closed during a study pause yields no press after resume until it was open for 8 fresh frames", () => {
    requestStudyPointer(true);
    const loop = createLoop();
    loop.frames(OPEN, 24);
    expect(get(studyPointer).pinchClosed).toBe(false);

    requestStudyPointer(false);
    // While paused GestureControl publishes nothing, but its hysteresis still sees the pinch close.
    loop.pinch.update(CLOSED);
    loop.pinch.update(CLOSED);
    expect(loop.pinch.closed).toBe(true);

    requestStudyPointer(true);
    loop.pinch.reset(); // what GestureControl does when `requested` turns true
    expect(get(studyPointer).pinchClosed).toBe(false);
    const afterResume = loop.published.length;

    loop.frames(CLOSED, 30);
    loop.frames(OPEN, 5);
    loop.frames(CLOSED, 6);
    expect(loop.published.slice(afterResume).every((closed) => !closed)).toBe(true);

    loop.frames(OPEN, 12);
    const beforeClose = loop.published.length;
    loop.frames(CLOSED, 4);
    const rise = loop.firstRise(afterResume);
    expect(rise).toBeGreaterThanOrEqual(beforeClose);
  });

  it("a pinch held closed from before the pause through the resume gives no press edge either", () => {
    requestStudyPointer(true);
    const loop = createLoop();
    loop.frames(OPEN, 24);
    loop.frames(CLOSED, 4);
    expect(get(studyPointer).pinchClosed).toBe(true);

    requestStudyPointer(false);
    requestStudyPointer(true);
    loop.pinch.reset();
    const afterResume = loop.published.length;

    loop.frames(CLOSED, 30);
    expect(loop.published.slice(afterResume).every((closed) => !closed)).toBe(true);
    loop.frames(OPEN, 12);
    const beforeClose = loop.published.length;
    loop.frames(CLOSED, 4);
    expect(loop.firstRise(afterResume)).toBeGreaterThanOrEqual(beforeClose);
  });
});

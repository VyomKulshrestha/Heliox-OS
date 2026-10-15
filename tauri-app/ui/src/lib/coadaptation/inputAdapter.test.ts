import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import {
  publishStudyPointerFrame,
  reportStudyPointerHandLost,
  requestStudyPointer,
  resetStudyPointer,
  studyPointer,
} from "../stores/studyPointer";
import {
  attachHandPointerInput,
  attachKeyboardInput,
  attachPointerInput,
  initialFocus,
  moveFocus,
  type ActivationInput,
  type HandPointerHandlers,
} from "./inputAdapter";
import { keyboardEndpoint } from "./targetPosterior";
import type { RectV1 } from "./types";

const RECTS: RectV1[] = [
  { cx: 100, cy: 100, w: 40, h: 40 },
  { cx: 150, cy: 100, w: 40, h: 40 },
  { cx: 100, cy: 150, w: 40, h: 40 },
  { cx: 150, cy: 150, w: 40, h: 40 },
];

function candidateAt(point: { x: number; y: number }): number | null {
  const index = RECTS.findIndex((r) => Math.abs(point.x - r.cx) <= r.w / 2 && Math.abs(point.y - r.cy) <= r.h / 2);
  return index < 0 ? null : index;
}

function pointer(type: string, x: number, y: number, extra: Record<string, unknown> = {}): Event {
  // jsdom has no PointerEvent: a MouseEvent with the pointer fields stands in for it.
  const event = new MouseEvent(type, { clientX: x, clientY: y, bubbles: true, button: 0 });
  for (const [key, value] of Object.entries({ pointerId: 7, isPrimary: true, pointerType: "mouse", ...extra })) {
    Object.defineProperty(event, key, { value });
  }
  return event;
}

/** Every handler as a spy, so tests can read the calls. */
type SpiedHandlers = { [K in keyof Required<HandPointerHandlers>]: Mock };

function handlers(): SpiedHandlers {
  return {
    candidateAt: vi.fn(candidateAt),
    onMove: vi.fn(),
    onPress: vi.fn(),
    onActivation: vi.fn(),
    onPressCancel: vi.fn(),
    onHoldRelease: vi.fn(),
    onHoldAbort: vi.fn(),
    onHoldComplete: vi.fn(),
    onInterrupt: vi.fn(),
    onPalmExit: vi.fn(),
  };
}

function surface(): HTMLElement {
  const el = document.createElement("div");
  Object.assign(el, { setPointerCapture: vi.fn(), releasePointerCapture: vi.fn() });
  document.body.appendChild(el);
  return el;
}

afterEach(() => {
  document.body.innerHTML = "";
  vi.useRealTimers();
});

describe("attachPointerInput", () => {
  it("takes pointer capture at press and gives no hit when press and release are on different candidates", () => {
    const el = surface();
    const h = handlers();
    const detach = attachPointerInput(el, h, { holdRegime: false });
    el.dispatchEvent(pointer("pointerdown", 100.4, 100.6));
    expect(el.setPointerCapture).toHaveBeenCalledWith(7);
    expect(h.onPress).toHaveBeenCalledWith(expect.objectContaining({ x: 100, y: 101, index: 0 }));
    expect(h.onActivation).not.toHaveBeenCalled();
    el.dispatchEvent(pointer("pointerup", 150, 100));
    const activation = h.onActivation.mock.calls[0][0] as ActivationInput;
    expect(activation).toMatchObject({ x: 150, y: 100, pressIndex: 0, hitIndex: null, kind: "release" });
    detach();
  });

  it("gives the literal hit for press and release on the same candidate, and -1 for a press on a gap", () => {
    const el = surface();
    const h = handlers();
    const detach = attachPointerInput(el, h, { holdRegime: false });
    el.dispatchEvent(pointer("pointerdown", 98, 99));
    el.dispatchEvent(pointer("pointerup", 105, 101));
    expect(h.onActivation.mock.calls[0][0]).toMatchObject({ pressIndex: 0, hitIndex: 0 });
    el.dispatchEvent(pointer("pointerdown", 125, 125));
    el.dispatchEvent(pointer("pointerup", 150, 150));
    expect(h.onActivation.mock.calls[1][0]).toMatchObject({ pressIndex: -1, hitIndex: null });
    for (const [call] of h.onActivation.mock.calls) {
      expect(Number.isInteger(call.x) && Number.isInteger(call.y)).toBe(true);
    }
    detach();
  });

  it("cancels a press without an activation on pointercancel or lostpointercapture", () => {
    for (const type of ["pointercancel", "lostpointercapture"]) {
      const el = surface();
      const h = handlers();
      const detach = attachPointerInput(el, h, { holdRegime: false });
      el.dispatchEvent(pointer("pointerdown", 100, 100));
      el.dispatchEvent(pointer(type, 100, 100));
      el.dispatchEvent(pointer("pointerup", 100, 100));
      expect(h.onPressCancel).toHaveBeenCalledWith(type);
      expect(h.onActivation).not.toHaveBeenCalled();
      detach();
    }
  });

  it("uses the press as the activation in hold regimes and aborts the hold on lost input", () => {
    for (const type of ["pointercancel", "lostpointercapture"]) {
      const el = surface();
      const h = handlers();
      const detach = attachPointerInput(el, h, { holdRegime: true, holdMs: 700 });
      el.dispatchEvent(pointer("pointerdown", 150, 150));
      expect(h.onActivation).toHaveBeenCalledTimes(1);
      expect(h.onActivation.mock.calls[0][0]).toMatchObject({ pressIndex: 3, hitIndex: 3, kind: "press" });
      el.dispatchEvent(pointer(type, 150, 150));
      expect(h.onHoldAbort).toHaveBeenCalledWith(type);
      expect(h.onHoldComplete).not.toHaveBeenCalled();
      detach();
    }
  });

  it("completes a hold held for hold_ms and reports an early release", () => {
    vi.useFakeTimers();
    const el = surface();
    const h = handlers();
    const detach = attachPointerInput(el, h, { holdRegime: true, holdMs: 400 });
    el.dispatchEvent(pointer("pointerdown", 100, 100));
    vi.advanceTimersByTime(399);
    expect(h.onHoldComplete).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(h.onHoldComplete).toHaveBeenCalledTimes(1);
    el.dispatchEvent(pointer("pointerup", 100, 100));
    expect(h.onHoldRelease).not.toHaveBeenCalled();

    el.dispatchEvent(pointer("pointerdown", 100, 100));
    vi.advanceTimersByTime(100);
    el.dispatchEvent(pointer("pointerup", 100, 100));
    expect(h.onHoldRelease).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1000);
    expect(h.onHoldComplete).toHaveBeenCalledTimes(1);
    detach();
  });

  it("interrupts on window blur and a hidden document", () => {
    const el = surface();
    const h = handlers();
    const detach = attachPointerInput(el, h, { holdRegime: false });
    window.dispatchEvent(new Event("blur"));
    expect(h.onInterrupt).toHaveBeenCalledWith("blur");
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    document.dispatchEvent(new Event("visibilitychange"));
    expect(h.onInterrupt).toHaveBeenCalledWith("hidden");
    visibility.mockRestore();
    detach();
    window.dispatchEvent(new Event("blur"));
    expect(h.onInterrupt).toHaveBeenCalledTimes(2);
  });
});

describe("keyboard input", () => {
  const origin = { x: 160, y: 160 };

  function keyHandlers(focus: { value: number | null }) {
    return {
      rects: () => RECTS,
      focused: () => focus.value,
      setFocus: vi.fn((i: number) => {
        focus.value = i;
      }),
      onPress: vi.fn(),
      onActivation: vi.fn(),
      onHoldRelease: vi.fn(),
      onHoldComplete: vi.fn(),
      onInterrupt: vi.fn(),
      onKey: vi.fn(),
    };
  }

  it("starts focus on the candidate nearest the origin, or the first menu item", () => {
    expect(initialFocus(RECTS, origin, "pointing")).toBe(3);
    expect(initialFocus(RECTS, { x: 125, y: 100 }, "pointing")).toBe(0);
    expect(initialFocus(RECTS, origin, "menu")).toBe(0);
    const el = surface();
    const focus = { value: null as number | null };
    const h = keyHandlers(focus);
    const detach = attachKeyboardInput(el, h, { holdRegime: false, origin, task: "pointing" });
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" }));
    expect(focus.value).toBe(3);
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" }));
    expect(focus.value).toBe(2);
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowUp" }));
    expect(focus.value).toBe(0);
    detach();
  });

  it("moves focus along rows and columns", () => {
    expect(moveFocus(RECTS, 0, "ArrowRight")).toBe(1);
    expect(moveFocus(RECTS, 0, "ArrowDown")).toBe(2);
    expect(moveFocus(RECTS, 0, "ArrowLeft")).toBe(0);
    expect(moveFocus(RECTS, 3, "ArrowUp")).toBe(1);
  });

  it("activates the focused candidate at its keyboardEndpoint and ignores key repeats", () => {
    const el = surface();
    const focus = { value: 1 as number | null };
    const h = keyHandlers(focus);
    const rects = [...RECTS];
    rects[1] = { cx: 150.5, cy: 99.5, w: 40, h: 40 };
    const detach = attachKeyboardInput(el, { ...h, rects: () => rects }, { holdRegime: false });
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", repeat: true }));
    el.dispatchEvent(new KeyboardEvent("keydown", { key: " ", repeat: true }));
    expect(h.onActivation).toHaveBeenCalledTimes(1);
    expect(h.onActivation.mock.calls[0][0]).toEqual({
      ...keyboardEndpoint(rects[1]),
      pressIndex: 1,
      hitIndex: 1,
      t: expect.any(Number),
      modality: "keyboard",
      kind: "key",
    });
    expect(h.onActivation.mock.calls[0][0]).toMatchObject({ x: 151, y: 100 });
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(h.onKey).toHaveBeenCalledWith("Escape");
    detach();
  });

  it("treats the keydown as the activation in hold regimes and an early keyup as a release", () => {
    const el = surface();
    const h = keyHandlers({ value: 0 });
    const detach = attachKeyboardInput(el, h, { holdRegime: true, holdMs: 700 });
    el.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
    el.dispatchEvent(new KeyboardEvent("keydown", { key: " ", repeat: true }));
    expect(h.onActivation).toHaveBeenCalledTimes(1);
    el.dispatchEvent(new KeyboardEvent("keyup", { key: " " }));
    expect(h.onHoldRelease).toHaveBeenCalledTimes(1);
    expect(h.onHoldComplete).not.toHaveBeenCalled();
    detach();
  });

  it("attaches no pointer listener in a keyboard session", () => {
    const el = surface();
    const added = vi.spyOn(el, "addEventListener");
    const detach = attachKeyboardInput(el, keyHandlers({ value: 0 }), { holdRegime: false });
    expect(added.mock.calls.map(([type]) => type).filter((type) => type.startsWith("pointer"))).toEqual([]);
    expect(added.mock.calls.map(([type]) => type).sort()).toEqual(["keydown", "keyup"]);
    detach();
  });
});

describe("attachHandPointerInput", () => {
  const RECT = { width: 640, height: 420 };
  let t = 0;
  const now = () => t;

  function frame(pinchClosed: boolean, dt = 33, nx = 100 / 640, ny = 100 / 420) {
    t += dt;
    publishStudyPointerFrame({ nx, ny, pinchClosed, quality: 0.9, t });
  }
  function frames(n: number, pinchClosed: boolean) {
    for (let i = 0; i < n; i += 1) frame(pinchClosed);
  }
  const handHandlers = handlers;

  beforeEach(() => {
    t = 0;
    resetStudyPointer();
    requestStudyPointer(true);
  });

  it("accepts no press at ready alone, only after the pinch was open for 8 fresh frames", () => {
    const h = handHandlers();
    const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime: false, now });
    frames(8, false);
    let ready = false;
    studyPointer.subscribe((s) => (ready = s.ready))();
    expect(ready).toBe(true);
    frame(true);
    frame(false);
    expect(h.onPress).not.toHaveBeenCalled();
    frames(8, false);
    frame(true);
    expect(h.onPress).toHaveBeenCalledTimes(1);
    frame(false);
    expect(h.onActivation).toHaveBeenCalledTimes(1);
    expect(h.onActivation.mock.calls[0][0]).toMatchObject({
      x: 100,
      y: 100,
      pressIndex: 0,
      hitIndex: 0,
      kind: "release",
    });
    detach();
  });

  it("never turns a pinch held through a hand loss and reacquisition into a press, in either regime", () => {
    for (const holdRegime of [false, true]) {
      t = 0;
      resetStudyPointer();
      requestStudyPointer(true);
      const h = handHandlers();
      const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime, holdMs: 400, now });
      frames(16, false);
      frame(true);
      expect(h.onPress).toHaveBeenCalledTimes(1);
      reportStudyPointerHandLost();
      if (holdRegime) {
        expect(h.onHoldAbort).toHaveBeenCalledWith("hand_lost");
        expect(h.onActivation).toHaveBeenCalledTimes(1);
      } else {
        expect(h.onPressCancel).toHaveBeenCalledWith("hand_lost");
        expect(h.onActivation).not.toHaveBeenCalled();
      }
      const activationsBefore = h.onActivation.mock.calls.length;
      frames(30, true);
      expect(h.onPress).toHaveBeenCalledTimes(1);
      expect(h.onActivation).toHaveBeenCalledTimes(activationsBefore);
      expect(h.onHoldComplete).not.toHaveBeenCalled();
      frames(8, false);
      frame(true);
      expect(h.onPress).toHaveBeenCalledTimes(2);
      detach();
    }
  });

  it("does the same for a pinch closed during a study pause", () => {
    const h = handHandlers();
    const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime: false, now });
    frames(16, false);
    requestStudyPointer(false);
    frames(5, true);
    requestStudyPointer(true);
    frames(30, true);
    expect(h.onPress).not.toHaveBeenCalled();
    frames(8, false);
    frame(true);
    expect(h.onPress).toHaveBeenCalledTimes(1);
    detach();
  });

  it("aborts a hold on a frame older than 100 ms and completes it on fresh frames", () => {
    const h = handHandlers();
    const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime: true, holdMs: 400, now });
    frames(16, false);
    frame(true);
    expect(h.onActivation.mock.calls[0][0]).toMatchObject({ kind: "press", hitIndex: 0 });
    frame(true, 150);
    expect(h.onHoldAbort).toHaveBeenCalledWith("stale_frame");
    frame(false);
    frame(true);
    frames(13, true);
    expect(h.onHoldComplete).toHaveBeenCalledTimes(1);
    detach();
  });

  it("cancels a non-hold press on a frame gap without an activation", () => {
    const h = handHandlers();
    const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime: false, now });
    frames(16, false);
    frame(true);
    frame(true, 140);
    frame(false);
    expect(h.onPressCancel).toHaveBeenCalledWith("stale_frame");
    expect(h.onActivation).not.toHaveBeenCalled();
    detach();
  });

  it("maps the normalised pointer onto integer surface pixels", () => {
    const h = handHandlers();
    const detach = attachHandPointerInput(RECT, studyPointer, h, { holdRegime: false, now });
    frame(false, 33, 0.33333, 0.77777);
    const [point] = h.onMove.mock.calls.at(-1)!;
    expect(point).toEqual({ x: Math.round(0.33333 * 640), y: Math.round(0.77777 * 420) });
    detach();
  });
});

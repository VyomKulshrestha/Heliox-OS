import { get, writable, type Readable } from "svelte/store";

/**
 * In-window hand pointer for the Selection Study (spec section 16.3). GestureControl writes filtered index-tip
 * coordinates and the measured pinch here while a hand-pointer study runs; it never moves or clicks the OS cursor in
 * this mode. This store alone decides when a press can start: after every hand acquisition the hand must be verified
 * for 8 frames (`ready`) and the pinch then seen open on 8 further consecutive fresh frames (`pressArmed`). Until
 * then the published pinch stays open, so a pinch held closed through a hand loss or a study pause never becomes a
 * press.
 */

export const HAND_ACQUIRE_FRAMES = 8;
/** A frame more than this long after the previous one is not fresh. */
export const FRESH_FRAME_GAP_MS = 100;

export interface StudyPointerState {
  requested: boolean;
  available: boolean;
  handPresent: boolean;
  ready: boolean;
  pressArmed: boolean;
  nx: number;
  ny: number;
  pinchClosed: boolean;
  quality: number;
  trackingLosses: number;
  palmExits: number;
  t: number;
}

const INITIAL: StudyPointerState = {
  requested: false,
  available: false,
  handPresent: false,
  ready: false,
  pressArmed: false,
  nx: 0.5,
  ny: 0.5,
  pinchClosed: false,
  quality: 0,
  trackingLosses: 0,
  palmExits: 0,
  t: 0,
};

const store = writable<StudyPointerState>({ ...INITIAL });

/** Consecutive frames since the last loss or reset (sets `ready` at the eighth). */
let verifiedFrames = 0;
/** Consecutive fresh frames with the measured pinch open after `ready` (sets `pressArmed` at the eighth). */
let openFrames = 0;
let lastFrameT: number | null = null;

function clearCounters(): void {
  verifiedFrames = 0;
  openFrames = 0;
  lastFrameT = null;
}

export const studyPointer: Readable<StudyPointerState> = { subscribe: store.subscribe };

/** Set by the study while a hand-pointer session runs and is not paused. Both values reset the press gate. */
export function requestStudyPointer(on: boolean): void {
  clearCounters();
  store.update((state) => ({ ...state, requested: on, ready: false, pressArmed: false, pinchClosed: false }));
}

/** Set by GestureControl when the camera and hand model start or stop. */
export function setStudyPointerAvailable(available: boolean): void {
  store.update((state) => ({ ...state, available }));
}

export function publishStudyPointerFrame(frame: {
  nx: number;
  ny: number;
  pinchClosed: boolean;
  quality: number;
  t: number;
}): void {
  const state = get(store);
  const fresh = lastFrameT !== null && frame.t - lastFrameT <= FRESH_FRAME_GAP_MS;
  lastFrameT = frame.t;
  let ready = state.ready;
  let pressArmed = state.pressArmed;

  if (!ready) {
    verifiedFrames += 1;
    if (verifiedFrames >= HAND_ACQUIRE_FRAMES) {
      ready = true;
      openFrames = 0;
    }
  } else if (!pressArmed) {
    openFrames = fresh && !frame.pinchClosed ? openFrames + 1 : 0;
    if (openFrames >= HAND_ACQUIRE_FRAMES) pressArmed = true;
  }

  store.set({
    ...state,
    handPresent: true,
    ready,
    pressArmed,
    nx: frame.nx,
    ny: frame.ny,
    // The measured pinch is published only once a press is armed; at the arming frame it is open.
    pinchClosed: pressArmed ? frame.pinchClosed : false,
    quality: frame.quality,
    t: frame.t,
  });
}

export function reportStudyPointerHandLost(): void {
  clearCounters();
  store.update((state) => ({
    ...state,
    handPresent: false,
    ready: false,
    pressArmed: false,
    pinchClosed: false,
    trackingLosses: state.requested ? state.trackingLosses + 1 : state.trackingLosses,
  }));
}

export function reportStudyPointerPalmExit(): void {
  store.update((state) => (state.requested ? { ...state, palmExits: state.palmExits + 1 } : state));
}

/** Clears the frame counters and the press gate; keeps `requested`, `available` and the loss counters. */
export function resetStudyPointer(): void {
  clearCounters();
  store.update((state) => ({ ...state, handPresent: false, ready: false, pressArmed: false, pinchClosed: false }));
}

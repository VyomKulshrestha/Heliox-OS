import { render } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MouthFrame } from "../face/mouthPlayer";

// The face component with the daemon's notices fed in by hand and a clock the test moves (performance.now, Date and
// the frame timer). The renderer is a stand-in that keeps every frame the component asks it to draw.
const mocks = vi.hoisted(() => ({
  listeners: [] as ((method: string, params: unknown) => void)[],
  draws: [] as { now: number; frame: MouthFrame }[],
}));

vi.mock("../api/daemon", () => ({
  onNotification: vi.fn((listener: (method: string, params: unknown) => void) => mocks.listeners.push(listener)),
  offNotification: vi.fn(),
  onConnectionState: vi.fn(),
  offConnectionState: vi.fn(),
}));

vi.mock("../stores/session", async () => {
  const { writable } = await import("svelte/store");
  return { session: writable({ interaction: null, loading: false }) };
});

vi.mock("../face/pack", () => ({
  readManifest: vi.fn(async () => ({
    format: 1,
    id: "test",
    name: "Test",
    credit: "A test face.",
    size: [512, 512],
    idle: { frames: ["idle.webp"] },
    mouth: { shapes: { sil: "sil.webp", PP: "pp.webp", aa: "aa.webp", E: "e.webp", O: "o.webp" } },
  })),
  packShapes: () => ["sil", "PP", "aa", "E", "O"],
}));

vi.mock("../face/faceRenderer", () => ({
  FaceRenderer: class {
    reducedMotion = false;
    async load() {}
    setColours() {}
    draw(now: number, _mode: string, frame: MouthFrame) {
      mocks.draws.push({ now, frame });
    }
    animating() {
      return false;
    }
    untilBlink() {
      return Infinity;
    }
  },
}));

import Face from "./Face.svelte";

/** A reply of 2.4 s as the daemon describes it: words, then a pause, then more words, loud throughout the words. */
function piece(reply: number) {
  const visemes: [string, number, number][] = [
    ["aa", 0, 280],
    ["PP", 280, 360],
    ["E", 360, 700],
    ["O", 700, 1000],
    ["sil", 1000, 1300],
    ["aa", 1300, 1700],
    ["E", 1700, 2400],
  ];
  const envelope = Array.from({ length: 60 }, (_, i) => (i * 40 >= 1000 && i * 40 < 1300 ? 0 : 12));
  return {
    reply,
    index: 0,
    seconds: 2.4,
    lead_ms: 0,
    sent_ms: Date.now(),
    voice: "local",
    mouth: "visemes",
    visemes,
    envelope,
    envelope_rate: 25,
  };
}

function say(method: string, params: unknown) {
  for (const listener of mocks.listeners) listener(method, params);
}

describe("Face: a cut", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance", "Date"] });
    mocks.listeners.length = 0;
    mocks.draws.length = 0;
  });
  afterEach(() => vi.useRealTimers());

  // Three places in the reply: wide open on a vowel, on a lip closure, and in the pause between phrases.
  it.each([
    ["mid-vowel", 520],
    ["on a lip closure", 300],
    ["in a pause", 1150],
  ])("leaves the mouth at rest at once, cut %s", async (_where, cutAt) => {
    render(Face, { pack: "test" });
    await vi.advanceTimersByTimeAsync(10); // the manifest and the stand-in renderer
    expect(mocks.draws.length).toBeGreaterThan(0);
    const reply = 7;
    say("voice_piece", piece(reply));
    await vi.advanceTimersByTimeAsync(cutAt);
    const before = mocks.draws.at(-1)!;
    expect(before.frame.speaking).toBe(true);
    const drawnBefore = mocks.draws.length;
    const cutNow = performance.now();
    say("voice_cut", { reply, index: 0, sent_ms: Date.now() });
    // Drawn on the spot, not at the next scheduled frame: the mouth at rest, the clock not moved.
    expect(mocks.draws.length).toBe(drawnBefore + 1);
    const at = mocks.draws.at(-1)!;
    expect(at.now).toBe(cutNow);
    expect(at.frame).toMatchObject({ speaking: false, shape: "sil", open: 0, weight: 0, nextWeight: 0 });
    // And it stays there for the rest of what the reply would have said.
    await vi.advanceTimersByTimeAsync(2500);
    const after = mocks.draws.slice(drawnBefore);
    expect(after.length).toBeGreaterThan(1);
    expect(after.every((d) => !d.frame.speaking && (d.frame.weight ?? 0) === 0 && d.frame.open === 0)).toBe(true);
  });

  it("is open before the cut at the vowel (the test's reply moves the mouth)", async () => {
    render(Face, { pack: "test" });
    await vi.advanceTimersByTimeAsync(10);
    say("voice_piece", piece(8));
    await vi.advanceTimersByTimeAsync(520);
    expect(mocks.draws.at(-1)!.frame.open).toBeGreaterThan(0.5);
  });
});

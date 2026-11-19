import { describe, expect, it } from "vitest";
import {
  calmShape,
  MouthPlayer,
  pieceStart,
  soundGate,
  stressPeaks,
  strengthOf,
  type MouthFrame,
  type Shape,
  type VoicePieceNotice,
} from "./mouthPlayer";

const LOUD = 13;

/** A piece as the daemon sends it: "map" (m, a, p) in 600 ms, first sample 50 ms after it was sent, loud throughout
 * the vowel. Loudness comes 25 values a second (40 ms each). */
function piece(over: Partial<VoicePieceNotice> = {}): VoicePieceNotice {
  return {
    reply: 1,
    index: 0,
    seconds: 0.6,
    lead_ms: 50,
    sent_ms: 10_000,
    mouth: "visemes",
    visemes: [
      ["PP", 0, 100],
      ["aa", 100, 400],
      ["PP", 400, 500],
      ["sil", 500, 600],
    ],
    envelope: [6, 6, 6, LOUD, LOUD, LOUD, LOUD, LOUD, LOUD, LOUD, 5, 5, 0, 0, 0],
    envelope_rate: 25,
    ...over,
  };
}

// The clock is the test's own: the notice arrives at 1000 ms, with Date.now() 2 ms after it was sent.
const RECEIVED = 1000;
const WALL = 10_002;
const START = 1048;

function weightOf(frame: MouthFrame, shape: Shape): number {
  if (frame.shape === shape) return frame.weight ?? 0;
  if (frame.next === shape) return frame.nextWeight ?? 0;
  return 0;
}

function played(notice: VoicePieceNotice, shapes?: string[]): MouthPlayer {
  const player = new MouthPlayer();
  if (shapes) player.setShapes(shapes);
  player.piece(notice, RECEIVED, WALL);
  return player;
}

describe("mouth player: timing", () => {
  it("starts a piece when its first sample is heard: receipt plus lead minus the transit already spent", () => {
    expect(pieceStart(piece(), RECEIVED, WALL)).toBe(1048);
    expect(pieceStart(piece({ sent_ms: undefined }), RECEIVED, WALL)).toBe(1050);
    expect(pieceStart(piece({ sent_ms: WALL + 5000 }), RECEIVED, WALL)).toBe(1050); // a clock not to trust
    expect(pieceStart(piece({ lead_ms: null }), RECEIVED, WALL)).toBe(998);
  });

  it("forms the vowel while it sounds and closes the lips for p, b and m fully and on time", () => {
    const player = played(piece());
    expect(player.sample(START - 10).speaking).toBe(false); // not heard yet
    expect(player.sample(START + 10)).toMatchObject({ speaking: true, open: 0 }); // the m: closed
    expect(weightOf(player.sample(START + 250), "aa")).toBeGreaterThan(0.9);
    // The p at 400 ms: the lips are together as it starts, and the vowel is all but gone.
    const closure = player.sample(START + 405);
    expect(closure.shape).toBe("PP");
    expect(closure.weight).toBeGreaterThan(0.85);
    expect(weightOf(closure, "aa")).toBeLessThan(0.2);
    expect(player.sample(START + 600).speaking).toBe(false); // ended by its own length
  });

  it("presses the lips for a closure, then lets a long one rest closed instead of pursed", () => {
    const player = played(
      piece({
        seconds: 0.8,
        visemes: [
          ["aa", 0, 200],
          ["PP", 200, 800],
        ],
        envelope: Array(20).fill(LOUD),
      }),
    );
    expect(player.sample(START + 230).shape).toBe("PP");
    const later = player.sample(START + 600);
    expect(weightOf(later, "PP")).toBeLessThan(0.1);
    expect(later.open).toBe(0);
  });

  it("pulls only part of the way to a wide shape that lasts a moment, and never reaches it", () => {
    const short = played(
      piece({
        visemes: [
          ["sil", 0, 100],
          ["aa", 100, 140],
          ["sil", 140, 600],
        ],
      }),
    );
    let highest = 0;
    for (let t = 0; t < 600; t += 4) highest = Math.max(highest, weightOf(short.sample(START + t), "aa"));
    expect(highest).toBeGreaterThan(0.1);
    expect(highest).toBeLessThan(0.75);
  });

  it("opens less for a quiet syllable and relaxes as the sound dies", () => {
    const quiet = played(piece({ envelope: Array(15).fill(3) }));
    const loud = played(piece({ envelope: Array(15).fill(LOUD) }));
    expect(weightOf(quiet.sample(START + 300), "aa")).toBeLessThan(weightOf(loud.sample(START + 300), "aa") - 0.3);
    expect(strengthOf(0)).toBeLessThan(strengthOf(15));
    // "too": U for 500 ms, but the sound is gone after 200 ms.
    const fading = played(
      piece({
        visemes: [
          ["U", 0, 500],
          ["sil", 500, 600],
        ],
        envelope: [12, 13, 12, 9, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      }),
    );
    expect(weightOf(fading.sample(START + 120), "U")).toBeGreaterThan(0.7);
    expect(weightOf(fading.sample(START + 350), "U")).toBeLessThan(0.1);
  });

  it("closes as a quiet stretch begins and reopens just before the sound returns, leaving a short stop alone", () => {
    const envelope = [10, 10, 0, 0, 0, 0, 10, 10]; // quiet from 80 to 240 ms
    expect(soundGate(envelope, 25, 40)).toBe(1);
    expect(soundGate(envelope, 25, 80)).toBeCloseTo(0.75, 5); // closing from 70 ms, at rest from 110 ms
    expect(soundGate(envelope, 25, 150)).toBe(0);
    expect(soundGate(envelope, 25, 235)).toBeCloseTo(0.5, 5); // reopening over the last 10 ms
    expect(soundGate(envelope, 25, 240)).toBe(1);
    expect(soundGate([10, 0, 10, 10], 25, 60)).toBe(1); // 40 ms: a stop closure inside a word
  });

  it("closes the mouth on the next frame after a cut, and ignores the rest of that reply", () => {
    const player = played(piece());
    expect(player.sample(START + 250).open).toBeGreaterThan(0.8);
    player.cut(1);
    expect(player.sample(START + 251)).toMatchObject({ speaking: false, shape: "sil", open: 0, weight: 0 });
    player.piece(piece({ index: 1 }), RECEIVED + 300, WALL + 300); // handed over before the cut, arrived after
    expect(player.sample(START + 400).speaking).toBe(false);
    expect(player.busy(START + 400)).toBe(false);
  });

  it("stays idle when a notice arrives too late to be heard, and never sticks open without an end notice", () => {
    const player = new MouthPlayer();
    player.piece(piece({ sent_ms: WALL - 900 }), RECEIVED, WALL); // 900 ms in transit: the 600 ms piece is over
    expect(player.sample(RECEIVED).speaking).toBe(false);
    const fresh = played(piece()); // no voice_end ever comes
    expect(fresh.sample(START + 5000)).toMatchObject({ speaking: false, open: 0 });
    expect(fresh.busy(START + 5000)).toBe(false);
    expect(new MouthPlayer().sample(5000).speaking).toBe(false); // no notice at all
  });

  it("drops an older reply's pieces when a newer reply starts", () => {
    const player = new MouthPlayer();
    player.piece(piece({ reply: 1, seconds: 10, visemes: [["aa", 0, 10_000]] }), RECEIVED, WALL);
    player.piece(piece({ reply: 2, lead_ms: 2000 }), RECEIVED + 100, WALL + 100);
    expect(player.sample(RECEIVED + 500).speaking).toBe(false);
    player.piece(piece({ reply: 1, index: 3 }), RECEIVED + 200, WALL + 200); // a late piece of the old reply
    expect(player.sample(RECEIVED + 300).speaking).toBe(false);
  });
});

describe("mouth player: shapes a pack does not have", () => {
  const has = (list: Shape[]) => (s: Shape) => list.includes(s);

  it("keeps a consonant without a picture near the vowel before it, or a small neutral opening", () => {
    const pack = has(["sil", "PP", "aa", "E", "O"]);
    expect(calmShape("DD", pack, "aa")).toEqual(["aa", 0.55]);
    expect(calmShape("nn", pack, null)).toEqual(["E", 0.3]);
    expect(calmShape("kk", has(["sil", "PP", "aa"]), null)).toEqual(["sil", 0]);
    expect(calmShape("O", pack, null)).toEqual(["O", 1]);
    expect(calmShape("U", pack, null)).toEqual(["O", 0.9]); // the nearest vowel it has
    expect(calmShape("FF", pack, null)).toEqual(["PP", 1]); // a lip shape stays a lip shape
    expect(calmShape("TH", has(["sil", "TH", "aa"]), "aa")[1]).toBeLessThan(1); // its own consonant, held back
    const calm = has(["sil", "FF", "SS", "nn", "aa", "E", "I", "O", "U"]); // a pack with a calm small opening, nn
    expect(calmShape("DD", calm, "aa")).toEqual(["nn", 0.7]); // t, d and th share it
    expect(calmShape("TH", calm, null)).toEqual(["nn", 0.7]);
    expect(calmShape("CH", calm, null)).toEqual(["SS", 0.7]);
    expect(calmShape("kk", calm, "O")).toEqual(["O", 0.55]); // k keeps the vowel's shape
    expect(calmShape("PP", calm, "aa")).toEqual(["sil", 0]); // no lip picture: the closed mouth
  });

  it("never jumps to an unrelated wide shape on a consonant inside a word", () => {
    const player = played(
      piece({
        visemes: [
          ["O", 0, 200],
          ["DD", 200, 300],
          ["O", 300, 500],
          ["sil", 500, 600],
        ],
      }),
      ["sil", "PP", "aa", "E", "O"],
    );
    for (let t = 0; t < 560; t += 8) expect(weightOf(player.sample(START + t), "aa")).toBe(0);
  });
});

describe("mouth player: loudness alone, the system voice, reduced motion", () => {
  it("drives the mouth from the loudness when there is no timeline", () => {
    const levels = [0, 0, 15, 15, 15, 15, 0, 0, 0, 0];
    const player = new MouthPlayer();
    player.piece(piece({ mouth: "envelope", visemes: undefined, envelope: levels, seconds: 0.4 }), 0, 10_000);
    const start = 50;
    expect(player.sample(start + 10)).toMatchObject({ source: "envelope", open: 0 });
    const loud = player.sample(start + 160);
    expect(loud.shape).toBe("aa");
    expect(loud.open).toBeGreaterThan(0.8);
    expect(player.sample(start + 380).open).toBeLessThan(0.2);
  });

  it("speaks with a still mouth for the system voice until its reply ends", () => {
    const player = new MouthPlayer();
    player.piece(piece({ mouth: "none", seconds: null, lead_ms: null, visemes: undefined }), 0, 10_000);
    expect(player.sample(500)).toMatchObject({ speaking: true, open: 0, shape: "sil" });
    player.end(1);
    expect(player.sample(600).speaking).toBe(false);
  });

  it("only opens or closes under reduced motion, with a still head", () => {
    const player = played(piece());
    expect(player.sample(START + 250, true)).toMatchObject({ shape: "aa", mix: 0, open: 1, nod: 0, sway: 0 });
    expect(player.sample(START + 20, true)).toMatchObject({ shape: "sil", open: 0 });
  });
});

describe("mouth player: the head and the pauses", () => {
  it("nods just before the loudest syllables and leans to one side for two of them, then the other", () => {
    const levels = [2, 4, 14, 6, 3, 2, 3, 5, 6, 15, 7, 3, 2, 2, 2, 2];
    const peaks = stressPeaks(levels, 25);
    expect(peaks.map(([at]) => at)).toEqual([100, 380]);
    const player = played(piece({ seconds: 0.64, envelope: levels, visemes: [["E", 0, 640]] }));
    const before = player.sample(START + 100 - 60);
    expect(before.nod).toBeGreaterThan(0.8);
    expect(before.sway).toBeGreaterThan(0.55);
    expect(player.sample(START + 380 - 60).sway).toBeGreaterThan(0.55); // the second: the same side
    expect(player.sample(START + 600).nod).toBeLessThan(0.05);
  });

  it("says how long the sound will now rest, so blinks fall between phrases", () => {
    const levels = [12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 12];
    const player = played(piece({ envelope: levels, visemes: [["E", 0, 600]] }));
    expect(player.sample(START + 60).pause).toBe(0);
    expect(player.sample(START + 200).pause).toBe(200); // quiet from 120 to 400 ms
  });
});

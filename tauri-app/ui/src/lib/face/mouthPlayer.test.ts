import { describe, expect, it } from "vitest";
import { CROSSFADE_MS, MouthPlayer, pieceStart, type VoicePieceNotice } from "./mouthPlayer";

/** A piece as the daemon sends it: "map" (m, a, p) in 600 ms, first sample 50 ms after it was sent. */
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
    ...over,
  };
}

// The clock is the test's own: the notice arrives at 1000 ms, with Date.now() 2 ms after it was sent.
const RECEIVED = 1000;
const WALL = 10_002;

describe("mouth player", () => {
  it("starts a piece when its first sample is heard: receipt plus lead minus the transit already spent", () => {
    expect(pieceStart(piece(), RECEIVED, WALL)).toBe(1048);
    expect(pieceStart(piece({ sent_ms: undefined }), RECEIVED, WALL)).toBe(1050);
    expect(pieceStart(piece({ sent_ms: WALL + 5000 }), RECEIVED, WALL)).toBe(1050); // a clock not to trust
    expect(pieceStart(piece({ lead_ms: null }), RECEIVED, WALL)).toBe(998);
  });

  it("shows the right shape at each moment of the timeline", () => {
    const player = new MouthPlayer();
    player.piece(piece(), RECEIVED, WALL);
    const start = 1048;
    expect(player.sample(start - 10).speaking).toBe(false); // not heard yet
    expect(player.sample(start + 10)).toMatchObject({ speaking: true, shape: "PP", mix: 0 });
    expect(player.sample(start + 250)).toMatchObject({ shape: "aa", next: "aa", mix: 0, open: 1 });
    expect(player.sample(start + 450)).toMatchObject({ shape: "PP", open: 0 });
    expect(player.sample(start + 590).shape).toBe("sil");
    expect(player.sample(start + 600).speaking).toBe(false); // ended by its own length
  });

  it("blends neighbouring shapes across the change, half before and half after it", () => {
    const player = new MouthPlayer();
    player.piece(piece(), RECEIVED, WALL);
    const change = 1048 + 100;
    const before = player.sample(change - CROSSFADE_MS / 4);
    expect(before.shape).toBe("PP");
    expect(before.next).toBe("aa");
    expect(before.mix).toBeCloseTo(0.25, 5);
    const at = player.sample(change);
    expect(at).toMatchObject({ shape: "PP", next: "aa" });
    expect(at.mix).toBeCloseTo(0.5, 5);
    expect(player.sample(change + CROSSFADE_MS / 4).mix).toBeCloseTo(0.75, 5);
    expect(player.sample(change + CROSSFADE_MS).mix).toBe(0);
  });

  it("closes the mouth on the next frame after a cut, and ignores the rest of that reply", () => {
    const player = new MouthPlayer();
    player.piece(piece(), RECEIVED, WALL);
    expect(player.sample(1048 + 250).open).toBe(1);
    player.cut(1);
    expect(player.sample(1048 + 251)).toMatchObject({ speaking: false, shape: "sil", open: 0 });
    player.piece(piece({ index: 1 }), RECEIVED + 300, WALL + 300); // handed over before the cut, arrived after
    expect(player.sample(1048 + 400).speaking).toBe(false);
    expect(player.busy(1048 + 400)).toBe(false);
  });

  it("stays idle when a notice arrives too late to be heard, and never sticks open without an end notice", () => {
    const player = new MouthPlayer();
    player.piece(piece({ sent_ms: WALL - 900 }), RECEIVED, WALL); // 900 ms in transit: the 600 ms piece is over
    expect(player.sample(RECEIVED).speaking).toBe(false);
    const fresh = new MouthPlayer();
    fresh.piece(piece(), RECEIVED, WALL); // no voice_end ever comes
    expect(fresh.sample(1048 + 5000)).toMatchObject({ speaking: false, open: 0 });
    expect(fresh.busy(1048 + 5000)).toBe(false);
    expect(new MouthPlayer().sample(5000).speaking).toBe(false); // no notice at all
  });

  it("drives the mouth from the loudness when there is no timeline", () => {
    const player = new MouthPlayer();
    const levels = [0, 15, 15, 0, 6];
    player.piece(piece({ mouth: "envelope", visemes: undefined, envelope: levels, envelope_rate: 25 }), 0, 10_000);
    const start = 50;
    expect(player.sample(start + 5)).toMatchObject({ source: "envelope", shape: "sil" });
    expect(player.sample(start + 45)).toMatchObject({ shape: "E", next: "aa", mix: 1, open: 1 });
    expect(player.sample(start + 125).open).toBeLessThan(0.1);
  });

  it("speaks with a still mouth for the system voice until its reply ends", () => {
    const player = new MouthPlayer();
    player.piece(piece({ mouth: "none", seconds: null, lead_ms: null, visemes: undefined }), 0, 10_000);
    expect(player.sample(500)).toMatchObject({ speaking: true, open: 0, shape: "sil" });
    player.end(1);
    expect(player.sample(600).speaking).toBe(false);
  });

  it("only opens or closes under reduced motion", () => {
    const player = new MouthPlayer();
    player.piece(piece(), RECEIVED, WALL);
    expect(player.sample(1048 + 150, true)).toMatchObject({ shape: "aa", mix: 0, open: 1 });
    expect(player.sample(1048 + 50, true)).toMatchObject({ shape: "sil", mix: 0, open: 0 });
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

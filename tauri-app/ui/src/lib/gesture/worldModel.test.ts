import { describe, it, expect } from "vitest";
import {
  toWristRelative3D,
  handSize3D,
  pinchDistance3D,
  detectPushPull3D,
  WorldModelFilterBank,
  type Landmark,
} from "./worldModel";

// A synthetic 21-landmark worldLandmarks reading in real-world meters,
// hand-center-relative (not wrist-relative) as HandLandmarker actually
// returns them -- deliberately NOT centered on the wrist, so
// toWristRelative3D()'s re-anchoring has something real to do.
function worldLandmarksHandCentered(): Landmark[] {
  const lm: Landmark[] = new Array(21).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));
  lm[0] = { x: 0.01, y: 0.06, z: 0.002 }; // wrist, offset from the hand-center origin
  lm[4] = { x: -0.03, y: 0.01, z: -0.01 }; // thumb tip
  lm[5] = { x: 0.02, y: -0.02, z: 0.0 }; // index MCP
  lm[8] = { x: 0.025, y: -0.07, z: -0.01 }; // index tip
  lm[9] = { x: 0.0, y: -0.02, z: 0.0 }; // middle MCP
  return lm;
}

describe("toWristRelative3D", () => {
  it("maps the wrist to the origin", () => {
    const out = toWristRelative3D(worldLandmarksHandCentered());
    expect(out[0].x).toBeCloseTo(0, 12);
    expect(out[0].y).toBeCloseTo(0, 12);
    expect(out[0].z ?? 0).toBeCloseTo(0, 12);
  });

  it("preserves relative offsets between non-wrist landmarks", () => {
    const raw = worldLandmarksHandCentered();
    const out = toWristRelative3D(raw);
    // Translation-invariant: the vector from index MCP to index tip is
    // unchanged by re-anchoring the whole set to the wrist.
    const rawDelta = { x: raw[8].x - raw[5].x, y: raw[8].y - raw[5].y, z: (raw[8].z ?? 0) - (raw[5].z ?? 0) };
    const outDelta = { x: out[8].x - out[5].x, y: out[8].y - out[5].y, z: (out[8].z ?? 0) - (out[5].z ?? 0) };
    expect(outDelta.x).toBeCloseTo(rawDelta.x, 12);
    expect(outDelta.y).toBeCloseTo(rawDelta.y, 12);
    expect(outDelta.z).toBeCloseTo(rawDelta.z, 12);
  });
});

describe("handSize3D / pinchDistance3D", () => {
  it("computes a positive metric hand size from wrist-relative landmarks", () => {
    const wristRelative = toWristRelative3D(worldLandmarksHandCentered());
    expect(handSize3D(wristRelative)).toBeGreaterThan(0);
  });

  it("computes a positive metric pinch distance", () => {
    const wristRelative = toWristRelative3D(worldLandmarksHandCentered());
    const raw = worldLandmarksHandCentered();
    const expected = Math.sqrt(
      (raw[4].x - raw[8].x) ** 2 + (raw[4].y - raw[8].y) ** 2 + ((raw[4].z ?? 0) - (raw[8].z ?? 0)) ** 2
    );
    expect(pinchDistance3D(wristRelative)).toBeCloseTo(expected, 12);
  });

  it("reports a smaller pinch distance for a closed pinch than an open hand", () => {
    const open = toWristRelative3D(worldLandmarksHandCentered());
    const closed = open.map((lm) => ({ ...lm }));
    closed[4] = { ...closed[8] }; // thumb tip collapsed onto index tip
    expect(pinchDistance3D(closed)).toBeLessThan(pinchDistance3D(open));
    expect(pinchDistance3D(closed)).toBeCloseTo(0, 12);
  });
});

describe("detectPushPull3D", () => {
  it("returns null with fewer than two readings", () => {
    expect(detectPushPull3D([])).toBeNull();
    expect(detectPushPull3D([{ x: 0, y: 0, z: 0 }])).toBeNull();
  });

  it("returns null when depth change stays within the threshold", () => {
    const history: Landmark[] = [
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 0.01 },
      { x: 0, y: 0, z: 0.02 },
    ];
    expect(detectPushPull3D(history)).toBeNull();
  });

  it("detects push for a sustained negative-z (toward camera) motion", () => {
    const history: Landmark[] = [
      { x: 0, y: 0, z: 0.1 },
      { x: 0, y: 0, z: 0.05 },
      { x: 0, y: 0, z: 0.02 }, // -0.08m from first reading
    ];
    expect(detectPushPull3D(history)).toBe("push");
  });

  it("detects pull for a sustained positive-z (away from camera) motion", () => {
    const history: Landmark[] = [
      { x: 0, y: 0, z: 0.02 },
      { x: 0, y: 0, z: 0.06 },
      { x: 0, y: 0, z: 0.09 }, // +0.07m from first reading
    ];
    expect(detectPushPull3D(history)).toBe("pull");
  });
});

describe("WorldModelFilterBank", () => {
  it("passes through the first frame unchanged", () => {
    const bank = new WorldModelFilterBank();
    const lm = worldLandmarksHandCentered();
    const out = bank.filter(lm, 0);
    expect(out[8].x).toBeCloseTo(lm[8].x, 10);
    expect(out[8].y).toBeCloseTo(lm[8].y, 10);
    expect(out[8].z ?? 0).toBeCloseTo(lm[8].z ?? 0, 10);
  });

  it("resets cleanly so a new hand doesn't inherit stale filter state", () => {
    const bank = new WorldModelFilterBank();
    const lm = worldLandmarksHandCentered();
    bank.filter(lm, 0);
    bank.filter(lm, 33);
    bank.reset();
    const farAway = lm.map((p) => ({ ...p, x: p.x + 0.2 }));
    const out = bank.filter(farAway, 100);
    expect(out[8].x).toBeCloseTo(farAway[8].x, 10);
  });

  describe("predictAhead", () => {
    it("returns null before any frame has been filtered", () => {
      const bank = new WorldModelFilterBank();
      expect(bank.predictAhead(50)).toBeNull();
    });

    it("predicts along the true 3D direction of a sustained diagonal motion", () => {
      // A landmark moving along a fixed 3D direction (1, 1, 1) normalized —
      // the coupled filter should extrapolate along that SAME direction,
      // unlike three independently-smoothed axes which can drift apart.
      const bank = new WorldModelFilterBank();
      const base = worldLandmarksHandCentered();
      const speed = 0.3; // meters/sec along (1,1,1)/sqrt(3)
      const dir = 1 / Math.sqrt(3);
      let t = 0;
      let lastFiltered: Landmark[] = [];
      for (let i = 0; i < 40; i++) {
        t += 33;
        const frame = base.map((p) => ({ ...p }));
        const disp = speed * dir * (t / 1000);
        frame[8] = { x: base[8].x + disp, y: base[8].y + disp, z: (base[8].z ?? 0) + disp };
        lastFiltered = bank.filter(frame, t);
      }

      const predicted = bank.predictAhead(100);
      expect(predicted).not.toBeNull();

      const delta = {
        x: predicted![8].x - lastFiltered[8].x,
        y: predicted![8].y - lastFiltered[8].y,
        z: (predicted![8].z ?? 0) - (lastFiltered[8].z ?? 0),
      };

      // All three axes should move in the same direction (positive) and by
      // very nearly the same magnitude, since the true motion is exactly
      // diagonal and the filter's velocity estimate is coupled (a single
      // 3D vector), not three independently-drifting per-axis estimates.
      expect(delta.x).toBeGreaterThan(0);
      expect(delta.y).toBeGreaterThan(0);
      expect(delta.z).toBeGreaterThan(0);

      const mag = Math.sqrt(delta.x ** 2 + delta.y ** 2 + delta.z ** 2);
      const cosineSimilarity =
        (delta.x * dir + delta.y * dir + delta.z * dir) / (mag * Math.sqrt(3 * dir * dir));
      expect(cosineSimilarity).toBeCloseTo(1, 6);
    });

    it("predicts no motion for a stationary hand", () => {
      const bank = new WorldModelFilterBank();
      const lm = worldLandmarksHandCentered();
      let t = 0;
      for (let i = 0; i < 10; i++) {
        t += 33;
        bank.filter(lm, t);
      }
      const predicted = bank.predictAhead(100);
      expect(predicted![8].x).toBeCloseTo(lm[8].x, 2);
      expect(predicted![8].z ?? 0).toBeCloseTo(lm[8].z ?? 0, 2);
    });
  });
});

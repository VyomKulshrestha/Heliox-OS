import { describe, it, expect } from "vitest";
import {
  handSize,
  isThumbExtended,
  thumbExtensionRatio,
  geometricQuality,
  computeHandQuality,
  LandmarkFilterBank,
  predictCursorTarget,
  trajectoryAgreement,
  type Landmark,
} from "./spatialModel";

// A synthetic "open palm" hand facing the camera, roughly matching MediaPipe's
// normalized [0,1] image-space + relative-z convention. Index values chosen
// so every finger's tip.y < pip.y < mcp.y (extended) and the thumb sits away
// from the palm on the right side (a right hand).
function openPalmRightHand(): Landmark[] {
  const lm: Landmark[] = new Array(21).fill(null).map(() => ({ x: 0.5, y: 0.5, z: 0 }));
  lm[0] = { x: 0.5, y: 0.7, z: 0 }; // wrist
  // Thumb: cmc, mcp, ip, tip — extended out to the side, away from index MCP
  lm[1] = { x: 0.46, y: 0.66, z: 0 };
  lm[2] = { x: 0.42, y: 0.6, z: 0 };
  lm[3] = { x: 0.38, y: 0.55, z: 0 };
  lm[4] = { x: 0.3, y: 0.5, z: 0 }; // thumb tip — far from index MCP (5)
  // Index: mcp, pip, dip, tip
  lm[5] = { x: 0.48, y: 0.55, z: 0 };
  lm[6] = { x: 0.48, y: 0.45, z: 0 };
  lm[7] = { x: 0.48, y: 0.38, z: 0 };
  lm[8] = { x: 0.48, y: 0.3, z: 0 };
  // Middle: mcp, pip, dip, tip
  lm[9] = { x: 0.5, y: 0.55, z: 0 };
  lm[10] = { x: 0.5, y: 0.45, z: 0 };
  lm[11] = { x: 0.5, y: 0.38, z: 0 };
  lm[12] = { x: 0.5, y: 0.28, z: 0 };
  // Ring: mcp, pip, dip, tip
  lm[13] = { x: 0.52, y: 0.55, z: 0 };
  lm[14] = { x: 0.52, y: 0.45, z: 0 };
  lm[15] = { x: 0.52, y: 0.38, z: 0 };
  lm[16] = { x: 0.52, y: 0.3, z: 0 };
  // Pinky: mcp, pip, dip, tip
  lm[17] = { x: 0.54, y: 0.56, z: 0 };
  lm[18] = { x: 0.54, y: 0.48, z: 0 };
  lm[19] = { x: 0.54, y: 0.42, z: 0 };
  lm[20] = { x: 0.54, y: 0.35, z: 0 };
  return lm;
}

// Mirror of openPalmRightHand across the x-axis around the wrist — a left
// hand in the same open-palm pose. Under the old `tip.x < ip.x` heuristic
// this would misclassify the thumb as NOT extended; the ratio-based check
// should still say it's extended since it's orientation/handedness-invariant.
function openPalmLeftHand(): Landmark[] {
  const right = openPalmRightHand();
  const wristX = right[0].x;
  return right.map((lm) => ({ ...lm, x: wristX - (lm.x - wristX) }));
}

// Same pose as openPalmRightHand(), but with non-zero, non-uniform z
// (depth) values. Every existing fixture in this file before this addition
// used z=0 everywhere, so dist3d()'s z term (used by handSize(),
// thumbExtensionRatio(), geometricQuality()) had zero real 3D regression
// coverage — a bug that silently dropped the z term would never have been
// caught. This fixture exercises it directly.
function openPalmRightHandWithDepth(): Landmark[] {
  const lm = openPalmRightHand();
  lm[0] = { ...lm[0], z: 0.02 }; // wrist
  lm[9] = { ...lm[9], z: 0.08 }; // middle MCP — used by handSize()
  lm[5] = { ...lm[5], z: 0.03 }; // index MCP — used by thumbExtensionRatio()
  lm[4] = { ...lm[4], z: -0.04 }; // thumb tip
  for (const [mcp, tip] of [
    [5, 8],
    [9, 12],
    [13, 16],
    [17, 20],
  ] as const) {
    lm[tip] = { ...lm[tip], z: lm[mcp].z! + 0.05 };
  }
  return lm;
}

function fistHand(): Landmark[] {
  const lm = openPalmRightHand();
  // Curl every fingertip below its pip (tip.y > pip.y) and tuck the thumb
  // against the palm near the index MCP.
  lm[4] = { x: lm[5].x + 0.01, y: lm[5].y + 0.01, z: 0 }; // thumb tip near index MCP
  lm[8] = { x: lm[6].x, y: lm[6].y + 0.1, z: 0 };
  lm[12] = { x: lm[10].x, y: lm[10].y + 0.1, z: 0 };
  lm[16] = { x: lm[14].x, y: lm[14].y + 0.1, z: 0 };
  lm[20] = { x: lm[18].x, y: lm[18].y + 0.1, z: 0 };
  return lm;
}

describe("handSize", () => {
  it("is positive for a normal hand pose", () => {
    expect(handSize(openPalmRightHand())).toBeGreaterThan(0);
  });

  it("incorporates the z (depth) term, not just x/y", () => {
    // wrist z=0.02, middle-MCP z=0.08 -> dz=0.06 on top of the existing
    // dy=0.15; a z=0-blind implementation would return 0.15 here instead.
    const expected = Math.sqrt(0.15 * 0.15 + 0.06 * 0.06);
    expect(handSize(openPalmRightHandWithDepth())).toBeCloseTo(expected, 10);
  });
});

describe("isThumbExtended", () => {
  it("reports extended for an open right-hand palm", () => {
    expect(isThumbExtended(openPalmRightHand())).toBe(true);
  });

  it("reports extended for the mirrored left-hand palm (handedness-invariant)", () => {
    // This is the exact case the old `tip.x < ip.x` heuristic got wrong.
    expect(isThumbExtended(openPalmLeftHand())).toBe(true);
  });

  it("reports not-extended for a tucked thumb (fist)", () => {
    expect(isThumbExtended(fistHand())).toBe(false);
  });

  it("still reports extended once a non-zero z offset is folded in", () => {
    expect(isThumbExtended(openPalmRightHandWithDepth())).toBe(true);
  });
});

describe("geometricQuality / computeHandQuality", () => {
  it("scores a normal hand pose highly", () => {
    expect(geometricQuality(openPalmRightHand())).toBeGreaterThan(0.5);
  });

  it("stays plausible once a non-zero z offset is folded into finger ratios", () => {
    expect(geometricQuality(openPalmRightHandWithDepth())).toBe(1);
  });

  it("penalizes a degenerate pose with implausible finger ratios", () => {
    const degenerate = openPalmRightHand();
    // Collapse all fingertips onto their MCP joints — near-zero apparent
    // finger length, well below the plausible ratio range.
    for (const [mcp, tip] of [
      [5, 8],
      [9, 12],
      [13, 16],
      [17, 20],
    ] as const) {
      degenerate[tip] = { ...degenerate[mcp] };
    }
    expect(geometricQuality(degenerate)).toBeLessThan(geometricQuality(openPalmRightHand()));
  });

  it("combines detection confidence and geometric quality multiplicatively", () => {
    const lm = openPalmRightHand();
    const full = computeHandQuality(lm, 1.0);
    const half = computeHandQuality(lm, 0.5);
    expect(half).toBeCloseTo(full * 0.5, 5);
  });
});

describe("LandmarkFilterBank", () => {
  it("passes through the first frame unchanged (no prior state to smooth against)", () => {
    const bank = new LandmarkFilterBank();
    const lm = openPalmRightHand();
    const out = bank.filter(lm, 0);
    expect(out[8].x).toBeCloseTo(lm[8].x, 10);
    expect(out[8].y).toBeCloseTo(lm[8].y, 10);
  });

  it("smooths a single-frame jitter spike toward the trend", () => {
    const bank = new LandmarkFilterBank();
    const base = openPalmRightHand();
    let t = 0;
    // Feed several stable frames so the filter settles.
    for (let i = 0; i < 10; i++) {
      t += 33;
      bank.filter(base, t);
    }
    // One noisy outlier frame.
    const jittered = base.map((p) => ({ ...p }));
    jittered[8] = { x: jittered[8].x + 0.2, y: jittered[8].y, z: 0 };
    t += 33;
    const out = bank.filter(jittered, t);
    // The filtered point should move toward the jitter but not fully jump to it.
    const jumpFraction = Math.abs(out[8].x - base[8].x) / Math.abs(jittered[8].x - base[8].x);
    expect(jumpFraction).toBeGreaterThan(0);
    expect(jumpFraction).toBeLessThan(1);
  });

  it("resets cleanly so a new hand doesn't inherit stale filter state", () => {
    const bank = new LandmarkFilterBank();
    const lm = openPalmRightHand();
    bank.filter(lm, 0);
    bank.filter(lm, 33);
    bank.reset();
    const farAway = lm.map((p) => ({ ...p, x: p.x + 0.4 }));
    const out = bank.filter(farAway, 100);
    // Right after reset, the first frame should pass through unchanged again.
    expect(out[8].x).toBeCloseTo(farAway[8].x, 10);
  });
});

describe("LandmarkFilterBank.predictAhead", () => {
  it("returns null before any frame has been filtered", () => {
    const bank = new LandmarkFilterBank();
    expect(bank.predictAhead(50)).toBeNull();
  });

  it("extrapolates in the direction of a sustained constant-velocity motion", () => {
    const bank = new LandmarkFilterBank();
    const base = openPalmRightHand();
    const velocity = 0.5; // normalized units/sec along x
    let t = 0;
    let lastFiltered: Landmark[] = [];
    for (let i = 0; i < 30; i++) {
      t += 33;
      const frame = base.map((p) => ({ ...p }));
      frame[8] = { ...frame[8], x: base[8].x + velocity * (t / 1000) };
      lastFiltered = bank.filter(frame, t);
    }

    const aheadMs = 100;
    const predicted = bank.predictAhead(aheadMs);
    expect(predicted).not.toBeNull();

    // Predicted position should be further along +x than the last filtered
    // position, in the same direction as the true velocity. Note this is a
    // wide, one-sided tolerance band, not a tight physical check: the
    // derivative estimate is computed as (raw_now - filtered_previous)/dt —
    // using the *filtered* previous value, which lags the raw ramp by a
    // roughly-constant offset — so for a sustained constant-velocity input
    // the steady-state velocity estimate is systematically amplified above
    // the true velocity (observed ~4x for these filter parameters), not an
    // unbiased estimate. That's a real, known property of this One Euro
    // filter formula (inherited from the existing filter() implementation,
    // not something this test tries to "fix") — this test only guards that
    // the sign and rough order of magnitude stay sane, not exact physics.
    const predictedDelta = predicted![8].x - lastFiltered[8].x;
    const expectedDelta = velocity * (aheadMs / 1000);
    expect(predictedDelta).toBeGreaterThan(expectedDelta * 0.5);
    expect(predictedDelta).toBeLessThan(expectedDelta * 6);
  });

  it("predicts no motion for a stationary hand", () => {
    const bank = new LandmarkFilterBank();
    const lm = openPalmRightHand();
    let t = 0;
    for (let i = 0; i < 10; i++) {
      t += 33;
      bank.filter(lm, t);
    }
    const predicted = bank.predictAhead(100);
    expect(predicted![8].x).toBeCloseTo(lm[8].x, 2);
  });
});

describe("predictCursorTarget", () => {
  const filtered: Landmark = { x: 0.2, y: 0.3, z: 0 };
  const predicted: Landmark = { x: 0.8, y: 0.9, z: 0 };

  it("returns the filtered position when blend=0", () => {
    expect(predictCursorTarget(filtered, predicted, 0)).toEqual(filtered);
  });

  it("returns the predicted position when blend=1", () => {
    const result = predictCursorTarget(filtered, predicted, 1);
    expect(result.x).toBeCloseTo(predicted.x);
    expect(result.y).toBeCloseTo(predicted.y);
  });

  it("linearly interpolates at blend=0.5", () => {
    const result = predictCursorTarget({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 0 }, 0.5);
    expect(result.x).toBeCloseTo(0.5);
    expect(result.y).toBeCloseTo(0.5);
  });

  it("clamps blend outside [0,1]", () => {
    const a = { x: 0, y: 0, z: 0 };
    const b = { x: 1, y: 1, z: 0 };
    expect(predictCursorTarget(a, b, -5).x).toBeCloseTo(0);
    expect(predictCursorTarget(a, b, 5).x).toBeCloseTo(1);
  });
});

// Pins exact numeric output for the untouched 2D static-pose path (handSize,
// thumbExtensionRatio, isThumbExtended, geometricQuality, computeHandQuality)
// against the fixed openPalmRightHand() fixture. The B1-B4 MediaPipe Tasks
// migration adds an entirely separate "tasks" backend/worldModel.ts path
// (see worldModel.test.ts) without touching any of these functions or their
// callers in classifyGesture() — this test exists to catch any accidental
// drift in that guarantee. Values captured from the current implementation;
// update deliberately (not silently) if a real behavior change is intended.
describe("numeric pinning — 2D static-pose path stays bit-identical", () => {
  it("pins handSize/thumbExtensionRatio/isThumbExtended/geometricQuality/computeHandQuality", () => {
    const lm = openPalmRightHand();
    expect(handSize(lm)).toBeCloseTo(0.15, 10);
    expect(thumbExtensionRatio(lm)).toBeCloseTo(1.245436112817961, 12);
    expect(isThumbExtended(lm)).toBe(true);
    expect(geometricQuality(lm)).toBe(1);
    expect(computeHandQuality(lm, 0.8)).toBeCloseTo(0.8, 12);
  });
});

describe("trajectoryAgreement", () => {
  it("returns 1 when predicted delta agrees in sign with observed delta", () => {
    expect(trajectoryAgreement(0.1, 0.05)).toBe(1);
    expect(trajectoryAgreement(-0.1, -0.02)).toBe(1);
  });

  it("returns a value below 1 when the predicted delta contradicts the observed direction", () => {
    expect(trajectoryAgreement(0.1, -0.05)).toBeLessThan(1);
  });

  it("treats a near-zero observed delta as nothing to contradict", () => {
    expect(trajectoryAgreement(0, 5)).toBe(1);
    expect(trajectoryAgreement(0, -5)).toBe(1);
  });
});

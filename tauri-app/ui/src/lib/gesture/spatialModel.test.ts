import { describe, it, expect } from "vitest";
import {
  handSize,
  isThumbExtended,
  geometricQuality,
  computeHandQuality,
  LandmarkFilterBank,
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
});

describe("geometricQuality / computeHandQuality", () => {
  it("scores a normal hand pose highly", () => {
    expect(geometricQuality(openPalmRightHand())).toBeGreaterThan(0.5);
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

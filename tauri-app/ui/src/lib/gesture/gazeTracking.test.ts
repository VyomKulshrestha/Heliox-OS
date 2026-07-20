import { describe, it, expect } from "vitest";
import { estimateGazeRegion, type FaceLandmark } from "./gazeTracking";

// Self-consistent synthetic 478-point face mesh: only the landmark indices
// estimateGazeRegion() actually reads are populated meaningfully (the rest
// are zeroed placeholders, since only array LENGTH matters beyond that).
// Left eye socket spans x=[0.30, 0.40], right eye spans x=[0.50, 0.60];
// both span y=[0.48, 0.52] (top=0.48, bottom=0.52).
function baseFaceMesh(): FaceLandmark[] {
  const landmarks: FaceLandmark[] = new Array(478).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));
  // Left eye: outer=33, inner=133, top=159, bottom=145, iris=468
  landmarks[33] = { x: 0.3, y: 0.5 };
  landmarks[133] = { x: 0.4, y: 0.5 };
  landmarks[159] = { x: 0.35, y: 0.48 };
  landmarks[145] = { x: 0.35, y: 0.52 };
  landmarks[468] = { x: 0.35, y: 0.5 }; // centered
  // Right eye: outer=263, inner=362, top=386, bottom=374, iris=473
  landmarks[263] = { x: 0.6, y: 0.5 };
  landmarks[362] = { x: 0.5, y: 0.5 };
  landmarks[386] = { x: 0.55, y: 0.48 };
  landmarks[374] = { x: 0.55, y: 0.52 };
  landmarks[473] = { x: 0.55, y: 0.5 }; // centered
  return landmarks;
}

function shiftIris(landmarks: FaceLandmark[], dxFractionOfWidth: number, dyFractionOfHeight: number): FaceLandmark[] {
  const out = landmarks.map((lm) => ({ ...lm }));
  // Left eye: width 0.10 (0.30-0.40), height 0.04 (0.48-0.52)
  out[468] = { x: 0.35 + dxFractionOfWidth * 0.1, y: 0.5 + dyFractionOfHeight * 0.04 };
  // Right eye: width 0.10 (0.50-0.60), height 0.04
  out[473] = { x: 0.55 + dxFractionOfWidth * 0.1, y: 0.5 + dyFractionOfHeight * 0.04 };
  return out;
}

describe("estimateGazeRegion", () => {
  it("returns null for a missing/too-short landmark array", () => {
    expect(estimateGazeRegion(null)).toBeNull();
    expect(estimateGazeRegion(undefined)).toBeNull();
    expect(estimateGazeRegion([{ x: 0, y: 0 }])).toBeNull();
  });

  it("reports center for an iris exactly at the socket midpoint", () => {
    const result = estimateGazeRegion(baseFaceMesh());
    expect(result).not.toBeNull();
    expect(result!.region).toBe("center");
    expect(result!.confidence).toBeGreaterThan(0);
  });

  it("stays center within the deadzone", () => {
    // A small shift (well under the 0.15 deadzone) should still read center.
    const landmarks = shiftIris(baseFaceMesh(), 0.05, 0);
    const result = estimateGazeRegion(landmarks);
    expect(result!.region).toBe("center");
  });

  it("reports a horizontal region once past the deadzone", () => {
    const shiftedPositive = estimateGazeRegion(shiftIris(baseFaceMesh(), 0.4, 0));
    const shiftedNegative = estimateGazeRegion(shiftIris(baseFaceMesh(), -0.4, 0));
    expect(shiftedPositive!.region).not.toBe("center");
    expect(shiftedNegative!.region).not.toBe("center");
    expect(shiftedPositive!.region).not.toBe(shiftedNegative!.region);
    expect(["left", "right"]).toContain(shiftedPositive!.region);
    expect(["left", "right"]).toContain(shiftedNegative!.region);
  });

  it("reports a vertical region once past the deadzone", () => {
    const shiftedPositive = estimateGazeRegion(shiftIris(baseFaceMesh(), 0, 0.4));
    const shiftedNegative = estimateGazeRegion(shiftIris(baseFaceMesh(), 0, -0.4));
    expect(["up", "down"]).toContain(shiftedPositive!.region);
    expect(["up", "down"]).toContain(shiftedNegative!.region);
    expect(shiftedPositive!.region).not.toBe(shiftedNegative!.region);
  });

  it("picks the axis with the larger deviation when both exceed the deadzone", () => {
    // Horizontal deviation (0.4) dominates vertical (0.2) here.
    const result = estimateGazeRegion(shiftIris(baseFaceMesh(), 0.4, 0.2));
    expect(["left", "right"]).toContain(result!.region);
  });

  it("confidence increases with distance from center", () => {
    const near = estimateGazeRegion(shiftIris(baseFaceMesh(), 0.2, 0))!;
    const far = estimateGazeRegion(shiftIris(baseFaceMesh(), 0.45, 0))!;
    expect(far.confidence).toBeGreaterThan(near.confidence);
  });

  it("averages both eyes rather than trusting one alone", () => {
    // Shift only the left eye's iris; the right eye stays centered, so the
    // averaged result should be a damped (not full-magnitude) reading.
    const landmarks = baseFaceMesh();
    landmarks[468] = { x: 0.39, y: 0.5 }; // left eye iris shifted hard toward inner corner
    const result = estimateGazeRegion(landmarks)!;
    const fullyShifted = estimateGazeRegion(shiftIris(baseFaceMesh(), 0.4, 0))!;
    expect(result.confidence).toBeLessThan(fullyShifted.confidence);
  });
});

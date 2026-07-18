import { describe, expect, it } from "vitest";
import { classifyControlGesture } from "./workflowControl";

describe("classifyControlGesture", () => {
  it("classifies thumbs_up as continue", () => {
    expect(classifyControlGesture("thumbs_up")).toBe("continue");
  });

  it("classifies palm as cancel (consistent with its existing universal stop meaning)", () => {
    expect(classifyControlGesture("palm")).toBe("cancel");
  });

  it("classifies thumbs_down as cancel", () => {
    expect(classifyControlGesture("thumbs_down")).toBe("cancel");
  });

  it("classifies an unrelated gesture as unknown", () => {
    expect(classifyControlGesture("swipe_left")).toBe("unknown");
    expect(classifyControlGesture("peace")).toBe("unknown");
    expect(classifyControlGesture("")).toBe("unknown");
  });
});

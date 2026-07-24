import { describe, expect, it } from "vitest";

import {
  isNearBottom,
  movedUpward,
  shouldFollowLatest,
} from "./scrollPolicy";

describe("chat scroll policy", () => {
  it("follows new content only while the user remains at the bottom", () => {
    expect(shouldFollowLatest(true)).toBe(true);
    expect(shouldFollowLatest(false)).toBe(false);
  });

  it("treats the configured bottom threshold as still following", () => {
    expect(
      isNearBottom({ scrollTop: 491, clientHeight: 500, scrollHeight: 999 }),
    ).toBe(true);
    expect(
      isNearBottom({ scrollTop: 490, clientHeight: 500, scrollHeight: 999 }),
    ).toBe(false);
  });

  it("detects deliberate upward movement during a programmatic scroll", () => {
    expect(movedUpward(600, 590)).toBe(true);
    expect(movedUpward(600, 599.5)).toBe(false);
    expect(movedUpward(600, 610)).toBe(false);
  });
});

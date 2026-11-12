import { describe, expect, it, vi } from "vitest";
import { handleGlobalShortcut } from "./waylandShortcut";

describe("Wayland global shortcut", () => {
  it("toggles the window on the daemon's toggle notification only", () => {
    const toggle = vi.fn(async () => undefined);
    expect(handleGlobalShortcut("global_shortcut", { id: "toggle" }, toggle)).toBe(true);
    expect(handleGlobalShortcut("global_shortcut", { id: "other" }, toggle)).toBe(false);
    expect(handleGlobalShortcut("live_card", { id: "toggle" }, toggle)).toBe(false);
    expect(handleGlobalShortcut("global_shortcut", null, toggle)).toBe(false);
    expect(toggle).toHaveBeenCalledTimes(1);
  });
});

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("Quick Actions screenshot", () => {
  const source = readFileSync(resolve("src/lib/components/QuickAction.svelte"), "utf8");
  const devServer = readFileSync(resolve("vite.config.ts"), "utf8");

  it("asks the daemon, which uses the Screenshot portal on Wayland", () => {
    // The desktop app captured with the Rust screenshots crate, which Rust 1.100 no longer compiles on Linux, and
    // saved into whatever folder the app was started in.
    expect(source).toContain('call<DaemonStatusResult & { path?: string }>("take_screenshot")');
    expect(source).not.toContain('invoke("take_screenshot")');
    expect(devServer).not.toContain('command === "take_screenshot"');
  });
});

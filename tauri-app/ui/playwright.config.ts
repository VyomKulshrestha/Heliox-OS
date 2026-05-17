import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright visual regression configuration.
 *
 * Tests run against the Vite dev server (http://localhost:1420) — no Tauri
 * binary required in CI. This keeps the suite fast and dependency-light while
 * still catching all CSS regressions across the three main panels.
 *
 * Baseline snapshots are stored in tests/visual/__snapshots__/ and committed
 * to the repo so every PR diffs against the last known-good state.
 */
export default defineConfig({
  testDir: "./tests/visual",
  snapshotDir: "./tests/visual/__snapshots__",

  /* Fail fast on CI — no retries for visual tests */
  retries: process.env.CI ? 0 : 0,
  workers: 1, // serial to avoid race conditions on the dev server

  use: {
    baseURL: "http://localhost:1420",

    /* Consistent viewport matching the Tauri window size in tauri.conf.json */
    viewport: { width: 700, height: 500 },

    /* Pixel diff tolerance — 0.2% of pixels may differ (anti-aliasing, fonts) */
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.002,
      threshold: 0.1, // per-pixel colour distance threshold (0–1)
      animations: "disabled", // freeze CSS animations for stable snapshots
    },

    /* Capture full trace on failure for debugging */
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  /* Start the Vite dev server automatically before running tests */
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});

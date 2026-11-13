/**
 * Every row in Settings has the row layout.
 *
 * Settings' row rules (setting-row, setting-info, setting-label, ...) live in SettingsPanel.svelte and reach every
 * component shown inside it. Svelte keeps a component's own styles to its own markup, so a component that uses the
 * class names from somewhere else gets nothing, and its rows run on as plain text: "Your profile and the model",
 * "Cloud Model", "Model for talking" and the Agent Gateway editor's 27 profiles did (CI on f6a1f8e). This test
 * fails the day a row is left without the layout, on every system the Visual Regression job runs on, without anyone
 * reading a screenshot. It opens Settings with every panel showing its rows (gotoApp's everyPanel: the daemon's
 * answers the panels wait for, the gesture cursor on, a headset with its own input), once with a cloud model and
 * once with Ollama since each shows rows the other does not, waits for every panel that loads late (LazyPanel),
 * and checks each row's computed style. A floor on the number of rows keeps a mock that shows less from passing.
 */

import { test, expect } from "@playwright/test";
import { gotoApp, clickTab } from "./helpers";

// Rows Settings shows with everyPanel on 0fb22f4 and the Settings rows fix; more is fine, fewer means a panel went
// missing from the page or from the mock.
const AT_LEAST: Record<string, number> = { cloud: 119, ollama: 121 };

test.describe("Settings rows", () => {
  for (const provider of ["cloud", "ollama"]) {
    test(`every row in Settings, the late panels included, has the row layout (${provider})`, async ({ page }) => {
      await gotoApp(page, { modelProvider: provider, everyPanel: true });
      await clickTab(page, "Settings");
      await page.waitForSelector(".settings-panel", { timeout: 5_000 });

      // The mock stubs requestAnimationFrame, so wait on a timer rather than on frames.
      await page.waitForFunction(
        () => document.querySelectorAll(".settings-panel .lazy-anchor").length === 0,
        undefined,
        { timeout: 30_000, polling: 250 },
      );
      await expect(page.locator(".settings-panel .lazy-failed")).toHaveCount(0);
      // Panels whose rows show only with the daemon's answers are there to be checked.
      await expect(page.getByTestId("profile-sharing")).toBeVisible();
      await expect(page.getByTestId("planner-model")).toBeVisible();
      if (provider === "cloud") await expect(page.getByTestId("talk-model")).toBeVisible();
      await expect(page.locator(".settings-panel .profile-card")).toHaveCount(27);
      await expect(
        page
          .locator(".settings-panel .setting-row")
          .filter({ hasText: "Speech recognition for other languages (Whisper small)" }),
      ).toHaveCount(1);

      const rows = await page.locator(".settings-panel .setting-row").evaluateAll((elements) =>
        elements.map((row) => {
          const style = getComputedStyle(row);
          const padding = ["padding-top", "padding-right", "padding-bottom", "padding-left"].map((side) =>
            parseFloat(style.getPropertyValue(side)),
          );
          return {
            display: style.display,
            padded: padding.some((value) => value > 0),
            words: (row.textContent ?? "").trim().replace(/\s+/g, " ").split(" ").slice(0, 6).join(" "),
          };
        }),
      );
      const shown = rows.filter((row) => row.display !== "none");
      const without = shown.filter((row) => !["flex", "grid"].includes(row.display) || !row.padded);

      expect(shown.length, `rows Settings shows with every panel (${provider})`).toBeGreaterThanOrEqual(
        AT_LEAST[provider],
      );
      expect(
        without.map((row) => `${row.words} (display ${row.display}, ${row.padded ? "padded" : "no padding"})`),
        "rows in Settings without the row layout",
      ).toEqual([]);
    });
  }
});

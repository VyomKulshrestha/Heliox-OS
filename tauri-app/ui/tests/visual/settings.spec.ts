/**
 * Visual regression tests — Settings Panel
 *
 * Covers all visible sections of SettingsPanel.svelte:
 *   - Appearance (theme toggle)
 *   - Security (root access, dry run, snapshot toggles)
 *   - Usage (token/cost display)
 *   - Screen Vision
 *   - Model (provider, mode, ollama model)
 *   - Cloud API (provider buttons, API key input)
 *   - Restrictions
 *   - Debug
 *
 * Also tests the light-mode variant to catch theme-switching regressions.
 */

import { test, expect } from "@playwright/test";
import { gotoApp, clickTab, freezeAnimations } from "./helpers";

test.describe("Settings Panel", () => {
  test.beforeEach(async ({ page }) => {
    await gotoApp(page);
    await clickTab(page, "Settings");
    await freezeAnimations(page);
    // Wait for the settings panel to be fully rendered
    await page.waitForSelector(".settings-panel", { timeout: 5_000 });
  });

  test("full settings panel matches baseline (dark mode)", async ({ page }) => {
    const panel = page.locator(".settings-panel");
    await expect(panel).toBeVisible();
    await expect(panel).toHaveScreenshot("settings-full-dark.png", {
      fullPage: false,
    });
  });

  test("appearance section matches baseline", async ({ page }) => {
    const section = page
      .locator(".settings-group")
      .filter({ hasText: "Appearance" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-appearance-section.png");
  });

  test("security section matches baseline", async ({ page }) => {
    const section = page
      .locator(".settings-group")
      .filter({ hasText: "Security" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-security-section.png");
  });

  test("model section matches baseline", async ({ page }) => {
    const section = page
      .locator(".settings-group")
      .filter({ hasText: "Model" })
      .first();
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-model-section.png");
  });

  test("cloud API section matches baseline", async ({ page }) => {
    const section = page
      .locator(".settings-group")
      .filter({ hasText: "Cloud API" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-cloud-section.png");
  });

  test("toggle active state renders correctly", async ({ page }) => {
    // Click the Root Access toggle and verify the active CSS class is applied
    const rootToggle = page
      .locator(".setting-row")
      .filter({ hasText: "Root Access" })
      .locator(".toggle");

    await expect(rootToggle).toBeVisible();
    // Capture inactive state
    await expect(rootToggle).toHaveScreenshot("settings-toggle-inactive.png");

    // Click to activate
    await rootToggle.click();
    await page.waitForTimeout(100);
    await expect(rootToggle).toHaveScreenshot("settings-toggle-active.png");
  });

  test("light mode settings panel matches baseline", async ({ page }) => {
    // Click the Light Mode toggle to switch themes
    const themeToggle = page
      .locator(".setting-row")
      .filter({ hasText: "Light Mode" })
      .locator(".toggle");

    await themeToggle.click();
    await page.waitForTimeout(200); // allow theme transition

    const panel = page.locator(".settings-panel");
    await expect(panel).toHaveScreenshot("settings-full-light.png");
  });

  test("restrictions section matches baseline", async ({ page }) => {
    const section = page
      .locator(".settings-group")
      .filter({ hasText: "Restrictions" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-restrictions-section.png");
  });

  test("full window with settings tab active matches baseline", async ({
    page,
  }) => {
    await expect(page.locator(".window")).toHaveScreenshot(
      "settings-full-window.png"
    );
  });
});

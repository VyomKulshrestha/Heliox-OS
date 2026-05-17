/**
 * Visual regression tests — Chat Interface
 *
 * Covers three states of the Command (chat) panel:
 *   1. Empty state  — no messages, suggestion chips visible
 *   2. Loading state — thinking indicator / loading skeleton
 *   3. Message thread — user message + system reply rendered
 *
 * Each test takes a screenshot and diffs it against the committed baseline.
 * A failing diff means a CSS regression was introduced in this PR.
 */

import { test, expect } from "@playwright/test";
import { gotoApp, clickTab, freezeAnimations } from "./helpers";

test.describe("Chat Interface", () => {
  test.beforeEach(async ({ page }) => {
    await gotoApp(page);
    await clickTab(page, "Command");
    await freezeAnimations(page);
  });

  test("empty state matches baseline", async ({ page }) => {
    // The empty state is shown when there are no messages and not loading
    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toBeVisible();

    // Confirm the empty-state element is present
    await expect(page.locator(".empty-state")).toBeVisible();

    await expect(chatPanel).toHaveScreenshot("chat-empty-state.png");
  });

  test("empty state suggestion chips are visible", async ({ page }) => {
    const suggestions = page.locator(".suggestions");
    await expect(suggestions).toBeVisible();

    // All three suggestion chips should be rendered
    const chips = suggestions.locator(".suggestion");
    await expect(chips).toHaveCount(3);

    await expect(suggestions).toHaveScreenshot("chat-suggestion-chips.png");
  });

  test("command input bar matches baseline", async ({ page }) => {
    const inputRow = page.locator(".input-row");
    await expect(inputRow).toBeVisible();

    await expect(inputRow).toHaveScreenshot("chat-input-bar.png");
  });

  test("command input focused state matches baseline", async ({ page }) => {
    // Focus the text input to trigger the accent border
    await page.click(".command-input input");
    await page.waitForTimeout(50);

    const inputWrapper = page.locator(".input-wrapper");
    await expect(inputWrapper).toHaveScreenshot("chat-input-focused.png");
  });

  test("user message renders correctly", async ({ page }) => {
    // Inject a user message directly into the session store via JS
    await page.evaluate(() => {
      const event = new CustomEvent("__test_inject_message__", {
        detail: {
          type: "user",
          text: "Show system information",
          timestamp: 1_000_000,
        },
      });
      window.dispatchEvent(event);
    });

    // Give Svelte a tick to re-render
    await page.waitForTimeout(200);

    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toHaveScreenshot("chat-user-message.png");
  });

  test("error message renders correctly", async ({ page }) => {
    await page.evaluate(() => {
      const event = new CustomEvent("__test_inject_message__", {
        detail: {
          type: "error",
          text: "Connection to daemon lost. Please restart.",
          timestamp: 1_000_001,
        },
      });
      window.dispatchEvent(event);
    });

    await page.waitForTimeout(200);

    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toHaveScreenshot("chat-error-message.png");
  });

  test("full chat panel layout matches baseline", async ({ page }) => {
    // Full-panel screenshot to catch any layout shifts
    await expect(page.locator(".window")).toHaveScreenshot("chat-full-panel.png");
  });
});

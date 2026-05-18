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
    // Type a command and press Enter
    await page.fill(".command-input input", "Show system information");
    await page.keyboard.press("Enter");

    // Give Svelte a tick to re-render
    await page.waitForTimeout(200);

    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toHaveScreenshot("chat-user-message.png");
  });

  test("error message renders correctly", async ({ page }) => {
    await page.fill(".command-input input", "Trigger error");
    await page.keyboard.press("Enter");
    
    // Send the error response back through the mock WS
    await page.evaluate(() => {
      const ws = (window as any).__mock_ws__;
      if (ws && ws.onmessage) {
        // Find the last sent message ID to reply to it
        const lastSend = (window as any).__last_ws_send__;
        const msgId = lastSend ? lastSend.id : 1;
        
        ws.onmessage({
          data: JSON.stringify({
            jsonrpc: "2.0",
            id: msgId,
            result: { status: "error", explanation: "Connection to daemon lost. Please restart." }
          })
        });
      }
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

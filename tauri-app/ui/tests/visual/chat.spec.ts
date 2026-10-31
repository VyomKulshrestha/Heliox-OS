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
import { gotoApp, clickTab, emitNotification, freezeAnimations } from "./helpers";

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
    await page.click(".command-input textarea");
    await page.waitForTimeout(50);

    const inputWrapper = page.locator(".input-wrapper");
    await expect(inputWrapper).toHaveScreenshot("chat-input-focused.png");
  });

  test("user message renders correctly", async ({ page }) => {
    // Type a command and press Enter
    await page.fill(".command-input textarea", "Show system information");
    await page.keyboard.press("Enter");

    // Give Svelte a tick to re-render
    await page.waitForTimeout(200);

    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toHaveScreenshot("chat-user-message.png");
  });

  test("error message renders correctly", async ({ page }) => {
    await page.fill(".command-input textarea", "Trigger error");
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
            result: { status: "error", message: "Connection to daemon lost. Please restart." },
          }),
        });
      }
    });

    await page.waitForTimeout(200);

    const errorMessage = page.locator(".error-msg");
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText("Connection to daemon lost. Please restart.");

    const chatPanel = page.locator(".chat-panel");
    await expect(chatPanel).toHaveScreenshot("chat-error-message.png");
  });

  test("keeps the reader's scroll position while new content streams", async ({ page }) => {
    const history = Array.from({ length: 80 }, (_, index) => ({
      type: index % 2 === 0 ? "user" : "system",
      text: `Long conversation message ${index}: ${"context ".repeat(12)}`,
      timestamp: 1716768000000 + index,
    }));
    await page.evaluate((messages) => {
      const sessionId = "visual-scroll-session";
      localStorage.setItem(
        "heliox_chat_sessions_v1",
        JSON.stringify([
          {
            id: sessionId,
            title: "Long conversation",
            createdAt: messages[0]?.timestamp ?? Date.now(),
            updatedAt: messages.at(-1)?.timestamp ?? Date.now(),
            messages,
            totalTokens: 0,
            estimatedCost: 0,
          },
        ]),
      );
      localStorage.setItem("heliox_active_chat_session", sessionId);
    }, history);
    await page.reload();

    const scroller = page.locator(".vl-scroller");
    await expect(scroller).toBeVisible();
    await expect
      .poll(() => scroller.evaluate((element) => element.scrollHeight - element.clientHeight))
      .toBeGreaterThan(500);

    await scroller.evaluate((element) => {
      element.scrollTop = Math.floor((element.scrollHeight - element.clientHeight) / 3);
      element.dispatchEvent(new Event("scroll"));
    });
    await page.waitForTimeout(100);
    const before = await scroller.evaluate((element) => element.scrollTop);

    await page.fill(".command-input textarea", "Continue with a new task");
    await page.keyboard.press("Enter");
    for (const token of ["New", " streamed", " content", " should", " stay below."]) {
      await emitNotification(page, "token_stream", { token });
    }

    const after = await scroller.evaluate((element) => element.scrollTop);
    expect(Math.abs(after - before)).toBeLessThanOrEqual(2);
    await expect(page.locator(".scroll-fab")).toHaveClass(/visible/);
  });

  test("world-model risk visibly interrupts execution before the action runs", async ({ page }) => {
    await emitNotification(page, "confirm_required", {
      plan_id: "world-plan",
      reason: "World model paused this plan at 80% predicted risk: predicted disk usage 96% exceeds the safe threshold",
      risk_assessment: {
        world_model_score: 0.8,
        prediction_sources: ["learned", "rule"],
        requires_confirmation: true,
      },
      actions: [
        {
          index: 0,
          action_type: "file_write",
          target: "C:\\Temp\\large-output.bin",
          parameters: {},
          destructive: false,
          irreversible: false,
          requires_root: false,
        },
      ],
    });

    const dialog = page.locator(".confirm-dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("alert")).toContainText("World model interruption · 80% predicted risk");
    await expect(dialog.getByRole("alert")).toContainText("learned + rule");
    await expect(dialog.getByRole("alert")).toContainText("predicted disk usage 96% exceeds the safe threshold");

    await dialog.getByRole("button", { name: "Deny" }).click();
    await expect
      .poll(() => page.evaluate(() => (window as any).__last_ws_send__))
      .toMatchObject({
        method: "confirm",
        params: { plan_id: "world-plan", confirmed: false },
      });
  });

  test("full shell exposes stable navigation and telemetry", async ({ page }) => {
    const shell = page.locator(".window");
    await expect(shell).toBeVisible();
    await expect(shell.getByRole("button", { name: "New chat" })).toBeVisible();
    await expect(shell.locator(".mini-monitor")).toContainText("CPU");
    await expect(shell.locator(".mini-monitor")).toContainText("RAM");
    await expect(shell.locator(".connection-hub")).toContainText(/Online|Connecting/);
  });
});

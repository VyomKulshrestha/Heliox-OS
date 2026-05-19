/**
 * Visual regression tests — Agent Thoughts Panel (ReActPipeline)
 *
 * The ReActPipeline component is only visible while the agent is processing
 * or has just completed a task. We drive it into each state by dispatching
 * the same WebSocket notification events the real daemon would send.
 *
 * States tested:
 *   1. Hidden (idle)  — pipeline not yet triggered
 *   2. Skeleton       — loading skeleton shown before first real event
 *   3. Active/thinking — stages lighting up mid-execution
 *   4. Completed      — all stages green, duration badge visible
 *   5. Thought stream — expanded agent thoughts accordion
 *   6. Collapsed      — auto-collapsed summary bar
 */

import { test, expect, type Page } from "@playwright/test";
import { gotoApp, clickTab, freezeAnimations, emitNotification } from "./helpers";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Agent Thoughts Panel (ReActPipeline)", () => {
  test.beforeEach(async ({ page }) => {
    await gotoApp(page);
    await clickTab(page, "Command");
    await freezeAnimations(page);
  });

  test("pipeline is hidden in idle state", async ({ page }) => {
    // Before any command is sent the pipeline should not be visible
    await expect(page.locator(".react-pipeline")).not.toBeVisible();

    // The chat panel without the pipeline should match baseline
    await expect(page.locator(".chat-panel")).toHaveScreenshot(
      "pipeline-idle-hidden.png"
    );
  });

  test("skeleton loading state matches baseline", async ({ page }) => {
    // Trigger the skeleton by emitting the first status event
    await emitNotification(page, "status", { phase: "receiving input" });
    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });

    // The skeleton is shown before real pipeline events arrive
    const pipeline = page.locator(".react-pipeline");
    await expect(pipeline).toBeVisible();
    await expect(pipeline).toHaveScreenshot("pipeline-skeleton.png");
  });

  test("active planning state matches baseline", async ({ page }) => {
    // Drive the pipeline through the planning phase
    await emitNotification(page, "status", { phase: "receiving input" });
    await emitNotification(page, "status", { phase: "recalling memory" });
    await emitNotification(page, "status", { phase: "routing agents" });
    await emitNotification(page, "agent_routing", {
      assigned_agents: ["system_agent"],
      is_multi_agent: false,
    });
    await emitNotification(page, "status", { phase: "planning" });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });
    await page.waitForTimeout(150);

    await expect(page.locator(".react-pipeline")).toHaveScreenshot(
      "pipeline-planning-active.png"
    );
  });

  test("execution state with sub-actions matches baseline", async ({
    page,
  }) => {
    await emitNotification(page, "status", { phase: "planning" });
    await emitNotification(page, "plan_preview", {
      explanation: "Read system information",
      actions: [{ action_type: "system_info", requires_confirmation: false }],
    });
    await emitNotification(page, "status", { phase: "executing" });
    await emitNotification(page, "action_start", {
      action: { action_type: "system_info", target: "" },
    });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });
    await page.waitForTimeout(150);

    await expect(page.locator(".react-pipeline")).toHaveScreenshot(
      "pipeline-executing-with-actions.png"
    );
  });

  test("multi-agent badge renders correctly", async ({ page }) => {
    await emitNotification(page, "status", { phase: "routing agents" });
    await emitNotification(page, "agent_routing", {
      assigned_agents: ["system_agent", "web_agent", "code_agent"],
      is_multi_agent: true,
    });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });
    await page.waitForTimeout(100);

    const badge = page.locator(".multi-agent-badge");
    await expect(badge).toBeVisible();
    await expect(badge).toHaveScreenshot("pipeline-multi-agent-badge.png");
  });

  test("thought stream expanded state matches baseline", async ({ page }) => {
    // Emit a reasoning event to populate the thought stream
    await emitNotification(page, "status", { phase: "planning" });
    await emitNotification(page, "reasoning_event", {
      event_type: "thought",
      event_name: "plan_thought",
      stage: "planning",
      duration_ms: 0,
      sequence: 1,
      data: { text: "Analysing user intent to determine the best action plan." },
    });
    await emitNotification(page, "reasoning_event", {
      event_type: "decision",
      event_name: "model_selected",
      stage: "planning",
      duration_ms: 120,
      sequence: 2,
      data: {
        description: "Model selection",
        chosen: "llama3.1:8b",
      },
    });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });

    // Click the thought toggle button to expand the stream
    const thoughtToggle = page.locator(".thought-toggle");
    await expect(thoughtToggle).toBeVisible();
    await thoughtToggle.click();
    await page.waitForTimeout(200);

    await expect(page.locator(".react-pipeline")).toHaveScreenshot(
      "pipeline-thought-stream-expanded.png"
    );
  });

  test("collapsed summary bar matches baseline", async ({ page }) => {
    // Drive to completion then collapse
    await emitNotification(page, "status", { phase: "planning" });
    await emitNotification(page, "status", { phase: "executing" });
    await emitNotification(page, "status", { phase: "verifying" });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });

    // Click the collapse toggle
    const collapseBtn = page.locator(".collapse-toggle");
    await expect(collapseBtn).toBeVisible();
    await collapseBtn.click();
    await page.waitForTimeout(150);

    await expect(page.locator(".react-pipeline")).toHaveScreenshot(
      "pipeline-collapsed.png"
    );
  });

  test("progress bar fills correctly at 50%", async ({ page }) => {
    // 4 of 9 stages complete → ~44% — close enough to test the fill width
    await emitNotification(page, "status", { phase: "planning" });
    await emitNotification(page, "agent_routing", {
      assigned_agents: ["system_agent"],
      is_multi_agent: false,
    });
    await emitNotification(page, "plan_preview", {
      explanation: "Test plan",
      actions: [],
    });

    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });
    await page.waitForTimeout(100);

    const progressTrack = page.locator(".progress-track");
    await expect(progressTrack).toBeVisible();
    await expect(progressTrack).toHaveScreenshot("pipeline-progress-partial.png");
  });

  test("full chat panel with active pipeline matches baseline", async ({
    page,
  }) => {
    await emitNotification(page, "status", { phase: "planning" });
    await page.waitForSelector(".react-pipeline", { timeout: 3_000 });
    await page.waitForTimeout(150);

    await expect(page.locator(".chat-panel")).toHaveScreenshot(
      "pipeline-chat-panel-with-pipeline.png"
    );
  });
});

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

  test("Air Handoff stays opt-in and produces a local pairing QR", async ({ page }) => {
    const section = page.locator(".handoff-card");
    await expect(section.getByRole("heading", { name: "Air Handoff" })).toBeVisible();
    await expect(section.getByRole("switch")).not.toBeChecked();

    await section.getByRole("switch").click();
    await expect(section.getByText("LAN receiver online")).toBeVisible();
    await section.getByRole("button", { name: "Pair a phone" }).click();
    await expect(section.getByAltText("Air Handoff phone pairing QR code")).toBeVisible();
    await expect(section.getByText("The one-time pairing link expires in five minutes.")).toBeVisible();
  });

  test("appearance section matches baseline", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Appearance" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-appearance-section.png");
  });

  test("security section matches baseline", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Security" });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-security-section.png");
  });

  test("model section exposes bounded local controls", async ({ page }) => {
    const section = page.locator(".settings-group").filter({
      has: page.getByRole("heading", { name: "Model", exact: true }),
    });
    await expect(section).toBeVisible();
    await expect(section).toContainText("Reasoning Mode");
    await expect(section).toContainText("GPU Memory Limit");
    await expect(section).toContainText("Idle model unload");
    await expect(section.getByRole("button", { name: "Light", exact: true })).toHaveClass(/active/);
  });

  test("cloud API section matches baseline", async ({ page }) => {
    const section = page.locator(".settings-group").filter({
      has: page.getByRole("heading", { name: "Cloud API (Fast)", exact: true }),
    });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-cloud-section.png");
  });

  test("subscription provider exposes verified login, model, and token controls", async ({ page }) => {
    const section = page.locator(".subscription-panel");
    await section.scrollIntoViewIfNeeded();
    await section.getByRole("button", { name: "Codex", exact: true }).click();
    await expect(section).toContainText("Subscription connected");
    await expect(section).toContainText("Codex ChatGPT subscription login");
    await expect(section.getByPlaceholder("Default Codex model")).toBeVisible();
    await expect(section).toContainText("13876 provider input");
    await expect(section).toContainText("4916 uncached");
    await expect(section).toContainText("8960 cached");

    await section.getByRole("button", { name: "Claude Code" }).click();
    await expect(section.getByPlaceholder("Default Claude model")).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => (window as any).__last_ws_send__))
      .toMatchObject({ method: "subscription_status", params: { provider: "claude" } });
  });

  test("toggle active state renders correctly", async ({ page }) => {
    // The mock returns an explicit daemon security status before the
    // privileged control is enabled.
    const rootToggle = page.locator(".setting-row").filter({ hasText: "Root Access" }).locator(".toggle");

    await expect(rootToggle).toBeVisible();
    await expect(rootToggle).toBeEnabled();
    // Capture inactive state
    await expect(rootToggle).toHaveScreenshot("settings-toggle-inactive.png");

    // Confirm the policy change before capturing its active state.
    await rootToggle.click();
    await page.getByRole("button", { name: "Confirm" }).click();
    await page.waitForTimeout(100);
    await expect(rootToggle).toHaveClass(/active/);
    await expect(rootToggle).toHaveAttribute("aria-pressed", "true");
  });

  test("light mode settings panel matches baseline", async ({ page }) => {
    // Click the Light Mode toggle to switch themes
    const themeToggle = page.locator(".setting-row").filter({ hasText: "Light Mode" }).locator(".toggle");

    await themeToggle.click();
    await page.waitForTimeout(200); // allow theme transition

    const panel = page.locator(".settings-panel");
    await expect(panel).toHaveScreenshot("settings-full-light.png");
  });

  test("restrictions section matches baseline", async ({ page }) => {
    const section = page.locator(".settings-group").filter({
      has: page.getByRole("heading", { name: "Restrictions", exact: true }),
    });
    await expect(section).toBeVisible();
    await expect(section).toHaveScreenshot("settings-restrictions-section.png");
  });

  test("hybrid world model exposes runtime status and saves its toggle", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Hybrid World Model" });
    await section.scrollIntoViewIfNeeded();

    await expect(section).toContainText("Loaded");
    await expect(section).toContainText("36,000 real samples");
    await expect(section).toContainText("5,400 held-out samples");
    await expect(section).toContainText("Calibration");
    await expect(section).toContainText("Active");
    await expect(section).toContainText("risk-mlp-v3-calibrated");
    await expect(section).toContainText("80% risk");
    await expect(section).toContainText("learned_calibrated + rule");
    await expect(section).toContainText("77% model confidence");

    const runtimeCard = section.locator(".status-card").filter({ hasText: "Runtime" });
    await expect(runtimeCard).toContainText("Enabled");
    await section.getByRole("button", { name: "Toggle Learned Risk World Model" }).click();
    await expect(runtimeCard).toContainText("Enabled");
    await section.getByRole("button", { name: "Save" }).click();
    await expect(runtimeCard).toContainText("Disabled");

    await expect
      .poll(() => page.evaluate(() => (window as any).__risk_gate_update__))
      .toMatchObject({
        method: "risk_gate_config_update",
        params: { enabled: false },
      });
  });

  test("gesture workflow bindings validate, save, and reload", async ({ page }) => {
    let section = page.locator(".settings-group").filter({ hasText: "Gesture Workflow Bindings" });
    await section.scrollIntoViewIfNeeded();

    await expect(section).toContainText(
      "Saved changes apply immediately, including while the camera is already running.",
    );
    await section.getByRole("button", { name: "Add Binding" }).click();
    await expect(section).toContainText("Every binding needs a workflow goal.");
    await expect(section.getByRole("button", { name: "Save" })).toBeDisabled();

    await section.getByRole("combobox", { name: "Gesture for binding 1" }).selectOption("swipe_up");
    await section.getByRole("textbox", { name: "Workflow goal for binding 1" }).fill("run my daily briefing");
    await section.getByRole("button", { name: "Toggle gesture workflow bindings" }).click();
    await section.getByRole("button", { name: "Save" }).click();

    await expect
      .poll(() => page.evaluate(() => (window as any).__gesture_workflow_update__))
      .toMatchObject({
        method: "gesture_workflow_bindings_update",
        params: {
          enabled: true,
          bindings: [
            {
              gesture_name: "swipe_up",
              goal_template: "run my daily briefing",
              enabled: true,
            },
          ],
        },
      });

    await clickTab(page, "Activity");
    await clickTab(page, "Settings");
    section = page.locator(".settings-group").filter({ hasText: "Gesture Workflow Bindings" });
    await section.scrollIntoViewIfNeeded();

    await expect(section.getByRole("button", { name: "Toggle gesture workflow bindings" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(section.getByRole("combobox", { name: "Gesture for binding 1" })).toHaveValue("swipe_up");
    await expect(section.getByRole("textbox", { name: "Workflow goal for binding 1" })).toHaveValue(
      "run my daily briefing",
    );
  });

  test("voice workflow panel submits a durable workflow goal", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Active Voice/Gesture Workflows" });
    await section.scrollIntoViewIfNeeded();

    await expect(section).toContainText("No active workflows right now.");
    await section.getByPlaceholder("Describe the multi-step goal").fill("open github and review notifications");
    await section.getByRole("button", { name: "Start Workflow" }).click();

    await expect
      .poll(() => page.evaluate(() => (window as any).__workflow_submit__))
      .toMatchObject({
        method: "voice_gesture_workflow_submit",
        params: {
          goal: "open github and review notifications",
          invocation_source: "voice",
        },
      });
  });

  test("autonomous healing shows live monitors and saves selected metrics", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Autonomous Healing" });
    await section.scrollIntoViewIfNeeded();

    await expect(section).toContainText("CPU > 80% · 24% · 12 checks");
    await expect(section).toContainText("RAM > 85% · 58% · 8 checks");
    await expect(section).toContainText("Disk > 90% · 71% · 2 checks");
    await section.getByRole("button", { name: "Toggle disk monitoring" }).click();
    await section.getByRole("button", { name: "Save" }).click();

    // Only the edited field is sent, so a change made elsewhere to the other fields is kept.
    await expect
      .poll(() => page.evaluate(() => (window as any).__self_healing_update__))
      .toMatchObject({ method: "self_healing_config_update" });
    expect(await page.evaluate(() => (window as any).__self_healing_update__.params)).toEqual({
      watched_metrics: ["cpu", "memory"],
    });
  });

  test("specialist mesh shows expanded agents and complete action coverage", async ({ page }) => {
    const section = page.locator(".settings-group").filter({ hasText: "Specialist Agent Mesh" });
    await section.scrollIntoViewIfNeeded();

    await expect(section).toContainText("21");
    await expect(section).toContainText("157 / 157");
    await expect(section).toContainText("Complete action coverage");
    await expect(section).toContainText("FileOperationsAgent");
    await expect(section).toContainText("GitAgent");
    await expect(section).toContainText("VisionAgent");
    await expect(section).not.toContainText("actions uncovered");
  });

  test("full shell keeps settings navigation and runtime status visible", async ({ page }) => {
    const shell = page.locator(".window");
    await expect(shell.getByRole("button", { name: "Settings", exact: true })).toHaveClass(/active/);
    await expect(shell.locator(".mini-monitor")).toContainText("CPU");
    await expect(shell.locator(".mini-monitor")).toContainText("RAM");
    await expect(shell.locator(".connection-hub")).toContainText(/Online|Connecting/);
  });
});

test.describe("Hybrid World Model compatibility", () => {
  test("an outdated daemon never looks like an empty fallback model", async ({ page }) => {
    await gotoApp(page, { riskGateStatusMissing: true });
    await clickTab(page, "Settings");

    const section = page.locator(".settings-group").filter({ hasText: "Hybrid World Model" });
    await section.scrollIntoViewIfNeeded();

    await expect(section.getByRole("alert")).toContainText(
      "This app is connected to an older Heliox daemon. Restart the daemon, then retry.",
    );
    await expect(section).not.toContainText("Rule fallback only");
    await expect(section).not.toContainText("0 real samples");
    await expect(section.getByRole("button", { name: "Toggle Learned Risk World Model" })).toBeDisabled();
  });
});

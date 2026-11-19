/**
 * The talking face (Settings, Appearance, Face) in the real page with the mock daemon. No screenshots here, so no
 * baselines: what is checked is behaviour. Off by default, nothing of the face is fetched; chosen in Settings, it
 * shows in the conversation view, speaks with voice_piece and closes its mouth on voice_cut.
 */

import { test, expect, type Page } from "@playwright/test";
import { clickTab, emitNotification, gotoApp } from "./helpers";

function faceRequests(page: Page): string[] {
  const seen: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/faces/")) seen.push(new URL(request.url()).pathname);
  });
  return seen;
}

const SPEECH = {
  reply: 7,
  index: 0,
  seconds: 20,
  lead_ms: 0,
  voice: "local",
  mouth: "visemes",
  visemes: [
    ["sil", 0, 100],
    ["aa", 100, 20_000],
  ],
};

test.describe("Face", () => {
  test("is off by default: no face is shown and none of its files is fetched", async ({ page }) => {
    const requests = faceRequests(page);
    await gotoApp(page);
    await page.waitForTimeout(500);
    await expect(page.getByTestId("face")).toHaveCount(0);
    await clickTab(page, "Settings");
    await expect(page.getByTestId("face-choice")).toBeVisible();
    await expect(page.getByLabel("Face", { exact: true })).toHaveValue("off");
    // Settings reads the list of packs and their manifests (to name them), never an image.
    expect(requests.filter((path) => !path.endsWith(".json"))).toEqual([]);
  });

  test("chosen in Settings, it shows beside the conversation, speaks, and stops at a cut", async ({ page }) => {
    const requests = faceRequests(page);
    await gotoApp(page);
    await clickTab(page, "Settings");
    const choice = page.getByLabel("Face", { exact: true });
    await expect(choice.locator("option", { hasText: "Line drawing" })).toHaveCount(1);
    await choice.selectOption("placeholder");
    await clickTab(page, "Command");
    const face = page.getByTestId("face");
    await expect(face).toBeVisible();
    await expect(face).toHaveAttribute("data-face-state", "idle");
    expect(requests).toContain("/faces/placeholder/mouth-aa.svg");

    await emitNotification(page, "voice_piece", SPEECH);
    await expect(face).toHaveAttribute("data-face-state", "speaking");
    await emitNotification(page, "voice_cut", { reply: 7, index: 0 });
    await expect(face).toHaveAttribute("data-face-state", "idle", { timeout: 1000 });

    await emitNotification(page, "interaction_state", { phase: "listening", message: "Listening" });
    await expect(face).toHaveAttribute("data-face-state", "listening");

    await clickTab(page, "Settings");
    await page.getByLabel("Face", { exact: true }).selectOption("off");
    await clickTab(page, "Command");
    await expect(page.getByTestId("face")).toHaveCount(0);
  });
});

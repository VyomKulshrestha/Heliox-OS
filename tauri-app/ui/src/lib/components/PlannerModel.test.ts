import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  sent: [] as unknown[],
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      return { status: "ok" };
    }
    if (method === "get_config") return mocks.config;
    if (method === "model_catalog") return { status: "ok", ...(await import("./modelCatalog.fixture")).CATALOG };
    return { status: "ok" };
  }),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

vi.mock("svelte-i18n", async () => {
  const { readable } = await import("svelte/store");
  return { _: readable((key: string) => (key === "settings.cloud_model" ? "Cloud Model" : key)) };
});

import { resetModelCatalogForTests } from "../stores/modelCatalog";
import { settings } from "../stores/settings";
import PlannerModel from "./PlannerModel.svelte";

async function saved(cloud_provider: string, cloud_model: string) {
  mocks.config = { model: { provider: "cloud", cloud_provider, cloud_model } };
  await settings.reloadFromDaemon();
}

function offered(): string[] {
  return Array.from(document.querySelectorAll("datalist option")).map((o) => (o as HTMLOptionElement).value);
}

describe("Cloud model (the planner)", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    resetModelCatalogForTests();
  });

  it("offers the daemon's current models only, with its default as the placeholder", async () => {
    await saved("openai", "");
    render(PlannerModel);
    await waitFor(() => expect(offered()).toEqual(["gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-luna"]));
    expect(screen.getByRole("combobox", { name: "Cloud model" }).getAttribute("placeholder")).toBe("gpt-6-luna");
    expect(screen.queryByTestId("planner-model-withdrawn")).toBeNull();
    expect(screen.getByText(/not yet checked against the live service by Heliox's tests/)).toBeTruthy();
  });

  it.each([
    ["claude", "claude-3-7-sonnet-latest", /was retired by Anthropic on 2026-02-19\. Suggested: claude-sonnet-5-5\./],
    ["openai", "o1", /o1 was retired by OpenAI on 2026-10-23\. Suggested: gpt-5\.6-sol\./],
    ["gemini", "gemini-2.5-flash", /deprecated by Google and stops working on 2026-12-01\. Suggested: gemini-3\.5/],
    ["openrouter", "openai/gpt-4o", /openai\/gpt-4o is deprecated by its provider .* Suggested: gpt-6-luna\./],
  ])("%s: a saved %s is named, kept, and replaced only when the person says", async (provider, model, line) => {
    await saved(provider, model);
    render(PlannerModel);
    await waitFor(() => expect(screen.getByTestId("planner-model-withdrawn").textContent).toMatch(line));
    expect((screen.getByRole("combobox", { name: "Cloud model" }) as HTMLInputElement).value).toBe(model);
    expect(mocks.sent).toEqual([]);
  });

  it("uses the suggestion on a click, and keeps any model id typed by hand", async () => {
    await saved("openai", "o1");
    render(PlannerModel);
    await waitFor(() => screen.getByRole("button", { name: "Use gpt-5.6-sol" }));
    await fireEvent.click(screen.getByRole("button", { name: "Use gpt-5.6-sol" }));
    await fireEvent.change(screen.getByRole("combobox", { name: "Cloud model" }), {
      target: { value: "my-private-model-7" },
    });
    await waitFor(() =>
      expect(mocks.sent).toEqual([
        { section: "model", values: { cloud_model: "gpt-5.6-sol" } },
        { section: "model", values: { cloud_model: "my-private-model-7" } },
      ]),
    );
  });
});

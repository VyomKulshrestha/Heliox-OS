import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CATALOG } from "./modelCatalog.fixture";

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

import { resetModelCatalogForTests } from "../stores/modelCatalog";
import { settings } from "../stores/settings";
import TalkModel from "./TalkModel.svelte";

async function withModel(model: Record<string, unknown>, conversation: Record<string, unknown> = {}) {
  mocks.config = { model: { provider: "cloud", cloud_model: "", ...model }, conversation };
  await settings.reloadFromDaemon();
}

describe("Model for talking", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    resetModelCatalogForTests();
  });

  it.each(Object.entries(CATALOG.providers).map(([provider, models]) => [provider, models.talk_default]))(
    "%s: a planner set alone still talks with the daemon's fast default, and says so",
    async (provider, fast) => {
      await withModel({ cloud_provider: provider, cloud_model: "a-slow-reasoning-planner" });
      render(TalkModel);
      await waitFor(() => expect(screen.getByText(`Talking with ${fast}`)).toBeTruthy());
      expect(screen.getByText(/plans and tasks use the cloud model above/)).toBeTruthy();
    },
  );

  it("shows the person's own talk model when they chose one, and saves a change under conversation", async () => {
    await withModel({ cloud_provider: "openai", cloud_model: "gpt-6.1-sol" }, { openai_model: "gpt-5.6-luna" });
    render(TalkModel);
    await waitFor(() => expect(screen.getByText("Talking with gpt-5.6-luna")).toBeTruthy());
    const box = screen.getByRole("combobox", { name: "Model for talking" });
    await fireEvent.change(box, { target: { value: "gpt-6-luna" } });
    await waitFor(() =>
      expect(mocks.sent).toEqual([{ section: "conversation", values: { openai_model: "gpt-6-luna" } }]),
    );
  });

  it("names a saved talk model the provider withdrew, and keeps it until the person picks another", async () => {
    await withModel({ cloud_provider: "openai" }, { openai_model: "gpt-4o-mini" });
    render(TalkModel);
    await waitFor(() =>
      expect(screen.getByTestId("talk-model-withdrawn").textContent).toMatch(
        /gpt-4o-mini is deprecated by OpenAI and stops working on 2027-01-14\. Suggested: gpt-6-luna\./,
      ),
    );
    expect(screen.getByText("Talking with gpt-4o-mini")).toBeTruthy();
    expect(mocks.sent).toEqual([]);
    await fireEvent.click(screen.getByRole("button", { name: "Use gpt-6-luna" }));
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "conversation", values: { openai_model: "" } }]));
  });

  it("is not shown for a model on this computer", async () => {
    mocks.config = { model: { provider: "ollama", cloud_provider: "openai" }, conversation: {} };
    await settings.reloadFromDaemon();
    render(TalkModel);
    expect(screen.queryByTestId("talk-model")).toBeNull();
  });
});

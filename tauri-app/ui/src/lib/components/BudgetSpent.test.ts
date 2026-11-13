import { render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ stats: (async () => ({})) as () => Promise<unknown> }));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string) => (method === "budget_stats" ? mocks.stats() : { status: "ok" })),
}));

import BudgetSpent from "./BudgetSpent.svelte";

describe("Spent this month", () => {
  beforeEach(() => {
    mocks.stats = async () => ({});
  });

  it("shows the month's cost against the limit", async () => {
    mocks.stats = async () => ({ cost_usd: 3.5, limit_usd: 25, estimated_high: [] });
    render(BudgetSpent);
    await waitFor(() => expect(screen.getByText(/\$3\.50 of \$25\.00, as Heliox counts it/)).toBeTruthy());
  });

  it("names every model costed without a current price", async () => {
    mocks.stats = async () => ({
      cost_usd: 0.25,
      limit_usd: 10,
      estimated_high: ["estimated high: no current price for gpt-9-preview"],
    });
    render(BudgetSpent);
    await waitFor(() => expect(screen.getByText("Estimated high: no current price for gpt-9-preview.")).toBeTruthy());
  });

  it("says so when the daemon does not answer", async () => {
    mocks.stats = async () => {
      throw new Error("down");
    };
    render(BudgetSpent);
    await waitFor(() => expect(screen.getByText(/could not be read/)).toBeTruthy());
  });
});

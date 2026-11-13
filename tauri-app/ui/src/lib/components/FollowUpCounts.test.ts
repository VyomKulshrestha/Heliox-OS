import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Reply = Record<string, unknown> | undefined;
const mocks = vi.hoisted(() => ({
  stats: (async () => ({})) as () => Promise<Reply>,
  reset: (async () => ({})) as () => Promise<Reply>,
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string) => {
    if (method === "voice_listener_stats") return mocks.stats();
    if (method === "voice_follow_up_counts_reset") return mocks.reset();
    return { status: "ok" };
  }),
}));

import FollowUpCounts from "./FollowUpCounts.svelte";

const COUNTS = { since: "2026-11-13", taken: 40, at_least_1: 3, at_least_2: 1, at_least_3: 0 };

describe("Follow-ups and talk Heliox left out", () => {
  beforeEach(() => {
    mocks.stats = async () => ({ running: true, follow_up_counts: COUNTS });
    mocks.reset = async () => ({
      status: "ok",
      follow_up_counts: { since: "2026-11-14", taken: 0, at_least_1: 0, at_least_2: 0, at_least_3: 0 },
    });
  });

  it("shows the four numbers in plain words, with the day counting began", async () => {
    render(FollowUpCounts);
    await waitFor(() =>
      expect(screen.getByText(/Follow-ups Heliox took: 40\. After something it left out: 3/)).toBeTruthy(),
    );
    expect(screen.getByText(/twice or more:\s*1, three times or more: 0/)).toBeTruthy();
    expect(screen.getByText(/Counting since/)).toBeTruthy();
  });

  it("starts again on a bare ok", async () => {
    render(FollowUpCounts);
    await waitFor(() => screen.getByText(/Follow-ups Heliox took: 40/));
    await fireEvent.click(screen.getByRole("button", { name: "Start again" }));
    await waitFor(() => expect(screen.getByText("Started again from today.")).toBeTruthy());
    expect(screen.getByText(/Follow-ups Heliox took: 0\./)).toBeTruthy();
  });

  it("says the daemon's reason when it refuses", async () => {
    mocks.reset = async () => ({ status: "error", message: "The state folder is read-only." });
    render(FollowUpCounts);
    await waitFor(() => screen.getByText(/Follow-ups Heliox took: 40/));
    await fireEvent.click(screen.getByRole("button", { name: "Start again" }));
    await waitFor(() => expect(screen.getByText("The state folder is read-only.")).toBeTruthy());
    expect(screen.getByText(/Follow-ups Heliox took: 40/)).toBeTruthy(); // unchanged
  });

  it("says so when there is no reply at all", async () => {
    mocks.stats = async () => undefined;
    mocks.reset = async () => undefined;
    render(FollowUpCounts);
    await waitFor(() => expect(screen.getByText("Heliox did not say how many follow-ups it took.")).toBeTruthy());
    await fireEvent.click(screen.getByRole("button", { name: "Start again" }));
    await waitFor(() => expect(screen.getByText("Heliox did not answer, so the counts were not reset.")).toBeTruthy());
  });
});

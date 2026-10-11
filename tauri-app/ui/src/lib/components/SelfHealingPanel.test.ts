import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../i18n";
import SelfHealingPanel from "./SelfHealingPanel.svelte";

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, ...daemonMocks };
});

afterEach(cleanup);

describe("self-healing approval results", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    daemonMocks.offNotification.mockReset();
  });

  it("shows when a stale healing approval is rejected", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "self_healing_status") {
        return Promise.resolve({
          status: "ok",
          enabled: true,
          auto_execute_max_tier: 1,
          watched_metrics: ["memory"],
          monitors: {},
          attempts: [
            {
              attempt_id: "attempt-1",
              metric: "memory",
              goal: "Reduce reversible memory pressure",
              plan_id: "plan-1",
              outcome: "proposed",
              max_tier: 1,
              irreversible: false,
              explanation: "A bounded cleanup is available.",
            },
          ],
        });
      }
      if (method === "confirm") {
        return Promise.resolve({ status: "error", message: "Confirmation already resolved for plan_id: plan-1" });
      }
      throw new Error(`unexpected method: ${method}`);
    });

    render(SelfHealingPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "Approve" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Confirmation already resolved");
    expect(daemonMocks.call).toHaveBeenCalledTimes(2);
  });

  it("keeps unsaved edits when a healing update refreshes the status, and saves only them", async () => {
    let daemonState = { enabled: false, auto_execute_max_tier: 1, watched_metrics: ["cpu", "memory", "disk"] };
    daemonMocks.call.mockImplementation((method: string, params?: Record<string, unknown>) => {
      if (method === "self_healing_status") {
        return Promise.resolve({ status: "ok", ...daemonState, monitors: {}, attempts: [] });
      }
      if (method === "self_healing_config_update") {
        daemonState = { ...daemonState, ...(params as object) };
        return Promise.resolve({ status: "ok" });
      }
      throw new Error(`unexpected method: ${method}`);
    });
    render(SelfHealingPanel);
    const toggle = await screen.findByRole("button", { name: "Toggle Autonomous Healing" });
    await vi.waitFor(() => expect(daemonMocks.onNotification).toHaveBeenCalled());

    await fireEvent.click(toggle); // switched on, not saved yet
    daemonState = { ...daemonState, auto_execute_max_tier: 2 }; // changed in another window
    for (const [handler] of daemonMocks.onNotification.mock.calls) {
      handler("self_healing_attempt_updated", {});
    }
    await vi.waitFor(() =>
      expect(daemonMocks.call.mock.calls.filter(([m]) => m === "self_healing_status").length).toBeGreaterThan(1),
    );
    expect(toggle.classList.contains("active")).toBe(true);

    await fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await vi.waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("self_healing_config_update", { enabled: true }),
    );
    expect(daemonState.auto_execute_max_tier).toBe(2);
  });
});

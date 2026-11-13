import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import RiskWorldModelPanel from "./RiskWorldModelPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

afterEach(cleanup);

const status = {
  status: "ok",
  enabled: true,
  weights_loaded: false,
  model_version: "risk-v1",
  training_samples: 0,
  validation_samples: 0,
  calibrated: false,
  validation_mae: null,
  embedding_size: 0,
  learnable_action_types: [],
  prediction_contract: null,
  last_evaluation: null,
};

describe("RiskWorldModelPanel result handling", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("does not render an incomplete response as world-model status", async () => {
    daemonMocks.call.mockResolvedValue({ enabled: true });

    render(RiskWorldModelPanel);

    expect((await screen.findByRole("alert")).textContent).toContain(
      "The daemon rejected the world-model status request",
    );
    expect(screen.queryByText("risk-v1")).toBeNull();
  });

  it("does not show saved when the daemon rejects a model setting", async () => {
    daemonMocks.call
      .mockResolvedValueOnce(status)
      .mockResolvedValueOnce({ status: "error", message: "risk gate configuration was not persisted" });

    render(RiskWorldModelPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "Save" }));

    expect((await screen.findByRole("alert")).textContent).toContain("risk gate configuration was not persisted");
    expect(screen.queryByText(/Saved/)).toBeNull();
  });

  it("puts the check back on when turning it off is not approved, with no error", async () => {
    daemonMocks.call.mockResolvedValueOnce(status).mockResolvedValueOnce({
      status: "cancelled",
      message: "Nothing was changed.",
      settings: ["gateway.risk_gate_enabled"],
    });

    render(RiskWorldModelPanel);
    const toggle = await screen.findByRole("button", { name: "Toggle Learned Risk World Model" });
    await fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect((await screen.findByRole("status")).textContent).toBe("Not changed.");
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(daemonMocks.call).toHaveBeenCalledWith("risk_gate_config_update", { enabled: false });
  });

  it("saves turning the check on directly", async () => {
    daemonMocks.call
      .mockResolvedValueOnce({ ...status, enabled: false })
      .mockResolvedValueOnce({ ...status, enabled: true });

    render(RiskWorldModelPanel);
    await fireEvent.click(await screen.findByRole("button", { name: "Toggle Learned Risk World Model" }));
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("✓ Saved")).toBeTruthy();
    expect(screen.queryByText("Not changed.")).toBeNull();
  });
});

import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../i18n";
import NarrationPanel from "./NarrationPanel.svelte";
import SupervisionPanel from "./SupervisionPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn(), onNotification: vi.fn(), offNotification: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return {
    ...actual,
    call: daemonMocks.call,
    onNotification: daemonMocks.onNotification,
    offNotification: daemonMocks.offNotification,
  };
});

const narrationStatus = {
  status: "ok",
  enabled: false,
  narrate_steps: true,
  interrupt_on_risk: true,
  proactive_review_enabled: true,
  live_corrections_enabled: true,
  follow_up_enabled: true,
  confirm_timeout_seconds: 120,
  advisory_timeout_seconds: 5,
};

afterEach(cleanup);

describe("companion settings result handling", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    daemonMocks.offNotification.mockReset();
  });

  it("saves only the narration fields changed here and follows outside changes while clean", async () => {
    let daemonState = { ...narrationStatus };
    daemonMocks.call.mockImplementation((method: string, params?: Record<string, unknown>) => {
      if (method === "narration_status") return Promise.resolve(daemonState);
      if (method === "proactive_learning_status") return Promise.resolve({ status: "ok", patterns: {} });
      if (method === "narration_config_update") {
        daemonState = { ...daemonState, ...(params as object) };
        return Promise.resolve({ status: "ok" });
      }
      throw new Error(`unexpected method: ${method}`);
    });
    render(NarrationPanel);
    const toggle = await screen.findByRole("button", { name: "Toggle Live Execution Narrator" });
    await vi.waitFor(() => expect(daemonMocks.onNotification).toHaveBeenCalled());
    const announce = (method: string) => {
      for (const [handler] of daemonMocks.onNotification.mock.calls) handler("config_updated", { method, section: "" });
    };

    daemonState = { ...daemonState, enabled: true }; // a voice command turned narration on
    announce("narration_config_update");
    await vi.waitFor(() => expect(toggle.classList.contains("active")).toBe(true));

    daemonState = { ...daemonState, narrate_steps: false }; // another window
    await fireEvent.click(toggle); // switched off here, not saved yet
    announce("narration_config_update");
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(toggle.classList.contains("active")).toBe(false); // the unsaved edit survives

    await fireEvent.click(screen.getByRole("button", { name: /save/i }));
    await vi.waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("narration_config_update", { enabled: false }),
    );
    expect(daemonState.narrate_steps).toBe(false);
  });

  it("does not claim narration settings were saved after daemon rejection", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "narration_status") {
        return Promise.resolve({
          status: "ok",
          enabled: false,
          narrate_steps: true,
          interrupt_on_risk: true,
          proactive_review_enabled: true,
          live_corrections_enabled: true,
          follow_up_enabled: true,
          confirm_timeout_seconds: 120,
          advisory_timeout_seconds: 5,
        });
      }
      if (method === "proactive_learning_status") return Promise.resolve({ status: "ok", patterns: {} });
      if (method === "narration_config_update") {
        return Promise.resolve({ status: "error", message: "Narration runtime rejected the setting" });
      }
      throw new Error(`unexpected method: ${method}`);
    });

    render(NarrationPanel);
    await fireEvent.click(await screen.findByRole("button", { name: /save/i }));

    expect((await screen.findByRole("alert")).textContent).toContain("Narration runtime rejected the setting");
    expect(screen.queryByText("✓ Saved")).toBeNull();
  });

  it("confirms and validates narration learning resets", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "narration_status") {
        return Promise.resolve({
          status: "ok",
          enabled: false,
          narrate_steps: true,
          interrupt_on_risk: true,
          proactive_review_enabled: true,
          live_corrections_enabled: true,
          follow_up_enabled: true,
          confirm_timeout_seconds: 120,
          advisory_timeout_seconds: 5,
        });
      }
      if (method === "proactive_learning_status") {
        return Promise.resolve({ status: "ok", patterns: { terminal_error: { shown: 2, accepted: 1, dismissed: 1 } } });
      }
      if (method === "proactive_learning_reset") {
        return Promise.resolve({ status: "error", message: "Learning reset was rejected" });
      }
      throw new Error(`unexpected method: ${method}`);
    });

    render(NarrationPanel);
    await fireEvent.click(await screen.findByRole("button", { name: /reset learning/i }));
    expect(screen.getByRole("button", { name: "Confirm reset learning" })).toBeTruthy();
    expect(daemonMocks.call).toHaveBeenCalledTimes(2);

    await fireEvent.click(screen.getByRole("button", { name: "Confirm reset learning" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Learning reset was rejected");
    expect(daemonMocks.call).toHaveBeenCalledTimes(3);
  });

  it("does not claim supervision settings were saved after daemon rejection", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "supervision_status") {
        return Promise.resolve({
          status: "ok",
          enabled: false,
          keyboard_mouse_hook_enabled: false,
          cognitive_coaching_enabled: true,
          risk_pattern_detection_enabled: true,
          hook_healthy: false,
        });
      }
      if (method === "supervision_config_update") {
        return Promise.resolve({ status: "error", message: "Input hook could not start" });
      }
      throw new Error(`unexpected method: ${method}`);
    });

    render(SupervisionPanel);
    await fireEvent.click(await screen.findByRole("button", { name: /save/i }));

    expect((await screen.findByRole("alert")).textContent).toContain("Input hook could not start");
    expect(screen.queryByText("✓ Saved")).toBeNull();
  });

  it("disables narration controls when status was not acknowledged", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "narration_status") return Promise.resolve({ enabled: true });
      if (method === "proactive_learning_status") return Promise.resolve({ status: "ok", patterns: {} });
      throw new Error(`unexpected method: ${method}`);
    });

    render(NarrationPanel);

    expect((await screen.findByRole("alert")).textContent).toContain("Narration status is unavailable");
    expect((screen.getByRole("button", { name: /save/i }) as HTMLButtonElement).disabled).toBe(true);
  });
});

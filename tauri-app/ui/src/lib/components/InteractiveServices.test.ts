import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call, onNotification: vi.fn(), offNotification: vi.fn() };
});

import InteractiveServices from "./InteractiveServices.svelte";

describe("InteractiveServices", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
  });

  it("summarises on one line and expands to each service, with off shown as off rather than failed", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "narration_status") return { enabled: true, narrate_steps: true, follow_up_enabled: false };
      if (method === "risk_gate_status") return { enabled: true, weights_loaded: true };
      return { gnomon: { state: "disabled" } };
    });
    const onconfigure = vi.fn();
    const { container } = render(InteractiveServices, { onconfigure });

    const toggle = await screen.findByRole("button", { name: /3 of 5 services on/ });
    const list = container.querySelector(`#${toggle.getAttribute("aria-controls")}`) as HTMLElement;
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(list.hidden).toBe(true);

    await fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(list.hidden).toBe(false);
    expect(screen.getByText("Follow-ups are off")).toBeTruthy();
    expect(container.querySelectorAll(".service.off").length).toBe(2);
    expect(container.querySelectorAll(".service.down").length).toBe(0);

    await fireEvent.click(screen.getByRole("button", { name: "Configure" }));
    expect(onconfigure).toHaveBeenCalled();
  });

  it("shows why Gnomon is not loaded and does not count it as on", async () => {
    const reason = "Needs 2.5 GB of free memory, 1.8 GB free now. Tries again on its own.";
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "narration_status") return { enabled: true, narrate_steps: true, follow_up_enabled: false };
      if (method === "risk_gate_status") return { enabled: true, weights_loaded: true };
      return { gnomon: { state: "unavailable", last_error: "At least 2.5 GiB of available system RAM", reason } };
    });
    render(InteractiveServices, { onconfigure: vi.fn() });

    await screen.findByRole("button", { name: /3 of 6 services on/ });
    expect(screen.getByText(reason).getAttribute("title")).toBe(reason);
  });

  it("says when the daemon cannot be reached", async () => {
    daemonMocks.call.mockRejectedValue(new Error("offline"));
    render(InteractiveServices, { onconfigure: vi.fn() });

    await waitFor(() => expect(screen.getByRole("button", { name: /Daemon unavailable/ })).toBeTruthy());
  });
});

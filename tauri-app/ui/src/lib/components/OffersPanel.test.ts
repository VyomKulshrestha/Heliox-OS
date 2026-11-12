import { render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: (...args: Parameters<typeof actual.call>) => daemonMocks.call(...args) };
});

import OffersPanel from "./OffersPanel.svelte";

describe("OffersPanel", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("stays up when the daemon answers without any offers fields", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok" });
    render(OffersPanel);
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("offers", { action: "status" }));
    expect(screen.getByTestId("offers-panel")).toBeTruthy();
    expect(screen.queryByTestId("offers-state")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("lists what was silenced, and copes with a reply that leaves the list out", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "ok",
      mode: "on",
      active: true,
      waiting_for_profile: false,
      per_hour: 1,
      per_day: 3,
      silenced: [{ key: "usual-app:outlook", label: "Opening Outlook at the usual time" }],
    });
    const first = render(OffersPanel);
    expect(await screen.findByText("Opening Outlook at the usual time")).toBeTruthy();
    expect(screen.getByTestId("offers-state").textContent).toContain("at most 1 an hour and 3 a day");
    first.unmount();

    daemonMocks.call.mockResolvedValue({ status: "ok", mode: "off", active: false, waiting_for_profile: false });
    render(OffersPanel);
    await waitFor(() => expect(screen.getByTestId("offers-state").textContent).toBe("Offers are off."));
    expect(screen.queryByText("Opening Outlook at the usual time")).toBeNull();
  });

  it("shows the daemon's reason when it refuses", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "Offers are not available." });
    render(OffersPanel);
    expect((await screen.findByRole("alert")).textContent).toBe("Offers are not available.");
  });
});

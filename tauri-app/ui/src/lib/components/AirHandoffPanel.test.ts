import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { afterEach, describe, expect, it, vi } from "vitest";
import AirHandoffPanel from "./AirHandoffPanel.svelte";

const handoffMocks = vi.hoisted(() => ({
  refresh: vi.fn().mockResolvedValue(undefined),
  revokeDevice: vi.fn().mockResolvedValue(undefined),
  configureRequests: vi.fn().mockResolvedValue("cancelled"),
}));

vi.mock("../stores/airHandoff", () => ({
  airHandoff: Object.assign(
    writable({
      enabled: true,
      running: true,
      receiver_url: "http://192.0.2.1:8766",
      paired_devices: [
        {
          device_id: "phone-1",
          name: "Test Phone",
          created_at: 1,
          last_seen_at: 1,
          credential_available: true,
        },
      ],
      pairing: null,
      draft: null,
      ready_transfers: 0,
      recent: [],
      secure_storage_available: true,
      selectedDeviceId: "phone-1",
      awaitingTransferId: "",
      awaitingTargetName: "",
      gestureArmed: false,
      busy: false,
      error: "",
      message: "",
    }),
    {
      ...handoffMocks,
      setEnabled: vi.fn(),
      startPairing: vi.fn(),
      cancelPairing: vi.fn(),
      selectDevice: vi.fn(),
    },
  ),
}));

afterEach(cleanup);

describe("AirHandoffPanel device revocation", () => {
  it("does not revoke a paired phone before explicit confirmation", async () => {
    render(AirHandoffPanel);

    await fireEvent.click(screen.getByRole("button", { name: "Revoke" }));
    expect(handoffMocks.revokeDevice).not.toHaveBeenCalled();

    await fireEvent.click(screen.getByRole("button", { name: "Revoke phone" }));
    expect(handoffMocks.revokeDevice).toHaveBeenCalledWith("phone-1");
  });
});

describe("AirHandoffPanel phone request switches", () => {
  it("a switch the person kept off in Heliox's dialog goes back off, with a note and no error", async () => {
    render(AirHandoffPanel);
    const accept = screen.getByRole("checkbox", { name: /Accept requests from paired phones/ }) as HTMLInputElement;

    await fireEvent.click(accept);

    expect(handoffMocks.configureRequests).toHaveBeenCalledWith(true, undefined);
    expect(await screen.findByText("Not changed: you did not approve it in Heliox's dialog.")).toBeTruthy();
    expect(accept.checked).toBe(false);
    expect(screen.queryByText(/Air Handoff failed/)).toBeNull();
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PeerMeshPanel from "./PeerMeshPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));

const offlineStatus = {
  status: "ok",
  enabled: false,
  configured_enabled: false,
  authenticated: false,
  secret_configured: true,
  reason: "Peer Mesh is disabled",
  peer_count: 0,
  skill_sync_enabled: false,
  collab_exec_enabled: false,
  port: 8786,
};

describe("PeerMeshPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "mesh_status") return Promise.resolve(offlineStatus);
      if (method === "mesh_configure") {
        return Promise.resolve({
          ...offlineStatus,
          enabled: true,
          configured_enabled: true,
          authenticated: true,
        });
      }
      throw new Error(`unexpected method: ${method}`);
    });
  });

  it("shows authenticated status and reconciles the runtime through the dedicated RPC", async () => {
    render(PeerMeshPanel);
    const toggle = await screen.findByRole("switch", { name: "Toggle Peer Mesh" });

    await fireEvent.click(toggle);

    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("mesh_configure", { enabled: true }));
  });

  it("does not revert a sharing switch that was changed elsewhere", async () => {
    let status = { ...offlineStatus };
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "mesh_status") return Promise.resolve(status);
      if (method === "mesh_configure") return Promise.resolve({ ...status, enabled: true, configured_enabled: true });
      throw new Error(`unexpected method: ${method}`);
    });
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(PeerMeshPanel);
      const toggle = await screen.findByRole("switch", { name: "Toggle Peer Mesh" });

      status = { ...offlineStatus, skill_sync_enabled: true }; // set from another window
      await vi.advanceTimersByTimeAsync(3100);
      await fireEvent.click(toggle);

      await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("mesh_configure", { enabled: true }));
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not claim success when the daemon rejects configuration", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "mesh_status") return Promise.resolve(offlineStatus);
      if (method === "mesh_configure") {
        return Promise.resolve({ status: "error", message: "shared secret required" });
      }
      throw new Error(`unexpected method: ${method}`);
    });
    render(PeerMeshPanel);

    await fireEvent.click(await screen.findByRole("switch", { name: "Toggle Peer Mesh" }));

    expect((await screen.findByRole("alert")).textContent).toContain("shared secret required");
  });

  it("keeps controls unavailable when status discovery is rejected", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "mesh_status") return Promise.resolve({ status: "error", reason: "mesh runtime unavailable" });
      throw new Error(`unexpected method: ${method}`);
    });

    render(PeerMeshPanel);

    expect((await screen.findByRole("alert")).textContent).toContain("mesh runtime unavailable");
    expect(((await screen.findByRole("switch", { name: "Toggle Peer Mesh" })) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });
});

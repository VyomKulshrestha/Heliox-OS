import { get } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn(), onNotification: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, ...daemonMocks };
});

describe("Air Handoff daemon contracts", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    vi.resetModules();
  });

  it("rejects a receiver state without an explicit acknowledgement", async () => {
    daemonMocks.call.mockResolvedValue({ enabled: true, running: true, paired_devices: [], recent: [] });
    const { airHandoff } = await import("./airHandoff");

    await airHandoff.refresh();

    expect(get(airHandoff)).toMatchObject({ running: false, gestureArmed: false });
    expect(get(airHandoff).error).toContain("Air Handoff status is unavailable");
  });

  it("rejects an incomplete pairing response even when its status is ok", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok" });
    const { airHandoff } = await import("./airHandoff");

    await expect(airHandoff.startPairing()).rejects.toThrow("pairing response is incomplete");
    expect(get(airHandoff).pairing).toBeNull();
  });

  it("requires the receiver to reach the requested enabled state", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", enabled: false, running: false });
    const { airHandoff } = await import("./airHandoff");

    await expect(airHandoff.setEnabled(true)).rejects.toThrow("did not reach the requested state");
    expect(get(airHandoff).enabled).toBe(false);
  });
});

describe("Phone requests", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    vi.resetModules();
  });

  const waiting = {
    request_id: "r1",
    device_id: "d1",
    device_name: "Pixel",
    text: "clean my downloads",
    created_at: 1,
    status: "waiting",
    job_id: "",
    result: "",
    read_only: false,
  };

  it("lists a request the phone sent and runs it only when Run is pressed", async () => {
    const { airHandoff } = await import("./airHandoff");
    const handler = daemonMocks.onNotification.mock.calls[0][0] as (method: string, params: unknown) => void;

    handler("phone_request", waiting);
    expect(get(airHandoff).requests).toEqual([waiting]);
    expect(daemonMocks.call).not.toHaveBeenCalled();

    daemonMocks.call.mockResolvedValue({ status: "ok", request: { ...waiting, status: "running", job_id: "j1" } });
    await airHandoff.runRequest("r1");

    expect(daemonMocks.call).toHaveBeenCalledWith("phone_request_run", { request_id: "r1" });
    expect(get(airHandoff).requests[0].status).toBe("running");
  });

  it("never asks for read-only autostart while requests are off", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", accept_requests: false, run_read_only_requests: false });
    const { airHandoff } = await import("./airHandoff");

    await airHandoff.configureRequests(false, true);

    expect(daemonMocks.call).toHaveBeenCalledWith("air_handoff_requests_configure", {
      accept_requests: false,
      run_read_only_requests: false,
    });
  });
});

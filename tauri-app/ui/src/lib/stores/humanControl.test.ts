import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let capturedHandler: NotificationHandler | null = null;
let connectionHandler: ((connected: boolean) => void) | null = null;
const callMock = vi.fn();

vi.mock("../api/daemon", () => ({
  call: (...args: unknown[]) => callMock(...args),
  onNotification: (handler: NotificationHandler) => {
    capturedHandler = handler;
  },
  offNotification: vi.fn(),
  onConnectionState: (handler: (connected: boolean) => void) => {
    connectionHandler = handler;
  },
  offConnectionState: vi.fn(),
}));

async function load() {
  const { humanControl } = await import("./humanControl");
  const read = () => {
    let state: any;
    humanControl.subscribe((value) => (state = value))();
    return state;
  };
  return { humanControl, read };
}

describe("humanControl store", () => {
  beforeEach(() => {
    vi.resetModules();
    capturedHandler = null;
    callMock.mockReset();
  });

  it("offers take-over only while an autonomous job runs", async () => {
    const { read } = await load();
    expect(read().activeJobs).toEqual([]);

    capturedHandler!("autonomous_started", { job_id: "j1", status: "decomposing" });
    expect(read().activeJobs).toEqual(["j1"]);
    capturedHandler!("autonomous_complete", { job_id: "j1", status: "success" });
    expect(read().activeJobs).toEqual([]);
  });

  it("also offers take-over while a voice workflow drives the desktop", async () => {
    const { read } = await load();

    capturedHandler!("voice_gesture_workflow_state", { workflow_id: "w1", state: "running" });
    expect(read().activeJobs).toEqual(["workflow:w1"]);
    capturedHandler!("voice_gesture_workflow_state", { workflow_id: "w1", state: "running" });
    expect(read().activeJobs).toEqual(["workflow:w1"]);
    capturedHandler!("voice_gesture_workflow_state", { workflow_id: "w1", state: "success" });
    expect(read().activeJobs).toEqual([]);
  });

  it("follows the daemon's human_control notifications", async () => {
    const { read } = await load();

    capturedHandler!("human_control", { holder: "human", reason: "physical mouse or keyboard input" });
    expect(read()).toMatchObject({ holder: "human", reason: "physical mouse or keyboard input" });
    capturedHandler!("human_control", { holder: "agent", reason: "" });
    expect(read()).toMatchObject({ holder: "agent", reason: "" });
  });

  it("takes and releases control through the daemon", async () => {
    const { humanControl, read } = await load();
    callMock.mockResolvedValueOnce({ status: "ok", holder: "human", reason: "taken from the Heliox app" });
    callMock.mockResolvedValueOnce({ status: "ok", holder: "agent", reason: "", generation: 1 });

    await humanControl.take();
    expect(callMock).toHaveBeenCalledWith("take_control", { reason: "taken from the Heliox app" });
    expect(read()).toMatchObject({ holder: "human", busy: false });

    await humanControl.release();
    expect(callMock).toHaveBeenLastCalledWith("release_control", {});
    expect(read()).toMatchObject({ holder: "agent", busy: false, error: "" });
  });

  it("shows a daemon refusal instead of changing state", async () => {
    const { humanControl, read } = await load();
    callMock.mockResolvedValueOnce({ status: "error", message: "reason must be a string" });

    await humanControl.take();

    expect(read()).toMatchObject({ holder: "agent", error: "reason must be a string", busy: false });
  });

  it("reads the current holder after connecting", async () => {
    const { read } = await load();
    callMock.mockResolvedValueOnce({ status: "ok", holder: "human", reason: "taken from the Heliox app" });

    connectionHandler!(true);
    await Promise.resolve();
    await Promise.resolve();

    expect(callMock).toHaveBeenCalledWith("human_control_status");
    expect(read()).toMatchObject({ holder: "human", reason: "taken from the Heliox app" });
  });
});

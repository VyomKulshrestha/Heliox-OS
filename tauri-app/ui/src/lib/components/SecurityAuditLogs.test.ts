import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import GatewayAuditLog from "./GatewayAuditLog.svelte";
import PermissionAuditLog from "./PermissionAuditLog.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

afterEach(cleanup);

describe("security audit availability", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("does not present an unavailable gateway audit as an empty log", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "error",
      message: "Gateway audit store is not initialized.",
      events: [],
    });

    render(GatewayAuditLog);

    expect((await screen.findByRole("alert")).textContent).toContain("Gateway audit store is not initialized");
    expect(screen.queryByText("No gateway decisions recorded yet.")).toBeNull();
  });

  it("does not present an unavailable permission audit as an empty log", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "error",
      message: "Permission audit store is not initialized.",
      events: [],
    });

    render(PermissionAuditLog);

    expect((await screen.findByRole("alert")).textContent).toContain("Permission audit store is not initialized");
    expect(screen.queryByText("No elevated permission decisions recorded yet.")).toBeNull();
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}

const permissionEvent = {
  id: 1,
  timestamp: "2026-10-11T08:00:00Z",
  plan_id: "p1",
  action_index: 0,
  action_type: "file_delete",
  target: "C:/Temp/old.txt",
  permission_tier: "DESTRUCTIVE",
  requires_root: false,
  destructive: true,
  confirmation_decision: "approved",
  critic_verdict: {},
  execution_success: true,
  execution_error: "",
};

const gatewayEvent = {
  id: 1,
  timestamp: "2026-10-11T08:00:00Z",
  plan_id: "p1",
  action_index: 0,
  action_type: "shell_command",
  action_family: "shell",
  target: "dir",
  source_profile: "unknown",
  permission_tier: "USER_WRITE",
  override_applied: false,
  override_restricted: false,
  decision: "allowed",
  denial_reason: "",
  dry_run: false,
  execution_success: null,
  execution_error: "",
};

describe("security audit refresh", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("reloads the permission audit log on Refresh and disables the button while loading", async () => {
    const second = deferred<unknown>();
    daemonMocks.call
      .mockResolvedValueOnce({ status: "ok", events: [permissionEvent] })
      .mockReturnValueOnce(second.promise);
    render(PermissionAuditLog);
    expect(await screen.findByText("file_delete")).toBeTruthy();

    const refresh = screen.getByRole("button", { name: "Refresh" }) as HTMLButtonElement;
    expect(refresh.disabled).toBe(false);
    await fireEvent.click(refresh);
    expect(refresh.disabled).toBe(true);

    second.resolve({ status: "ok", events: [{ ...permissionEvent, id: 2, action_type: "registry_write" }] });
    expect(await screen.findByText("registry_write")).toBeTruthy();
    expect(refresh.disabled).toBe(false);
    expect(daemonMocks.call.mock.calls.filter(([method]) => method === "list_permission_events")).toHaveLength(2);
  });

  it("reloads the gateway audit log on Refresh and disables the button while loading", async () => {
    const second = deferred<unknown>();
    let listCalls = 0;
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "gateway_policy_get") return Promise.resolve({ status: "ok", profiles: {} });
      listCalls += 1;
      return listCalls === 1 ? Promise.resolve({ status: "ok", events: [gatewayEvent] }) : second.promise;
    });
    render(GatewayAuditLog);
    expect(await screen.findByText("shell_command")).toBeTruthy();

    const refresh = screen.getByRole("button", { name: "Refresh" }) as HTMLButtonElement;
    await fireEvent.click(refresh);
    expect(refresh.disabled).toBe(true);

    second.resolve({ status: "ok", events: [{ ...gatewayEvent, id: 2, action_type: "browser_navigate" }] });
    expect(await screen.findByText("browser_navigate")).toBeTruthy();
    expect(refresh.disabled).toBe(false);
  });
});

describe("gateway audit source filter", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("lists the sources from the daemon's gateway policy plus sources seen in events", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "gateway_policy_get") {
        return Promise.resolve({
          status: "ok",
          profiles: { interactive: {}, autonomous: {}, code_agent: {} },
        });
      }
      return Promise.resolve({ status: "ok", events: [gatewayEvent] });
    });
    render(GatewayAuditLog);
    await screen.findByText("shell_command");

    const filter = screen.getByLabelText("Filter by source") as HTMLSelectElement;
    await waitFor(() =>
      expect(Array.from(filter.options).map((option) => option.value)).toEqual([
        "",
        "interactive",
        "autonomous",
        "code_agent",
        "unknown",
      ]),
    );

    await fireEvent.change(filter, { target: { value: "code_agent" } });
    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("list_gateway_events", {
        limit: 100,
        source_profile: "code_agent",
      }),
    );
  });
});

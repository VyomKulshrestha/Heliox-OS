import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import GatewayPolicyEditor from "./GatewayPolicyEditor.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn(), onNotification: vi.fn(), offNotification: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, ...daemonMocks };
});

afterEach(cleanup);

type Profile = { max_tier: Record<string, number>; deny_action_types: string[]; allow_root: boolean };

const autonomous: Profile = {
  max_tier: { shell: 1, browsing: 2, system_control: 1, other: 1 },
  deny_action_types: [],
  allow_root: false,
};

function serve(profiles: Record<string, Profile>, update: Record<string, unknown> = { status: "ok" }) {
  daemonMocks.call.mockImplementation((method: string) => {
    if (method === "gateway_policy_get") return Promise.resolve({ status: "ok", profiles });
    if (method === "gateway_policy_update") return Promise.resolve(update);
    return Promise.resolve({ status: "ok" });
  });
}

function select(label: string): HTMLSelectElement {
  return screen.getByLabelText(label) as HTMLSelectElement;
}

function announce(change: { method: string; section?: string }) {
  const handler = daemonMocks.onNotification.mock.calls[0][0] as (method: string, params: unknown) => void;
  handler("config_updated", { section: "", ...change });
}

function policyLoads(): number {
  return daemonMocks.call.mock.calls.filter(([method]) => method === "gateway_policy_get").length;
}

describe("gateway policy result handling", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    daemonMocks.offNotification.mockReset();
  });

  it("does not claim a rejected security policy was saved", async () => {
    serve({ autonomous }, { status: "error", message: "Policy validation failed" });

    render(GatewayPolicyEditor);
    await fireEvent.change(await screen.findByLabelText("shell ceiling for autonomous"), { target: { value: "2" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Policy validation failed");
    expect(daemonMocks.call.mock.calls).toEqual([
      ["gateway_policy_get"],
      ["gateway_policy_update", expect.any(Object)],
    ]);
    expect(screen.queryByText(/Saved/)).toBeNull();
  });
});

describe("gateway policy editing", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.onNotification.mockReset();
    daemonMocks.offNotification.mockReset();
  });

  it("shows READ_ONLY (tier 0) for a family the daemon policy leaves unset", async () => {
    serve({ autonomous: { ...autonomous, max_tier: { shell: 1 } } });
    render(GatewayPolicyEditor);
    await screen.findByLabelText("shell ceiling for autonomous");

    for (const family of ["browsing", "system_control", "other"]) {
      const box = select(`${family} ceiling for autonomous`);
      expect(box.value).toBe("0");
      expect(box.selectedOptions[0].textContent).toBe("READ_ONLY");
    }
    expect(select("shell ceiling for autonomous").value).toBe("1");
    // Showing the default is not an edit, so there is nothing to save yet.
    expect((screen.getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("sends tier 4 when chosen and leaves the unset families out of the save", async () => {
    serve({ autonomous: { ...autonomous, max_tier: { shell: 1 } } });
    render(GatewayPolicyEditor);

    await fireEvent.change(await screen.findByLabelText("shell ceiling for autonomous"), { target: { value: "4" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("gateway_policy_update", {
        profile: "autonomous",
        max_tier: { shell: 4 },
      }),
    );
  });

  it("sends only the fields the person changed", async () => {
    serve({ autonomous });
    render(GatewayPolicyEditor);

    await fireEvent.change(await screen.findByLabelText("browsing ceiling for autonomous"), { target: { value: "0" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("gateway_policy_update", {
        profile: "autonomous",
        max_tier: { browsing: 0 },
      }),
    );

    // After a save the profile is clean again, so the next save carries only the new change.
    await fireEvent.click(screen.getByLabelText("Toggle root access for autonomous"));
    const save = screen.getByRole("button", { name: /Save/ }) as HTMLButtonElement;
    expect(save.disabled).toBe(false);
    await fireEvent.click(save);
    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenLastCalledWith("gateway_policy_update", {
        profile: "autonomous",
        allow_root: true,
      }),
    );
  });

  it("reloads when the policy changes elsewhere and there are no unsaved edits", async () => {
    serve({ autonomous });
    render(GatewayPolicyEditor);
    await screen.findByLabelText("shell ceiling for autonomous");

    serve({ autonomous: { ...autonomous, max_tier: { ...autonomous.max_tier, shell: 3 } } });
    announce({ method: "gateway_policy_update" });

    await waitFor(() => expect(select("shell ceiling for autonomous").value).toBe("3"));
    expect(policyLoads()).toBe(2);
  });

  it("keeps unsaved edits when the policy changes elsewhere", async () => {
    serve({ autonomous });
    render(GatewayPolicyEditor);
    await fireEvent.change(await screen.findByLabelText("shell ceiling for autonomous"), { target: { value: "2" } });

    serve({ autonomous: { ...autonomous, max_tier: { ...autonomous.max_tier, shell: 3 } } });
    announce({ method: "gateway_policy_update" });
    await Promise.resolve();

    expect(policyLoads()).toBe(1);
    expect(select("shell ceiling for autonomous").value).toBe("2");
  });

  it("ignores changes to unrelated settings and stops listening when closed", async () => {
    serve({ autonomous });
    const { unmount } = render(GatewayPolicyEditor);
    await screen.findByLabelText("shell ceiling for autonomous");

    announce({ method: "update_config", section: "voice" });
    expect(policyLoads()).toBe(1);

    const handler = daemonMocks.onNotification.mock.calls[0][0];
    unmount();
    expect(daemonMocks.offNotification).toHaveBeenCalledWith(handler);
  });
});

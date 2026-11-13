import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SshIntegrationPanel from "./SshIntegrationPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, ...daemonMocks };
});

afterEach(cleanup);

const NOT_APPROVED = "Not changed: you did not approve it in Heliox's dialog.";

describe("SSH hosts and Heliox's approval dialog", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "ssh_list_hosts") return Promise.resolve({ status: "ok", enabled: false, hosts: [] });
      if (method === "ssh_save_host" || method === "ssh_set_enabled") {
        return Promise.resolve({ status: "cancelled", message: "Not approved." });
      }
      throw new Error(`unexpected method: ${method}`);
    });
  });

  it("a host the person did not approve is not an error, and what was typed stays", async () => {
    render(SshIntegrationPanel);
    await fireEvent.input(screen.getByPlaceholderText("build-box"), { target: { value: "build-box" } });
    await fireEvent.input(screen.getByPlaceholderText("10.0.0.7"), { target: { value: "10.0.0.7" } });
    await fireEvent.input(screen.getByPlaceholderText("builder"), { target: { value: "builder" } });
    const key = screen.getByPlaceholderText("-----BEGIN OPENSSH PRIVATE KEY-----") as HTMLTextAreaElement;
    await fireEvent.input(key, { target: { value: "not a real key" } });

    await fireEvent.click(screen.getByRole("button", { name: "Save host" }));

    const message = await screen.findByText(NOT_APPROVED);
    expect(message.classList.contains("error")).toBe(false);
    expect(key.value).toBe("not a real key");
    // Nothing was saved, so the list is not reloaded: only the load on opening.
    expect(daemonMocks.call.mock.calls.filter(([method]) => method === "ssh_list_hosts")).toHaveLength(1);
  });

  it("turning SSH on and not approving it leaves the switch off, without an error", async () => {
    render(SshIntegrationPanel);
    const toggle = await screen.findByRole("switch", { name: "Toggle SSH integration" });

    await fireEvent.click(toggle);

    const message = await screen.findByText(NOT_APPROVED);
    expect(message.classList.contains("error")).toBe(false);
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });
});

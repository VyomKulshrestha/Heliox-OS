import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import McpServersPanel from "./McpServersPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));

const configured = {
  status: "ok",
  tool_count: 12,
  config_path: "C:\\Users\\me\\AppData\\Roaming\\heliox-os\\mcp_servers.json",
  servers: [
    { name: "github", running: true, tools: 12, error: "" },
    { name: "notion", running: false, tools: 0, error: "could not start MCP server 'notion'" },
  ],
};

describe("McpServersPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
  });

  it("lists configured servers with their tool counts and start errors", async () => {
    daemonMocks.call.mockResolvedValue(configured);
    render(McpServersPanel);

    expect(await screen.findByText("github")).toBeTruthy();
    expect(screen.getByText("12 tools")).toBeTruthy();
    expect(screen.getByText("Not running")).toBeTruthy();
    expect(screen.getByText("could not start MCP server 'notion'")).toBeTruthy();
    expect(screen.getByText(configured.config_path)).toBeTruthy();
    expect(daemonMocks.call).toHaveBeenCalledWith("mcp_client_status");
  });

  it("restarts the servers through mcp_client_refresh", async () => {
    daemonMocks.call.mockImplementation((method: string) =>
      Promise.resolve(method === "mcp_client_status" ? { ...configured, servers: [] } : configured),
    );
    render(McpServersPanel);
    expect(await screen.findByText(/No servers configured/)).toBeTruthy();

    await fireEvent.click(screen.getByRole("button", { name: "Reload servers" }));

    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("mcp_client_refresh"));
    expect(await screen.findByText("github")).toBeTruthy();
  });

  it("shows the daemon's error instead of an empty list", async () => {
    daemonMocks.call.mockRejectedValue(new Error("Not connected to the Heliox daemon"));
    render(McpServersPanel);

    expect((await screen.findByRole("alert")).textContent).toContain("Not connected to the Heliox daemon");
  });
});

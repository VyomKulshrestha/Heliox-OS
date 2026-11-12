import { render, screen } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  handlers: new Set<(method: string, params: unknown) => void>(),
  call: vi.fn(),
}));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: (method: string, params: unknown) => void) => mocks.handlers.add(handler),
  offNotification: (handler: (method: string, params: unknown) => void) => mocks.handlers.delete(handler),
  requireOkResult: (result: { status?: string; message?: string }, fallback: string) => {
    if (result.status !== "ok") throw new Error(result.message || fallback);
    return result;
  },
}));

import AgentMeshPanel from "./AgentMeshPanel.svelte";

const STARTING = { status: "starting", message: "The agent mesh is still starting. Ask again in a moment." };

describe("a late part's panel while the part is still starting", () => {
  beforeEach(() => {
    mocks.handlers.clear();
    mocks.call.mockReset();
  });

  it("shows the daemon's sentence as waiting, not an error, and asks again when the part is announced", async () => {
    mocks.call.mockResolvedValue(STARTING);
    render(AgentMeshPanel);

    expect(await screen.findByText(STARTING.message)).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(mocks.call).toHaveBeenCalledTimes(1);

    for (const handler of [...mocks.handlers]) handler("late_parts", { late_parts: { agent_mesh: "ready" } });
    await vi.waitFor(() => expect(mocks.call).toHaveBeenCalledTimes(2));
  });
});

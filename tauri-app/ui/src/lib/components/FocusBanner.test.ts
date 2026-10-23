import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const daemonMocks = vi.hoisted(() => ({ call: vi.fn(), handlers: [] as Handler[] }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return {
    ...actual,
    call: daemonMocks.call,
    onNotification: (handler: Handler) => daemonMocks.handlers.push(handler),
    offNotification: vi.fn(),
  };
});

import FocusBanner from "./FocusBanner.svelte";

describe("FocusBanner", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.handlers.length = 0;
  });

  it("shows a running session and stops it", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", active: true, until: 1792760400, held: 2 });
    render(FocusBanner);

    await waitFor(() => expect(screen.getByTestId("focus-banner").textContent).toContain("(2 held)"));
    await fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("focus_stop");

    daemonMocks.handlers.forEach((handler) => handler("focus_ended", { text: "Focus session done." }));
    await waitFor(() => expect(screen.queryByTestId("focus-banner")).toBeNull());
  });

  it("stays hidden when no session runs", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", active: false, until: null, held: 0 });
    render(FocusBanner);

    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("focus_status"));
    expect(screen.queryByTestId("focus-banner")).toBeNull();
  });
});

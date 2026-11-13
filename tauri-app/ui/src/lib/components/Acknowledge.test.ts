import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  sent: [] as unknown[],
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      return { status: "ok" };
    }
    if (method === "get_config") return mocks.config;
    return { status: "ok" };
  }),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

import { settings } from "../stores/settings";
import Acknowledge from "./Acknowledge.svelte";

describe('Say "Mm" while thinking', () => {
  beforeEach(() => {
    mocks.sent.length = 0;
  });

  it("is off by default, and says what it is for and that it is new", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    render(Acknowledge);
    const toggle = screen.getByRole("button", { name: "Say a short sound while thinking" });
    expect(toggle.classList.contains("active")).toBe(false);
    expect(screen.getByText(/so you are not left in silence/)).toBeTruthy();
    expect(screen.getByText(/Off for now while the sounds are being checked/)).toBeTruthy();
  });

  it("turns on with one press, saved under conversation", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    render(Acknowledge);
    await fireEvent.click(screen.getByRole("button", { name: "Say a short sound while thinking" }));
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "conversation", values: { acknowledge: true } }]));
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Reply = Record<string, unknown> | undefined;
const mocks = vi.hoisted(() => ({
  update: (async () => ({ status: "ok" })) as (params: unknown) => Promise<Reply>,
  sent: [] as unknown[],
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      return mocks.update(params);
    }
    if (method === "get_config") return { status: "ok", config: {} };
    return { status: "ok" };
  }),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

import ProfileSharing from "./ProfileSharing.svelte";

describe("Your profile and the model", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    mocks.update = async () => ({ status: "ok" });
  });

  it("says in words what a cloud model is sent, and never health, money or secrets", () => {
    render(ProfileSharing);
    expect(screen.getByText(/your name, your answer style and the parts of your profile/)).toBeTruthy();
    expect(screen.getByText(/Never health, money, secrets/)).toBeTruthy();
  });

  it("turns it off on a bare ok", async () => {
    render(ProfileSharing);
    await fireEvent.click(screen.getByRole("button", { name: "Send your profile to a cloud model" }));
    await waitFor(() => expect(screen.getByText("Done: nothing about you goes to a cloud model now.")).toBeTruthy());
    expect(mocks.sent).toEqual([{ section: "conversation", values: { share_profile: "off" } }]);
    expect(screen.getByText("Off: nothing about you goes to a cloud model")).toBeTruthy();
  });

  it("says the daemon's reason when it refuses", async () => {
    mocks.update = async () => ({ status: "error", message: "Your config file could not be read." });
    render(ProfileSharing);
    await fireEvent.click(screen.getByRole("button", { name: "Send your profile to a cloud model" }));
    await waitFor(() => expect(screen.getByText("Your config file could not be read.")).toBeTruthy());
  });

  it("says nothing changed when there is no reply at all", async () => {
    mocks.update = async () => undefined;
    render(ProfileSharing);
    await fireEvent.click(screen.getByRole("button", { name: "Send your profile to a cloud model" }));
    await waitFor(() => expect(screen.getByText("The daemon rejected the setting.")).toBeTruthy());
  });
});

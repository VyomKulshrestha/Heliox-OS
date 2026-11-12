import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({ call: vi.fn(), handlers: [] as Handler[] }));

vi.mock("../api/daemon", () => ({
  call: mocks.call,
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-clipboard-manager", () => ({ writeText: vi.fn() }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn(async () => true),
  requestPermission: vi.fn(async () => "granted"),
  sendNotification: vi.fn(),
}));
vi.mock("../stores/session", () => ({ session: { sendCommand: vi.fn() } }));

import { offers } from "../stores/offers";
import LiveCards from "./LiveCards.svelte";
import GettingToKnowForm from "./GettingToKnowForm.svelte";

const notify = (method: string, params: unknown) => mocks.handlers.forEach((handler) => handler(method, params));

const intro = {
  id: "i1",
  key: "get_to_know",
  kind: "get_to_know",
  text: "Would you like to tell me a little about yourself? Three short questions, and you can skip any of them.",
  request: "get to know me",
};

describe("getting to know the person", () => {
  beforeEach(() => {
    mocks.call.mockReset();
    offers.reset();
  });

  it("offers the chat, a form, not now and don't ask again", async () => {
    mocks.call.mockResolvedValue({ status: "ok", reply: "Okay." });
    render(LiveCards);
    notify("offer", intro);
    await tick();
    for (const name of ["Ask me in the chat", "Use a form", "Not now", "Why?", "Don't ask again"]) {
      expect(screen.getByRole("button", { name })).toBeTruthy();
    }
    await fireEvent.click(screen.getByRole("button", { name: "Use a form" }));
    expect(screen.getByTestId("gtk-form")).toBeTruthy();
    expect(mocks.call).not.toHaveBeenCalled(); // opening the form answers nothing yet
  });

  it("saves only what was filled in, shows every stored line and removes one", async () => {
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.action === "save")
        return {
          status: "ok",
          stored: [
            { id: "call_me", text: "I call you Sam" },
            { id: "style:length", text: "you want short answers" },
          ],
          problems: [],
          reply: "Thanks.",
        };
      if (params?.action === "remove") return { status: "ok", profile_empty: false };
      return { status: "ok" };
    });
    const done = vi.fn();
    render(GettingToKnowForm, { ondone: done });
    await fireEvent.input(screen.getByLabelText("What should I call you?"), { target: { value: "Sam" } });
    await fireEvent.click(screen.getByLabelText("Short answers"));
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByTestId("gtk-stored")).toBeTruthy());
    expect(mocks.call).toHaveBeenCalledWith("getting_to_know", {
      action: "save",
      name: "Sam",
      use: "",
      length: "short",
      speaking: "",
    });
    expect(screen.getByText("I call you Sam")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Remove: you want short answers" }));
    await waitFor(() => expect(screen.queryByText("you want short answers")).toBeNull());
    expect(mocks.call).toHaveBeenCalledWith("getting_to_know", { action: "remove", id: "style:length" });
    await fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(done).toHaveBeenCalled();
  });
});

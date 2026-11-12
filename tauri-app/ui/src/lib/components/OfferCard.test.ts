import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
const mocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Handler[],
}));

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

import { live } from "../stores/live";
import { offers } from "../stores/offers";
import LiveCards from "./LiveCards.svelte";
import OffersPanel, { stateLine } from "./OffersPanel.svelte";

const notify = (method: string, params: unknown) => mocks.handlers.forEach((handler) => handler(method, params));

const offer = {
  id: "o1",
  key: "usual_app:outlook.exe",
  kind: "usual_app",
  text: "You usually open Outlook about now. Want me to open it?",
  request: "open Outlook",
};

const status = (over: Record<string, unknown> = {}) => ({
  status: "ok",
  mode: "auto",
  active: true,
  waiting_for_profile: false,
  silenced: [],
  per_hour: 1,
  per_day: 3,
  ...over,
});

describe("offer cards", () => {
  beforeEach(() => {
    mocks.call.mockReset();
    offers.reset();
    live.reset();
  });

  it("shows an offer with its four answers and asks why without taking the card down", async () => {
    mocks.call.mockResolvedValue(status({ reply: "On 4 of your last 5 weekdays you first opened Outlook at 9:00." }));
    render(LiveCards);
    notify("offer", offer);
    await tick();
    expect(screen.getByTestId("offer-card").textContent).toContain("You usually open Outlook about now.");
    for (const name of ["Yes", "Not today", "Why?", "Don't offer this"]) {
      expect(screen.getByRole("button", { name })).toBeTruthy();
    }
    await fireEvent.click(screen.getByRole("button", { name: "Why?" }));
    await waitFor(() => expect(screen.getByTestId("offer-why").textContent).toContain("4 of your last 5 weekdays"));
    expect(mocks.call).toHaveBeenCalledWith("offers", { action: "answer", id: "o1", answer: "why" });
    expect(screen.queryByRole("button", { name: "Why?" })).toBeNull();
    expect(screen.getByTestId("offer-card")).toBeTruthy();
  });

  it.each([
    ["Yes", "yes"],
    ["Not today", "not_today"],
    ["Don't offer this", "silence"],
  ])("%s answers the offer and takes the card down", async (label, answer) => {
    mocks.call.mockResolvedValue(status({ reply: "Okay." }));
    render(LiveCards);
    notify("offer", offer);
    await tick();
    await fireEvent.click(screen.getByRole("button", { name: label }));
    expect(mocks.call).toHaveBeenCalledWith("offers", { action: "answer", id: "o1", answer });
    await tick();
    expect(screen.queryByTestId("offer-card")).toBeNull();
  });

  it("goes away when the offer is answered by voice", async () => {
    render(LiveCards);
    notify("offer", offer);
    await tick();
    notify("offer_withdrawn", { id: "o1", reason: "answered" });
    await tick();
    expect(screen.queryByTestId("offer-card")).toBeNull();
  });
});

describe("the offers settings", () => {
  beforeEach(() => mocks.call.mockReset());

  it("says in plain words what happens now", () => {
    expect(stateLine(status({ mode: "off", active: false }) as never)).toBe("Offers are off.");
    expect(stateLine(status({ active: false, waiting_for_profile: true }) as never)).toMatch(/^Waiting until Heliox/);
    expect(stateLine(status() as never)).toBe("On: at most 1 an hour and 3 a day, and only when you're not busy.");
  });

  it("changes the mode and lets a silenced offer be made again", async () => {
    const silenced = [{ key: "habit:reading", label: "Logging reading" }];
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.action === "status") return status({ silenced });
      if (params?.action === "set") return status({ mode: params.mode, silenced });
      if (params?.action === "unsilence") return status({ mode: "off", silenced: [] });
      return status();
    });
    render(OffersPanel);
    await waitFor(() => expect(screen.getByText("Logging reading")).toBeTruthy());
    await fireEvent.click(screen.getByRole("radio", { name: /^Off/ }));
    await waitFor(() => expect(mocks.call).toHaveBeenCalledWith("offers", { action: "set", mode: "off" }));
    await waitFor(() => expect(screen.getByTestId("offers-state").textContent).toBe("Offers are off."));
    await fireEvent.click(screen.getByRole("button", { name: "Offer again" }));
    await waitFor(() => expect(screen.queryByText("Logging reading")).toBeNull());
    expect(mocks.call).toHaveBeenCalledWith("offers", { action: "unsilence", key: "habit:reading" });
    expect(screen.getByText(/None\. "Don't offer this"/)).toBeTruthy();
  });

  it("shows the daemon's error", async () => {
    mocks.call.mockResolvedValue({ status: "error", message: "mode must be auto, on or off" });
    render(OffersPanel);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("mode must be auto, on or off"));
  });
});

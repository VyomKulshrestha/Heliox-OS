import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  sent: [] as unknown[],
  answer: { status: "ok" } as unknown,
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string, params: unknown) => {
    if (method === "update_config") {
      mocks.sent.push(params);
      if (mocks.answer === undefined) throw new Error("no daemon");
      return mocks.answer;
    }
    if (method === "get_config") return mocks.config;
    return { status: "ok" };
  }),
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

import { settings } from "../stores/settings";
import Disagree from "./Disagree.svelte";

const LABEL = "When Heliox pushes back on decisions";

describe("Push back on decisions", () => {
  beforeEach(() => {
    mocks.sent.length = 0;
    mocks.answer = { status: "ok" };
  });

  it("shows When it matters when nothing is set, and says it never refuses or approves", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    render(Disagree);
    const select = screen.getByRole("combobox", { name: LABEL }) as HTMLSelectElement;
    expect(select.value).toBe("when_it_matters");
    expect(screen.getByText(/It never refuses, and it\s+never approves anything for you/)).toBeTruthy();
  });

  it("shows the saved value", async () => {
    mocks.config = { conversation: { disagree: "often" } };
    await settings.reloadFromDaemon();
    render(Disagree);
    expect((screen.getByRole("combobox", { name: LABEL }) as HTMLSelectElement).value).toBe("often");
  });

  it("saves a choice under conversation", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    render(Disagree);
    await fireEvent.change(screen.getByRole("combobox", { name: LABEL }), { target: { value: "off" } });
    await waitFor(() => expect(mocks.sent).toEqual([{ section: "conversation", values: { disagree: "off" } }]));
  });

  it("goes back to the saved choice when the daemon refuses, and says so", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    mocks.answer = { status: "error", message: "Refused." };
    render(Disagree);
    const select = screen.getByRole("combobox", { name: LABEL }) as HTMLSelectElement;
    const saved = select.value;
    expect(saved).not.toBe("often");
    await fireEvent.change(select, { target: { value: "often" } });
    await waitFor(() => expect(screen.getByText("Refused.")).toBeTruthy());
    expect(select.value).toBe(saved);
  });

  it("renders with a bare answer, a refusal and no daemon at all", async () => {
    mocks.config = { conversation: {} };
    await settings.reloadFromDaemon();
    for (const answer of [{ status: "ok" }, { status: "error", message: "Refused." }, undefined]) {
      mocks.answer = answer;
      const { unmount } = render(Disagree);
      await fireEvent.change(screen.getByRole("combobox", { name: LABEL }), { target: { value: "often" } });
      await waitFor(() =>
        expect((screen.getByRole("combobox", { name: LABEL }) as HTMLSelectElement).disabled).toBe(false),
      );
      unmount();
    }
  });
});

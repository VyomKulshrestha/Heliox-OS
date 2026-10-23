import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call, onNotification: vi.fn(), offNotification: vi.fn() };
});

import VoiceShortcutsPanel from "./VoiceShortcutsPanel.svelte";

describe("VoiceShortcutsPanel", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("adds, lists and removes a shortcut, and shows a refusal", async () => {
    let shortcuts: { phrase: string; command: string }[] = [];
    daemonMocks.call.mockImplementation(async (method: string, params: { phrase?: string; command?: string }) => {
      if (method === "shortcut_add") {
        if (params.phrase === "stop") return { status: "error", message: '"stop" already means something to Heliox' };
        shortcuts = [{ phrase: params.phrase ?? "", command: params.command ?? "" }];
      }
      if (method === "shortcut_remove") shortcuts = [];
      return { status: "ok", shortcuts };
    });
    render(VoiceShortcutsPanel);
    await waitFor(() => expect(screen.getByText("No shortcuts yet.")).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("Shortcut phrase"), { target: { value: "start work" } });
    await fireEvent.input(screen.getByLabelText("Shortcut command"), { target: { value: "open VS Code" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(screen.getByText('"start work"')).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Remove start work" }));
    await waitFor(() => expect(screen.getByText("No shortcuts yet.")).toBeTruthy());

    await fireEvent.input(screen.getByLabelText("Shortcut phrase"), { target: { value: "stop" } });
    await fireEvent.input(screen.getByLabelText("Shortcut command"), { target: { value: "x" } });
    await fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("already means something"));
  });
});

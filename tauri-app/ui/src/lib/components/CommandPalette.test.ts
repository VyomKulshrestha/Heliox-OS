import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

import CommandPalette from "./CommandPalette.svelte";

describe("CommandPalette", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "receipts_list") return { status: "ok", receipts: [{ request: "tidy my downloads" }] };
      if (method === "command_guide")
        return {
          status: "ok",
          groups: [
            { group: "Reminders and timers", about: "", examples: ["set a timer for 10 minutes"] },
            { group: "Notes", about: "", examples: ["read my notes"] },
          ],
        };
      return { status: "ok" };
    });
  });

  it("lists recent requests and guide examples, filters, and picks with the keyboard", async () => {
    const onpick = vi.fn();
    render(CommandPalette, { onpick, onclose: vi.fn() });

    await waitFor(() => expect(screen.getByText("tidy my downloads")).toBeTruthy());
    expect(screen.getByText("set a timer for 10 minutes")).toBeTruthy();

    const input = screen.getByLabelText("Search commands");
    await fireEvent.input(input, { target: { value: "timer" } });
    expect(screen.queryByText("read my notes")).toBeNull();
    await fireEvent.keyDown(input, { key: "Enter" });

    expect(onpick).toHaveBeenCalledWith("set a timer for 10 minutes");
  });

  it("uses free text when nothing matches, and Esc closes", async () => {
    const onpick = vi.fn();
    const onclose = vi.fn();
    render(CommandPalette, { onpick, onclose });
    const input = screen.getByLabelText("Search commands");

    await fireEvent.input(input, { target: { value: "zzz unknown" } });
    await fireEvent.keyDown(input, { key: "Enter" });
    expect(onpick).toHaveBeenCalledWith("zzz unknown");

    await fireEvent.keyDown(input, { key: "Escape" });
    expect(onclose).toHaveBeenCalled();
  });
});

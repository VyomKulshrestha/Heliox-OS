import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ call: vi.fn() }));
// A vi.fn whose promise rejects fails the test under vitest 4 even when the component catches it, so the mock
// resolves to an Error and this plain wrapper throws it.
vi.mock("../api/daemon", () => ({
  call: async (...args: unknown[]) => {
    const result = await mocks.call(...args);
    if (result instanceof Error) throw result;
    return result;
  },
  onNotification: vi.fn(),
  offNotification: vi.fn(),
}));

import LiveShellCheck from "./LiveShellCheck.svelte";

const zsh = {
  installed: false,
  script: "/home/a/.local/share/heliox/shell/heliox-live.zsh",
  startup_file: "/home/a/.zshrc",
  line: '[ -f "/home/a/.local/share/heliox/shell/heliox-live.zsh" ] && source "..."  # Heliox Live terminal check',
  remove: "Delete the line ending in '# Heliox Live terminal check ...' from /home/a/.zshrc.",
};
const fish = {
  installed: false,
  script: "/home/a/.config/fish/conf.d/heliox-live.fish",
  startup_file: "",
  line: "",
  remove: "Delete /home/a/.config/fish/conf.d/heliox-live.fish.",
};
const status = (over: Record<string, unknown> = {}) => ({
  status: "ok",
  supported: true,
  listening: true,
  problem: "",
  shells: { zsh, fish },
  bash: "bash only offers hooks after Enter; with ble.sh it could work like zsh, which is not built yet.",
  ...over,
});

describe("Live's terminal check in Settings", () => {
  beforeEach(() => mocks.call.mockReset());

  it("shows exactly what Add does before anything is added, then adds and removes on click", async () => {
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.action === "install") return status({ shells: { zsh: { ...zsh, installed: true }, fish } });
      if (params?.action === "uninstall") return status();
      return status();
    });
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getByTestId("live-shell-check")).toBeTruthy());
    expect(mocks.call).toHaveBeenCalledTimes(1); // only the status: nothing added by opening Settings
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[0]);
    const preview = screen.getByTestId("live-shell-preview-zsh").textContent ?? "";
    expect(preview).toContain("adds this one line to /home/a/.zshrc");
    expect(preview).toContain("# Heliox Live terminal check");
    expect(preview).toContain("To take it out yourself:");
    await fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(screen.getByText("zsh: added")).toBeTruthy());
    expect(mocks.call).toHaveBeenCalledWith("live_shell", { action: "install", shell: "zsh" });
    await fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(screen.getByText("zsh: not added")).toBeTruthy());
    expect(screen.getByText(/bash: bash only offers hooks after Enter/)).toBeTruthy();
  });

  it("fish changes no start-up file, and says so", async () => {
    mocks.call.mockResolvedValue(status());
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Show what Add does" }).length).toBe(2));
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[1]);
    expect(screen.getByTestId("live-shell-preview-fish").textContent).toContain("no start-up file is changed");
  });

  it.each([
    ["a bare ok", async () => ({ status: "ok" })],
    ["a refusal", async () => ({ status: "error", message: "not ready" })],
    ["no answer", async () => new Error("Heliox is not connected")],
    [
      "Windows",
      async () => ({ status: "ok", supported: false, message: "On Windows Heliox reads the terminal itself." }),
    ],
  ])("renders nothing and does not throw with %s", async (_label, answer) => {
    mocks.call.mockImplementation(answer);
    render(LiveShellCheck);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByTestId("live-shell-check")).toBeNull();
  });
});

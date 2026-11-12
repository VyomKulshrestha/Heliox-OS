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
const passed = { ok: true, problem: "", install: "", answered_ms: 30, silent_ms: null, bound_ms: 0 };
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
      if (params?.action === "test") return status({ ready: { zsh: passed } });
      if (params?.action === "uninstall") return status();
      return status();
    });
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getByTestId("live-shell-check")).toBeTruthy());
    expect(mocks.call).toHaveBeenCalledTimes(1); // only the status: nothing added by opening Settings
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[0]);
    expect(mocks.call).toHaveBeenCalledWith("live_shell", { action: "test", shell: "zsh" }); // tried here first
    await waitFor(() => expect(screen.getByRole("button", { name: "Add" })).toBeTruthy());
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

  it("fish changes no start-up file, says so, and says how long Enter can wait", async () => {
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) =>
      params?.action === "test"
        ? status({ ready: { fish: { ...passed, answered_ms: 40, silent_ms: 1010, bound_ms: 1000 } } })
        : status(),
    );
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Show what Add does" }).length).toBe(2));
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[1]);
    await waitFor(() => expect(screen.getByTestId("live-shell-bound")).toBeTruthy());
    expect(screen.getByTestId("live-shell-preview-fish").textContent).toContain("no start-up file is changed");
    const bound = screen.getByTestId("live-shell-bound").textContent ?? "";
    expect(bound).toContain("waits at most 1 second");
    expect(bound).toContain("If Heliox is not running, Enter does not wait at all.");
    expect(bound).toContain("gave up after 1010 ms");
  });

  it("says plainly what to install when the hook could not reach Heliox, and offers no Add", async () => {
    const cannot = {
      ok: false,
      problem: "fish needs nc (netcat) to reach Heliox, and there is none here.",
      install: "Install the OpenBSD netcat: sudo apt install netcat-openbsd.",
    };
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) =>
      params?.action === "test" ? status({ ready: { fish: cannot } }) : status(),
    );
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Show what Add does" }).length).toBe(2));
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[1]);
    await waitFor(() => expect(screen.getByTestId("live-shell-cannot-fish")).toBeTruthy());
    const text = screen.getByTestId("live-shell-cannot-fish").textContent ?? "";
    expect(text).toContain("fish needs nc");
    expect(text).toContain("sudo apt install netcat-openbsd");
    expect(screen.queryByRole("button", { name: "Add" })).toBeNull();
    expect(mocks.call).not.toHaveBeenCalledWith("live_shell", { action: "install", shell: "fish" });
  });

  it("a refused install shows the reason and keeps Add away", async () => {
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.action === "test") return status({ ready: { zsh: passed } });
      if (params?.action === "install")
        return {
          status: "error",
          message: "Not added: This zsh cannot open a socket.",
          ready: { zsh: { ok: false, problem: "This zsh cannot open a socket." } },
        };
      return status();
    });
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Show what Add does" }).length).toBe(2));
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[0]);
    await waitFor(() => expect(screen.getByRole("button", { name: "Add" })).toBeTruthy());
    await fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Not added"));
    expect(screen.queryByRole("button", { name: "Add" })).toBeNull();
    expect(screen.getByText("zsh: not added")).toBeTruthy();
  });

  it("a test that gets no answer leaves Add away", async () => {
    mocks.call.mockImplementation(async (_method: string, params?: Record<string, unknown>) =>
      params?.action === "test" ? new Error("Heliox is not connected") : status(),
    );
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Show what Add does" }).length).toBe(2));
    await fireEvent.click(screen.getAllByRole("button", { name: "Show what Add does" })[0]);
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Add" })).toBeNull();
  });

  it("says what kitty and WezTerm need, in one line each", async () => {
    mocks.call.mockResolvedValue(
      status({
        terminals: {
          wezterm: { found: true, needs: "", problem: "" },
          kitty: {
            found: true,
            ready: false,
            needs: "kitty answers only with remote control on: add 'allow_remote_control socket-only'.",
            problem: "",
          },
        },
      }),
    );
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getByTestId("live-terminals")).toBeTruthy());
    const text = screen.getByTestId("live-terminals").textContent ?? "";
    expect(text).toContain("WezTerm: read directly while Live is on, nothing to add.");
    expect(text).toContain("kitty: kitty answers only with remote control on");
  });

  it("a terminal it cannot find, or no terminal list at all, is said plainly or left out", async () => {
    mocks.call.mockResolvedValue(status({ terminals: { wezterm: { found: false } } }));
    render(LiveShellCheck);
    await waitFor(() => expect(screen.getByTestId("live-terminals")).toBeTruthy());
    expect(screen.getByTestId("live-terminals").textContent).toContain("WezTerm: not found on this computer.");
    expect(screen.getByTestId("live-terminals").textContent).toContain("kitty: not found on this computer.");
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

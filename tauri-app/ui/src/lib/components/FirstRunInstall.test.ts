import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it, vi } from "vitest";
import {
  catalog,
  defaultChoice,
  failedGroups,
  formatMb,
  pythonSentence,
  screenFor,
  totalMb,
  type SetupState,
} from "../firstRun";
import FirstRunInstall from "./FirstRunInstall.svelte";

const invoke = vi.fn(async (..._args: unknown[]) => undefined);
vi.mock("../api/invoke", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

function state(partial: Partial<SetupState>): SetupState {
  return {
    phase: "needs_choice",
    python: { kind: "system", version: "3.12.3", path: "/usr/bin/python3.12", download_bytes: 0, source: "" },
    steps: [],
    log_tail: [],
    ...partial,
  };
}

describe("first-run choices", () => {
  it("offers the platform readers only on macOS and Linux and preselects them and voice with a microphone", () => {
    expect(catalog("Win32").some((g) => g.id === "platform")).toBe(false);
    const mac = catalog("MacIntel");
    expect(mac[0].label).toBe("macOS readers");
    expect(defaultChoice(mac, false)).toEqual(["platform"]);
    expect(defaultChoice(mac, true)).toEqual(["platform", "voice"]);
    expect(defaultChoice(catalog("Win32"), false)).toEqual([]);
  });

  it("says what Python will be used and what a private one costs", () => {
    expect(pythonSentence(undefined)).toContain("Checking");
    expect(pythonSentence(state({}).python)).toContain("Python 3.12.3 already on this computer");
    const download = pythonSentence({
      kind: "download",
      version: "",
      path: "",
      download_bytes: 16_682_639,
      source: "x",
    });
    expect(download).toContain("about 47 MB");
    expect(download).toContain("github.com/astral-sh/uv");
    expect(pythonSentence({ kind: "none", version: "", path: "", download_bytes: 0, source: "" })).toContain(
      "Install Python 3.11",
    );
    expect(formatMb(900)).toBe("about 900 MB");
    expect(formatMb(1500)).toBe("about 1.5 GB");
    const groups = catalog("Linux x86_64");
    expect(totalMb(groups, ["platform"], state({}).python)).toBe(111);
  });

  it("maps every setup phase to a screen", () => {
    expect(screenFor(null)).toBe("hidden");
    expect(screenFor(state({ phase: "installed" }))).toBe("hidden");
    expect(screenFor(state({ phase: "needs_choice" }))).toBe("choose");
    expect(screenFor(state({ phase: "running" }))).toBe("installing");
    expect(screenFor(state({ phase: "failed" }))).toBe("failed");
    expect(screenFor(state({ phase: "done" }))).toBe("hidden");
    const partly = state({
      phase: "done",
      steps: [
        { id: "core", status: "done", message: "", hint: "", detail: "" },
        { id: "voice", status: "failed", message: "Could not be installed.", hint: "Press Retry.", detail: "" },
      ],
    });
    expect(screenFor(partly)).toBe("done");
    expect(failedGroups(partly)).toEqual(["voice"]);
  });
});

describe("FirstRunInstall", () => {
  it("installs nothing until the person presses the button, then sends only the chosen groups", async () => {
    invoke.mockClear();
    render(FirstRunInstall, { initial: state({}) });
    expect(screen.getByRole("heading", { name: "Set up Heliox" })).toBeTruthy();
    expect(invoke).not.toHaveBeenCalled();
    await fireEvent.click(screen.getByRole("checkbox", { name: /Browser automation/ }));
    await fireEvent.click(screen.getByRole("button", { name: "Set up Heliox" }));
    expect(invoke).toHaveBeenCalledWith("setup_run", { groups: ["browser"] });
  });

  it("shows a failed step's reason and what to do, with a retry", async () => {
    invoke.mockClear();
    render(FirstRunInstall, {
      initial: state({
        phase: "failed",
        steps: [
          {
            id: "python",
            status: "failed",
            message: "Could not create the environment with /usr/bin/python3.12.",
            hint: "On Debian or Ubuntu install python3-venv.",
            detail: "",
          },
        ],
        log_tail: ["Error: ensurepip is not available"],
      }),
    });
    expect(screen.getByText("Setup stopped")).toBeTruthy();
    expect(screen.getByText(/install python3-venv/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText(/ensurepip is not available/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(invoke).toHaveBeenCalledWith("setup_run", { groups: [] });
  });

  it("shows progress while installing and lets a partly finished setup continue", async () => {
    const running = render(FirstRunInstall, {
      initial: state({
        phase: "running",
        steps: [
          { id: "core", status: "running", message: "Installing Heliox's core", hint: "", detail: "Collecting numpy" },
        ],
      }),
    });
    expect(screen.getByText("Setting up Heliox")).toBeTruthy();
    expect(screen.getByText("Collecting numpy")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    running.unmount();

    render(FirstRunInstall, {
      initial: state({
        phase: "done",
        steps: [
          { id: "voice", status: "failed", message: "Could not be installed.", hint: "Press Retry.", detail: "" },
        ],
      }),
    });
    expect(screen.getByText("Heliox is ready")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.queryByText("Heliox is ready")).toBeNull();
  });
});

describe("OptionalParts", () => {
  it("installs one group when its button is pressed", async () => {
    const { default: OptionalParts } = await import("./OptionalParts.svelte");
    invoke.mockClear();
    render(OptionalParts, { force: true });
    await fireEvent.click(screen.getAllByRole("button", { name: "Install or repair" })[0]);
    expect(invoke).toHaveBeenCalledWith("setup_run", { groups: [expect.any(String)] });
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { protectedPackageExample } from "../firstRun";
import SetupWizard from "./SetupWizard.svelte";

const mocks = vi.hoisted(() => ({ call: vi.fn(), updateSectionDetailed: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: mocks.call };
});
vi.mock("../stores/settings", () => ({
  settings: {
    subscribe(run: (value: unknown) => void) {
      run({ restrictions: { protected_folders: [], protected_packages: [], blocked_commands: [] } });
      return () => undefined;
    },
    updateSectionDetailed: mocks.updateSectionDetailed,
  },
}));

describe("Setup wizard protected packages", () => {
  const source = readFileSync(resolve("src/lib/components/SetupWizard.svelte"), "utf8");

  it("starts with nothing protected on every system", () => {
    // It started as "firefox, nautilus" everywhere, so a Windows first run saved two Linux names into
    // restrictions.toml, and every action whose path held "firefox" scored as touching a protected package.
    expect(source).toContain('let protectedPackages = $state("");');
    expect(source).not.toContain('$state("firefox');
  });

  it("shows example names this system's package manager uses", () => {
    expect(protectedPackageExample("Win32")).toBe("Mozilla.Firefox, Git.Git");
    expect(protectedPackageExample("MacIntel")).toBe("firefox, git");
    expect(protectedPackageExample("Linux x86_64")).toBe("firefox, nautilus, gnome-shell");
    expect(source).toContain("placeholder={packageExample}");
    expect(source).not.toContain('placeholder="firefox, nautilus, gnome-shell"');
  });

  it("gives a protected-folder example that names a real folder", () => {
    expect(source).toContain('placeholder={"~/Documents/private\\n~/.ssh"}');
  });
});

describe("Setup wizard model approval", () => {
  beforeEach(() => {
    localStorage.clear();
    mocks.call.mockReset();
    mocks.updateSectionDetailed.mockReset();
    mocks.call.mockImplementation(async (method: string) => {
      if (method === "list_ollama_models") return { status: "ok", models: ["qwen2.5:7b"], available: true };
      if (method === "subscription_status") return { status: "ok", subscription: false, message: "" };
      return { status: "ok" };
    });
  });

  async function goToReady(provider: "ollama" | "cloud") {
    const oncomplete = vi.fn();
    render(SetupWizard, { oncomplete });
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    if (provider === "cloud") {
      await fireEvent.click(screen.getByRole("radio", { name: /Cloud API/ }));
      await fireEvent.change(screen.getByLabelText("Cloud Provider"), { target: { value: "openai" } });
    } else {
      await waitFor(() =>
        expect((screen.getByLabelText("Ollama Model") as HTMLSelectElement).value).toBe("qwen2.5:7b"),
      );
    }
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByRole("heading", { name: "All Set" })).toBeTruthy();
    return oncomplete;
  }

  it("stays on Ready, saves nothing more and does not finish when a cloud model is not approved", async () => {
    mocks.updateSectionDetailed.mockResolvedValue({
      ok: false,
      cancelled: true,
      message: "Not changed.",
      restartRequired: false,
    });
    const oncomplete = await goToReady("cloud");

    await fireEvent.click(screen.getByRole("button", { name: "Launch Heliox OS" }));

    const note = await screen.findByRole("status");
    expect(note.textContent).toBe("Nothing was changed; choose a model on this computer or approve the cloud one.");
    expect(note.classList.contains("finish-error")).toBe(false);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("heading", { name: "All Set" })).toBeTruthy();
    expect(mocks.updateSectionDetailed).toHaveBeenCalledTimes(1);
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith(
      "model",
      expect.objectContaining({ provider: "cloud", cloud_provider: "openai" }),
    );
    expect(oncomplete).not.toHaveBeenCalled();
    expect(localStorage.getItem("heliox_first_run_complete")).toBeNull();
  });

  it("finishes with a local Ollama model without any approval", async () => {
    mocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    const oncomplete = await goToReady("ollama");

    await fireEvent.click(screen.getByRole("button", { name: "Launch Heliox OS" }));

    await waitFor(() => expect(oncomplete).toHaveBeenCalledTimes(1));
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith(
      "model",
      expect.objectContaining({ provider: "ollama", ollama_model: "qwen2.5:7b" }),
    );
    expect(localStorage.getItem("heliox_first_run_complete")).toBe("true");
    expect(screen.queryByRole("status")).toBeNull();
  });
});

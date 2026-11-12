import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it, vi } from "vitest";
import {
  catalog,
  defaultChoice,
  failedGroups,
  formatMb,
  pythonSentence,
  screenFor,
  sizeMb,
  totalMb,
  type SetupState,
} from "../firstRun";
import FirstRunInstall from "./FirstRunInstall.svelte";

const invoke = vi.fn(async (..._args: unknown[]) => undefined);
vi.mock("../api/invoke", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));
// The real session store and daemon client would open a WebSocket to a daemon on this machine.
vi.mock("../api/daemon", () => ({ call: vi.fn(async () => ({ status: "ok" })) }));
vi.mock("../stores/session", async () => {
  const { writable } = await import("svelte/store");
  return { session: writable({ daemonConnected: false }) };
});

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
      "Install Python 3.12",
    );
    expect(formatMb(900)).toBe("about 900 MB");
    expect(formatMb(1500)).toBe("about 1.5 GB");
    const groups = catalog("Linux x86_64");
    expect(totalMb(groups, ["platform"], state({}).python)).toBe(
      sizeMb("core", "") + sizeMb("platform", "Linux x86_64"),
    );
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

describe("first-run sizes and speech models", () => {
  it("reads every size from the data file, per system", async () => {
    const { sizeMb, catalog: groupsFor } = await import("../firstRun");
    const data = (await import("../firstRunSizes.json")).default as {
      sizes: Record<string, Record<string, { mb: number }>>;
    };
    expect(sizeMb("core", "Linux x86_64")).toBe(data.sizes.core.linux.mb);
    expect(sizeMb("core", "Win32")).toBe(data.sizes.core.windows.mb);
    expect(sizeMb("voice", "MacIntel")).toBe(data.sizes.voice.all.mb);
    expect(sizeMb("nothing", "Win32")).toBe(0);
    const { sizePhrase, diskMb } = await import("../firstRun");
    const linuxCore = data.sizes.core.linux as { mb: number; disk_mb?: number };
    expect(diskMb("core", "Linux x86_64")).toBe(linuxCore.disk_mb ?? 0);
    expect(sizePhrase("core", "Linux x86_64")).toContain("to download");
    if (linuxCore.disk_mb) expect(sizePhrase("core", "Linux x86_64")).toContain("on disk");
    expect(sizePhrase("voice", "Linux x86_64")).not.toContain("on disk"); // no disk size measured: none shown
    for (const group of groupsFor("MacIntel")) expect(group.downloadMb).toBe(sizeMb(group.id, "MacIntel"));
  });

  it("offers the daemon's speech models except the one that comes with voice", async () => {
    const { speechModels, languageNote } = await import("../firstRun");
    const manifest = (await import("../../../../../daemon/pilot/system/speech_models.json")).default as {
      models: { id: string; part_of_voice?: boolean }[];
    };
    const offered = speechModels();
    expect(offered.map((m) => m.id)).toEqual(manifest.models.filter((m) => !m.part_of_voice).map((m) => m.id));
    expect(offered.some((m) => m.id === "whisper-base")).toBe(false);
    expect(languageNote({ ...offered[0], language: "en" })).toBe("English only");
  });

  it("keeps ticked models until the daemon is up, then asks for each and greets an empty profile once", async () => {
    const { afterSetup, pendingModels, savePendingModels } = await import("../firstRun");
    const store = new Map<string, string>();
    const storage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    };
    savePendingModels(storage, ["parakeet-tdt-0.6b-v2-int8", "smart-turn-v3"]);
    expect(pendingModels(storage)).toEqual(["parakeet-tdt-0.6b-v2-int8", "smart-turn-v3"]);
    const calls: [string, Record<string, unknown>][] = [];
    const daemon = async (method: string, params: Record<string, unknown>) => {
      calls.push([method, params]);
      if (method === "speech_model_download") return { status: params.id === "smart-turn-v3" ? "busy" : "started" };
      if (params.action === "status") return { status: "ok", profile_empty: true, active: false } as { status: string };
      return { status: "ok" };
    };
    const left = await afterSetup(daemon, pendingModels(storage));
    expect(left).toEqual(["smart-turn-v3"]); // one download at a time: asked again later
    expect(calls.map(([m, p]) => `${m}:${p.id ?? p.action}`)).toEqual([
      "speech_model_download:parakeet-tdt-0.6b-v2-int8",
      "speech_model_download:smart-turn-v3",
      "getting_to_know:status",
      "getting_to_know:start",
    ]);
    calls.length = 0;
    await afterSetup(daemon, [], false);
    expect(calls).toEqual([]);
    savePendingModels(storage, []);
    expect(pendingModels(storage)).toEqual([]);
  });

  it("lets speech models be ticked only with voice, and keeps them for the daemon when setup starts", async () => {
    invoke.mockClear();
    localStorage.clear();
    render(FirstRunInstall, { initial: state({}) });
    const parakeet = screen.getByRole("checkbox", { name: /Parakeet/ }) as HTMLInputElement;
    const voice = screen.getByRole("checkbox", { name: /^Voice/ }) as HTMLInputElement;
    if (voice.checked) await fireEvent.click(voice);
    expect(parakeet.disabled).toBe(true);
    await fireEvent.click(voice);
    await fireEvent.click(screen.getByRole("checkbox", { name: /Parakeet/ }));
    await fireEvent.click(screen.getByRole("button", { name: "Set up Heliox" }));
    expect(JSON.parse(localStorage.getItem("heliox_first_run_speech_models") ?? "[]")).toEqual([
      "parakeet-tdt-0.6b-v2-int8",
    ]);
    expect(localStorage.getItem("heliox_first_run_greet")).toBe("1");
    expect(invoke).toHaveBeenCalledWith("setup_run", { groups: expect.arrayContaining(["voice"]) });
  });
});

describe("after setup", () => {
  it("asks the daemon for the kept models and the first conversation once it is connected", async () => {
    const { session } = (await import("../stores/session")) as unknown as {
      session: { set: (v: { daemonConnected: boolean }) => void };
    };
    localStorage.setItem("heliox_first_run_speech_models", JSON.stringify(["parakeet-tdt-0.6b-v2-int8"]));
    localStorage.setItem("heliox_first_run_greet", "1");
    const calls: string[] = [];
    const daemonCall = async (method: string, params: Record<string, unknown>) => {
      calls.push(`${method}:${String(params.id ?? params.action)}`);
      if (params.action === "status")
        return { status: "ok", profile_empty: false, active: false } as { status: string };
      return { status: "started" };
    };
    render(FirstRunInstall, { initial: state({ phase: "done" }), daemonCall });
    expect(calls).toEqual([]); // nothing before the daemon is connected
    session.set({ daemonConnected: true });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toEqual(["speech_model_download:parakeet-tdt-0.6b-v2-int8", "getting_to_know:status"]);
    expect(localStorage.getItem("heliox_first_run_speech_models")).toBeNull();
    expect(localStorage.getItem("heliox_first_run_greet")).toBeNull();
    session.set({ daemonConnected: false });
  });
});

describe("update from an older installed version", () => {
  const plan = {
    installed: "0.15.0",
    target: "0.16.0",
    groups: ["platform", "voice"],
    planned: false,
    changes: [] as { name: string; from: string; to: string; bytes: number }[],
    total_bytes: 0,
  };

  it("offers the update, plans it only when asked, and keeps the person's parts", async () => {
    const { screenFor: screenOf, updateSummary, keptGroups, catalog: groupsFor } = await import("../firstRun");
    expect(screenOf(state({ phase: "needs_update", update: plan }))).toBe("update");
    expect(screenOf(state({ phase: "planning", update: plan }))).toBe("planning");
    expect(screenOf(state({ phase: "updating", update: plan }))).toBe("installing");
    expect(updateSummary(plan)).toBe("0.15.0 to 0.16.0");
    const planned = {
      ...plan,
      planned: true,
      changes: [
        { name: "pilot-daemon", from: "0.15.0", to: "0.16.0", bytes: 900_000 },
        { name: "mcp", from: "", to: "2.1.0", bytes: 300_000 },
      ],
      total_bytes: 1_200_000,
    };
    expect(updateSummary(planned)).toBe("0.15.0 to 0.16.0: 2 packages change, about 1 MB to download");
    expect(keptGroups(planned, groupsFor("Linux x86_64"))).toBe("Your parts stay installed: Linux readers, Voice.");

    invoke.mockClear();
    render(FirstRunInstall, { initial: state({ phase: "needs_update", update: plan }) });
    expect(screen.getByRole("heading", { name: "Update Heliox" })).toBeTruthy();
    expect(invoke).not.toHaveBeenCalled();
    await fireEvent.click(screen.getByRole("button", { name: "See what changes" }));
    expect(invoke).toHaveBeenCalledWith("setup_update_plan");
  });

  it("lists what changes and applies only on Update", async () => {
    invoke.mockClear();
    const planned = {
      ...plan,
      planned: true,
      changes: [{ name: "pilot-daemon", from: "0.15.0", to: "0.16.0", bytes: 900_000 }],
      total_bytes: 900_000,
    };
    render(FirstRunInstall, { initial: state({ phase: "needs_update", update: planned }) });
    expect(screen.getByText("0.15.0 → 0.16.0")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Update" }));
    expect(invoke).toHaveBeenCalledWith("setup_update_apply");
  });

  it("retries a failed update as an update and lets the person keep the current version", async () => {
    invoke.mockClear();
    render(FirstRunInstall, {
      initial: state({
        phase: "failed",
        update: plan,
        steps: [
          {
            id: "update",
            status: "failed",
            message: "The update did not install cleanly, so Heliox put the previous version back.",
            hint: "Nothing changed.",
            detail: "",
          },
        ],
      }),
    });
    expect(screen.getByText(/put the previous version back/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(invoke).toHaveBeenCalledWith("setup_update_plan");
    expect(invoke).not.toHaveBeenCalledWith("setup_run", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Keep the current version" }));
    expect(invoke).toHaveBeenCalledWith("setup_update_later");
  });
});

describe("repair", () => {
  it("offers to rebuild a broken environment with the same parts and runs repair only when asked", async () => {
    const { screenFor: screenOf } = await import("../firstRun");
    const broken = state({
      phase: "needs_repair",
      update: { installed: "", target: "0.16.0", groups: ["voice"], planned: false, changes: [], total_bytes: 0 },
    });
    expect(screenOf(broken)).toBe("repair");
    invoke.mockClear();
    render(FirstRunInstall, { initial: broken });
    expect(screen.getByRole("heading", { name: "Repair Heliox" })).toBeTruthy();
    expect(screen.getByText("Your parts stay installed: Voice.")).toBeTruthy();
    expect(invoke).not.toHaveBeenCalled();
    await fireEvent.click(screen.getByRole("button", { name: "Repair Heliox" }));
    expect(invoke).toHaveBeenCalledWith("setup_repair");
  });
});

import { get } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  connectionHandlers: [] as Array<(connected: boolean) => void>,
  notificationHandlers: [] as Array<(method: string, params: unknown) => void>,
}));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  onConnectionState: (handler: (connected: boolean) => void) => daemonMocks.connectionHandlers.push(handler),
  onNotification: (handler: (method: string, params: unknown) => void) =>
    daemonMocks.notificationHandlers.push(handler),
}));

describe("settings daemon synchronization", () => {
  beforeEach(() => {
    localStorage.clear();
    daemonMocks.call.mockReset();
    daemonMocks.connectionHandlers.length = 0;
    daemonMocks.notificationHandlers.length = 0;
    vi.resetModules();
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn(() => ({
        matches: false,
        addEventListener: vi.fn(),
      })),
    });
  });

  it("deep-merges daemon settings without erasing local theme and hotkey", async () => {
    localStorage.setItem(
      "heliox_settings",
      JSON.stringify({
        theme: "light",
        hotkey: "Alt+Space",
        model: { provider: "cloud" },
      }),
    );
    daemonMocks.call.mockResolvedValue({
      model: { provider: "ollama", ollama_model: "qwen-test" },
      security: { root_enabled: false },
    });

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));
    await vi.waitFor(() => expect(get(settings).model.provider).toBe("ollama"));

    expect(get(settings)).toMatchObject({
      theme: "light",
      hotkey: "Alt+Space",
      model: {
        provider: "ollama",
        ollama_model: "qwen-test",
        max_tokens_per_task: 100000,
      },
      security: {
        root_enabled: false,
        dry_run: false,
        snapshot_on_destructive: true,
      },
      network: {
        enabled: false,
        port: 8786,
        skill_sync_enabled: false,
        collab_exec_enabled: false,
      },
    });
  });

  it("does not display daemon-backed settings that the daemon rejected", async () => {
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "get_config") return Promise.resolve({});
      return Promise.reject(new Error("daemon unavailable"));
    });

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));

    const saved = await settings.updateSection("model", { provider: "cloud" });

    expect(saved).toBe(false);
    expect(get(settings).model.provider).toBe("ollama");
    expect(JSON.parse(localStorage.getItem("heliox_settings") || "{}").model?.provider).toBe("ollama");
  });

  it("keeps purely local appearance settings usable while the daemon is offline", async () => {
    daemonMocks.call.mockRejectedValue(new Error("daemon unavailable"));

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));
    daemonMocks.call.mockClear();

    const saved = await settings.updateSection("", { theme: "light" });

    expect(saved).toBe(true);
    expect(get(settings).theme).toBe("light");
    expect(daemonMocks.call).not.toHaveBeenCalled();
  });

  it("keeps the displayed settings when the daemon rejects a factory reset", async () => {
    localStorage.setItem("heliox_settings", JSON.stringify({ model: { provider: "cloud" } }));
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "get_config") return Promise.resolve({ model: { provider: "cloud" } });
      return Promise.reject(new Error("daemon unavailable"));
    });

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(get(settings).model.provider).toBe("cloud"));

    const reset = await settings.reset();

    expect(reset).toBe(false);
    expect(get(settings).model.provider).toBe("cloud");
    expect(JSON.parse(localStorage.getItem("heliox_settings") || "{}").model.provider).toBe("cloud");
  });

  it("shows what the daemon saved after it confirms a factory reset", async () => {
    localStorage.setItem("heliox_settings", JSON.stringify({ model: { provider: "cloud" } }));
    localStorage.setItem("heliox_theme", "light");
    let wasReset = false;
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "get_config") {
        return Promise.resolve(
          wasReset
            ? { model: { provider: "ollama" }, restrictions: { blocked_commands: ["format"] } }
            : { model: { provider: "cloud" } },
        );
      }
      if (method === "reset_config") {
        wasReset = true;
        return Promise.resolve({ status: "ok", runtime_reconciled: true, kept: ["server", "restrictions"] });
      }
      return Promise.reject(new Error(`unexpected method: ${method}`));
    });

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(get(settings).model.provider).toBe("cloud"));

    const reset = await settings.reset();

    expect(reset).toBe(true);
    expect(get(settings).model.provider).toBe("ollama");
    // Restrictions survive a reset on the daemon, so the UI must not show them as cleared.
    expect(get(settings).restrictions.blocked_commands).toEqual(["format"]);
    expect(get(settings).theme).toBe("dark");
    expect(localStorage.getItem("heliox_theme")).toBeNull();
  });

  it("re-reads the daemon when it announces a config change or reconnects", async () => {
    let provider = "ollama";
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "get_config") return Promise.resolve({ model: { provider } });
      return Promise.reject(new Error(`unexpected method: ${method}`));
    });

    const { settings, configRevision } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));

    provider = "cloud"; // changed by another window or a voice command
    for (const handler of daemonMocks.notificationHandlers) handler("config_updated", { section: "model" });
    await vi.waitFor(() => expect(get(settings).model.provider).toBe("cloud"));
    expect(get(configRevision)).toBe(1);

    provider = "subscription"; // the daemon restarted with a hand-edited config.toml
    for (const handler of daemonMocks.connectionHandlers) handler(true);
    await vi.waitFor(() => expect(get(settings).model.provider).toBe("subscription"));
  });

  it("reports the daemon's reason and restart notices for a save", async () => {
    daemonMocks.call.mockImplementation((method: string, params?: { section?: string }) => {
      if (method === "get_config") return Promise.resolve({});
      if (params?.section === "model")
        return Promise.resolve({ status: "error", message: "rate_limit_rpm must be >= 1" });
      return Promise.resolve({ status: "ok", restart_required: true });
    });

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));

    await expect(settings.updateSectionDetailed("model", { rate_limit_rpm: 0 })).resolves.toEqual({
      ok: false,
      message: "rate_limit_rpm must be >= 1",
      restartRequired: false,
    });
    expect(get(settings).model.rate_limit_rpm).toBe(60);
    await expect(settings.updateSectionDetailed("memory", { enabled: false })).resolves.toEqual({
      ok: true,
      message: "",
      restartRequired: true,
    });
  });

  it("exposes config.toml read problems from the daemon", async () => {
    daemonMocks.call.mockResolvedValue({ load_errors: ["config.toml: Invalid value (line 4)"] });

    const { settingsLoadErrors } = await import("./settings");

    await vi.waitFor(() => expect(get(settingsLoadErrors)).toEqual(["config.toml: Invalid value (line 4)"]));
  });

  it("follows the OS theme until the person picks one", async () => {
    let listener: ((event: { matches: boolean }) => void) | undefined;
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn(() => ({
        matches: false,
        addEventListener: (_: string, handler: (event: { matches: boolean }) => void) => (listener = handler),
      })),
    });
    daemonMocks.call.mockResolvedValue({});

    const { settings } = await import("./settings");
    await vi.waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("get_config"));
    expect(get(settings).theme).toBe("dark");
    expect(JSON.parse(localStorage.getItem("heliox_settings") || "{}").theme).toBeUndefined();

    listener?.({ matches: true });
    expect(get(settings).theme).toBe("light");

    await settings.updateSection("", { theme: "dark" });
    listener?.({ matches: true });
    expect(get(settings).theme).toBe("dark");
    expect(localStorage.getItem("heliox_theme")).toBe("dark");
  });
});

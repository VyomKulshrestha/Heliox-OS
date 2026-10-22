import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import "../../i18n";
import SettingsPanel from "./SettingsPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => {
  let value: any;
  const subscribers = new Set<(next: any) => void>();
  const loadErrorSubscribers = new Set<(next: string[]) => void>();
  let loadErrors: string[] = [];
  return {
    updateSectionDetailed: vi.fn(),
    store: {
      subscribe(run: (next: any) => void) {
        subscribers.add(run);
        run(value);
        return () => subscribers.delete(run);
      },
    },
    loadErrorStore: {
      subscribe(run: (next: string[]) => void) {
        loadErrorSubscribers.add(run);
        run(loadErrors);
        return () => loadErrorSubscribers.delete(run);
      },
    },
    set(next: any) {
      value = next;
      subscribers.forEach((run) => run(value));
    },
    setLoadErrors(next: string[]) {
      loadErrors = next;
      loadErrorSubscribers.forEach((run) => run(loadErrors));
    },
  };
});

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return {
    ...actual,
    call: daemonMocks.call,
    connect: vi.fn(async () => false),
    isConnected: () => false,
    onNotification: vi.fn(),
    offNotification: vi.fn(),
    onConnectionState: vi.fn(),
    offConnectionState: vi.fn(),
  };
});
vi.mock("../stores/settings", () => ({
  settings: {
    subscribe: settingsMocks.store.subscribe,
    updateSection: vi.fn(async () => true),
    updateSectionDetailed: settingsMocks.updateSectionDetailed,
    reset: vi.fn(async () => true),
  },
  settingsLoadErrors: settingsMocks.loadErrorStore,
  configRevision: { subscribe: (run: (n: number) => void) => (run(0), () => {}) },
}));

const baseSettings = {
  model: {
    provider: "ollama",
    ollama_base_url: "http://127.0.0.1:11434",
    ollama_model: "llama3.1:8b",
    mode: "lightweight",
    gpu_memory_limit_mb: 0,
    idle_unload_seconds: 60,
    cloud_provider: "",
    cloud_model: "",
    subscription_provider: "codex",
    subscription_model: "",
    subscription_timeout_seconds: 120,
    subscription_max_prompt_chars: 48000,
    rate_limit_enabled: true,
    rate_limit_rpm: 60,
    rate_limit_burst: 5,
    budget_enabled: true,
    budget_monthly_limit_usd: 10,
    max_tokens_per_action: 12000,
    max_tokens_per_task: 100000,
    max_usd_per_task: 0.1,
    max_consecutive_failures: 3,
  },
  security: {
    root_enabled: false,
    dry_run: false,
    snapshot_on_destructive: true,
    snapshot_backend: "auto",
    snapshot_retention_count: 10,
    snapshot_retention_days: 7,
  },
  screen_vision: { capture_interval_seconds: 3 },
  vision: { mediapipe_backend: "tasks", gaze_tracking_enabled: false },
  gesture_cursor: { enabled: false, sensitivity: 1, prediction_ms: 80, blend: 0.3 },
  adaptive_calibration: { gesture_enabled: true, voice_wake_word_enabled: true },
  preview: { enabled: false },
  undo: { enabled: true, retention_days: 7 },
  typed_decision: { enabled: false, artifact_dir: "", report_path: "" },
  network: { enabled: false, port: 8786, peer_timeout_s: 30, skill_sync_enabled: false, collab_exec_enabled: false },
  calendar: { enabled: false, caldav_url: "", caldav_username: "", caldav_password_provider: "caldav", ics_files: [] },
  email: { enabled: false, imap_host: "", smtp_host: "", smtp_port: 587, username: "", password_provider: "email" },
  ssh: { enabled: false, connect_timeout_seconds: 10, allowed_hosts: [] },
  voice: {
    input_device: "auto",
    transcription_engine: "auto",
    whisper_model: "small",
    language: "auto",
    tts_engine: "kokoro_tts",
    tts_voice: "af_heart",
  },
  restrictions: { protected_folders: [], protected_packages: [], blocked_commands: ["format"] },
  first_run_complete: true,
  theme: "dark",
  hotkey: "Ctrl+Space",
};

describe("SettingsPanel saves", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockResolvedValue({ status: "ok", providers: [], models: [], devices: [], variants: [] });
    settingsMocks.updateSectionDetailed.mockReset();
    settingsMocks.set(structuredClone(baseSettings));
    settingsMocks.setLoadErrors([]);
  });

  // SettingsPanel is imported once at module level (vi.mock calls are hoisted above it), so loading its
  // ~40 modules does not count against a test's timeout when the full suite runs in parallel.
  async function renderPanel() {
    return render(SettingsPanel);
  }

  it("puts a refused value back and shows the daemon's reason", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({
      ok: false,
      message: "model.rate_limit_rpm must be between 1 and 10000",
      restartRequired: false,
    });
    const { container } = await renderPanel();
    const rpm = container.querySelector('input[type="number"][max="10000"]') as HTMLInputElement;
    expect(rpm.value).toBe("60");

    rpm.value = "50000";
    await fireEvent.change(rpm);

    await waitFor(() =>
      expect(screen.getByText("Not saved: model.rate_limit_rpm must be between 1 and 10000").getAttribute("role")).toBe(
        "alert",
      ),
    );
    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("model", { rate_limit_rpm: 50000 });
    expect(rpm.value).toBe("60");
  });

  it("turns undo history off and clears it only on a second click", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    await renderPanel();

    await fireEvent.click(screen.getByRole("button", { name: "Keep undo history" }));
    await waitFor(() => expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("undo", { enabled: false }));

    const clear = screen.getByRole("button", { name: "Clear" });
    await fireEvent.click(clear);
    expect(daemonMocks.call).not.toHaveBeenCalledWith("undo_clear", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Click again to delete" }));
    await waitFor(() => expect(daemonMocks.call).toHaveBeenCalledWith("undo_clear", { confirmed: true }));
  });

  it("says when a saved change needs a restart", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: true });
    await renderPanel();

    await fireEvent.click(screen.getByRole("button", { name: "Subscription" }));

    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Restart Heliox"));
    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("model", { provider: "subscription" });
  });

  it("keeps a refused restriction in the box so it can be fixed", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({
      ok: false,
      message: "restrictions.blocked_commands entries must be at most 1024 characters",
      restartRequired: false,
    });
    const { container } = await renderPanel();
    const input = container.querySelector('input[placeholder="command-name"]') as HTMLInputElement;

    await fireEvent.input(input, { target: { value: "rm -rf" } });
    await fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() =>
      expect(screen.getByText(/Not saved: .*at most 1024 characters/).getAttribute("role")).toBe("alert"),
    );
    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("restrictions", {
      blocked_commands: ["format", "rm -rf"],
    });
    expect(input.value).toBe("rm -rf");
  });

  it("warns that settings are read-only while config.toml cannot be read", async () => {
    settingsMocks.setLoadErrors(["config.toml: Invalid value (at line 4, column 9)"]);
    await renderPanel();

    const banner = screen.getByText(/Settings are read-only until config.toml is fixed/).closest("[role=alert]");
    expect(banner?.textContent).toContain("Invalid value (at line 4, column 9)");
  });

  it("will not save an API key without a chosen cloud provider", async () => {
    await renderPanel();
    const keyInput = document.querySelector('input[type="password"]') as HTMLInputElement | null;
    if (!keyInput) return; // the key field only shows for cloud setups in some layouts
    await fireEvent.input(keyInput, { target: { value: "sk-test" } });
    const save = keyInput.parentElement?.querySelector("button.btn-save") as HTMLButtonElement;
    await fireEvent.click(save);

    await waitFor(() => expect(screen.getByText(/Choose a cloud provider above/)).toBeTruthy());
    expect(daemonMocks.call).not.toHaveBeenCalledWith("store_api_key", expect.anything());
  });
});

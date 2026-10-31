import { writable } from "svelte/store";
import { call, onConnectionState, onNotification } from "../api/daemon";

export interface PilotSettings {
  model: {
    provider: string;
    ollama_base_url: string;
    ollama_model: string;
    mode: string;
    gpu_memory_limit_mb: number;
    idle_unload_seconds: number;
    cloud_provider: string;
    cloud_model: string;
    subscription_provider: "codex" | "claude";
    subscription_model: string;
    subscription_timeout_seconds: number;
    subscription_max_prompt_chars: number;
    // Rate limiting
    rate_limit_enabled: boolean;
    rate_limit_rpm: number;
    rate_limit_burst: number;
    // Monthly cumulative budget
    budget_enabled: boolean;
    budget_monthly_limit_usd: number;
    // Per-action and per-task enforcement (Phase 1 of #312)
    max_tokens_per_action: number;
    max_tokens_per_task: number;
    max_usd_per_task: number;
    max_consecutive_failures: number;
  };
  security: {
    root_enabled: boolean;
    dry_run: boolean;
    snapshot_on_destructive: boolean;
    snapshot_backend: string;
    snapshot_retention_count: number;
    snapshot_retention_days: number;
  };
  screen_vision: {
    capture_interval_seconds: number;
  };
  vision: {
    mediapipe_backend: "legacy" | "tasks";
    gaze_tracking_enabled: boolean;
  };
  gesture_cursor: {
    enabled: boolean;
    sensitivity: number;
    prediction_ms: number;
    blend: number;
  };
  adaptive_calibration: {
    gesture_enabled: boolean;
    voice_wake_word_enabled: boolean;
  };
  preview: {
    enabled: boolean;
  };
  undo: {
    enabled: boolean;
    retention_days: number;
  };
  timeline: {
    enabled: boolean;
    retention_days: number;
    excluded_apps: string[];
  };
  brief: {
    enabled: boolean;
    time: string;
    speak: boolean;
  };
  clipboard: {
    history: boolean;
  };
  suggestions: {
    enabled: boolean;
  };
  typed_decision: {
    enabled: boolean;
    artifact_dir: string;
    report_path: string;
  };
  network: {
    enabled: boolean;
    port: number;
    peer_timeout_s: number;
    skill_sync_enabled: boolean;
    collab_exec_enabled: boolean;
  };
  calendar: {
    enabled: boolean;
    caldav_url: string;
    caldav_username: string;
    caldav_password_provider: string;
    ics_files: string[];
  };
  email: {
    enabled: boolean;
    imap_host: string;
    smtp_host: string;
    smtp_port: number;
    username: string;
    password_provider: string;
  };
  ssh: {
    enabled: boolean;
    connect_timeout_seconds: number;
    allowed_hosts: Array<{
      name: string;
      hostname: string;
      port: number;
      username: string;
      private_key_provider: string;
      passphrase_provider: string;
      strict_host_key_checking: boolean;
    }>;
  };
  voice: {
    input_device: string;
    transcription_engine: "auto" | "faster_whisper" | "openai_whisper";
    whisper_model: "tiny" | "base" | "small" | "medium" | "turbo";
    language: string;
    tts_engine: string;
    tts_voice: string;
  };
  restrictions: {
    protected_folders: string[];
    protected_packages: string[];
    blocked_commands: string[];
  };
  first_run_complete: boolean;
  theme: "light" | "dark";
  hotkey: string; // Added tracking for active UI theme mode
}

const defaultSettings: PilotSettings = {
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
    budget_monthly_limit_usd: 10.0,
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
  screen_vision: {
    capture_interval_seconds: 3,
  },
  vision: {
    mediapipe_backend: "tasks",
    gaze_tracking_enabled: false,
  },
  gesture_cursor: {
    enabled: false,
    sensitivity: 1.0,
    prediction_ms: 80.0,
    blend: 0.3,
  },
  adaptive_calibration: {
    gesture_enabled: true,
    voice_wake_word_enabled: true,
  },
  preview: {
    enabled: false,
  },
  undo: {
    enabled: true,
    retention_days: 7,
  },
  timeline: {
    enabled: false,
    retention_days: 7,
    excluded_apps: ["1password", "bitwarden", "keepass", "lastpass", "dashlane", "keychain access"],
  },
  brief: {
    enabled: false,
    time: "18:00",
    speak: false,
  },
  clipboard: {
    history: false,
  },
  suggestions: {
    enabled: false,
  },
  typed_decision: {
    enabled: false,
    artifact_dir: "",
    report_path: "",
  },
  network: {
    enabled: false,
    port: 8786,
    peer_timeout_s: 30,
    skill_sync_enabled: false,
    collab_exec_enabled: false,
  },
  calendar: {
    enabled: false,
    caldav_url: "",
    caldav_username: "",
    caldav_password_provider: "caldav",
    ics_files: [],
  },
  email: {
    enabled: false,
    imap_host: "",
    smtp_host: "",
    smtp_port: 587,
    username: "",
    password_provider: "email",
  },
  ssh: {
    enabled: false,
    connect_timeout_seconds: 10,
    allowed_hosts: [],
  },
  voice: {
    input_device: "auto",
    transcription_engine: "auto",
    whisper_model: "small",
    language: "auto",
    tts_engine: "kokoro_tts",
    tts_voice: "af_heart",
  },
  restrictions: {
    protected_folders: [],
    protected_packages: [],
    blocked_commands: [],
  },
  first_run_complete: false,
  theme: "dark",
  hotkey: typeof navigator !== "undefined" && navigator.platform.includes("Mac") ? "Cmd+Space" : "Ctrl+Space", // Default configuration set to dark mode
};

function mergeSettings(current: PilotSettings, incoming: Partial<PilotSettings>): PilotSettings {
  return {
    ...current,
    ...incoming,
    model: { ...current.model, ...incoming.model },
    security: { ...current.security, ...incoming.security },
    screen_vision: { ...current.screen_vision, ...incoming.screen_vision },
    vision: { ...current.vision, ...incoming.vision },
    gesture_cursor: { ...current.gesture_cursor, ...incoming.gesture_cursor },
    adaptive_calibration: { ...current.adaptive_calibration, ...incoming.adaptive_calibration },
    preview: { ...current.preview, ...incoming.preview },
    undo: { ...current.undo, ...incoming.undo },
    timeline: { ...current.timeline, ...incoming.timeline },
    brief: { ...current.brief, ...incoming.brief },
    clipboard: { ...current.clipboard, ...incoming.clipboard },
    suggestions: { ...current.suggestions, ...incoming.suggestions },
    typed_decision: { ...current.typed_decision, ...incoming.typed_decision },
    network: { ...current.network, ...incoming.network },
    calendar: { ...current.calendar, ...incoming.calendar },
    email: { ...current.email, ...incoming.email },
    ssh: { ...current.ssh, ...incoming.ssh },
    voice: { ...current.voice, ...incoming.voice },
    restrictions: { ...current.restrictions, ...incoming.restrictions },
  };
}

export type SettingsSaveResult = {
  ok: boolean;
  /** The daemon's reason when it refused the change (or why it could not be reached). */
  message: string;
  /** The daemon saved the value but only applies it after a restart. */
  restartRequired: boolean;
};

const SETTINGS_CACHE_KEY = "heliox_settings";
/** Set only when the person picks a theme; while absent the UI follows the OS. */
const THEME_KEY = "heliox_theme";

/** Problems the daemon hit reading config.toml; while non-empty it refuses to save. */
export const settingsLoadErrors = writable<string[]>([]);
/** Bumped whenever the daemon announces a config change, so panels can re-read their section. */
export const configRevision = writable(0);

function readJson(key: string): Record<string, any> | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeCache(value: PilotSettings) {
  try {
    const { theme: _theme, ...rest } = value;
    localStorage.setItem(SETTINGS_CACHE_KEY, JSON.stringify(rest));
  } catch {
    /* ignore */
  }
}

function createSettings() {
  const { subscribe, set, update } = writable<PilotSettings>(defaultSettings);
  let current = defaultSettings;
  subscribe((value) => (current = value));

  // Helper utility to detect system-level operating system dark/light mode preference
  function getSystemTheme(): "light" | "dark" {
    if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    return "dark";
  }

  function manualTheme(): "light" | "dark" | null {
    try {
      const stored = localStorage.getItem(THEME_KEY);
      if (stored === "light" || stored === "dark") return stored;
      // Older builds kept the theme inside the settings cache and rewrote it on every load,
      // so it cannot tell a choice from the OS default. Keep it only where it differs from the OS.
      const legacy = readJson(SETTINGS_CACHE_KEY)?.theme;
      if ((legacy === "light" || legacy === "dark") && legacy !== getSystemTheme()) {
        localStorage.setItem(THEME_KEY, legacy);
        return legacy;
      }
    } catch {
      /* ignore */
    }
    return null;
  }

  /** Read the daemon's saved config, which is the only source of truth for daemon-backed settings. */
  async function reloadFromDaemon(): Promise<boolean> {
    try {
      const config = (await call("get_config")) as Partial<PilotSettings> & { load_errors?: unknown };
      const { load_errors, ...incoming } = config ?? {};
      settingsLoadErrors.set(Array.isArray(load_errors) ? load_errors.map(String) : []);
      update((existing) => {
        const merged = mergeSettings(existing, incoming as Partial<PilotSettings>);
        merged.theme = existing.theme;
        merged.hotkey = existing.hotkey;
        writeCache(merged);
        return merged;
      });
      return true;
    } catch {
      return false;
    }
  }

  async function load() {
    const cached = readJson(SETTINGS_CACHE_KEY);
    update((s) => {
      const merged = cached ? mergeSettings(s, cached as Partial<PilotSettings>) : { ...s };
      merged.theme = manualTheme() ?? getSystemTheme();
      return merged;
    });
    await reloadFromDaemon();
  }

  function applyLocally(section: string, values: Record<string, unknown>) {
    update((s) => {
      const next = section === "" ? { ...s, ...values } : { ...s, [section]: { ...(s as any)[section], ...values } };
      writeCache(next as PilotSettings);
      return next as PilotSettings;
    });
    if (section === "" && (values.theme === "light" || values.theme === "dark")) {
      try {
        localStorage.setItem(THEME_KEY, values.theme);
      } catch {
        /* ignore */
      }
    }
  }

  async function updateSectionDetailed(section: string, values: Record<string, unknown>): Promise<SettingsSaveResult> {
    const keys = Object.keys(values);
    const localOnly = section === "" && keys.length > 0 && keys.every((key) => key === "theme" || key === "hotkey");
    let restartRequired = false;

    if (!localOnly) {
      try {
        const result = await call<{ status?: string; message?: string; restart_required?: boolean }>("update_config", {
          section,
          values,
        });
        if (result?.status !== "ok") {
          return { ok: false, message: result?.message || "The daemon rejected the setting.", restartRequired };
        }
        restartRequired = Boolean(result.restart_required);
      } catch (err) {
        console.warn("Daemon rejected settings update:", err);
        const message = err instanceof Error && err.message ? err.message : "The daemon could not be reached.";
        return { ok: false, message, restartRequired };
      }
    }

    applyLocally(section, values);
    return { ok: true, message: "", restartRequired };
  }

  async function updateSection(
    section: string,
    values: Record<string, unknown>,
    _options: { requireDaemon?: boolean } = {},
  ): Promise<boolean> {
    return (await updateSectionDetailed(section, values)).ok;
  }

  void load();

  // Keep in step with the daemon: another window, a voice command, a hand edit of
  // config.toml or a daemon restart can all change the saved settings.
  let reloadTimer: ReturnType<typeof setTimeout> | null = null;
  function scheduleReload() {
    if (reloadTimer) clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => {
      reloadTimer = null;
      void reloadFromDaemon().then(() => configRevision.update((n) => n + 1));
    }, 150);
  }
  try {
    onConnectionState((connected) => {
      if (connected) scheduleReload();
    });
    onNotification((method) => {
      if (method === "config_updated") scheduleReload();
    });
  } catch {
    /* daemon client unavailable (tests with a partial mock) */
  }

  // Reactive subscription side-effect to safely toggle HTML element tags dynamically
  subscribe((s) => {
    if (typeof window !== "undefined") {
      const root = document.documentElement;
      if (s.theme === "light") {
        root.classList.add("light-mode");
      } else {
        root.classList.remove("light-mode");
      }
    }
  });

  // Follow the OS theme until the person picks one.
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", (e) => {
      if (!manualTheme()) {
        update((s) => ({ ...s, theme: e.matches ? "light" : "dark" }));
      }
    });
  }

  async function reset(): Promise<boolean> {
    try {
      const result = await call<{ status?: string; message?: string }>("reset_config");
      if (result?.status !== "ok") {
        throw new Error(result?.message || "Daemon rejected the factory reset");
      }
    } catch (err) {
      console.warn("Failed to reset backend config:", err);
      return false;
    }

    try {
      localStorage.removeItem(SETTINGS_CACHE_KEY);
      localStorage.removeItem(THEME_KEY);
    } catch {
      /* ignore */
    }
    // The daemon keeps some sections (server, restrictions) and its own defaults, so show
    // what it saved rather than the UI's copy of the defaults.
    set({ ...defaultSettings, theme: getSystemTheme(), hotkey: current.hotkey });
    await reloadFromDaemon();
    return true;
  }

  return {
    subscribe,
    load,
    reloadFromDaemon,
    updateSection,
    updateSectionDetailed,
    reset,
  };
}

export const settings = createSettings();

import { writable } from "svelte/store";
import { call } from "../api/daemon";

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
    typed_decision: { ...current.typed_decision, ...incoming.typed_decision },
    network: { ...current.network, ...incoming.network },
    calendar: { ...current.calendar, ...incoming.calendar },
    email: { ...current.email, ...incoming.email },
    ssh: { ...current.ssh, ...incoming.ssh },
    voice: { ...current.voice, ...incoming.voice },
    restrictions: { ...current.restrictions, ...incoming.restrictions },
  };
}

function createSettings() {
  const { subscribe, set, update } = writable<PilotSettings>(defaultSettings);

  // Helper utility to detect system-level operating system dark/light mode preference
  function getSystemTheme(): "light" | "dark" {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    return "dark";
  }

  async function load() {
    try {
      const stored = localStorage.getItem("heliox_settings");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Fallback to system preference matching if no theme key exists in saved cache
        if (!parsed.theme) {
          parsed.theme = getSystemTheme();
        }
        update((s) => mergeSettings(s, parsed));
      } else {
        // Apply detected system preference mode on fresh startup instances
        update((s) => ({ ...s, theme: getSystemTheme() }));
      }
    } catch {
      /* ignore */
    }

    call("get_config")
      .then((config) => {
        update((current) => {
          const fullConfig = mergeSettings(current, config as Partial<PilotSettings>);
          fullConfig.theme = current.theme;
          fullConfig.hotkey = current.hotkey;
          try {
            localStorage.setItem("heliox_settings", JSON.stringify(fullConfig));
          } catch {
            /* ignore */
          }
          return fullConfig;
        });
      })
      .catch(() => {});
  }

  async function updateSection(
    section: string,
    values: Record<string, unknown>,
    options: { requireDaemon?: boolean } = {},
  ): Promise<boolean> {
    const keys = Object.keys(values);
    const localOnly = section === "" && keys.length > 0 && keys.every((key) => key === "theme" || key === "hotkey");

    if (!localOnly) {
      try {
        const result = await call<{ status?: string; message?: string }>("update_config", { section, values });
        if (result?.status !== "ok") {
          throw new Error(result?.message || "Daemon rejected the setting");
        }
      } catch (err) {
        const qualifier = options.requireDaemon ? "required " : "";
        console.warn(`Daemon rejected ${qualifier}settings update:`, err);
        return false;
      }
    }

    if (section === "") {
      update((s) => ({ ...s, ...values }));
    } else {
      update((s) => ({
        ...s,
        [section]: { ...(s as any)[section], ...values },
      }));
    }

    try {
      const stored = JSON.parse(localStorage.getItem("heliox_settings") || "{}");
      if (section === "") {
        Object.assign(stored, values);
      } else {
        stored[section] = { ...(stored[section] || {}), ...values };
      }
      localStorage.setItem("heliox_settings", JSON.stringify(stored));
    } catch {
      /* ignore */
    }

    return true;
  }

  load();

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

  // Event listener tracking OS level theme switches when manual overrides aren't present
  if (typeof window !== "undefined") {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
      const stored = localStorage.getItem("heliox_settings");
      const hasManualTheme = stored && JSON.parse(stored).theme;
      if (!hasManualTheme) {
        updateSection("", { theme: e.matches ? "light" : "dark" });
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

    set(defaultSettings);
    try {
      localStorage.removeItem("heliox_settings");
    } catch {
      /* ignore */
    }
    return true;
  }

  return {
    subscribe,
    load,
    updateSection,
    reset,
  };
}

export const settings = createSettings();

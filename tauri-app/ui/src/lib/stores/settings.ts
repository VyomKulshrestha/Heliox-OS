import { writable } from "svelte/store";
import { call } from "../api/daemon";

export interface PilotSettings {
  model: {
    provider: string;
    ollama_base_url: string;
    ollama_model: string;
    mode: string;
    gpu_memory_limit_mb: number;
    cloud_provider: string;
    cloud_model: string;
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
    confirm_tier2: boolean;
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
  voice: {
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
    cloud_provider: "",
    cloud_model: "",
    rate_limit_enabled: true,
    rate_limit_rpm: 60,
    rate_limit_burst: 5,
    budget_enabled: true,
    budget_monthly_limit_usd: 10.0,
    max_tokens_per_action: 4000,
    max_tokens_per_task: 50000,
    max_usd_per_task: 0.10,
    max_consecutive_failures: 3,
  },
  security: {
    root_enabled: false,
    confirm_tier2: true,
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
    mediapipe_backend: "legacy",
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
  voice: {
    tts_engine: "pocket_tts",
    tts_voice: "alba",
  },
  restrictions: {
    protected_folders: [],
    protected_packages: [],
    blocked_commands: [],
  },
  first_run_complete: false,
  theme: "dark",
  hotkey: typeof navigator !== "undefined" && navigator.platform.includes("Mac") 
    ? "Cmd+Space" 
    : "Ctrl+Space", // Default configuration set to dark mode
};

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
        update((s) => ({ ...s, ...parsed }));
      } else {
        // Apply detected system preference mode on fresh startup instances
        update((s) => ({ ...s, theme: getSystemTheme() }));
      }
    } catch { /* ignore */ }

    call("get_config")
      .then((config) => {
        const fullConfig = config as PilotSettings;
        // Keep localized store UI theme value if backend daemon returns empty config properties
        if (!fullConfig.theme) {
          subscribe(s => { fullConfig.theme = s.theme; })();
        }
        set(fullConfig);
        try {
          localStorage.setItem("heliox_settings", JSON.stringify(fullConfig));
        } catch { /* ignore */ }
      })
      .catch(() => {});
  }

  async function updateSection(section: string, values: Record<string, unknown>) {
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
    } catch { /* ignore */ }

    call("update_config", { section, values }).catch((err) => {
      console.warn("Daemon unreachable, settings saved locally:", err);
    });
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
  async function reset() {
  // Reset app state immediately
  set(defaultSettings);

  // Remove cached local settings
  try {
    localStorage.removeItem("heliox_settings");
  } catch {
    /* ignore */
  }

  // Tell backend/daemon to reset config
  call("reset_config").catch((err) => {
    console.warn("Failed to reset backend config:", err);
  });
}

  return {
    subscribe,
    load,
    updateSection,
    reset,
  };
}

export const settings = createSettings();
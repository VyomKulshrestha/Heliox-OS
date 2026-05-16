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
  restrictions: {
    protected_folders: string[];
    protected_packages: string[];
    blocked_commands: string[];
  };
  first_run_complete: boolean;
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
  restrictions: {
    protected_folders: [],
    protected_packages: [],
    blocked_commands: [],
  },
  first_run_complete: false,
};

function createSettings() {
  const { subscribe, set, update } = writable<PilotSettings>(defaultSettings);

  async function load() {
    // Load from localStorage first (instant, always available)
    try {
      const stored = localStorage.getItem("heliox_settings");
      if (stored) {
        const parsed = JSON.parse(stored);
        update((s) => ({ ...s, ...parsed }));
      }
    } catch { /* ignore */ }

    // Then try to sync from daemon in background (non-blocking)
    call("get_config")
      .then((config) => {
        set(config as PilotSettings);
        // Update localStorage with daemon's authoritative copy
        try {
          localStorage.setItem("heliox_settings", JSON.stringify(config));
        } catch { /* ignore */ }
      })
      .catch(() => {
        // daemon not available, localStorage values are fine
      });
  }

  async function updateSection(section: string, values: Record<string, unknown>) {
    // Always update the local store immediately so the UI never hangs
    if (section === "") {
      update((s) => ({ ...s, ...values }));
    } else {
      update((s) => ({
        ...s,
        [section]: { ...(s as any)[section], ...values },
      }));
    }

    // Persist to localStorage as a reliable fallback
    try {
      const stored = JSON.parse(localStorage.getItem("heliox_settings") || "{}");
      if (section === "") {
        Object.assign(stored, values);
      } else {
        stored[section] = { ...(stored[section] || {}), ...values };
      }
      localStorage.setItem("heliox_settings", JSON.stringify(stored));
    } catch { /* ignore */ }

    // Try to sync to daemon in background (non-blocking)
    call("update_config", { section, values }).catch((err) => {
      console.warn("Daemon unreachable, settings saved locally:", err);
    });
  }

  load();

  return {
    subscribe,
    load,
    updateSection,
  };
}

export const settings = createSettings();

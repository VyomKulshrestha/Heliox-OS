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
  restrictions: {
    protected_folders: string[];
    protected_packages: string[];
    blocked_commands: string[];
  };
  first_run_complete: boolean;
  theme: "light" | "dark"; // Added tracking for active UI theme mode
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
  restrictions: {
    protected_folders: [],
    protected_packages: [],
    blocked_commands: [],
  },
  first_run_complete: false,
  theme: "dark", // Default configuration set to dark mode
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

  return {
    subscribe,
    load,
    updateSection,
  };
}

export const settings = createSettings();
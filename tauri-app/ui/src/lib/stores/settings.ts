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
};

function createSettings() {
  const { subscribe, set, update } = writable<PilotSettings>(defaultSettings);

  async function load() {
    try {
      const config = (await call("get_config")) as PilotSettings;
      set(config);
    } catch {
      // use defaults
    }
  }

  async function updateSection(section: string, values: Record<string, unknown>) {
    try {
      await call("update_config", { section, values });
      if (section === "") {
        update((s) => ({ ...s, ...values }));
      } else {
        update((s) => ({
          ...s,
          [section]: { ...(s as any)[section], ...values },
        }));
      }
    } catch (err) {
      console.error("Failed to update config:", err);
    }
  }

  load();

  return {
    subscribe,
    load,
    updateSection,
  };
}

export const settings = createSettings();

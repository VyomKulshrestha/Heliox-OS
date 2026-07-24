export function defaultHotkey(platform = typeof navigator !== "undefined" ? navigator.platform : ""): string {
  return platform.includes("Mac") ? "Cmd+Space" : "Ctrl+Space";
}

export function normalizeHotkeyValue(value: unknown, fallback = defaultHotkey()): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  if (value && typeof value === "object") {
    for (const key of ["shortcut", "hotkey", "value"]) {
      const candidate = (value as Record<string, unknown>)[key];
      if (typeof candidate === "string" && candidate.trim()) {
        return candidate.trim();
      }
    }
  }

  return fallback;
}

export function isNativeTauriRuntime(): boolean {
  return typeof window !== "undefined" && Boolean((window as any).__TAURI_INTERNALS__);
}

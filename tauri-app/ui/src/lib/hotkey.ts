// Free on a stock install: on macOS Cmd+Space is Spotlight and Ctrl+Space switches input sources (see hotkey.rs).
export function defaultHotkey(platform = typeof navigator !== "undefined" ? navigator.platform : ""): string {
  return platform.includes("Mac") ? "Alt+Space" : "Ctrl+Space";
}

export type HotkeyStatus = { shortcut: string; default: string; error: string | null };

export function hotkeyFailureMessage(status: Partial<HotkeyStatus> | null | undefined): string {
  if (!status?.error) return "";
  const shortcut = status.shortcut || "The shortcut";
  return `${shortcut} could not be registered (${status.error}). Another app may be using it: choose a different shortcut below.`;
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

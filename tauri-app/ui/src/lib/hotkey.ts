function currentPlatform(): string {
  return typeof navigator !== "undefined" ? navigator.platform : "";
}

function isMac(platform: string): boolean {
  return platform.includes("Mac");
}

// Free on a stock install: on macOS Cmd+Space is Spotlight and Ctrl+Space switches input sources (see hotkey.rs).
// This is the string Tauri registers; show it to people through displayHotkey().
export function defaultHotkey(platform = currentPlatform()): string {
  return isMac(platform) ? "Alt+Space" : "Ctrl+Space";
}

// Modifier spellings people type or Tauri stores, by what they mean.
const MODIFIERS: Record<string, "alt" | "ctrl" | "meta" | "shift"> = {
  alt: "alt",
  option: "alt",
  opt: "alt",
  "⌥": "alt",
  ctrl: "ctrl",
  control: "ctrl",
  "⌃": "ctrl",
  cmd: "meta",
  command: "meta",
  "⌘": "meta",
  super: "meta",
  meta: "meta",
  win: "meta",
  windows: "meta",
  shift: "shift",
  "⇧": "shift",
};

// What each system's keyboard calls the modifier.
const MAC_NAMES = { alt: "Option", ctrl: "Control", meta: "Command", shift: "Shift" };
const WINDOWS_NAMES = { alt: "Alt", ctrl: "Ctrl", meta: "Win", shift: "Shift" };
const LINUX_NAMES = { alt: "Alt", ctrl: "Ctrl", meta: "Super", shift: "Shift" };
// What Tauri's global-shortcut parser accepts.
const TAURI_NAMES = { alt: "Alt", ctrl: "Ctrl", meta: "Super", shift: "Shift" };

function keyName(part: string): string {
  const key = part.trim();
  if (key.toLowerCase() === "space") return "Space";
  return key.length === 1 ? key.toUpperCase() : key;
}

function convert(shortcut: string, names: Record<"alt" | "ctrl" | "meta" | "shift", string>): string {
  return shortcut
    .split("+")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const modifier = MODIFIERS[part.toLowerCase()];
      return modifier ? names[modifier] : keyName(part);
    })
    .join("+");
}

/** A shortcut as this system's keyboard names it: "Option+Space" on a Mac, "Ctrl+Space" on Windows. */
export function displayHotkey(shortcut: string, platform = currentPlatform()): string {
  if (isMac(platform)) return convert(shortcut, MAC_NAMES);
  return convert(shortcut, platform.startsWith("Win") ? WINDOWS_NAMES : LINUX_NAMES);
}

/** What the person typed, in either spelling ("Option+Space", "⌥+Space", "Alt+Space"), as Tauri registers it. */
export function toTauriHotkey(input: string): string {
  return convert(input, TAURI_NAMES);
}

export type HotkeyStatus = { shortcut: string; default: string; error: string | null };

export function hotkeyFailureMessage(
  status: Partial<HotkeyStatus> | null | undefined,
  platform = currentPlatform(),
): string {
  if (!status?.error) return "";
  // Wayland: not a failure; the desktop owns global shortcuts there (see hotkey.rs and wayland_shortcut.py).
  if (status.error.startsWith("Wayland:")) return status.error.slice("Wayland:".length).trim();
  const shortcut = status.shortcut ? displayHotkey(status.shortcut, platform) : "The shortcut";
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

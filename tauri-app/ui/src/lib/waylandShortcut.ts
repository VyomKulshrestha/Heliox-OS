// Wayland: the global shortcut arrives from the daemon (the desktop's GlobalShortcuts portal, see
// daemon/pilot/system/wayland_shortcut.py) as a "global_shortcut" notification; it shows or hides the window the
// same way the X11, Windows and macOS shortcut does in hotkey.rs.
import { onNotification } from "./api/daemon";
import { invoke } from "./api/invoke";

export function handleGlobalShortcut(method: string, params: unknown, toggle = () => invoke("toggle_window")): boolean {
  const id = (params as { id?: unknown } | null)?.id;
  if (method !== "global_shortcut" || id !== "toggle") return false;
  void toggle().catch(() => undefined);
  return true;
}

let installed = false;

export function installWaylandShortcut(): void {
  if (installed) return;
  installed = true;
  onNotification((method: string, params: unknown) => {
    handleGlobalShortcut(method, params);
  });
}

import { invoke as tauriInvoke } from "@tauri-apps/api/core";

export async function invoke<T = any>(command: string, args?: any): Promise<T> {
  // First check if native Tauri IPC bridge is present
  if (typeof window !== "undefined" && (window as any).__TAURI_INTERNALS__) {
    try {
      return await tauriInvoke<T>(command, args);
    } catch (e) {
      console.error(`Tauri native invoke error (${command}):`, e);
      throw e;
    }
  }

  // System-wide shortcuts are a native desktop capability. Keep browser
  // development truthful instead of returning an empty object and pretending
  // that a shortcut was registered.
  if (command === "get_hotkey") {
    return "Ctrl+Space" as unknown as T;
  }
  if (command === "get_hotkey_status") {
    return { shortcut: "Ctrl+Space", default: "Ctrl+Space", error: null } as unknown as T;
  }
  if (command === "set_hotkey") {
    throw new Error("Global hotkeys are only available in the Heliox desktop app.");
  }

  if (command === "get_auth_token") {
    try {
      const res = await fetch("/api/auth_token");
      if (res.ok) return (await res.text()).trim() as unknown as T;
    } catch (error) {
      console.error("Dev server auth-token request failed:", error);
    }
    return ((import.meta as any).env?.VITE_DAEMON_TOKEN ?? "") as unknown as T;
  }

  // Fallback for browser dev mode (npm run dev running in standard Chrome/Edge)
  try {
    const res = await fetch("/api/tauri_invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args }),
    });
    if (!res.ok) {
      let detail = "";
      try {
        const payload = await res.json();
        detail = typeof payload?.error === "string" ? payload.error : "";
      } catch {
        detail = await res.text().catch(() => "");
      }
      throw new Error(detail || `Browser development command '${command}' is unavailable (${res.status}).`);
    }
    return await res.json();
  } catch (error) {
    console.error(`Dev server fallback error (${command}):`, error);
    if (error instanceof Error) throw error;
    throw new Error(`Browser development command '${command}' failed.`);
  }
}

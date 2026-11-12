import { describe, expect, it } from "vitest";
import { defaultHotkey, displayHotkey, hotkeyFailureMessage, normalizeHotkeyValue, toTauriHotkey } from "./hotkey";

describe("hotkey settings", () => {
  it("uses the platform default", () => {
    expect(defaultHotkey("Win32")).toBe("Ctrl+Space");
    expect(defaultHotkey("MacIntel")).toBe("Alt+Space");
  });

  it("normalizes native and legacy response shapes", () => {
    expect(normalizeHotkeyValue(" Alt+H ", "Ctrl+Space")).toBe("Alt+H");
    expect(normalizeHotkeyValue({ shortcut: "Ctrl+Shift+H" }, "Ctrl+Space")).toBe("Ctrl+Shift+H");
    expect(normalizeHotkeyValue({ hotkey: "Alt+Space" }, "Ctrl+Space")).toBe("Alt+Space");
  });

  it("never renders object values into the input", () => {
    expect(normalizeHotkeyValue({}, "Ctrl+Space")).toBe("Ctrl+Space");
    expect(normalizeHotkeyValue(null, "Ctrl+Space")).toBe("Ctrl+Space");
  });

  it("says when the shortcut could not be registered", () => {
    expect(hotkeyFailureMessage({ shortcut: "Cmd+Space", error: "HotKey already registered" }, "MacIntel")).toContain(
      "Command+Space could not be registered (HotKey already registered)",
    );
    expect(hotkeyFailureMessage({ shortcut: "Ctrl+Space", error: null })).toBe("");
    expect(hotkeyFailureMessage(null)).toBe("");
  });

  it("shows each system's own key names and accepts either spelling", () => {
    expect(displayHotkey("Alt+Space", "MacIntel")).toBe("Option+Space");
    expect(displayHotkey("Cmd+Shift+h", "MacIntel")).toBe("Command+Shift+H");
    expect(displayHotkey("Ctrl+Space", "MacIntel")).toBe("Control+Space");
    expect(displayHotkey("Ctrl+Space", "Win32")).toBe("Ctrl+Space");
    expect(displayHotkey("Super+space", "Win32")).toBe("Win+Space");
    expect(displayHotkey("Super+Space", "Linux x86_64")).toBe("Super+Space");
    expect(toTauriHotkey("Option+Space")).toBe("Alt+Space");
    expect(toTauriHotkey(" ⌥ + space ")).toBe("Alt+Space");
    expect(toTauriHotkey("Command+Shift+H")).toBe("Super+Shift+H");
    expect(toTauriHotkey("Control+Alt+K")).toBe("Ctrl+Alt+K");
    expect(toTauriHotkey("   ")).toBe("");
  });
});

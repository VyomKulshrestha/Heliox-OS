import { describe, expect, it } from "vitest";
import { defaultHotkey, hotkeyFailureMessage, normalizeHotkeyValue } from "./hotkey";

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
    expect(hotkeyFailureMessage({ shortcut: "Cmd+Space", error: "HotKey already registered" })).toContain(
      "Cmd+Space could not be registered (HotKey already registered)",
    );
    expect(hotkeyFailureMessage({ shortcut: "Ctrl+Space", error: null })).toBe("");
    expect(hotkeyFailureMessage(null)).toBe("");
  });
});

import { describe, expect, it } from "vitest";
import { defaultHotkey, normalizeHotkeyValue } from "./hotkey";

describe("hotkey settings", () => {
  it("uses the platform default", () => {
    expect(defaultHotkey("Win32")).toBe("Ctrl+Space");
    expect(defaultHotkey("MacIntel")).toBe("Cmd+Space");
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
});

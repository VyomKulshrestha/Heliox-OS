"""An AppKit window with a text field and a secure text field, driven over stdin, for the macOS check.

Commands: ``focus entry``, ``focus password``, ``title <text>``, ``insert <text>``, ``presses`` (prints
``presses <n>``, how many times the button's own action ran), ``quit``. Prints ``ready`` once the window is in
front with the text field focused. The fields have placeholders ("message", "password"), as their labels.
"""

from __future__ import annotations

import queue
import sys
import threading

import AppKit
from Foundation import NSObject, NSTimer

commands: queue.Queue[str] = queue.Queue()


class Driver(NSObject):
    def setup(self):  # noqa: D401
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        rect = AppKit.NSMakeRect(200, 200, 420, 140)
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
        )
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Heliox AX fixture")
        self.entry = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 80, 380, 24))
        self.password = AppKit.NSSecureTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 30, 380, 24))
        self.entry.setPlaceholderString_("message")
        self.password.setPlaceholderString_("password")
        self.presses = 0
        self.button = AppKit.NSButton.buttonWithTitle_target_action_("Press me", self, "pressed:")
        self.button.setFrame_(AppKit.NSMakeRect(300, 105, 100, 28))
        self.window.contentView().addSubview_(self.button)
        self.window.contentView().addSubview_(self.entry)
        self.window.contentView().addSubview_(self.password)
        self.window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        self.window.makeFirstResponder_(self.entry)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.1, self, "tick:", None, True)
        print("ready", flush=True)
        return self

    def pressed_(self, _sender):
        self.presses += 1
        print("pressed", flush=True)

    def tick_(self, _timer):
        while not commands.empty():
            command, _, arg = commands.get().partition(" ")
            if command == "focus":
                self.window.makeFirstResponder_(self.password if arg == "password" else self.entry)
            elif command == "title":
                self.window.setTitle_(arg)
            elif command == "insert":
                editor = self.window.fieldEditor_forObject_(True, None)
                responder = self.window.firstResponder()
                (responder if hasattr(responder, "insertText_") else editor).insertText_(arg)
            elif command == "presses":
                print(f"presses {self.presses}", flush=True)
            elif command == "quit":
                AppKit.NSApplication.sharedApplication().terminate_(None)


def main() -> None:
    app = AppKit.NSApplication.sharedApplication()
    driver = Driver.alloc().init().setup()

    def reader() -> None:
        for raw in sys.stdin:
            commands.put(raw.rstrip("\n"))
        commands.put("quit")

    threading.Thread(target=reader, daemon=True).start()
    app.run()
    del driver  # kept alive while the app runs


if __name__ == "__main__":
    main()

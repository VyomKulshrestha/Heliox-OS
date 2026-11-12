"""An AppKit window with a text field and a secure text field, driven over stdin, for Heliox Live's checks on macOS.

Like ``macos_fixture.py`` (the reader check's), plus ``front``: an earlier step can leave another app in front, and a
window that is not key has no focused field to read, so the checks ask for it until the daemon reads the field.

Commands: ``front``, ``focus entry``, ``focus password``, ``title <text>``, ``insert <text>``, ``quit``. Prints ``ready``
once shown, and ``ok <command>`` after each command.
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
        style = AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable | AppKit.NSWindowStyleMaskResizable
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(200, 200, 420, 140), style, AppKit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Heliox Live fixture")
        self.entry = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 80, 380, 24))
        self.password = AppKit.NSSecureTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 30, 380, 24))
        self.window.contentView().addSubview_(self.entry)
        self.window.contentView().addSubview_(self.password)
        self.front()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.1, self, "tick:", None, True)
        print("ready", flush=True)
        return self

    def front(self):
        app = AppKit.NSApplication.sharedApplication()
        if hasattr(app, "activate"):
            app.activate()  # macOS 14 and later: cooperative activation
        app.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)
        self.window.orderFrontRegardless()
        self.window.makeFirstResponder_(self.entry)

    def tick_(self, _timer):
        while not commands.empty():
            line = commands.get()
            command, _, arg = line.partition(" ")
            if command == "front":
                self.front()
            elif command == "focus":
                self.window.makeFirstResponder_(self.password if arg == "password" else self.entry)
            elif command == "title":
                self.window.setTitle_(arg)
            elif command == "insert":
                editor = self.window.fieldEditor_forObject_(True, None)
                responder = self.window.firstResponder()
                (responder if hasattr(responder, "insertText_") else editor).insertText_(arg)
            elif command == "quit":
                AppKit.NSApplication.sharedApplication().terminate_(None)
            print(f"ok {line}", flush=True)


def main() -> None:
    app = AppKit.NSApplication.sharedApplication()
    driver = Driver.alloc().init().setup()

    def reader() -> None:
        for raw in sys.stdin:
            commands.put(raw.rstrip("\n"))
        commands.put("quit")

    threading.Thread(target=reader, daemon=True).start()
    app.run()
    del driver


if __name__ == "__main__":
    main()

"""An AppKit window for ``click_hang_check.py``: a text field, and a "Click me" label (static text, no AXPress)
that counts mouse clicks on it, so a click by element ref can only reach it as a real click at its centre.

Commands on stdin: ``clicks`` (prints ``clicks <n>``), ``hang <s>`` (the main thread stops answering for that many
seconds, as a hung app does), ``insert <text>``, ``quit``. Prints ``ready`` once the text field has focus.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import AppKit
from Foundation import NSObject, NSTimer

commands: queue.Queue[str] = queue.Queue()
counts = {"clicks": 0}


class ClickLabel(AppKit.NSTextField):
    def mouseDown_(self, _event):  # noqa: N802
        counts["clicks"] += 1
        print("clicked", flush=True)


class Driver(NSObject):
    def setup(self):  # noqa: D401
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        rect = AppKit.NSMakeRect(200, 200, 420, 180)
        style = AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Heliox click fixture")
        self.entry = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(20, 120, 380, 24))
        self.label = ClickLabel.labelWithString_("Click me")
        self.label.setFrame_(AppKit.NSMakeRect(20, 30, 200, 60))
        self.window.contentView().addSubview_(self.entry)
        self.window.contentView().addSubview_(self.label)
        self.window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        self.window.makeFirstResponder_(self.entry)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.1, self, "tick:", None, True)
        print("ready", flush=True)
        return self

    def tick_(self, _timer):
        while not commands.empty():
            command, _, arg = commands.get().partition(" ")
            if command == "clicks":
                print(f"clicks {counts['clicks']}", flush=True)
            elif command == "hang":
                print("ok hang", flush=True)
                time.sleep(float(arg or 5))
            elif command == "insert":
                editor = self.window.fieldEditor_forObject_(True, None)
                responder = self.window.firstResponder()
                (responder if hasattr(responder, "insertText_") else editor).insertText_(arg)
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
    del driver


if __name__ == "__main__":
    main()

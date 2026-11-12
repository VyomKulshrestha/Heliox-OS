"""A window that shows one line of text in a large font and nothing to type in, for Live's screen-reading check.

    python live_text_window.py "<title>" "<text>"

GTK 3 on Linux (run with the system Python that has PyGObject), AppKit on macOS (the daemon's Python with pyobjc).
Prints ``ready`` once shown; quits when stdin closes.
"""

from __future__ import annotations

import sys
import threading


def gtk(title: str, text: str) -> None:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk

    window = Gtk.Window(title=title)
    window.set_default_size(1100, 200)
    label = Gtk.Label()
    label.set_markup(f'<span font="28" font_family="monospace">{GLib.markup_escape_text(text)}</span>')
    window.add(label)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()

    def ready() -> bool:
        print("ready", flush=True)
        return False

    def wait() -> None:
        sys.stdin.read()
        GLib.idle_add(Gtk.main_quit)

    GLib.timeout_add(500, ready)
    threading.Thread(target=wait, daemon=True).start()
    Gtk.main()


def appkit(title: str, text: str) -> None:
    import AppKit
    from Foundation import NSObject

    class Quit(NSObject):
        def quit_(self, _arg):
            AppKit.NSApplication.sharedApplication().terminate_(None)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(100, 300, 1100, 200),
        AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    window.setTitle_(title)
    label = AppKit.NSTextField.labelWithString_(text)
    label.setFont_(AppKit.NSFont.monospacedSystemFontOfSize_weight_(28, 0))
    label.setFrame_(AppKit.NSMakeRect(20, 70, 1060, 60))
    window.contentView().addSubview_(label)
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    quitter = Quit.alloc().init()
    print("ready", flush=True)

    def wait() -> None:
        sys.stdin.read()
        quitter.performSelectorOnMainThread_withObject_waitUntilDone_("quit:", None, False)

    threading.Thread(target=wait, daemon=True).start()
    app.run()


if __name__ == "__main__":
    (appkit if sys.platform == "darwin" else gtk)(sys.argv[1], sys.argv[2])

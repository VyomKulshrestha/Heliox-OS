"""A GTK 4 window with one text entry whose title can change, for Live's never-read check on GTK 4.

    /usr/bin/python3 live_gtk4_window.py "<title>"

Run with the system Python that has PyGObject and GTK 4 (``gir1.2-gtk-4.0``). Commands on stdin, one per line:
``title <text>``, ``focus``, ``insert <text>`` (added to the entry from inside the app), ``quit``. Prints ``ready``
once the entry has focus, and ``ok <command>`` after each command.
"""

from __future__ import annotations

import sys
import threading

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402


def main(title: str) -> None:
    app = Gtk.Application(application_id="org.heliox.LiveGtk4Window")

    def activate(application: Gtk.Application) -> None:
        window = Gtk.ApplicationWindow(application=application, title=title)
        window.set_default_size(420, 120)
        entry = Gtk.Entry()
        window.set_child(entry)
        window.present()

        def run(line: str) -> bool:
            command, _, arg = line.partition(" ")
            if command == "title":
                window.set_title(arg)
            elif command == "focus":
                window.present()
                entry.grab_focus()
            elif command == "insert":
                entry.set_text(entry.get_text() + arg)
                entry.set_position(-1)
            elif command == "quit":
                application.quit()
            print(f"ok {line}", flush=True)
            return False

        def reader() -> None:
            for raw in sys.stdin:
                GLib.idle_add(run, raw.rstrip("\n"))
            GLib.idle_add(application.quit)

        def ready() -> bool:
            entry.grab_focus()
            print("ready", flush=True)
            return False

        GLib.timeout_add(500, ready)
        threading.Thread(target=reader, daemon=True).start()

    app.connect("activate", activate)
    app.run([])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Heliox Live GTK 4 window")

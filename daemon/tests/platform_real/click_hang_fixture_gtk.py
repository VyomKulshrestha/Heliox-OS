"""A GTK 3 window for ``click_hang_check.py``: a text entry, and a "Click me" label that offers no accessibility
action but counts mouse clicks on it, so a click by element ref can only reach it as a real click at its centre.

Run with the system Python that has PyGObject. Commands on stdin: ``clicks`` (prints ``clicks <n>``), ``front``
(brings the window forward), ``hang <s>`` (the main loop stops answering for that many seconds, as a hung app
does), ``quit``. Prints ``ready`` once the entry has focus, and ``ok <command>`` after each command.
"""

from __future__ import annotations

import sys
import threading
import time

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

TITLE = "Heliox click fixture"


def main() -> None:
    window = Gtk.Window(title=TITLE)
    window.set_default_size(420, 200)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    entry = Gtk.Entry()
    entry.get_accessible().set_name("message")
    clicks = [0]
    target = Gtk.EventBox()
    label = Gtk.Label(label="Click me")
    label.set_size_request(200, 60)
    target.add(label)

    def clicked(_widget, _event) -> bool:
        clicks[0] += 1
        print("clicked", flush=True)
        return True

    target.connect("button-press-event", clicked)
    box.pack_start(entry, False, False, 0)
    box.pack_start(target, False, False, 0)
    window.add(box)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()

    def run(line: str) -> bool:
        command, _, arg = line.partition(" ")
        if command == "clicks":
            print(f"clicks {clicks[0]}", flush=True)
        elif command == "front":
            window.present()
            entry.grab_focus()
        elif command == "hang":
            print(f"ok {line}", flush=True)
            time.sleep(float(arg or 5))
            return False
        elif command == "quit":
            Gtk.main_quit()
        print(f"ok {line}", flush=True)
        return False

    def reader() -> None:
        for raw in sys.stdin:
            GLib.idle_add(run, raw.rstrip("\n"))
        GLib.idle_add(Gtk.main_quit)

    def ready() -> bool:
        entry.grab_focus()
        print("ready", flush=True)
        return False

    GLib.timeout_add(600, ready)
    threading.Thread(target=reader, daemon=True).start()
    Gtk.main()


if __name__ == "__main__":
    main()

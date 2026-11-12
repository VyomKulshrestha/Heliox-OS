"""A GTK 3 window with a text entry, a password entry and a label, driven over stdin, for the AT-SPI check.

Run with the system Python that has PyGObject (``python3-gi`` and ``gir1.2-gtk-3.0``), not the daemon's venv.
Commands, one per line: ``focus entry``, ``focus password``, ``title <text>``, ``insert <text>`` (types into the
focused entry from inside the app, for sessions where no input tool can send keys), ``quit``. Prints ``ready``
once the window is shown with the entry focused, and ``ok <command>`` after each command.
"""

from __future__ import annotations

import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

TITLE = "Heliox AT-SPI fixture"


def main() -> None:
    window = Gtk.Window(title=TITLE)
    window.set_default_size(420, 160)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    label = Gtk.Label(label="Type here")
    entry = Gtk.Entry()
    entry.get_accessible().set_name("message")
    password = Gtk.Entry()
    password.set_visibility(False)
    password.get_accessible().set_name("password")
    button = Gtk.Button(label="Press me")
    button.connect("clicked", lambda _b: print("pressed", flush=True))
    for widget in (label, entry, password, button):
        box.pack_start(widget, False, False, 0)
    window.add(box)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.present()

    def focused() -> Gtk.Entry:
        return password if password.has_focus() else entry

    def run(line: str) -> bool:
        command, _, arg = line.partition(" ")
        if command == "focus":
            (password if arg == "password" else entry).grab_focus()
        elif command == "title":
            window.set_title(arg)
        elif command == "insert":
            target = focused()
            position = target.get_position()
            target.insert_text(arg, position)
            target.set_position(position + len(arg))
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

    GLib.timeout_add(500, ready)
    threading.Thread(target=reader, daemon=True).start()
    Gtk.main()


if __name__ == "__main__":
    main()

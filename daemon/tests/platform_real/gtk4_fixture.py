"""A GTK 4 window with a text entry and a password entry, for ``toolkit_check.py``.

Run with the system Python that has PyGObject and GTK 4 (``gir1.2-gtk-4.0``). Commands on stdin, one per line:
``focus entry``, ``focus password``, ``text`` (what the keys reached), ``quit``. Prints ``ready`` once the entry
has focus, ``ok <command>`` after each.
"""

from __future__ import annotations

import sys
import threading

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

TITLE = "Heliox GTK 4 fixture"


def main() -> None:
    app = Gtk.Application(application_id="org.heliox.Gtk4Fixture")

    def activate(application: Gtk.Application) -> None:
        window = Gtk.ApplicationWindow(application=application, title=TITLE)
        window.set_default_size(420, 160)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        entry = Gtk.Entry()
        password = Gtk.PasswordEntry()
        for widget, label in ((entry, "message"), (password, "password")):
            try:
                widget.update_property([Gtk.AccessibleProperty.LABEL], [label])
            except (TypeError, AttributeError):
                pass  # older PyGObject: the field stays unnamed, which the check does not need
        box.append(entry)
        box.append(password)
        window.set_child(box)
        window.present()

        def run(line: str) -> bool:
            command, _, arg = line.partition(" ")
            if command == "focus":
                (password if arg == "password" else entry).grab_focus()
            elif command == "text":
                # What the keys actually reached: the entry's text, and only the length of the password's. In GTK 4
                # the focus sits on the text widget inside an entry, so "has focus" means the window's focus widget
                # is the entry or inside it.
                focus = window.get_focus()

                def within(widget: Gtk.Widget) -> bool:
                    return focus is not None and (focus == widget or focus.is_ancestor(widget))

                print(
                    f"text entry={entry.get_text()!r} password_length={len(password.get_text())} "
                    f"entry_focus={within(entry)} password_focus={within(password)} "
                    f"window_active={window.is_active()}",
                    flush=True,
                )
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

        GLib.timeout_add(700, ready)
        threading.Thread(target=reader, daemon=True).start()

    app.connect("activate", activate)
    app.run([])


if __name__ == "__main__":
    main()

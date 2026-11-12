"""A GTK 3 window that fills the screen with one line of large black text on white, for ``ocr_check.py``.

Run with the system Python that has PyGObject. Prints ``ready`` once shown; quits when stdin closes or it is killed.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

LINE = "Heliox reads this line"


def main() -> None:
    window = Gtk.Window(title="Heliox OCR fixture")
    window.set_default_size(1280, 800)
    window.override_background_color(Gtk.StateFlags.NORMAL, Gdk_rgba(1.0, 1.0, 1.0))
    label = Gtk.Label()
    label.set_markup(f'<span font="Sans 56" foreground="black">{LINE}</span>')
    window.add(label)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    window.move(0, 0)

    def ready() -> bool:
        print("ready", flush=True)
        return False

    GLib.timeout_add(700, ready)
    Gtk.main()


def Gdk_rgba(red: float, green: float, blue: float):  # noqa: N802
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk

    color = Gdk.RGBA()
    color.red, color.green, color.blue, color.alpha = red, green, blue, 1.0
    return color


if __name__ == "__main__":
    main()

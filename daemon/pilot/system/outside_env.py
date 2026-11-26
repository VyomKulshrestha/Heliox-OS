"""The daemon's own environment, cleaned of an AppImage's start-up settings, so every program it starts (an app the
person asks to open, a shell command, the terminal helpers) sees the desktop's settings.

The desktop app gives the daemon this environment already (tauri-app/src-tauri/src/outside_env.rs); this is the same
rule, applied once when the daemon starts, for a daemon started some other way from inside an AppImage (an older app,
a start path that skipped the rule). It acts only when the environment carries the AppImage runtime's marks
(APPIMAGE or APPDIR); on Windows, macOS, the .deb, the .rpm and a daemon the app started cleanly nothing changes.

The rule, as in outside_env.rs:

- the runtime's own names (APPIMAGE, APPDIR, ARGV0, OWD) and Python's search settings (PYTHONHOME, PYTHONPATH) go;
- a fixed toolkit setting the launcher made goes, or gets the value recorded before the start-up
  (APPIMAGE_ORIGINAL_<NAME> or <NAME>_ORIG): GDK_BACKEND=x11 and GTK_THEME=Adwaita[:light|:dark] when the GTK hook
  ran (GTK_DATA_PREFIX in the folder), PYTHONDONTWRITEBYTECODE=1 when AppRun.c ran (PYTHONHOME in the folder),
  GST_REGISTRY_REUSE_PLUGIN_SCANNER=no when the GStreamer hook ran; any other value is the person's and stays;
- a value pointing into the folder gets its recorded value, else loses its entries under the folder (and the
  /usr/share the GTK hook put after its own in XDG_DATA_DIRS); nothing left, or a mention left, means unset;
- every other variable is kept as it is.

The folder is $APPDIR: without it nothing here can tell the AppImage's paths apart, and only the runtime's names go.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, MutableMapping

logger = logging.getLogger("pilot.system.outside_env")

DROPPED = ("PYTHONHOME", "PYTHONPATH")
RUNTIME = ("APPIMAGE", "APPDIR", "ARGV0", "OWD")
_NAME_CHARS = "/._-~+@"


def from_appimage(env: Mapping[str, str]) -> bool:
    return bool(env.get("APPIMAGE") or env.get("APPDIR"))


def _inside(entry: str, folder: str) -> bool:
    return entry.startswith(folder) and (len(entry) == len(folder) or entry[len(folder)] == "/")


def _mentions(value: str, folder: str) -> bool:
    start = value.find(folder)
    while start >= 0:
        before = value[start - 1] if start > 0 else ""
        end = start + len(folder)
        after = value[end] if end < len(value) else ""
        if (not before or not (before.isalnum() or before in _NAME_CHARS)) and after in ("", "/", ":"):
            return True
        start = value.find(folder, start + 1)
    return False


def _cleaned(value: str, folder: str) -> str | None:
    if not _mentions(value, folder):
        return value
    kept = [entry for entry in value.split(":") if not _inside(entry, folder)]
    joined = ":".join(kept)
    if all(not entry for entry in kept) or _mentions(joined, folder):
        return None
    return joined


def _ran(env: Mapping[str, str], mark: str, folder: str) -> bool:
    return _mentions(env.get(mark, ""), folder)


def set_by_the_launcher(name: str, value: str, env: Mapping[str, str], folder: str) -> bool:
    if name == "GDK_BACKEND":
        return _ran(env, "GTK_DATA_PREFIX", folder) and value == "x11"
    if name == "GTK_THEME":
        return _ran(env, "GTK_DATA_PREFIX", folder) and (
            value in ("Adwaita", "Adwaita:light", "Adwaita:dark") or env.get("APPIMAGE_GTK_THEME") == value
        )
    if name == "PYTHONDONTWRITEBYTECODE":
        return _ran(env, "PYTHONHOME", folder) and value == "1"
    if name == "GST_REGISTRY_REUSE_PLUGIN_SCANNER":
        scanner = _ran(env, "GST_PLUGIN_SCANNER", folder) or _ran(env, "GST_PLUGIN_SCANNER_1_0", folder)
        return scanner and value == "no"
    return False


def _the_hooks_share(entry: str, folder: str) -> bool:
    rest = entry[len(folder) :] if entry.startswith(folder) else None
    return rest is not None and rest.startswith("/") and rest.strip("/") == "usr/share"


def without_the_hooks_share(value: str, folder: str) -> str:
    entries = value.split(":")
    kept = [
        entry
        for at, entry in enumerate(entries)
        if not (at > 0 and _the_hooks_share(entries[at - 1], folder) and entry.rstrip("/") == "/usr/share")
    ]
    return ":".join(kept)


def outside(env: Mapping[str, str]) -> dict[str, str]:
    """The environment for programs started outside the AppImage (see the module's comment)."""
    if not from_appimage(env):
        return dict(env)
    folder = (env.get("APPDIR") or "").rstrip("/")
    result: dict[str, str] = {}
    for name, value in env.items():
        if name in DROPPED or name in RUNTIME:
            continue
        if not folder:
            result[name] = value
            continue
        recorded = env.get(f"APPIMAGE_ORIGINAL_{name}", env.get(f"{name}_ORIG"))
        if set_by_the_launcher(name, value, env, folder):
            if recorded:
                result[name] = recorded
            continue
        if not _mentions(value, folder):
            result[name] = value
            continue
        if recorded is not None:
            source = recorded
        elif name == "XDG_DATA_DIRS" and _ran(env, "GTK_DATA_PREFIX", folder):
            source = without_the_hooks_share(value, folder)
        else:
            source = value
        kept = _cleaned(source, folder)
        if kept:
            result[name] = kept
    return result


def clean_this_process(env: MutableMapping[str, str] | None = None, platform: str | None = None) -> list[str]:
    """Apply ``outside`` to this process's environment (or ``env``) in place, on Linux only. The names it changed or
    removed, for one log line; never their values."""
    target = os.environ if env is None else env
    if (platform or sys.platform) != "linux" or not from_appimage(target):
        return []
    wanted = outside(dict(target))
    changed = sorted(name for name in target if wanted.get(name) != target[name])
    for name in changed:
        if name in wanted:
            target[name] = wanted[name]
        else:
            del target[name]
    if changed:
        logger.info("Left the AppImage's start-up settings out of the daemon's environment: %s", ", ".join(changed))
    return changed


__all__ = ["clean_this_process", "from_appimage", "outside", "set_by_the_launcher", "without_the_hooks_share"]

"""Keeps tools that run outside the test suite out of the person's real Heliox folders.

Anything that starts a Heliox server or opens Heliox's stores outside the daemon itself (the conversation bench, the
session eval, measuring scripts, replay tools that build a real listener) calls ``refuse_real_folders(purpose)``
first. It returns where the folders are and raises ``RealFoldersRefused`` unless one of these holds:

- the folders are a test root (``use_scratch_folders(root)``, the same marked folder the test suite uses: every
  Heliox folder, including ``~/.cache/heliox`` and setup's ``~/.config/heliox-os``, moves under it);
- every Heliox folder is under a fake home, a home folder that is not this account's own (HOME and USERPROFILE
  pointed at an empty folder, and on Linux XDG_RUNTIME_DIR inside it too);
- the caller passed ``allow_real=True``, which a tool sets only from an explicit ``--real-folders`` on its command
  line. The daemon's own entry point is the one caller that sets it without a flag.

The settings are covered too: PilotConfig.save refuses outside the daemon on the real folders, paid run or not.

A paid run, under a scratch root, reads the person's provider settings with ``apply_owner_model_settings(config)``
(copied onto its own config, whose save writes its own folder) and records what it spent with
``real_budget_tracker(config.model)``, the one intended write to the real folders.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REAL_FOLDERS_FLAG = "--real-folders"

_daemon_entry = False


class RealFoldersRefused(RuntimeError):
    """A tool outside the daemon was about to use the person's real Heliox folders."""


def enter_as_daemon() -> None:
    """Called by the daemon's own entry point (pilot.server.main): its folders are the real ones."""
    global _daemon_entry
    _daemon_entry = True


def is_daemon() -> bool:
    return _daemon_entry


def wants_real_folders(argv: list[str] | None = None) -> bool:
    """True when the command line carries the explicit --real-folders flag."""
    return REAL_FOLDERS_FLAG in (sys.argv[1:] if argv is None else argv)


def require_real_folders_flag(tool: str, argv: list[str] | None = None) -> None:
    """For the owner's own tools that read or write their real Heliox data on purpose: without --real-folders on
    the command line they stop here and say why (OWNER_TOOLS in tests/test_real_folders.py lists them)."""
    if not wants_real_folders(argv):
        raise SystemExit(
            f"{tool} reads or writes your real Heliox data. Run it again with {REAL_FOLDERS_FLAG} if that is what "
            "you want."
        )


def use_scratch_folders(root: Path | None = None) -> Path:
    """Lay every Heliox folder out under ``root`` for this process and the processes it starts.

    Two lines at the top of a script, before anything imports pilot.config (the folders are fixed when it loads):

        from pilot.real_folders import use_scratch_folders
        use_scratch_folders()

    Without ``root``, a new temporary folder that is removed when the process ends. With one (a bench that keeps
    its downloaded models between runs), that folder, kept. Writes the marker pilot.config looks for, so a stray
    HELIOX_TEST_ROOT without it is still ignored. Under a test root already (a tool imported by the test suite, or
    started by a tool that set one up), that root stays and is returned.
    """
    from_config = sys.modules.get("pilot.config")
    active = getattr(from_config, "_TEST_ROOT", None) if from_config is not None else _marked_root()
    if active is not None:
        return active
    if from_config is not None:
        raise RuntimeError("use_scratch_folders() must run before pilot.config is imported")
    if root is None:
        import atexit
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="heliox-scratch-"))
        atexit.register(shutil.rmtree, root, True)
    root = Path(root).resolve()
    for layout in (".config", ".local/share", ".local/state", "runtime"):
        (root / layout / "heliox-os").mkdir(parents=True, exist_ok=True)
    (root / ".heliox-test-root").write_text("A scratch root for a Heliox tool; see pilot/real_folders.py", "utf-8")
    os.environ["HELIOX_TEST_ROOT"] = str(root)
    return root


def _marked_root() -> Path | None:
    """The test root HELIOX_TEST_ROOT names, when it carries the marker (pilot.config's own rule)."""
    named = os.environ.get("HELIOX_TEST_ROOT", "").strip()
    return Path(named) if named and (Path(named) / ".heliox-test-root").is_file() else None


def account_home() -> Path | None:
    """This account's own home folder, whatever HOME and USERPROFILE say; None when the system cannot tell."""
    if sys.platform == "win32":
        try:
            import ctypes
            import uuid
            from ctypes import wintypes

            folder_id = uuid.UUID("5E6C858F-0E22-4760-9AFE-EA3317B67173")  # FOLDERID_Profile
            guid = (ctypes.c_byte * 16).from_buffer_copy(folder_id.bytes_le)
            found = ctypes.c_wchar_p()
            shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
            if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, wintypes.HANDLE(0), ctypes.byref(found)) != 0:
                return None
            try:
                return Path(found.value) if found.value else None
            finally:
                ctypes.windll.ole32.CoTaskMemFree(found)  # type: ignore[attr-defined]
        except (OSError, AttributeError, ValueError):
            return None
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError, OSError):
        return None


def heliox_folders() -> dict[str, Path]:
    """Every folder Heliox reads or writes, as pilot.config resolved them for this process."""
    from pilot import config

    return {
        "config": config.CONFIG_DIR,
        "data": config.DATA_DIR,
        "state": config.STATE_DIR,
        "runtime": config.RUNTIME_DIR,
        "cache": config.CACHE_DIR,
        "setup": config.SETUP_DIR,
    }


def _key(path: Path) -> str:
    text = os.path.normpath(os.path.abspath(os.path.realpath(path)))
    return text.casefold() if sys.platform in ("win32", "darwin") else text


def _under(path: Path, root: Path) -> bool:
    inner, outer = _key(path), _key(root)
    return inner == outer or inner.startswith(outer.rstrip(os.sep) + os.sep)


def where_folders_are() -> tuple[str, list[str]]:
    """("test root" | "fake home" | "real", the folders outside the fake home when that is what made it real)."""
    from pilot import config

    if config._TEST_ROOT is not None:
        return "test root", []
    return classify(Path.home(), account_home(), heliox_folders())


def classify(home: Path, own: Path | None, folders: dict[str, Path]) -> tuple[str, list[str]]:
    """ "fake home" when ``home`` is not the account's own and every folder is under it; "real" otherwise, with the
    folders that are outside the fake home (on Linux the runtime folder, on Windows LOCALAPPDATA's, an XDG one)."""
    if own is None or _key(home) == _key(own):
        return "real", []
    outside = [f"{name} {path}" for name, path in folders.items() if not _under(path, home)]
    return ("fake home", []) if not outside else ("real", outside)


def refuse_real_folders(purpose: str, *, allow_real: bool = False, honour_flag: bool = True) -> str:
    """Raise RealFoldersRefused unless Heliox's folders are a test root, a fake home, or real ones explicitly allowed.

    ``honour_flag=False``: --real-folders does not allow it either; only the daemon does (the wake-word calibration,
    which a replay tool reading the person's settings on purpose must still not change).
    Returns "test root", "fake home" or "real" (the last only when allowed)."""
    kind, outside = where_folders_are()
    if kind != "real" or allow_real or _daemon_entry or (honour_flag and wants_real_folders()):
        return kind
    detail = f" ({'; '.join(outside)} would still be the real ones)" if outside else ""
    raise RealFoldersRefused(
        f"{purpose} would use this person's real Heliox folders{detail}. Run it under a scratch root "
        "(pilot.real_folders.use_scratch_folders, before importing pilot.config) or a fake home, or pass "
        f"{REAL_FOLDERS_FLAG} to use the real ones on purpose."
    )


def _owner_folder(variable: str, fallback: str) -> Path:
    """One of the person's real Heliox folders, from this account's home and the XDG variable (pilot.config's rule
    without a test root)."""
    home = account_home() or Path.home()
    value = os.environ.get(variable, "").strip()
    return (Path(value) if value and Path(value).is_absolute() else home / fallback) / "heliox-os"


def real_budget_db() -> Path:
    """The person's real budget tracker store, for a paid run that must count what it spent where the daemon does."""
    return _owner_folder("XDG_DATA_HOME", ".local/share") / "pilot.db"


def real_budget_tracker(model_config: Any) -> Any:
    """The one write a paid run makes to the real folders: its spend, through the budget tracker, into the person's
    own store. Nothing else of the real data folder is opened; the run's server and stores stay in its own folder."""
    from pilot.models.budget_tracker import BudgetTracker

    return BudgetTracker(model_config, str(real_budget_db()))


# The sections a run outside the daemon may copy from the person's settings: the provider and model ([model]), how
# the voice hears and speaks ([voice]) and the conversation's models and reply length ([conversation]).
OWNER_SECTIONS = ("model", "voice", "conversation")
# Never copied: this computer's devices. A bench feeds recorded files and chooses its own device.
NOT_COPIED = {"voice": frozenset({"input_device", "headset_input_choices", "tts_device"})}


def owner_settings(section: str) -> dict[str, Any]:
    """One section of the person's settings, read only and returned as a copy; {} when missing or unreadable.

    A run outside the daemon (a paid eval, the bench) needs the provider, the model and the voice the person chose.
    It reads them here and applies them to a config of its own folder, so it never holds a config object whose save
    would write the person's file. The file is found from the account's home as the system names it, not from HOME.
    """
    import tomllib

    if section not in OWNER_SECTIONS:
        raise ValueError(f"not a section a run may copy: {section!r} (one of {', '.join(OWNER_SECTIONS)})")
    path = _owner_folder("XDG_CONFIG_HOME", ".config") / "config.toml"
    try:
        found = tomllib.loads(path.read_text(encoding="utf-8")).get(section, {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    return dict(found) if isinstance(found, dict) else {}


def _owner_heliox_folders() -> list[Path]:
    home = account_home() or Path.home()
    return [
        _owner_folder("XDG_CONFIG_HOME", ".config"),
        _owner_folder("XDG_DATA_HOME", ".local/share"),
        _owner_folder("XDG_STATE_HOME", ".local/state"),
        home / ".config" / "heliox-os",
        home / ".cache" / "heliox",
        home / "AppData" / "Local" / "heliox-os",
        home / "Library" / "Caches" / "heliox-os",
    ]


def _points_into_owner_folders(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError):
        return False
    return path.is_absolute() and any(_under(path, folder) for folder in _owner_heliox_folders())


def apply_owner_settings(
    config: Any,
    sections: tuple[str, ...] = ("model",),
    *,
    only: tuple[str, ...] = (),
    skip: tuple[str, ...] = (),
) -> list[str]:
    """Copy fields of the person's settings onto ``config`` (one whose files are in the run's own folder).

    ``only`` names the fields to copy as "section.name". For [voice] and [conversation] it must name at least one: a
    setting added to those sections later is then not copied into a run by default (ValueError otherwise). [model]
    with no field named copies every field this version knows. ``skip`` leaves out "section.name" fields or whole
    sections. Returns "section.name" for each field copied.

    Never copied, even when named: a field this version does not know, the device fields (NOT_COPIED), and a text value
    that is a path inside the person's Heliox folders (a model folder in their data folder, say): through it the run
    would read or write their folder again."""
    copied = []
    for section in sections:
        if section in skip:
            continue
        named = {entry.split(".", 1)[1] for entry in only if entry.startswith(f"{section}.")}
        if not named and section != "model":
            raise ValueError(f"name the [{section}] fields to copy with only=, as 'section.name'")
        target = getattr(config, section)
        for name, value in owner_settings(section).items():
            if (named and name not in named) or f"{section}.{name}" in skip or name in NOT_COPIED.get(section, ()):
                continue
            if not hasattr(target, name) or _points_into_owner_folders(value):
                continue
            setattr(target, name, value)
            copied.append(f"{section}.{name}")
    return copied


def owner_model_settings() -> dict[str, Any]:
    """The [model] section of the person's settings, read and copied (see owner_settings)."""
    return owner_settings("model")


def apply_owner_model_settings(config: Any) -> list[str]:
    """The person's [model] settings copied onto ``config``; returns the bare names (see apply_owner_settings)."""
    return [name.split(".", 1)[1] for name in apply_owner_settings(config, ("model",))]

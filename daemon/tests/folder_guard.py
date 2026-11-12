"""Keeps the test suite out of the real Heliox folders of the person running it.

``install()`` runs from ``tests/conftest.py`` before any pilot module is imported:

1. It works out the real folders the way ``pilot.config`` does (config, data, state and runtime, each ``heliox-os``),
   plus the home-folder places removal also counts as Heliox's data (``~/.heliox``, ``~/.cache/heliox``,
   ``~/.pilot``), and lists what is in them.
2. It makes one temporary folder for the session, marks it (``.heliox-test-root``) and names it in HELIOX_TEST_ROOT,
   so ``pilot.config`` lays all of Heliox's folders out there (config, data, state, runtime, ``~/.cache/heliox`` and
   setup's ``~/.config/heliox-os``), for this process and every daemon a test starts. Without the marker the variable
   is ignored, so a real install never picks it up.
3. An audit hook refuses (PermissionError) any open, SQLite connect, folder creation, rename or removal under the
   real folders, and records which test asked. A refused open never touches the file.

At the end of the session ``report()`` lists the tests that were refused and anything under the real folders that
was created or changed since the start (a subprocess a test started is not seen by the audit hook, but its writes
show here). On CI either fails the session. Elsewhere a change without a refused test is only reported: a Heliox
daemon running on the same computer writes there too.

Importing this module has no effect.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

_EVENTS_ONE_PATH = {
    "open": 0,
    "sqlite3.connect": 0,
    "os.mkdir": 0,
    "os.remove": 0,
    "os.rmdir": 0,
    "os.truncate": 0,
    "os.chmod": 0,
    "shutil.rmtree": 0,
    "shutil.copyfile": 1,
    "shutil.copytree": 1,
}
_EVENTS_TWO_PATHS = {"os.rename", "os.replace", "os.link", "os.symlink", "shutil.move"}

_state: dict = {"roots": (), "skip": (), "refused": {}, "before": {}, "session": None, "active": False}


def real_folders(env: dict[str, str] | None = None) -> list[Path]:
    """Heliox's folders for this person, computed as pilot.config does, without importing it."""
    env = dict(os.environ if env is None else env)
    home = Path(env.get("USERPROFILE") if sys.platform == "win32" and env.get("USERPROFILE") else Path.home())

    def heliox(variable: str, fallback: str) -> Path:
        return (Path(env[variable]) if variable in env else home / fallback) / "heliox-os"

    runtime_root = env.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_root:
        runtime = Path(runtime_root) / "heliox-os"
    elif sys.platform == "darwin":
        runtime = home / "Library" / "Caches" / "heliox-os" / "runtime"
    elif sys.platform == "win32":
        runtime = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "heliox-os" / "runtime"
    else:
        uid = os.getuid()
        run_user = Path(f"/run/user/{uid}")
        runtime = (
            run_user / "heliox-os"
            if run_user.is_dir() and os.access(run_user, os.W_OK)
            else Path(env.get("TMPDIR", "/tmp")) / f"heliox-os-runtime-{uid}"
        )
    return [
        heliox("XDG_CONFIG_HOME", ".config"),
        heliox("XDG_DATA_HOME", ".local/share"),
        heliox("XDG_STATE_HOME", ".local/state"),
        home / ".config" / "heliox-os",  # setup's folder, which does not follow XDG_CONFIG_HOME
        runtime,
        home / ".heliox",
        home / ".cache" / "heliox",
        home / ".pilot",
    ]


_FOLD = sys.platform in ("win32", "darwin")  # case-insensitive by default there (NTFS, APFS)
_busy = threading.local()


def _key(path) -> str:
    """Absolute, normalised and, where the file system ignores case, case-folded. No file system access."""
    text = os.path.normpath(os.path.abspath(os.fsdecode(path)))
    return text.casefold() if _FOLD else text


def _resolved(path) -> str:
    """As _key, after symbolic links and junctions are followed (os.path.realpath)."""
    text = os.path.realpath(os.fsdecode(path))
    return text.casefold() if _FOLD else text


def _under(text: str, roots: tuple[str, ...]) -> bool:
    return any(text == root or text.startswith(root + os.sep) for root in roots)


def _inside(path) -> str | None:
    """The path, when it is under a real folder, written or reached through a link; else None."""
    if isinstance(path, int) or path is None:
        return None
    try:
        text = _key(path)
    except (TypeError, ValueError):
        return None
    if _under(text, _state["roots"]):
        return text
    if _under(text, _state["skip"]):  # Python's own files and this checkout: never a real Heliox folder
        return None
    try:
        real = _resolved(path)
    except (OSError, ValueError):
        return None
    return real if real != text and _under(real, _state["roots"]) else None


def _hook(event: str, args: tuple) -> None:
    if not _state["active"] or getattr(_busy, "on", False):
        return
    if event in _EVENTS_ONE_PATH:
        paths = (args[_EVENTS_ONE_PATH[event]],) if len(args) > _EVENTS_ONE_PATH[event] else ()
    elif event in _EVENTS_TWO_PATHS:
        paths = args[:2]
    else:
        return
    _busy.on = True  # resolving a link must not come back here
    try:
        for path in paths:
            if event == "sqlite3.connect" and isinstance(path, (str, bytes)):
                text = os.fsdecode(path)
                if text.startswith("file:"):
                    path = text[5:].split("?", 1)[0]
                elif text == ":memory:" or not text:
                    continue
            hit = _inside(path)
            if hit is None:
                continue
            test = os.environ.get("PYTEST_CURRENT_TEST", "outside any test").split(" (")[0]
            _state["refused"].setdefault(test, set()).add(f"{event} {hit}")
            raise PermissionError(
                f"{test} reached Heliox's real folders: {event} {os.fsdecode(path)} (refused by tests/folder_guard.py; "
                "use tmp_path, or the session folders pilot.config already points at)"
            )
    finally:
        _busy.on = False


def listing(roots) -> dict[str, tuple[int, int]]:
    """Every file under the folders: (size, modification time in ns)."""
    found: dict[str, tuple[int, int]] = {}
    for root in roots:
        for folder, _dirs, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                found[path] = (info.st_size, info.st_mtime_ns)
    return found


def install() -> Path:
    """Before any pilot import: list the real folders, point Heliox at a session folder, refuse the real ones."""
    if _state["session"] is not None:
        return _state["session"]
    roots = [root for root in real_folders()]
    session = Path(tempfile.mkdtemp(prefix="heliox-tests-"))
    # pilot.config's test root, not the XDG variables: PulseAudio, the accessibility bus and other programs the
    # platform tests talk to find their sockets and cookies through those. The marker file is what makes pilot.config
    # use it; daemons a test starts inherit the variable and find the same marker.
    for layout in (".config", ".local/share", ".local/state", "runtime"):
        (session / layout / "heliox-os").mkdir(parents=True)  # laid out as in a home folder
    (session / ".heliox-test-root").write_text(
        "Heliox's test suite; see daemon/tests/folder_guard.py", encoding="utf-8"
    )
    os.environ["HELIOX_TEST_ROOT"] = str(session)
    session_key = _key(session)
    # A real folder that contains the session folder (TMPDIR under ~/.pilot, say) would refuse the session itself.
    kept = [root for root in roots if not session_key.startswith(_key(root) + os.sep)]
    _state["roots"] = tuple(dict.fromkeys([*map(_key, kept), *map(_resolved, kept)]))  # as written, and as linked
    checkout = Path(__file__).resolve().parents[2]
    _state["skip"] = tuple(
        dict.fromkeys(_key(folder) for folder in (sys.prefix, sys.base_prefix, sys.exec_prefix, checkout))
    )
    _state["before"] = listing(_state["roots"])
    _state["session"] = session
    _state["active"] = True
    sys.addaudithook(_hook)
    return session


def failure_for(test: str) -> str | None:
    """Why ``test`` must fail, when the guard refused something it did; None otherwise.

    The refusal itself is a PermissionError where the access happened, and product code may rightly swallow it (a
    missing or unreadable voice calibration must not stop the listener). So the test is failed here, at its
    teardown, whatever the code under test did with the error."""
    hits = _state["refused"].get(test)
    if not hits:
        return None
    lines = [
        f"{test} reached Heliox's real folders of the person running the tests (refused by tests/folder_guard.py):",
        *(f"  {hit}" for hit in sorted(hits)[:5]),
        "Use tmp_path, or the session folders pilot.config already points at (CACHE_DIR, DATA_DIR, ...).",
    ]
    return "\n".join(lines)


def report() -> tuple[list[str], bool]:
    """Lines for the end of the session, and whether the session must fail."""
    _state["active"] = False
    after = listing(_state["roots"])
    before = _state["before"]
    changed = sorted(path for path, stamp in after.items() if before.get(path) != stamp)
    refused = _state["refused"]
    lines = [
        f"FOLDER-GUARD {len(refused)} tests reached Heliox's real folders and were refused; "
        f"{len(changed)} files there were created or changed during the run"
    ]
    for test, hits in sorted(refused.items()):
        lines.append(f"FOLDER-GUARD refused {test}: {'; '.join(sorted(hits)[:3])}")
    for path in changed[:40]:
        lines.append(f"FOLDER-GUARD changed {path}")
    on_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    return lines, bool(refused) or (bool(changed) and on_ci)

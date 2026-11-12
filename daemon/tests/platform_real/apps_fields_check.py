"""Fields where people actually type: a browser, an Electron app and a Qt app, read by Heliox's Linux reader (X11).

    python apps_fields_check.py chrome-default chrome-switch chrome-forced chrome-forced-switch chrome-env \\
        chrome-late electron-default electron-switch electron-forced electron-forced-switch electron-env \\
        electron-late qt-default qt-switch qt-forced

Each target shows a text field and a password field (field_fixture.html in the browser and Electron, qt_fixture.py
for Qt). For each:

- the text typed into the text field must be read back (``<target>/reads-typed-text``);
- only after that first successful read, every read for five seconds after typing into the password field must be
  refused and none may carry the typed text (``<target>/password-refused``);
- DIAG lines dump, for the field and its two parents, the role name, numeric role, states and attributes, and
  whether the session says toolkit accessibility is on (org.a11y.Status IsEnabled).

A target is the app and its setup joined by "-":
- ``switch`` turns the desktop's toolkit accessibility on, the setting Live asks the person for. It is off
  otherwise.
- ``forced`` adds the app's own switch: Chrome and Electron ``--force-renderer-accessibility``, Qt
  ``QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1``.
- ``env`` starts the app with ``ACCESSIBILITY_ENABLED=1``, which Heliox could do for an app it starts itself.
- ``late`` starts the app with the desktop switch off and turns it on once the window is shown: whether an app that
  is already running joins the accessibility bus without a restart.
- ``default`` is the app as a person has it.

Every target with ``switch``, and ``qt-forced``, must read the field (a CHECK). The others answer whether that setup
gives Heliox the field (an ANSWER line, yes or no). The password check counts wherever the field was read.

When the field is not read, DIAG lines show what the reader's search saw:
- the app's windows;
- the answer when the active window is asked for its focused element (Collection.GetMatches);
- a walk of the window without the reader's budget: how many objects, where the focused one is, the roles and
  states on the way, and the calls that failed.

Run in an X session with a session bus, AT-SPI and a window manager. CI-only (``tests/platform_real``).
Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux_atspi_check import read_until  # noqa: E402
from toolkit_check import diag_focused, diag_reader, find_window, type_keys  # noqa: E402

HERE = Path(__file__).resolve().parent
TITLE = "Heliox field fixture"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


SETTING = ["org.gnome.desktop.interface", "toolkit-accessibility"]


def read_toolkit_accessibility() -> str:
    """The session's toolkit-accessibility value as gsettings prints it ("true", "false"), written down before any
    change so it can be put back."""
    done = subprocess.run(["gsettings", "get", *SETTING], capture_output=True, text=True, timeout=10, check=False)
    return done.stdout.strip()


def toolkit_accessibility(value: str | None) -> None:
    """Set the session's toolkit-accessibility switch to ``value`` ("true", "false"); None resets it."""
    args = ["gsettings", "reset", *SETTING] if value is None else ["gsettings", "set", *SETTING, value]
    subprocess.run(args, capture_output=True, timeout=10, check=False)


def chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def launch(target: str, workdir: Path) -> subprocess.Popen | None:
    """Start the fixture for ``target`` in its own process group; None when its program is not on the runner."""
    page = workdir / "field_fixture.html"
    env = {key: value for key, value in os.environ.items() if key != "NO_AT_BRIDGE"}
    kind, *parts = target.split("-")
    if "env" in parts:
        env["ACCESSIBILITY_ENABLED"] = "1"
    if kind == "chrome":
        program = chrome()
        if program is None:
            return None
        args = [
            program,
            f"--user-data-dir={workdir / 'chrome-profile'}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=Translate",
            "--new-window",
        ]
        if "forced" in parts:
            args.append("--force-renderer-accessibility")
        args.append(page.as_uri())
    elif kind == "electron":
        electron = os.environ.get("HELIOX_ELECTRON") or shutil.which("electron")
        if not electron:
            return None
        env["HELIOX_FIXTURE_PAGE"] = str(page)
        # Ubuntu 24.04 restricts unprivileged user namespaces, so Electron's sandbox cannot start on the runner
        # without a SUID helper; --no-sandbox changes nothing about accessibility.
        args = [electron, str(HERE / "electron_fixture.js"), "--no-sandbox"]
        if "forced" in parts:
            args.append("--force-renderer-accessibility")
    elif kind == "qt":
        args = [sys.executable, str(HERE / "qt_fixture.py")]
        if "forced" in parts:
            env["QT_LINUX_ACCESSIBILITY_ALWAYS_ON"] = "1"
    else:
        raise ValueError(target)
    log = (workdir / f"{target}.log").open("w", encoding="utf-8")
    return subprocess.Popen(args, stdout=log, stderr=log, env=env, start_new_session=True)


def end(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def run(target: str) -> None:
    from pilot.live import focused_text_linux as L

    parts = target.split("-")[1:]
    must_read = "switch" in parts or target == "qt-forced"
    toolkit_accessibility("true" if "switch" in parts else "false")
    workdir = Path(tempfile.mkdtemp(prefix="heliox-fields-", dir=Path.home()))
    shutil.copyfile(HERE / "field_fixture.html", workdir / "field_fixture.html")
    process = None
    try:
        process = launch(target, workdir)
        if process is None:
            print(f"ANSWER {target}: not run, its program is not on this runner", flush=True)
            return
        window = find_window(TITLE, timeout=60.0)
        if not check(f"{target}/window-shown", bool(window), "" if window else f"exit {process.poll()}"):
            log = (workdir / f"{target}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            print(f"DIAG {target} output:\n  " + "\n  ".join(log), flush=True)
            return
        if "late" in parts:
            # The desktop switch turned on after the app started: does a running app join the bus without a restart?
            toolkit_accessibility("true")
        subprocess.run(["xdotool", "windowactivate", "--sync", window], timeout=10, check=False)
        time.sleep(3.0 if "late" in parts else 2.0)  # the page (or Qt's bridge) settles; a browser loads the file
        print(f"DIAG {target} toolkit accessibility on: {L.toolkit_accessibility_enabled()!r}", flush=True)
        word = f"hello {target}"
        type_keys(word, False)
        typed = read_until(lambda r: (r.text or "").endswith(word), timeout=10.0)
        seen = typed is not None and (typed.text or "").endswith(word)
        diag_focused(f"{target} text field")
        if must_read:
            check(f"{target}/reads-typed-text", seen, repr(typed))
        else:
            print(f"ANSWER {target}: Heliox reads the field with this setup: {'yes' if seen else 'no'}", flush=True)
        if not seen:
            diag_reader(target)
            diag_search(target, process.pid)
            return
        # Only after a first successful read: the password field must be refused, and nothing typed may leak.
        subprocess.run(["xdotool", "key", "Tab"], timeout=10, check=False)
        time.sleep(0.6)
        type_keys("s3cret", False)
        diag_focused(f"{target} password field")
        reads = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            reads.append(L.read_focused())
            time.sleep(0.3)
        leaked = [r for r in reads if r is not None and "s3cret" in (r.text or "")]
        refused = [r for r in reads if r is not None and r.password]
        check(
            f"{target}/password-refused",
            not leaked and bool(refused),
            f"{len(refused)} of {len(reads)} reads refused, leaked {leaked[:1]!r}",
        )
    finally:
        if process is not None:
            end(process)
        toolkit_accessibility(ORIGINAL.get("value") or None)
        shutil.rmtree(workdir, ignore_errors=True)
        time.sleep(1.0)


def diag_search(prefix: str, root_pid: int) -> None:
    """DIAG: what the reader's search for the focused element meets in the fixture's active window, the window of an
    application whose process is the fixture or one below it (Chrome's and Electron's windows belong to a child)."""
    from pilot.live import focused_text_linux as L

    with L.SHARED.lock:
        client = L.SHARED.get()
        if client is None:
            return
        try:
            apps = client._children((L.REGISTRY, L.ROOT_PATH))
        except Exception as exc:
            print(f"DIAG {prefix} search: applications: {type(exc).__name__}: {exc}", flush=True)
            return
        family = process_family(root_pid)
        mine = [app for app in apps if client.pid(app[0]) in family]
        print(f"DIAG {prefix} search: the app is on the accessibility bus: {'yes' if mine else 'no'}", flush=True)
        for app in mine:
            try:
                windows = client._children(app)
            except Exception as exc:
                print(f"DIAG {prefix} search: windows of {app}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for window in windows:
                try:
                    if L.STATE_ACTIVE not in client._state(window):
                        continue
                except Exception:
                    continue
                started = time.monotonic()
                found = client.focused_in(window)
                took = time.monotonic() - started
                print(
                    f"DIAG {prefix} search: asked {window} for its focused element: {found!r} in {took:.2f} s",
                    flush=True,
                )
                walk(prefix, client, window)


def walk(prefix: str, client: Any, window: tuple[str, str]) -> None:
    """Walk ``window`` without the reader's budget and without skipping objects lacking SHOWING, and say where the
    focused object is and whether the reader's own walk (SHOWING only, MANAGES_DESCENDANTS not entered, at most
    SEARCH_BUDGET objects) would reach it."""
    from pilot.live import focused_text_linux as L

    started = time.monotonic()
    visited = failed = reader_order = 0
    focused_at = None
    stack: list[tuple[tuple[str, str], tuple, bool]] = [(window, (), True)]
    while stack and visited < 5000 and time.monotonic() - started < 30.0:
        ref, path, reader_reaches = stack.pop()
        visited += 1
        reader_order += 1 if reader_reaches else 0
        try:
            state = client._state(ref)
            step = (client._role(ref), client._role_number(ref), sorted(state))
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"DIAG {prefix} walk: {ref} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        if L.STATE_FOCUSED in state:
            focused_at = (ref, (*path, step), reader_order if reader_reaches else None)
            break
        reaches = reader_reaches and L.STATE_MANAGES_DESCENDANTS not in state
        reaches = reaches and (ref == window or L.STATE_SHOWING in state)
        try:
            children = client._children(ref)
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"DIAG {prefix} walk: children of {ref} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        stack.extend((child, (*path, step), reaches) for child in reversed(children))
    took = time.monotonic() - started
    print(
        f"DIAG {prefix} walk of {window}: {visited} objects in {took:.1f} s, {failed} calls failed; "
        f"the reader's budget is {L.SEARCH_BUDGET} objects",
        flush=True,
    )
    if focused_at is None:
        print(f"DIAG {prefix} walk: no focused object found", flush=True)
        return
    ref, path, order = focused_at
    where = f"the reader's walk reaches it as object {order}" if order else "the reader's walk never enters it"
    print(f"DIAG {prefix} walk: focused object {ref} at depth {len(path) - 1}; {where}", flush=True)
    for depth, step in enumerate(path):
        print(f"DIAG {prefix} walk:   {depth}: {step}", flush=True)


def process_family(root: int) -> set[int]:
    """``root`` and the processes below it, four generations deep."""
    parents: dict[int, int] = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            line = next(line for line in status.read_text().splitlines() if line.startswith("PPid:"))
            parents[int(status.parent.name)] = int(line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
    family = {root}
    for _ in range(4):
        family |= {pid for pid, ppid in parents.items() if ppid in family}
    return family


ORIGINAL: dict[str, str] = {}


def main(argv: list[str]) -> int:
    ORIGINAL["value"] = read_toolkit_accessibility()
    print(f"toolkit-accessibility before the check: {ORIGINAL['value'] or 'unset'}", flush=True)
    try:
        for target in argv:
            try:
                run(target)
            except Exception as exc:  # one target failing must not stop the others
                check(f"{target}/ran", False, f"{type(exc).__name__}: {exc}")
    finally:
        toolkit_accessibility(ORIGINAL["value"] or None)
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("changes the toolkit-accessibility setting and types into Chrome, Electron and Qt windows it opens")
    sys.exit(main(sys.argv[1:]))

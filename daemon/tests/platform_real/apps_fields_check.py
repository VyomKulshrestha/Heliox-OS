"""Fields where people actually type: a browser, an Electron app and a Qt app, read by Heliox's Linux reader (X11).

    python apps_fields_check.py chrome-default chrome-switch chrome-forced electron-default electron-forced \\
        qt-default qt-forced

Each target shows a text field and a password field (field_fixture.html in the browser and Electron, qt_fixture.py
for Qt). For each:

- the text typed into the text field must be read back (``<target>/reads-typed-text``);
- only after that first successful read, every read for five seconds after typing into the password field must be
  refused and none may carry the typed text (``<target>/password-refused``);
- DIAG lines dump, for the field and its two parents, the role name, numeric role, states and attributes, and
  whether the session says toolkit accessibility is on (org.a11y.Status IsEnabled).

The variants answer "does a person's default setup give Heliox the field":
- ``*-default``: as a person has it (toolkit accessibility off, no flags). The read is an ANSWER line, yes or no,
  not a pass or a fail; the password check still counts when the field was read;
- ``chrome-switch``: toolkit accessibility turned on, the switch Live asks the person for;
- ``*-forced``: the toolkit's own switch (Chrome and Electron ``--force-renderer-accessibility``, Qt
  ``QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1``). These must read the field.

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
    kind, _, variant = target.partition("-")
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
        if variant == "forced":
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
        if variant == "forced":
            args.append("--force-renderer-accessibility")
    elif kind == "qt":
        args = [sys.executable, str(HERE / "qt_fixture.py")]
        if variant == "forced":
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

    variant = target.partition("-")[2]
    toolkit_accessibility("true" if variant == "switch" else "false")
    workdir = Path(tempfile.mkdtemp(prefix="heliox-fields-", dir=Path.home()))
    shutil.copyfile(HERE / "field_fixture.html", workdir / "field_fixture.html")
    process = launch(target, workdir)
    if process is None:
        print(f"ANSWER {target}: not run, its program is not on this runner", flush=True)
        return
    try:
        window = find_window(TITLE, timeout=60.0)
        if not check(f"{target}/window-shown", bool(window), "" if window else f"exit {process.poll()}"):
            log = (workdir / f"{target}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            print(f"DIAG {target} output:\n  " + "\n  ".join(log), flush=True)
            return
        subprocess.run(["xdotool", "windowactivate", "--sync", window], timeout=10, check=False)
        time.sleep(2.0)  # the page (or Qt's bridge) settles; a browser loads the file
        print(f"DIAG {target} toolkit accessibility on: {L.toolkit_accessibility_enabled()!r}", flush=True)
        word = f"hello {target}"
        type_keys(word, False)
        typed = read_until(lambda r: (r.text or "").endswith(word), timeout=10.0)
        seen = typed is not None and (typed.text or "").endswith(word)
        diag_focused(f"{target} text field")
        if variant == "default":
            print(
                f"ANSWER {target}: Heliox reads the field with the default setup: {'yes' if seen else 'no'}", flush=True
            )
        else:
            check(f"{target}/reads-typed-text", seen, repr(typed))
        if not seen:
            diag_reader(target)
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
        end(process)
        toolkit_accessibility(ORIGINAL.get("value") or None)
        shutil.rmtree(workdir, ignore_errors=True)
        time.sleep(1.0)


ORIGINAL: dict[str, str] = {}


def main(argv: list[str]) -> int:
    ORIGINAL["value"] = read_toolkit_accessibility()
    print(f"toolkit-accessibility before the check: {ORIGINAL['value'] or 'unset'}", flush=True)
    try:
        for target in argv:
            run(target)
    finally:
        toolkit_accessibility(ORIGINAL["value"] or None)
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("changes the toolkit-accessibility setting and types into Chrome, Electron and Qt windows it opens")
    sys.exit(main(sys.argv[1:]))

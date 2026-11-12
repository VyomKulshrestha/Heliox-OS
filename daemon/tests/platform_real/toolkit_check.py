"""The Linux reader against toolkits other than GTK 3, under Xvfb with a session bus and AT-SPI, for CI.

    python toolkit_check.py gtk4 --python /usr/bin/python3   # gtk4_fixture.py: typed text read, password refused
    python toolkit_check.py firefox                          # a Firefox textarea, with toolkit accessibility on

Keys are typed with xdotool into the focused window, and ``focused_text_linux.read_focused`` must read them back.
Firefox is told to expose itself through ``GNOME_ACCESSIBILITY=1`` and the session's toolkit-accessibility
setting, the same switch Live asks the person to turn on.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux_atspi_check import read_until  # noqa: E402

HERE = Path(__file__).resolve().parent
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def focus_and_type(title: str, text: str) -> None:
    found = subprocess.run(["xdotool", "search", "--sync", "--name", title], capture_output=True, text=True, timeout=30)
    window = (found.stdout.split() or [""])[0]
    if window:
        subprocess.run(["xdotool", "windowactivate", "--sync", window], timeout=10, check=False)
        subprocess.run(["xdotool", "windowfocus", "--sync", window], timeout=10, check=False)
    subprocess.run(["xdotool", "type", "--delay", "30", text], timeout=60, check=False)
    time.sleep(0.6)


def gtk4(system_python: str) -> None:
    env = {key: value for key, value in os.environ.items() if key != "NO_AT_BRIDGE"}
    fixture = subprocess.Popen(
        [system_python, str(HERE / "gtk4_fixture.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert fixture.stdin is not None and fixture.stdout is not None
    try:
        if not check("gtk4/fixture-started", fixture.stdout.readline().startswith("ready")):
            return
        focus_and_type("Heliox GTK 4 fixture", "hello gtk four")
        typed = read_until(lambda r: r.text == "hello gtk four")
        check("gtk4/reads-typed-text", typed is not None and typed.text == "hello gtk four", repr(typed))
        fixture.stdin.write("focus password\n")
        fixture.stdin.flush()
        time.sleep(0.8)
        subprocess.run(["xdotool", "type", "--delay", "30", "s3cret"], timeout=30, check=False)
        secret = read_until(lambda r: r.password)
        check("gtk4/password-refused", secret is not None and secret.password and secret.text == "", repr(secret))
    finally:
        fixture.kill()


PAGE = "data:text/html,<title>Heliox Firefox fixture</title><textarea autofocus aria-label=message></textarea>"


def firefox() -> None:
    program = shutil.which("firefox")
    if not check("firefox/installed-on-the-runner", program is not None, str(program)):
        return
    subprocess.run(
        ["gsettings", "set", "org.gnome.desktop.interface", "toolkit-accessibility", "true"],
        capture_output=True,
        check=False,
    )
    profile = Path(tempfile.mkdtemp(prefix="heliox-firefox-"))
    (profile / "user.js").write_text(
        'user_pref("accessibility.force_disabled", -1);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n',
        encoding="utf-8",
    )
    env = {**os.environ, "GNOME_ACCESSIBILITY": "1", "MOZ_ENABLE_WAYLAND": "0"}
    browser = subprocess.Popen(
        [program, "--new-instance", "--profile", str(profile), PAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        time.sleep(12.0)
        focus_and_type("Heliox Firefox fixture", "hello firefox")
        typed = read_until(lambda r: r.text.endswith("hello firefox"), timeout=30.0)
        check("firefox/reads-typed-text", typed is not None and typed.text.endswith("hello firefox"), repr(typed))
    finally:
        browser.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("toolkit", choices=["gtk4", "firefox"])
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    if args.toolkit == "gtk4":
        gtk4(args.python)
    else:
        firefox()
    return failures


if __name__ == "__main__":
    sys.exit(main())

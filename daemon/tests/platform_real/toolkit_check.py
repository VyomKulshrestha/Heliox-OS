"""The Linux reader against toolkits other than GTK 3, with a session bus and AT-SPI, for CI.

    python toolkit_check.py gtk4 --python /usr/bin/python3             # under Xvfb: keys typed with xdotool
    python toolkit_check.py gtk4 --python /usr/bin/python3 --wayland   # under sway: keys typed with wtype
    python toolkit_check.py firefox                                    # a Firefox textarea, toolkit a11y on

``focused_text_linux.read_focused`` must read the typed text back, and must refuse the password entry. For the
password entry the check prints DIAG lines with what AT-SPI says about the focused object and its two parents
(role name, numeric role, states, object attributes, interfaces), so each run records how the toolkit marks it.
GTK 4 marks a password entry only in its numeric role (CI on 008f1a0 read one as a "text box").

Firefox is told to expose itself through ``GNOME_ACCESSIBILITY=1`` and the session's toolkit-accessibility setting,
the same switch Live asks the person to turn on; its own output is printed when its window does not appear.

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


def find_window(title: str, timeout: float = 30.0) -> str:
    try:
        found = subprocess.run(
            ["xdotool", "search", "--sync", "--name", title], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return ""
    return (found.stdout.split() or [""])[0]


def type_keys(text: str, wayland: bool) -> None:
    if wayland:
        subprocess.run(["wtype", "-d", "30", text], timeout=60, check=False)
    else:
        subprocess.run(["xdotool", "type", "--delay", "30", text], timeout=60, check=False)
    time.sleep(0.6)


def focus_window(title: str, wayland: bool) -> bool:
    if wayland:
        return True  # sway focuses a new window; the fixture also grabs focus for its entry
    window = find_window(title)
    if window:
        subprocess.run(["xdotool", "windowactivate", "--sync", window], timeout=10, check=False)
        subprocess.run(["xdotool", "windowfocus", "--sync", window], timeout=10, check=False)
    return bool(window)


def diag_focused(prefix: str) -> None:
    """DIAG lines: what AT-SPI says about the focused object and its two parents."""
    from pilot.live import focused_text_linux as L

    with L.SHARED.lock:
        client = L.SHARED.get()
        if client is None:
            print(f"DIAG {prefix}: no AT-SPI client: {L.SHARED.error}")
            return
        ref = client.focused()
        for level in range(3):
            if ref is None:
                break
            facts = {"ref": f"{ref[0]}{ref[1]}"}
            for key, call in (
                ("role_name", lambda r: client._role(r)),
                ("role_number", lambda r: client._role_number(r)),
                ("states", lambda r: sorted(client._state(r))),
                ("attributes", lambda r: dict(client._t.call(r[0], r[1], L.ACCESSIBLE, "GetAttributes")[0])),
                ("interfaces", lambda r: list(client._t.call(r[0], r[1], L.ACCESSIBLE, "GetInterfaces")[0])),
                ("name", lambda r: client._get(r, L.ACCESSIBLE, "Name")),
            ):
                try:
                    facts[key] = call(ref)
                except Exception as exc:
                    facts[key] = f"{type(exc).__name__}: {exc}"
            print(f"DIAG {prefix} level {level}: {facts}", flush=True)
            try:
                bus, path = client._get(ref, L.ACCESSIBLE, "Parent")
                ref = (str(bus), str(path))
            except Exception:
                break


def ask_fixture(fixture: subprocess.Popen, command: str, answer: str, timeout: float = 5.0) -> str:
    """Send a command to the fixture and return its line that starts with ``answer`` ("" when none came)."""
    assert fixture.stdin is not None and fixture.stdout is not None
    fixture.stdin.write(command + "\n")
    fixture.stdin.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = fixture.stdout.readline()
        if not line:
            return ""
        if line.startswith(answer):
            return line.strip()
    return ""


def diag_reader(prefix: str) -> None:
    """DIAG lines for a read that saw nothing: the reader's foreground, its focused object, and every application
    on the accessibility bus with its windows' role, numeric role, states and title, so a run shows whether the
    keys or the reader is where it stops."""
    from pilot.live import focused_text_linux as L

    try:
        print(f"DIAG {prefix} reader foreground: {L.foreground_window()!r}", flush=True)
    except Exception as exc:
        print(f"DIAG {prefix} reader foreground: {type(exc).__name__}: {exc}", flush=True)
    with L.SHARED.lock:
        client = L.SHARED.get()
        if client is None:
            print(f"DIAG {prefix}: no AT-SPI client: {L.SHARED.error}", flush=True)
            return
        try:
            print(f"DIAG {prefix} reader focused object: {client.focused()!r}", flush=True)
        except Exception as exc:
            print(f"DIAG {prefix} reader focused object: {type(exc).__name__}: {exc}", flush=True)
        try:
            apps = client._children((L.REGISTRY, L.ROOT_PATH))
        except Exception as exc:
            print(f"DIAG {prefix} applications: {type(exc).__name__}: {exc}", flush=True)
            return
        for app in apps[:12]:
            try:
                name = client._get(app, L.ACCESSIBLE, "Name")
                windows = []
                for window in client._children(app)[:6]:
                    windows.append(
                        (
                            client._role(window),
                            client._role_number(window),
                            sorted(client._state(window)),
                            client._get(window, L.ACCESSIBLE, "Name"),
                        )
                    )
                print(f"DIAG {prefix} application {name!r} {app}: windows {windows}", flush=True)
            except Exception as exc:
                print(f"DIAG {prefix} application {app}: {type(exc).__name__}: {exc}", flush=True)


def fixture_focus_within(fixture: subprocess.Popen, field: str, wayland: bool, timeout: float = 10.0) -> str:
    """Wait until the fixture's window is active and ``field`` ("entry" or "password") has focus; the fixture's last
    answer. Under sway the window is asked forward with swaymsg on the way."""
    answer = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        answer = ask_fixture(fixture, "text", "text ")
        if f"{field}_focus=True" in answer and "window_active=True" in answer:
            return answer
        if wayland and shutil.which("swaymsg"):
            subprocess.run(
                ["swaymsg", '[title="Heliox GTK 4 fixture"] focus'], capture_output=True, timeout=5, check=False
            )
        time.sleep(0.5)
    return answer


def gtk4(system_python: str, wayland: bool) -> None:
    prefix = "gtk4-wayland" if wayland else "gtk4"
    env = {key: value for key, value in os.environ.items() if key != "NO_AT_BRIDGE"}
    keeper = None
    if wayland and shutil.which("wtype"):
        # A headless sway seat has no keyboard until a virtual one is open, and without it no window gets keyboard
        # focus (CI on e8bf67c: the GTK 4 window was never active, and the first typed letter was lost). The GTK 3
        # check holds one open for its own run only; this one holds its own.
        keeper = subprocess.Popen(["wtype", "-s", "600000"])
        time.sleep(1.0)
    fixture = subprocess.Popen(
        [system_python, str(HERE / "gtk4_fixture.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert fixture.stdin is not None and fixture.stdout is not None
    try:
        if not check(f"{prefix}/fixture-started", fixture.stdout.readline().startswith("ready")):
            return
        focus_window("Heliox GTK 4 fixture", wayland)
        state = fixture_focus_within(fixture, "entry", wayland)
        if not check(
            f"{prefix}/window-active-with-the-entry-focused",
            "entry_focus=True" in state and "window_active=True" in state,
            state or "no answer",
        ):
            diag_reader(prefix)
            return
        type_keys("hello gtk four", wayland)
        typed = read_until(lambda r: r.text == "hello gtk four")
        check(f"{prefix}/reads-typed-text", typed is not None and typed.text == "hello gtk four", repr(typed))
        reached = ask_fixture(fixture, "text", "text ")
        print(f"DIAG {prefix} what the keys reached: {reached or 'no answer'}", flush=True)
        if typed is None or typed.text != "hello gtk four":
            diag_reader(prefix)
        fixture.stdin.write("focus password\n")
        fixture.stdin.flush()
        time.sleep(0.8)
        state = fixture_focus_within(fixture, "password", wayland)
        print(f"DIAG {prefix} before typing the password: {state or 'no answer'}", flush=True)
        type_keys("s3cret", wayland)
        diag_focused(f"{prefix} password entry")
        # Read for a while: every read in that time must be refused, none may carry the typed text.
        reads = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            from pilot.live import focused_text_linux as L

            reads.append(L.read_focused())
            time.sleep(0.3)
        leaked = [r for r in reads if r is not None and "s3cret" in (r.text or "")]
        refused = [r for r in reads if r is not None and r.password]
        check(
            f"{prefix}/password-refused",
            not leaked and bool(refused),
            f"{len(refused)} of {len(reads)} reads refused, leaked {leaked[:1]!r}",
        )
    finally:
        fixture.kill()
        if keeper is not None:
            keeper.kill()


PAGE = (
    "<!doctype html><title>Heliox Firefox fixture</title><textarea autofocus aria-label=message></textarea>"
    "<input type=password aria-label=password><input type=password id=f3>\n"
)


def firefox() -> None:
    program = shutil.which("firefox")
    if not check("firefox/installed-on-the-runner", program is not None, str(program)):
        return
    subprocess.run(
        ["gsettings", "set", "org.gnome.desktop.interface", "toolkit-accessibility", "true"],
        capture_output=True,
        check=False,
    )
    # In the home folder, not /tmp: a snap Firefox (Ubuntu's) sees a private /tmp of its own. The page is a file,
    # as Firefox refuses a top-level data: URL given on the command line (CI on 8d513fd: no window ever had the
    # page's title, and Firefox printed nothing).
    profile = Path(tempfile.mkdtemp(prefix="heliox-firefox-", dir=Path.home()))
    page = profile / "fixture.html"
    page.write_text(PAGE, encoding="utf-8")
    print(f"DIAG firefox program {program} -> {Path(program).resolve()}", flush=True)
    (profile / "user.js").write_text(
        'user_pref("accessibility.force_disabled", -1);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n',
        encoding="utf-8",
    )
    log = profile / "firefox.log"
    env = {**os.environ, "GNOME_ACCESSIBILITY": "1", "MOZ_ENABLE_WAYLAND": "0"}
    with log.open("w", encoding="utf-8") as out:
        browser = subprocess.Popen(
            [program, "--new-instance", "--profile", str(profile), page.as_uri()], stdout=out, stderr=out, env=env
        )
    try:
        window = find_window("Heliox Firefox fixture", timeout=60.0)
        if not check("firefox/window-shown", bool(window), "" if window else f"exit {browser.poll()}"):
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            print("DIAG firefox output:\n  " + "\n  ".join(tail))
            print(f"DIAG windows now: {run_quiet(['xdotool', 'search', '--name', '.'])}")
            return
        focus_window("Heliox Firefox fixture", False)
        type_keys("hello firefox", False)
        typed = read_until(lambda r: r.text.endswith("hello firefox"), timeout=30.0)
        check("firefox/reads-typed-text", typed is not None and typed.text.endswith("hello firefox"), repr(typed))
        # Each password field: Tab to it, type, and every read for five seconds must be refused. The second has no
        # label and a neutral id, so the refusal cannot depend on what the page calls it.
        from pilot.live import focused_text_linux as L

        for name, secret in (("password-refused", "s3cret"), ("unnamed-password-refused", "n3utral")):
            subprocess.run(["xdotool", "key", "Tab"], timeout=10, check=False)
            time.sleep(0.5)
            type_keys(secret, False)
            diag_focused(f"firefox {name} field")
            reads = []
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                reads.append(L.read_focused())
                time.sleep(0.3)
            leaked = [r for r in reads if r is not None and secret in (r.text or "")]
            refused = [r for r in reads if r is not None and r.password]
            check(
                f"firefox/{name}",
                not leaked and bool(refused),
                f"{len(refused)} of {len(reads)} reads refused, leaked {leaked[:1]!r}",
            )
    finally:
        browser.kill()
        shutil.rmtree(profile, ignore_errors=True)


def run_quiet(args: list[str]) -> str:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        return "timed out"
    names = []
    for window in done.stdout.split()[:15]:
        title = subprocess.run(["xdotool", "getwindowname", window], capture_output=True, text=True, check=False)
        names.append(title.stdout.strip())
    return repr(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("toolkit", choices=["gtk4", "firefox"])
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--wayland", action="store_true")
    args = parser.parse_args()
    if args.toolkit == "gtk4":
        gtk4(args.python, args.wayland)
    else:
        firefox()
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens windows (and Firefox) on this desktop and types into them")
    sys.exit(main())

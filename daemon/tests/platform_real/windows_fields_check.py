"""Fields where people type on Windows: Edge and Chrome, read by Heliox's Windows reader (UI Automation).

    python windows_fields_check.py edge-default chrome-default chrome-forced

Each target opens field_fixture.html (a text area and a password input) in a new browser profile of its own. For
each:

- the window is brought to the front, and the route that worked is printed (``<target>/in-front``);
- the text typed into the text area must be read back through ``WindowsFocusedTextReader``, the reader Live uses
  (``<target>/reads-typed-text``);
- only after that first read, every read for five seconds after typing into the password input must be refused,
  and none may carry the typed text (``<target>/password-refused``). An ANSWER line names the mark UI Automation
  gave: IsPassword on the focused element;
- DIAG lines dump the focused element and its two parents: control type, class, framework, automation id, name,
  IsPassword, the patterns it supports and the value pattern's IsReadOnly (never a value or text).

``*-default`` is the browser as a person has it: its read is an ANSWER line, yes or no; the password check still
counts when the field was read. ``chrome-forced`` adds ``--force-renderer-accessibility`` and must read the field.

Typing goes through pyautogui (SendInput). The check changes no setting on the machine: each browser runs with a
profile in a temporary folder, and its process tree is ended and the folder removed after each target. One target
failing does not stop the others. CI-only (``ci_only.require_runner``).
Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
TITLE = "Heliox field fixture"
BROWSERS = {
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ],
}
failures = 0

# The focused element and its two parents, as UI Automation describes them; never their value or text.
DIAG_SCRIPT = r"""
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$el = [System.Windows.Automation.AutomationElement]::FocusedElement
$levels = @()
for ($i = 0; $i -lt 3 -and $el -ne $null; $i++) {
  $c = $el.Current
  $vp = $null
  $ro = $null
  if ($el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$vp)) { $ro = $vp.Current.IsReadOnly }
  $levels += @{
    level = $i; type = $c.ControlType.ProgrammaticName; class = $c.ClassName; framework = $c.FrameworkId;
    automation_id = $c.AutomationId; name = $c.Name; is_password = $c.IsPassword; pid = $c.ProcessId;
    patterns = @($el.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName }); value_read_only = $ro
  }
  $el = $walker.GetParent($el)
}
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
ConvertTo-Json -Compress -Depth 4 @($levels)
"""


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def diag_focused(label: str) -> list[dict]:
    """The DIAG dump. The script goes in as -EncodedCommand: given on stdin ("-Command -"), PowerShell runs it line
    by line, the multi-line table broke, and the first run on the runner printed nothing, which the check then
    reported as "IsPassword False" (CI on 2aeeab2). An empty answer is now said as such."""
    encoded = base64.b64encode(DIAG_SCRIPT.encode("utf-16-le")).decode("ascii")
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=30,
        check=False,
    )
    output = done.stdout.decode("utf-8", errors="replace").strip()
    try:
        levels = json.loads(output) if output else None
    except ValueError:
        levels = None
    if not levels:
        print(f"DIAG {label}: UI Automation gave no answer ({done.stderr.decode('utf-8', 'replace').strip()[:300]!r})")
        return []
    levels = levels if isinstance(levels, list) else [levels]
    for level in levels:
        print(f"DIAG {label} level {level.get('level')}: {level}", flush=True)
    return levels


user32 = ctypes.WinDLL("user32", use_last_error=True) if os.name == "nt" else None


def windows_titled(text: str) -> list[tuple[int, int, str]]:
    """(hwnd, pid, title) of the visible top-level windows whose title contains ``text``."""
    found: list[tuple[int, int, str]] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if text in buffer.value:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            found.append((int(hwnd), int(pid.value), buffer.value))
        return True

    user32.EnumWindows(proc(visit), 0)
    return found


def foreground_title() -> str:
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def bring_to_front(hwnd: int) -> str:
    """The route that put ``hwnd`` in front, or "". Windows lets a background process take the foreground only in
    some cases; pressing and releasing Alt first is the usual way a tool earns it."""
    if user32.GetForegroundWindow() == hwnd:
        return "already"
    vk_menu, keyup = 0x12, 0x0002

    def alt_then_set() -> None:
        user32.keybd_event(vk_menu, 0, 0, 0)
        user32.keybd_event(vk_menu, 0, keyup, 0)
        user32.SetForegroundWindow(hwnd)

    def attach_then_set() -> None:
        foreground = user32.GetForegroundWindow()
        mine = ctypes.windll.kernel32.GetCurrentThreadId()
        theirs = user32.GetWindowThreadProcessId(foreground, None)
        user32.AttachThreadInput(mine, theirs, True)
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            user32.AttachThreadInput(mine, theirs, False)

    routes = [
        ("set-foreground", lambda: user32.SetForegroundWindow(hwnd)),
        ("alt-then-set", alt_then_set),
        ("attach-thread-input", attach_then_set),
    ]
    for name, route in routes:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        route()
        time.sleep(0.8)
        if user32.GetForegroundWindow() == hwnd:
            return name
    print(f"DIAG front: still in front: {foreground_title()!r}", flush=True)
    return ""


def end_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=30, check=False)


def run(target: str, reader) -> None:
    import pyautogui

    kind, _, variant = target.partition("-")
    program = next((path for path in BROWSERS[kind] if Path(path).exists()), None)
    if program is None:
        print(f"ANSWER {target}: not run, {kind} is not on this runner", flush=True)
        return
    workdir = Path(tempfile.mkdtemp(prefix="heliox-fields-"))
    page = workdir / "field_fixture.html"
    shutil.copyfile(HERE / "field_fixture.html", page)
    args = [
        program,
        f"--user-data-dir={workdir / 'profile'}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
    ]
    if variant == "forced":
        args.append("--force-renderer-accessibility")
    process = subprocess.Popen([*args, page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60.0
        windows: list[tuple[int, int, str]] = []
        while not windows and time.monotonic() < deadline:
            windows = windows_titled(TITLE)
            time.sleep(0.5)
        if not check(f"{target}/window-shown", bool(windows), windows[0][2] if windows else f"exit {process.poll()}"):
            print(f"DIAG {target}: windows now {[w[2] for w in windows_titled('')][:15]}", flush=True)
            return
        hwnd = windows[0][0]
        route = bring_to_front(hwnd)
        check(f"{target}/in-front", bool(route), f"route {route or 'none'}")
        time.sleep(2.0)  # the page loads and its text area takes focus (autofocus)
        word = f"hello {target}"
        pyautogui.write(word, interval=0.03)
        time.sleep(0.6)
        seen = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            seen = reader.read()
            if seen is not None and (seen.text or "").endswith(word):
                break
            time.sleep(0.3)
        ok = seen is not None and (seen.text or "").endswith(word)
        diag_focused(f"{target} text field")
        if variant == "default":
            print(f"ANSWER {target}: Heliox reads the field with this setup: {'yes' if ok else 'no'}", flush=True)
        else:
            check(f"{target}/reads-typed-text", ok, repr(seen))
        if not ok:
            print(f"DIAG {target}: last read {seen!r}", flush=True)
            return
        # Only after a first successful read: each password input must be refused, and nothing typed may leak.
        # The second has a neutral id, no label and no placeholder: the refusal must not depend on what the page
        # calls the field.
        password_check(target, "password-refused", "s3cret", reader)
        password_check(target, "unnamed-password-refused", "n3utral", reader)
    finally:
        end_tree(process.pid)
        time.sleep(1.5)
        shutil.rmtree(workdir, ignore_errors=True)


def password_check(target: str, name: str, secret: str, reader) -> None:
    """Tab to the next field, type ``secret``, and read for five seconds: every read refused, none carrying it."""
    import pyautogui

    pyautogui.press("tab")
    time.sleep(0.6)
    pyautogui.write(secret, interval=0.03)
    time.sleep(0.6)
    levels = diag_focused(f"{target} {name} field")
    if levels:
        focused = levels[0]
        print(
            f"ANSWER {target}: {name} field: UI Automation IsPassword {focused.get('is_password')!r} on the focused "
            f"element (control {focused.get('type')!r}, automation id {focused.get('automation_id')!r})",
            flush=True,
        )
    reads = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        reads.append(reader.read())
        time.sleep(0.3)
    leaked = [r for r in reads if r is not None and secret in (r.text or "")]
    refused = [r for r in reads if r is not None and r.password]
    # The Windows reader refuses only on UI Automation's IsPassword of the focused element (focused_text._HELPER),
    # so a refused read means UI Automation marked that element a password.
    check(
        f"{target}/{name}",
        not leaked and bool(refused),
        f"{len(refused)} of {len(reads)} reads refused (the reader's mark: IsPassword), leaked {leaked[:1]!r}",
    )


def main(argv: list[str]) -> int:
    # The console of a Windows runner is not UTF-8 (a window title with U+200B stopped the first run, CI on 2aeeab2).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    from pilot.live.focused_text import WindowsFocusedTextReader

    for target in argv:
        reader = _OneLoop(WindowsFocusedTextReader())  # a reader (and its helper process) per target
        try:
            run(target, reader)
        except Exception as exc:  # one target failing must not stop the others
            check(f"{target}/ran", False, f"{type(exc).__name__}: {exc}")
        finally:
            reader.close()
    return failures


class _OneLoop:
    """The reader driven from one event loop for the whole target (its subprocess pipes belong to that loop)."""

    def __init__(self, reader) -> None:
        self._reader = reader
        self._loop = asyncio.new_event_loop()

    def read(self):
        return self._loop.run_until_complete(self._reader.read())

    def close(self) -> None:
        self._loop.run_until_complete(self._reader.close())
        self._loop.close()


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens Edge and Chrome windows and types into them")
    sys.exit(main(sys.argv[1:]))

"""Launching apps and naming them for real, on the macOS runner and on Linux, for the ``platform-readers`` job.

macOS: TextEdit is found among the installed bundles and launched through ``applications.launch_application``
(the planner's "open an app"); its process must appear, and the names Heliox shows come from real bundles
(TextEdit, Safari) and from the running process.

Linux (under Xvfb): a test desktop entry in ``~/.local/share/applications`` starts a small program that writes the
arguments it was given; Heliox launches it by the entry's shown name with a file name containing a space, and the
program must have run with exactly those arguments. The names come from that entry, and from a desktop entry the
runner itself has installed.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import one_loop  # noqa: E402  (next to this script)

from pilot.system import app_names, applications  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.3)
    return False


def running(name: str) -> bool:
    return subprocess.run(["pgrep", "-x", name], capture_output=True, check=False).returncode == 0


def macos() -> None:
    textedit = "/System/Applications/TextEdit.app/Contents/MacOS/TextEdit"
    check("apps/macos-bundle-name-textedit", app_names.bundle_display_name(textedit) == "TextEdit")
    safari = "/Applications/Safari.app/Contents/MacOS/Safari"
    if Path(safari).exists():
        check("apps/macos-bundle-name-safari", app_names.bundle_display_name(safari) == "Safari")
    bundles = applications.mac_application_bundles()
    found = applications.resolve_mac_application("textedit", bundles)
    check("apps/macos-finds-textedit-among-bundles", found == "/System/Applications/TextEdit.app", repr(found))

    subprocess.run(["osascript", "-e", 'quit app "TextEdit"'], capture_output=True, check=False)
    wait_for(lambda: not running("TextEdit"), 5.0)
    try:
        output = one_loop.run(applications.launch_application("TextEdit"))
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
    started = wait_for(lambda: running("TextEdit"))
    check("apps/macos-launch-textedit", started and output.startswith("Launched TextEdit"), output)
    if started:
        app_names.describe.cache_clear()
        label = app_names.describe("TextEdit")
        check("apps/macos-names-the-running-process", label == "TextEdit", repr(label))
    subprocess.run(["osascript", "-e", 'quit app "TextEdit"'], capture_output=True, check=False)


PROBE = """#!/bin/sh
for argument in "$@"; do printf '%s\\n' "$argument"; done > "$HELIOX_PROBE_OUT"
sleep 2
"""


def linux() -> None:
    home = Path.home()
    bin_dir = home / ".local" / "bin"
    apps_dir = home / ".local" / "share" / "applications"
    bin_dir.mkdir(parents=True, exist_ok=True)
    apps_dir.mkdir(parents=True, exist_ok=True)
    probe = bin_dir / "heliox-probe-editor"
    probe.write_text(PROBE, encoding="utf-8")
    probe.chmod(0o755)
    out = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "heliox-probe-args.txt"
    out.unlink(missing_ok=True)
    os.environ["HELIOX_PROBE_OUT"] = str(out)
    entry = apps_dir / "org.heliox.ProbeEditor.desktop"
    entry.write_text(
        f"[Desktop Entry]\nType=Application\nName=Heliox Probe Editor\nExec={probe} --new-window %F\nTerminal=false\n",
        encoding="utf-8",
    )
    try:
        app_names._desktop_entries.cache_clear()
        entries = app_names._desktop_entries()
        check(
            "apps/linux-names-a-desktop-entry",
            app_names.desktop_entry_name("heliox-probe-editor", entries) == "Heliox Probe Editor",
        )
        system = [(stem, text) for stem, text in entries if Path(f"/usr/share/applications/{stem}.desktop").exists()]
        named = 0
        for stem, text in system:
            keys = app_names.parse_desktop_entry(text)
            program = app_names._exec_name(keys.get("Exec", ""))
            if keys.get("Name") and program and keys.get("NoDisplay", "").lower() != "true":
                got = app_names.desktop_entry_name(program, entries)
                print(f"installed entry {stem}: Exec {program!r} named {got!r} (entry Name {keys['Name']!r})")
                named += 1
                if named >= 3:
                    break
        check("apps/linux-installed-entries-found", bool(system), f"{len(system)} in /usr/share/applications")

        arguments = ["/tmp/notes with space.txt", "second.txt"]
        try:
            output = one_loop.run(applications.launch_application("Heliox Probe Editor", arguments))
        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
        wrote = wait_for(out.exists, 10.0)
        seen = out.read_text(encoding="utf-8").splitlines() if wrote else []
        check(
            "apps/linux-launch-by-desktop-entry-with-arguments",
            seen == ["--new-window", *arguments] and "desktop entry org.heliox.ProbeEditor" in output,
            f"{output}; the program got {seen!r}",
        )
    finally:
        entry.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)


def main() -> int:
    if sys.platform == "darwin":
        import mac_front

        mac_front.diag("at the start of apps_check")
        macos()
        mac_front.diag("at the end of apps_check")
    else:
        linux()
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("launches apps (TextEdit on macOS) on this desktop")
    sys.exit(main())

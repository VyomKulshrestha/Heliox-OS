"""What starts by itself on a Windows runner, before setup, after setup and after removal, for ``real-removal``.

    python windows_autostart_check.py save BEFORE    # before setup
    python windows_autostart_check.py left BEFORE    # after setup (or after removal): what is new, and whose

Compared: the scheduled tasks (``schtasks /query``), the Run and RunOnce keys of the current user and of the
machine, and the Startup folders of the user and of all users. Heliox says it creates none of these; anything new
with Heliox's names (heliox, pilot) is a finding and fails the check, anything else new is listed.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Heliox's names: the app, and the daemon's module or package ("pilot.server", "pilot-daemon"); not a bare
# "pilot", which Windows' own Copilot entries contain.
HELIOX = re.compile(r"heliox|pilot[.\-_](server|daemon)", re.IGNORECASE)
RUN_KEYS = (
    r"Software\Microsoft\Windows\CurrentVersion\Run",
    r"Software\Microsoft\Windows\CurrentVersion\RunOnce",
)


def entries() -> set[str]:
    import winreg

    found: set[str] = set()
    done = subprocess.run(["schtasks", "/query", "/fo", "csv", "/nh"], capture_output=True, text=True, check=False)
    for row in csv.reader(io.StringIO(done.stdout)):
        if row:
            found.add(f"task {row[0]}")
    for hive, label in ((winreg.HKEY_CURRENT_USER, "HKCU"), (winreg.HKEY_LOCAL_MACHINE, "HKLM")):
        for key_path in RUN_KEYS:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    index = 0
                    while True:
                        try:
                            name, value, _kind = winreg.EnumValue(key, index)
                        except OSError:
                            break
                        found.add(f"{label}\\{key_path}: {name} = {value}")
                        index += 1
            except OSError:
                continue
    startup = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup",
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "StartUp",
    ]
    for folder in startup:
        if folder.is_dir():
            found.update(f"startup {item}" for item in folder.iterdir())
    return found


def main(argv: list[str]) -> int:
    mode, record = argv[0], Path(argv[1])
    if mode == "save":
        now = entries()
        record.write_text(json.dumps(sorted(now)), encoding="utf-8")
        print(f"{len(now)} autostart entries recorded")
        return 0
    before = set(json.loads(record.read_text(encoding="utf-8")))
    new = sorted(entries() - before)
    heliox = [item for item in new if HELIOX.search(item)]
    print(f"new, not Heliox's: {[item for item in new if item not in heliox]}")
    print(f"CHECK removal/no-autostart-entry-of-heliox: {'FAIL' if heliox else 'PASS'} - {heliox}", flush=True)
    return 1 if heliox else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

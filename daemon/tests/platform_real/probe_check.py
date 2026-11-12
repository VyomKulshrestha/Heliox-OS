"""The first-run probe when the only "python" on PATH is one Heliox must never run, for CI.

    python probe_check.py   # macOS or Windows runner, from the repository root

macOS: the probe runs under ``sh -x`` with PATH limited to /usr/bin:/bin:/usr/sbin:/sbin, where ``python3`` is
Apple's stub (on a Mac without the developer tools it opens their installer). The trace must show
/usr/bin/python3 never run, and the probe must still print its plan. The probe also looks in Homebrew's and
python.org's folders, so on a runner, which has Homebrew's Python, the plan can be "system" with that Python; the
"download" plan for a Mac with nothing installed cannot be shown on a hosted runner.

Windows: the probe runs with PATH holding only a folder named like the Store's (``...\\Microsoft\\WindowsApps``)
with a ``python.exe`` in it, and the system folders PowerShell needs. Neither ``py`` nor any other Python is on
PATH, so the plan must be "download". What runs in its place is a copy of ``whoami.exe``: if the probe ran it as
a Python, ``@@python system`` would name that path.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "tauri-app" / "src-tauri" / "setup"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def versions_env() -> dict[str, str]:
    import json

    data = json.loads((SETUP / "python-versions.json").read_text(encoding="utf-8"))
    return {
        "HELIOX_PYTHON_MIN": data["min"],
        "HELIOX_PYTHON_MAX": data["max"],
        "HELIOX_PYTHON_PRIVATE": data["private"],
        "HELIOX_PYTHON": "auto",
    }


def macos() -> None:
    home = tempfile.mkdtemp(prefix="heliox-probe-home-")
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": home, "HELIOX_SETUP_DIR": f"{home}/.config/heliox-os"}
    env.update(versions_env())
    done = subprocess.run(
        ["/bin/sh", "-x", str(SETUP / "first-run.sh"), "probe"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    shutil.rmtree(home, ignore_errors=True)
    trace = done.stderr.splitlines()
    ran_stub = [line for line in trace if line.lstrip("+ ").startswith("/usr/bin/python3 ")]
    plan = [line for line in done.stdout.splitlines() if line.startswith("@@python")]
    check("probe-macos/never-runs-usr-bin-python3", not ran_stub, repr(ran_stub[:3]))
    check("probe-macos/prints-a-plan", bool(plan) and "/usr/bin/python3" not in plan[0], repr(plan))


def windows() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="heliox-probe-"))
    folder = scratch / "Microsoft" / "WindowsApps"
    folder.mkdir(parents=True)
    system32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    shutil.copyfile(system32 / "whoami.exe", folder / "python.exe")
    path = os.pathsep.join([str(folder), str(system32), str(system32 / "WindowsPowerShell" / "v1.0")])
    env = {key: value for key, value in os.environ.items() if key.upper() not in {"PATH", "PYTHONHOME", "PYTHONPATH"}}
    env.update(versions_env())
    env["PATH"] = path
    env["HELIOX_SETUP_DIR"] = str(scratch / "setup")
    powershell = system32 / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    done = subprocess.run(
        [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SETUP / "first-run.ps1"), "probe"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    plan = [line for line in done.stdout.splitlines() if line.startswith("@@python")]
    check(
        "probe-windows/only-the-store-stub-means-download",
        bool(plan) and plan[0].startswith("@@python download"),
        f"{plan!r}; exit {done.returncode}; {done.stderr.strip()[:200]}",
    )


def main() -> int:
    if sys.platform == "darwin":
        macos()
    elif sys.platform == "win32":
        windows()
    else:
        print("probe_check runs on macOS and Windows only")
    return failures


if __name__ == "__main__":
    sys.exit(main())

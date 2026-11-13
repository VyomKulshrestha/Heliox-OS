"""The uninstaller's end-heliox.ps1, run for real against stand-in process trees on a Windows runner.

    end_heliox_check.py

What the Windows uninstaller runs before it removes Heliox (tauri-app/src-tauri/windows/end-heliox.ps1, from
installer-hooks.nsh) must end the installed app and the daemon of Heliox's own environment, each with its process
tree, and nothing else: a developer's daemon from a source checkout (even one cloned as "Heliox-OS") and a copy of
the app installed elsewhere survive. Stand-ins, all under the runner's temp folder, none of them Heliox:

- the installed app: a copy of cmd.exe named heliox-os.exe in <temp>\\install, running ping (a child);
- another install: the same copy in <temp>\\other-install, which must survive;
- Heliox's daemon: a venv at <temp>\\home\\.config\\heliox-os\\env running ``-m pilot.server``, where pilot.server is a
  stand-in that starts a child and sleeps;
- a developer's daemon: a venv at <temp>\\Heliox-OS\\daemon\\.venv running the same stand-in, which must survive.

Each check prints one CHECK line; every wait is bounded; whatever is left at the end is ended.
CI-only (``ci_only.require_runner``): it runs the uninstall script, which ends processes on the machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "tauri-app" / "src-tauri" / "windows" / "end-heliox.ps1"
STAND_IN = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "time.sleep(600)\n"
)
failures = 0


def check(name: str, ok: bool, detail: str) -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK end-heliox/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    return ok


def tree(pid: int) -> list[int]:
    import psutil

    try:
        process = psutil.Process(pid)
        return [pid, *(child.pid for child in process.children(recursive=True))]
    except psutil.Error:
        return []


def alive(pids: list[int]) -> list[int]:
    import psutil

    return [pid for pid in pids if psutil.pid_exists(pid)]


def daemon_stand_in(venv: Path) -> subprocess.Popen:
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=180)
    package = venv.parent / "stand-in" / "pilot"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "server.py").write_text(STAND_IN, encoding="utf-8")
    return subprocess.Popen([str(venv / "Scripts" / "python.exe"), "-m", "pilot.server"], cwd=package.parent)


def app_stand_in(folder: Path) -> subprocess.Popen:
    folder.mkdir(parents=True, exist_ok=True)
    copy = folder / "heliox-os.exe"
    shutil.copy2(Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "cmd.exe", copy)
    return subprocess.Popen([str(copy), "/c", "ping", "-n", "600", "127.0.0.1"], stdout=subprocess.DEVNULL)


def main() -> int:
    base = Path(os.environ["RUNNER_TEMP"]) / "end-heliox"
    shutil.rmtree(base, ignore_errors=True)
    started: list[subprocess.Popen] = []
    try:
        app = app_stand_in(base / "install")
        other_app = app_stand_in(base / "other-install")
        heliox = daemon_stand_in(base / "home" / ".config" / "heliox-os" / "env")
        developer = daemon_stand_in(base / "Heliox-OS" / "daemon" / ".venv")
        started = [app, other_app, heliox, developer]
        time.sleep(5)  # each stand-in starts its child
        trees = {name: tree(p.pid) for name, p in zip(("app", "other-app", "heliox", "developer"), started)}
        print(f"DIAG the trees before: {trees}", flush=True)
        env = {**os.environ, "HELIOX_APP": str(base / "install" / "heliox-os.exe")}
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and alive(trees["app"] + trees["heliox"]):
            time.sleep(0.5)
        check(
            "ends-the-installed-app-and-its-tree",
            len(trees["app"]) > 1 and not alive(trees["app"]),
            f"exit {done.returncode}; tree {trees['app']}, still alive {alive(trees['app'])}",
        )
        check(
            "ends-helioxs-daemon-and-its-tree",
            len(trees["heliox"]) > 1 and not alive(trees["heliox"]),
            f"tree {trees['heliox']}, still alive {alive(trees['heliox'])}",
        )
        check(
            "leaves-a-developers-daemon-from-a-checkout",
            len(trees["developer"]) > 1 and alive(trees["developer"]) == trees["developer"],
            f"a checkout named Heliox-OS: tree {trees['developer']}, alive {alive(trees['developer'])}",
        )
        check(
            "leaves-another-copy-of-the-app",
            len(trees["other-app"]) > 1 and alive(trees["other-app"]) == trees["other-app"],
            f"tree {trees['other-app']}, alive {alive(trees['other-app'])}",
        )
    finally:
        for process in started:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("runs the uninstaller's script, which ends processes on the machine")
    if sys.platform != "win32":
        print("NOT RUN: this check is for Windows, and changed nothing here.")
        sys.exit(3)
    try:
        sys.exit(main())
    except Exception as error:  # every exit path prints a CHECK line
        check("check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))

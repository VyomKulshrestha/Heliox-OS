"""The Windows installer a person downloads, installed, used and uninstalled on a windows-latest runner.

    windows_package_check.py <the NSIS setup .exe> <the installer.nsi Tauri generated for it>

The installer is the one the release builds (Tauri's NSIS bundle, release build). Everything runs as the runner's
own user, in its real profile, because the installer and uninstaller work on the real profile and registry. Each
check is one CHECK line, every wait is bounded, and any other failure is a check-itself FAIL line:

- ``installs-for-the-current-user``: ``/S /CURRENTUSER`` installs it under %LOCALAPPDATA% and registers it under
  HKCU, not HKLM;
- ``app-starts``: the installed app is still running 20 s after it is started;
- ``setup-creates-environment``: the first-run script the app embeds (setup/first-run.ps1, the same file) creates
  the environment. The button that starts it in the app is not pressed here: WebDriver does not attach to the app's
  WebView2 on the hosted runner (the App launch job). The CI "first-run" job proves the script on both Python paths;
- ``page-reaches-daemon``: the installed app, opened again, starts the daemon from that environment and its page
  signs in and stays connected (the daemon's own log);
- ``daemon-answers-with-token``: first_run_rpc_check.py, with the environment's Python, reads the token and gets
  answers;
- ``open-before-removal``: the app and its daemon are running;
- ``remove-ends-app-and-daemon``: the uninstaller, run silently (``/S``), ends both (the PREUNINSTALL hook);
- ``remove-takes-the-program``: the install folder, the uninstall key and the shortcuts are gone;
- ``remove-left-no-locked-file``: once the processes are gone (bounded wait), every file left in Heliox's folders
  can be renamed, so nothing still holds it open;
- ``remove-leaves-no-program-files-outside-the-user-environment``: what is left, in FIRST_RUN.md's three kinds:
  program files (inside the environment folder ~/.config/heliox-os), the app's working files, the person's data;
- ``remove-says-what-is-left``: read, not run. A silent uninstall shows nothing; the built installer's script
  includes Heliox's hook and Heliox's English text, whose end-of-uninstall message names what is left;
- ``app-data-box-says-what-it-removes``: read, not run. The built uninstaller's "delete app data" box removes
  exactly %APPDATA%\\com.helioxos.app and %LOCALAPPDATA%\\com.helioxos.app, and its label says so and that the
  person's settings and data are kept.

CI-only (``ci_only.require_runner``): it installs and uninstalls an app and writes in the real profile.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

PRODUCT = "Heliox-OS"
BINARY = "heliox-os.exe"
BUNDLE_ID = "com.helioxos.app"
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{PRODUCT}"
OUT = Path(os.environ.get("RUNNER_TEMP", ".")) / "windows-package"
failures = 0

HOME = Path(os.path.expanduser("~"))
LOCAL = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData/Local"))
ROAMING = Path(os.environ.get("APPDATA", HOME / "AppData/Roaming"))
ENVIRONMENT = HOME / ".config/heliox-os"
PROGRAM_PARTS = ("env", "python", "tools", "uv-cache", "setup", "setup.json")
WORKING_FILES = (HOME / ".local/state/heliox-os", LOCAL / "heliox-os", LOCAL / BUNDLE_ID, ROAMING / BUNDLE_ID)
PERSONS = (ENVIRONMENT, HOME / ".local/share/heliox-os", HOME / ".cache/heliox", HOME / ".heliox", HOME / ".pilot")
PROGRAM_ELSEWHERE = (HOME / ".local/share/heliox-os/models",)


def check(name: str, ok: bool, detail: str) -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK windows-package/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    return ok


def app_processes(install: Path | None = None) -> list:
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "name", "exe"]):
        if (process.info.get("name") or "").lower() != BINARY:
            continue
        if install is None or str(process.info.get("exe") or "").lower().startswith(str(install).lower()):
            found.append(process)
    return found


def daemon_processes() -> list:
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "cmdline", "exe"]):
        command = " ".join(process.info.get("cmdline") or [])
        if "-m pilot.server" in command and ("\\heliox-os\\" in command or "\\heliox-os\\" in str(process.info["exe"])):
            found.append(process)
    return found


def end_tree(processes: list) -> None:
    for process in processes:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=30, check=False)


def registered(hive) -> str | None:
    import winreg

    try:
        with winreg.OpenKey(hive, UNINSTALL_KEY) as key:
            return str(winreg.QueryValueEx(key, "UninstallString")[0])
    except OSError:
        return None


def wait_for(probe, timeout: float, every: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = probe()
        if found:
            return found
        time.sleep(every)
    return probe()


def files_under(folder: Path, limit: int = 50_000) -> list[Path]:
    found: list[Path] = []
    for root, _, names in os.walk(folder):
        found.extend(Path(root) / name for name in names)
        if len(found) >= limit:
            break
    return found


def ui_client_stays(log: Path, since: int, timeout: float, stay: float = 10) -> str | None:
    """A ui client that signed in after line ``since`` of the daemon's log and stayed ``stay`` seconds."""
    first_seen: dict[str, float] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[since:]
        except OSError:
            lines = []
        text = "\n".join(lines)
        for line in lines:
            found = re.search(r"Client authenticated: (\(.*?\)) role=ui", line)
            if found:
                first_seen.setdefault(found.group(1), time.monotonic())
        for remote, seen in first_seen.items():
            if f"Client disconnected: {remote}" not in text and time.monotonic() - seen >= stay:
                return remote
        time.sleep(1)
    return None


def log_lines(log: Path) -> int:
    try:
        return len(log.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def start_app(executable: Path, name: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring"}
    handle = open(OUT / name, "w", encoding="utf-8")  # noqa: SIM115 - open for the life of the app
    return subprocess.Popen([str(executable)], stdout=handle, stderr=subprocess.STDOUT, env=env)


def install(setup_exe: str) -> Path | None:
    import winreg

    done = subprocess.run([setup_exe, "/S", "/CURRENTUSER"], capture_output=True, text=True, timeout=600, check=False)
    user = wait_for(lambda: registered(winreg.HKEY_CURRENT_USER), 30)
    machine = registered(winreg.HKEY_LOCAL_MACHINE)
    folder = Path(user.strip('"')).parent if user else None
    executable = folder / BINARY if folder else None
    ok = done.returncode == 0 and bool(user) and not machine and executable is not None and executable.is_file()
    check(
        "installs-for-the-current-user",
        ok and str(folder).lower().startswith(str(LOCAL).lower()),
        f"exit {done.returncode}; HKCU {user}; HKLM {machine}; app {executable}",
    )
    return folder if ok else None


def first_run(folder: Path) -> bool:
    versions = json.loads((REPO / "tauri-app/src-tauri/setup/python-versions.json").read_text(encoding="utf-8"))
    env = {
        **os.environ,
        "HELIOX_SETUP_DIR": str(ENVIRONMENT),
        "HELIOX_CORE_REQ": str(REPO / "daemon"),
        "HELIOX_GROUPS": "",
        "HELIOX_PYTHON_MIN": versions["min"],
        "HELIOX_PYTHON_MAX": versions["max"],
        "HELIOX_PYTHON_PRIVATE": versions["private"],
    }
    script = REPO / "tauri-app/src-tauri/setup/first-run.ps1"
    started = time.monotonic()
    done = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "run"],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
        check=False,
    )
    (OUT / "first-run.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    python = ENVIRONMENT / "env/Scripts/python.exe"
    return check(
        "setup-creates-environment",
        done.returncode == 0 and python.is_file(),
        f"exit {done.returncode} after {time.monotonic() - started:.0f} s; {python} "
        f"{'exists' if python.is_file() else 'missing'}; the script the app embeds, run directly (the button: not "
        "pressed, WebDriver does not attach on the hosted runner)",
    )


def daemon_side(folder: Path) -> None:
    log = HOME / ".local/state/heliox-os/pilot.log"
    since = log_lines(log)
    start_app(folder / BINARY, "app-after-setup.log")
    remote = ui_client_stays(log, since, 180)
    check(
        "page-reaches-daemon",
        remote is not None,
        f"ui client {remote} signed in and stayed 10 s" if remote else "no ui client stayed 10 s in 180 s",
    )
    python = ENVIRONMENT / "env/Scripts/python.exe"
    asked = subprocess.run(
        [str(python), str(HERE / "first_run_rpc_check.py"), "240"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    for line in asked.stdout.splitlines():
        print(f"DIAG first_run_rpc_check: {line}", flush=True)
    last = [line for line in asked.stdout.splitlines() if line.startswith("CHECK")]
    check("daemon-answers-with-token", asked.returncode == 0, last[-1] if last else asked.stderr.strip()[-300:])


def uninstall(folder: Path) -> None:
    import winreg

    import psutil

    apps, daemons = app_processes(folder), daemon_processes()
    check(
        "open-before-removal",
        bool(apps) and bool(daemons),
        f"app {[p.pid for p in apps]}, daemon {[p.pid for p in daemons]}",
    )
    command = registered(winreg.HKEY_CURRENT_USER) or str(folder / "uninstall.exe")
    uninstaller = command.strip('"')
    started = time.monotonic()
    subprocess.run([uninstaller, "/S"], capture_output=True, timeout=120, check=False)

    # The uninstaller copies itself to the temp folder and runs from there: wait for that copy to finish.
    def finished() -> bool:
        running = [
            p for p in psutil.process_iter(["name"]) if re.fullmatch(r"(un_a|au_)\.exe", (p.info["name"] or "").lower())
        ]
        return not running and registered(winreg.HKEY_CURRENT_USER) is None and not Path(uninstaller).exists()

    done = wait_for(finished, 300)
    gone = wait_for(lambda: not app_processes() and not daemon_processes(), 30)
    left = []
    for process in app_processes() + daemon_processes():
        try:
            left.append(f"{process.pid} {' '.join(process.cmdline())[:160]}")
        except psutil.Error:
            pass
    check(
        "remove-ends-app-and-daemon",
        bool(done) and bool(gone),
        f"uninstaller finished: {bool(done)} after {time.monotonic() - started:.0f} s; still running 30 s later: "
        f"{left or 'nothing'}",
    )

    leftovers = files_under(folder) if folder.exists() else []
    shortcuts = [
        path
        for path in (ROAMING / f"Microsoft/Windows/Start Menu/Programs/{PRODUCT}.lnk", HOME / f"Desktop/{PRODUCT}.lnk")
        if path.exists()
    ]
    key = registered(winreg.HKEY_CURRENT_USER)
    check(
        "remove-takes-the-program",
        not leftovers and not shortcuts and key is None,
        f"install folder files left: {[str(p) for p in leftovers[:10]] or 'none'}; shortcuts: "
        f"{[str(p) for p in shortcuts] or 'none'}; uninstall key: {key or 'gone'}",
    )

    # Files that something still holds open cannot be renamed on Windows.
    places = [folder, ENVIRONMENT, *WORKING_FILES, *PERSONS[1:]]
    locked: list[str] = []
    deadline = time.monotonic() + 180
    tried = 0
    for place in places:
        for path in files_under(place) if place.exists() else []:
            if time.monotonic() > deadline:
                break
            tried += 1
            trial = path.with_name(path.name + ".heliox-lock-check")
            try:
                path.rename(trial)
                trial.rename(path)
            except OSError as error:
                locked.append(f"{path} ({error.winerror if hasattr(error, 'winerror') else error})")
    timed_out = time.monotonic() > deadline
    check(
        "remove-left-no-locked-file",
        not locked and not timed_out,
        f"{tried} files renamed and back{' (stopped at the 180 s limit)' if timed_out else ''}; locked: "
        f"{locked[:10] or 'none'}",
    )

    groups: dict[str, dict[str, int]] = {kind: {} for kind in ("program", "elsewhere", "working", "persons")}

    def add(kind: str, label: str, path: Path) -> None:
        groups[kind][label] = groups[kind].get(label, 0) + path.stat().st_size

    for path in files_under(HOME / ".config/heliox-os") if ENVIRONMENT.exists() else []:
        part = path.relative_to(ENVIRONMENT).parts[0]
        add("program" if part in PROGRAM_PARTS else "persons", f"~/.config/heliox-os/{part}", path)
    for kind, places in (
        ("elsewhere", (folder, *PROGRAM_ELSEWHERE)),
        ("working", WORKING_FILES),
        ("persons", PERSONS[1:]),
    ):
        for place in places:
            for path in files_under(place) if place.exists() else []:
                if kind == "persons" and any(path.is_relative_to(p) for p in PROGRAM_ELSEWHERE):
                    continue
                add(kind, str(place), path)
    known = {str(p).lower() for p in (folder, ENVIRONMENT, *WORKING_FILES, *PERSONS)}
    unknown = [
        str(entry)
        for parent in (HOME, LOCAL, ROAMING, HOME / ".config", HOME / ".local/share", HOME / ".local/state")
        if parent.exists()
        for entry in parent.iterdir()
        if "heliox" in entry.name.lower() and str(entry).lower() not in known
    ]

    def sizes(found: dict[str, int]) -> str:
        return ", ".join(f"{name} {count / 1e6:.0f} MB" for name, count in sorted(found.items())) or "none"

    check(
        "remove-leaves-no-program-files-outside-the-user-environment",
        not groups["elsewhere"] and not unknown,
        f"program files in the environment folder: {sizes(groups['program'])}; program files elsewhere: "
        f"{sizes(groups['elsewhere'])}; Heliox-named places of no known kind: {unknown or 'none'}; the app's working "
        f"files: {sizes(groups['working'])}; the person's data: {sizes(groups['persons'])}",
    )


def read_the_built_installer(script: Path) -> None:
    """What a silent uninstall cannot show: the built installer's own script and the text it includes."""
    text = script.read_text(encoding="utf-8-sig", errors="replace")
    includes = re.findall(r'!include "([^"]+)"', text)
    hooks = [path for path in includes if path.replace("\\", "/").endswith("installer-hooks.nsh")]
    languages = [Path(path) for path in includes if path.lower().endswith(".nsh") and "english" in path.lower()]
    strings: dict[str, str] = {}
    for language in languages:
        if language.exists():
            content = language.read_text(encoding="utf-8-sig", errors="replace")
            strings.update(re.findall(r'^LangString (\w+) \$\{LANG_ENGLISH\} "(.*)"$', content, flags=re.MULTILINE))
    message = strings.get("helioxLeftBehind", "")
    names_it = all(
        part in message
        for part in (".config\\heliox-os", "env", "uv-cache", ".local\\state\\heliox-os", BUNDLE_ID, "kept")
    )
    check(
        "remove-says-what-is-left",
        bool(hooks) and names_it,
        f"read, not run (a silent uninstall shows nothing): hook included: {bool(hooks)}; the interactive "
        f"uninstall's last message: {message[:500]!r}",
    )
    bundle = re.search(r'!define BUNDLEID "([^"]+)"', text)
    removed = re.findall(r'RmDir /r "(\$(?:APPDATA|LOCALAPPDATA)\\\$\{BUNDLEID\})"', text, flags=re.IGNORECASE)
    other = re.findall(r'RmDir /r "([^"]+)"', text, flags=re.IGNORECASE)
    label = strings.get("deleteAppData", "")
    exact = sorted(removed) == ["$APPDATA\\${BUNDLEID}", "$LOCALAPPDATA\\${BUNDLEID}"] and len(other) == len(removed)
    check(
        "app-data-box-says-what-it-removes",
        exact and bool(bundle) and bundle.group(1) == BUNDLE_ID and "web view data" in label and "kept" in label,
        f"read, not run: BUNDLEID {bundle.group(1) if bundle else None}; the box removes {removed}; every RmDir /r in "
        f"the script: {other}; its label: {label!r}",
    )


def main(setup_exe: str, generated: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    read_the_built_installer(Path(generated))
    folder = install(setup_exe)
    if folder is None:
        return failures
    first = start_app(folder / BINARY, "app-first-start.log")
    time.sleep(20)
    running = app_processes(folder)
    check("app-starts", bool(running), f"running after 20 s: {bool(running)}; exit code {first.poll()}")
    end_tree(app_processes(folder))
    if first_run(folder):
        daemon_side(folder)
    else:
        check("page-reaches-daemon", False, "not run: no environment")
        check("daemon-answers-with-token", False, "not run: no environment")
    uninstall(folder)
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("installs and uninstalls the desktop app and writes in the real profile")
    try:
        sys.exit(main(*sys.argv[1:3]))
    except Exception as error:  # every exit path prints a CHECK line
        check("check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))
    finally:
        end_tree(app_processes() + daemon_processes())

"""The AppImage, set up and then removed through the app's own removal, the only one an AppImage has.

    appimage_check.py <the .AppImage in the person's home> <folder with tauri-driver>

Runs as root inside the throwaway container after appimage_check.sh, with the standard library only, and with
deb_install_check.py's person, window and daemon-log helpers. Each check is one CHECK line named appimage/<check>:

- ``app-starts``, ``first-run-setup``, ``page-reaches-daemon``, ``daemon-answers-with-token``: as for the .deb, with
  the AppImage as the app;
- ``removal-names-the-appimage-file``: the removal screen's plan (heliox_files_plan, the command the screen calls)
  names the AppImage file, by its path, as the person's to delete;
- ``removal-ends-the-daemon``: heliox_files_remove with "keep my data", called from the app's page as the screen
  calls it, ends the daemon. The daemon here is one the app did not start (it was running before the window
  opened, as after the app was ended abruptly), which the removal used to leave running;
- ``removal-removes-the-apps-files-keeps-the-persons``: the program files and the app's working files are gone and
  the person's data is still there (FIRST_RUN.md's three kinds);
- ``removal-leaves-the-appimage-file``: the AppImage file itself is untouched.

The screen's clicks are not driven: vitest covers the screen; this covers what its commands do in a real AppImage.
CI-only (``ci_only.require_runner``), and only as root in a container.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deb_install_check as base  # noqa: E402

PAGE_READY = "return document.readyState === 'complete' && !!document.querySelector('main')"


def call(driver, command: str, args: dict, timeout: float) -> object:
    """A Tauri command, called from the app's page as its own screens call it: started by one script, its answer
    read by later ones, so no script waits inside the driver."""
    from app_launch_check import wait_until

    driver.run(
        "window.__helioxCall = 'pending';"
        "window.__TAURI_INTERNALS__.invoke(arguments[0], arguments[1])"
        ".then(r => { window.__helioxCall = JSON.stringify({ok: r}); },"
        " e => { window.__helioxCall = JSON.stringify({error: String(e)}); });"
        "return 1;",
        command,
        args,
    )

    def answered():
        value = base.once_more(driver, f"reading {command}'s answer", lambda: driver.run("return window.__helioxCall"))
        return value if value != "pending" else None

    value = wait_until(answered, timeout, every=1)
    if not isinstance(value, str):
        raise TimeoutError(f"{command} did not answer in {timeout:.0f} s ({value!r})")
    return json.loads(value)


def kinds(home: Path) -> dict[str, list[str]]:
    """Which of Heliox's places in the home folder hold files, by FIRST_RUN.md's kind."""
    setup = home / base.ENVIRONMENT
    found: dict[str, list[str]] = {"program": [], "working": [], "persons": []}
    for part in base.PROGRAM_PARTS:
        if (setup / part).exists():
            found["program"].append(f"{base.ENVIRONMENT}/{part}")
    for place in base.WORKING_FILES:
        if any((home / place).rglob("*")) if (home / place).is_dir() else (home / place).exists():
            found["working"].append(place)
    for place in base.PERSONS:
        folder = home / place
        if not folder.exists():
            continue
        names = [p for p in folder.iterdir() if not (place == base.ENVIRONMENT and p.name in base.PROGRAM_PARTS)]
        if place == ".local/share/heliox-os":
            names = [p for p in names if p.name != "models"]
        if names:
            found["persons"].append(place)
    return found


def environ(pid: int) -> dict[str, str]:
    """A process's environment, from /proc (the check runs as root)."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    pairs = (entry.split("=", 1) for entry in raw.decode(errors="replace").split("\0") if "=" in entry)
    return dict(pairs)


def into_the_appimage(env: dict[str, str], appdir: str) -> dict[str, str]:
    """The entries that point into the mounted AppImage, and the Python search settings, which no program outside
    it should inherit (the app's outside_env rule removes both)."""
    return {
        name: value
        for name, value in sorted(env.items())
        if (appdir and appdir in value) or name in ("PYTHONHOME", "PYTHONPATH")
    }


APPDIR: list[str] = [""]  # where the AppImage's files are for the app, read at its first start


def image_folder(pid: int, env: dict[str, str]) -> str:
    """Where the app's AppImage files are: APPDIR when its start-up set it, else the folder its program runs from
    (…/usr/bin/heliox-os). Extracted to run (APPIMAGE_EXTRACT_AND_RUN, no FUSE in the container), the runtime sets
    APPIMAGE and not always APPDIR: on 0fb22f4 this check found no APPDIR and could not judge the daemon."""
    if env.get("APPDIR"):
        return env["APPDIR"]
    try:
        program = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""
    head, marker, _ = program.partition("/usr/bin/")
    return head if marker else str(Path(program).parent)


def the_app_environment() -> None:
    """Print what the AppImage's start-up gave the app: the entries that point into the AppImage."""
    for pid in base.app_pids():
        env = environ(pid)
        APPDIR[0] = image_folder(pid, env) or APPDIR[0]
        print(
            f"DIAG the app's environment: {len(env)} entries; APPIMAGE {env.get('APPIMAGE', '(not set)')!r}, "
            f"APPDIR {env.get('APPDIR', '(not set)')!r}; the AppImage's files: {APPDIR[0] or '(not found)'}",
            flush=True,
        )
        print(
            f"DIAG the app's own environment, pointing into the AppImage: {into_the_appimage(env, APPDIR[0])}",
            flush=True,
        )
        return
    print("DIAG the app's own environment: no app process to read", flush=True)


def the_daemons_environment() -> None:
    """The daemon the app started from the environment it set up must carry none of the AppImage's entries
    (src-tauri/src/outside_env.rs): the line that proves the app starts programs outside the AppImage cleanly."""
    daemons = base.daemon_pids()
    if not daemons or not APPDIR[0]:
        base.check(
            "daemon-environment-outside-the-appimage", False, f"not read: daemon {daemons}, APPDIR {APPDIR[0]!r}"
        )
        return
    leftovers = {pid: into_the_appimage(environ(pid), APPDIR[0]) for pid in daemons}
    base.check(
        "daemon-environment-outside-the-appimage",
        not any(leftovers.values()),
        f"APPDIR {APPDIR[0]}; the daemon's entries pointing into it: {leftovers}",
    )


def removal(person, appimage: str, tools: str) -> None:
    from app_launch_check import wait_until

    # A daemon the app did not start: end the app abruptly, so its daemon keeps running, or start one from the
    # environment if the app had stopped it.
    base.end(base.app_pids())
    time.sleep(2)
    if not base.daemon_pids():
        environment = person.home / base.ENVIRONMENT / "env/bin/python3"
        person.start([str(environment), "-m", "pilot.server"], "daemon-before-removal.log")
        print("DIAG the daemon was started by the check from the environment", flush=True)
    else:
        print("DIAG the daemon kept running after the app was ended", flush=True)
    daemons = wait_until(base.daemon_pids, 60, every=1)
    time.sleep(10)  # listening
    window = base.Window(person, appimage, tools)
    with window as driver:
        wait_until(lambda: driver.run(PAGE_READY), 90, every=2)
        plan = call(driver, "heliox_files_plan", {}, 60)
        steps = (plan.get("ok") or {}).get("manual_steps", [])
        base.check(
            "removal-names-the-appimage-file",
            bool(steps) and appimage in steps[0] and "yours to delete" in steps[0],
            f"the screen's first step: {steps[0] if steps else plan}",
        )
        before = kinds(person.home)
        report = call(driver, "heliox_files_remove", {"keepData": True, "confirm": "remove"}, 180)
        print(f"DIAG the removal's report: {json.dumps(report)[:1500]}", flush=True)
        gone = wait_until(lambda: not base.daemon_pids(), 20, every=1)
        base.check(
            "removal-ends-the-daemon",
            gone is True and bool(daemons),
            f"daemon before {daemons}, after {base.daemon_pids()}; the report's failures: "
            f"{(report.get('ok') or {}).get('failed', report)}",
        )
        after = kinds(person.home)
        base.check(
            "removal-removes-the-apps-files-keeps-the-persons",
            bool(before["program"])
            and not after["program"]
            and not after["working"]
            and after["persons"] == before["persons"]
            and bool(before["persons"]),
            f"before: {before}; after: {after}",
        )
    base.check(
        "removal-leaves-the-appimage-file",
        Path(appimage).is_file(),
        f"{appimage} {'is still there' if Path(appimage).is_file() else 'is gone'}",
    )


def main(appimage: str, tools: str) -> int:
    base.LABEL[0] = "appimage"
    base.OUT.mkdir(parents=True, exist_ok=True)
    display = subprocess.Popen(["Xvfb", ":99", "-ac", "-screen", "0", "1280x800x24"], stderr=subprocess.DEVNULL)
    for _ in range(30):
        if Path("/tmp/.X11-unix/X99").exists():
            break
        time.sleep(0.5)
    try:
        person = base.Person()
        person.env["APPIMAGE_EXTRACT_AND_RUN"] = "1"  # no FUSE in the container
        log = base.DaemonLog(person.home)
        first = person.start([appimage], "app-first-start.log")
        time.sleep(20)
        output = (base.OUT / "app-first-start.log").read_text(encoding="utf-8", errors="replace")
        running = base.app_pids()
        the_app_environment()
        base.check(
            "app-starts",
            bool(running) and "error while loading shared libraries" not in output,
            f"running after 20 s: {bool(running)}; exit code {first.poll()}; last output: {output[-300:]!r}",
        )
        base.end(base.app_pids())
        if base.first_run_through_the_window(person, appimage, tools, log, the_daemons_environment):
            removal(person, appimage, tools)
        else:
            for name in (
                "daemon-environment-outside-the-appimage",
                "page-reaches-daemon",
                "daemon-answers-with-token",
                "removal-names-the-appimage-file",
                "removal-ends-the-daemon",
                "removal-removes-the-apps-files-keeps-the-persons",
            ):
                base.check(name, False, "not run: the setup did not finish")
    finally:
        print(f"DIAG driver retries in this run: {len(base.RETRIES)} ({', '.join(base.RETRIES) or 'none'})", flush=True)
        base.end([pid for pid, _, _, uid in base.processes() if uid != 0], wait=5)
        display.terminate()
    return base.failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("installs packages, creates a user and starts the desktop app")
    if not Path("/.dockerenv").exists() or os.geteuid() != 0:
        print("NOT RUN: this check runs only as root inside a throwaway container, and changed nothing here.")
        sys.exit(3)
    try:
        sys.exit(main(*sys.argv[1:3]))
    except Exception as error:  # every exit path prints a CHECK line
        base.LABEL[0] = "appimage"
        base.check("check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(base.failures, 1))

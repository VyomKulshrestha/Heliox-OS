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
        value = driver.run("return window.__helioxCall")
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
        base.check(
            "app-starts",
            bool(running) and "error while loading shared libraries" not in output,
            f"running after 20 s: {bool(running)}; exit code {first.poll()}; last output: {output[-300:]!r}",
        )
        base.end(base.app_pids())
        if base.first_run_through_the_window(person, appimage, tools, log):
            removal(person, appimage, tools)
        else:
            for name in (
                "page-reaches-daemon",
                "daemon-answers-with-token",
                "removal-names-the-appimage-file",
                "removal-ends-the-daemon",
                "removal-removes-the-apps-files-keeps-the-persons",
            ):
                base.check(name, False, "not run: the setup did not finish")
    finally:
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

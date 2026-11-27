"""The installed .deb, used the way a person uses it, after deb_install_check.sh installed it on a clean system.

    deb_install_check.py <app binary> <package name> <folder with tauri-driver> <label> <full|offer>

Runs as root inside the throwaway container, with the standard library only. The person is a new user, "sam", with
an empty home folder, a virtual display and a session bus. Each check is one CHECK line named <label>/<check>, and
every wait is bounded. Both modes:

- ``app-starts``: the installed binary starts under the virtual display and is still running 20 s later, with no
  missing library.

"offer" (a system whose own Python Heliox cannot use, such as Ubuntu 22.04's 3.10):

- ``setup-offers-private-python``: the app's setup screen, read in its own window (tauri-driver and
  WebKitWebDriver), offers Heliox's private Python with the download's size and its sources, and nothing has been
  fetched: the button is not pressed. The download itself is proved by the first-run job's "private" leg.

"full":

- ``first-run-setup``: "Set up Heliox" is pressed with the choices the screen made, and the screen reaches "Heliox
  is ready": the environment is created and the app starts the daemon it installed;
- ``page-reaches-daemon``: a ui client signs in to that daemon with its token and stays connected (the app's page;
  the app's own Rust calls sign in and close at once);
- ``daemon-answers-with-token``: first_run_rpc_check.py, run with the new environment's Python, reads the token
  the daemon wrote and gets answers to ping and startup_status;
- ``open-before-removal``: the app, opened again as a person opens it, and its daemon are running;
- ``remove-ends-app-and-daemon``: ``apt-get remove`` ends both (the package's pre-remove script);
- ``remove-takes-the-program``: every file the package installed is gone;
- ``remove-leaves-no-program-files-outside-the-user-environment``: every file Heliox left in the home folder is,
  in the words of docs/FIRST_RUN.md's removal table, the person's data, the app's working files (logs, the web
  view's storage and cache), or a program file inside the environment folder ~/.config/heliox-os (env, python,
  tools, uv-cache, setup); shared caches (pip, Hugging Face) and the desktop libraries' own files are named, not
  counted;
- ``remove-says-what-is-left``: apt-get remove printed, from the package's post-remove script, the environment
  folder, its rough size and the command that deletes it, and named every working-file folder left.

The keyring is off (PYTHON_KEYRING_BACKEND fail), as on a Linux machine without a secret service, as in the
first-run job. CI-only (``ci_only.require_runner``), and only as root in a container.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PERSON = "sam"
OUT = Path("/out")
LABEL = ["deb"]
failures = 0

ENVIRONMENT = ".config/heliox-os"
# The kinds of docs/FIRST_RUN.md's table ("Removing Heliox's files"), in its words: program files, the app's working
# files, the person's data.
# Program files that the first-run setup installs in the environment folder.
PROGRAM_PARTS = ("env", "python", "tools", "uv-cache", "setup", "setup.json")
# Program files Heliox downloads later, outside the environment folder (speech models; not in this run's setup).
PROGRAM_ELSEWHERE = (".local/share/heliox-os/models",)
# The app's working files: logs, the web view's storage and its cache. Safe to delete; not the person's data.
WORKING_FILES = (".local/state/heliox-os", ".local/share/com.helioxos.app", ".cache/com.helioxos.app")
# The person's data: settings, keys, memory, timeline, persona, calibration, screenshots.
PERSONS = (".config/heliox-os", ".local/share/heliox-os", ".cache/heliox", ".heliox", ".pilot")
SHARED_CACHES = (".cache/pip", ".cache/huggingface", ".cache/uv")
# Written by the desktop libraries for any program that opens a window, not by Heliox.
DESKTOP_LIBRARIES = (
    ".cache/fontconfig",
    ".cache/mesa_shader_cache",
    ".cache/mesa_shader_cache_db",
    ".config/dconf",
    ".config/pulse",
    ".local/share/recently-used.xbel",
    ".cache/gstreamer-1.0",
    ".local/share/gvfs-metadata",
    ".dbus",
    ".local/share/webkitgtk",
    ".cache/webkitgtk",
)


def check(name: str, ok: bool, detail: str) -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {LABEL[0]}/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    return ok


def processes() -> list[tuple[int, str, str, int]]:
    """(pid, name, command line, uid) of every living process, from /proc. A zombie (ended, not yet reaped by its
    parent) is not running and is left out: in a container without an init, one the check ended earlier stays a
    zombie with its name, and CI on f806416 counted it as the app still running after apt-get remove."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            if "\nState:\tZ" in status:
                continue
            name = (entry / "comm").read_text().strip()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            uid = status.split("\nUid:")[1].split()[0]
        except (OSError, IndexError):
            continue
        found.append((int(entry.name), name, command, int(uid)))
    return found


# The window's processes. /proc/<pid>/comm keeps 15 characters, so the web view's two helpers appear there as
# "WebKitWebProces" and "WebKitNetworkPr": before 27 November this listing compared the full names and never saw them,
# and "processes changed: none" said nothing about the web process.
WINDOW_PROCESSES = ("heliox-os", "WebKitWebProcess", "WebKitNetworkProcess", "WebKitWebDriver", "tauri-driver")


def window_process(name: str) -> str | None:
    """The full name of one of WINDOW_PROCESSES whose /proc comm is ``name``, else None."""
    return next((full for full in WINDOW_PROCESSES if name == full[:15]), None)


def listing() -> dict[str, list[int]]:
    """The processes by name, as app_launch_check.aftermath compares them before and after a dropped connection."""
    found: dict[str, list[int]] = {}
    for pid, name, _, _ in processes():
        full = window_process(name)
        if full is not None and full != "tauri-driver":
            found.setdefault(full.casefold(), []).append(pid)
    return {name: sorted(pids) for name, pids in found.items()}


RETRIES: list[str] = []  # the driver's dropped connections tried again (app_launch_check's rule), counted in DIAG


def once_more(driver, what: str, call):
    """Run ``call``; when the driver's connection drops and only that happened (app running, same session, page mark
    kept, no process changed: app_launch_check.aftermath), run it once more, and count it. Anything else raises."""
    from app_launch_check import aftermath, mark_page

    before, mark = listing(), mark_page(driver)
    try:
        return call()
    except Exception as error:
        _, state, only_the_driver = aftermath(driver, mark, before, listing)
        if not only_the_driver:
            raise RuntimeError(f"{type(error).__name__}: {error}; {state}") from error
        RETRIES.append(f"{what} ({type(error).__name__})")
        print(f"DIAG {what}: the driver dropped its connection with nothing else changed; tried once more", flush=True)
        return call()


def app_pids() -> list[int]:
    return [pid for pid, name, _, _ in processes() if name == "heliox-os"]


def daemon_pids() -> list[int]:
    return [
        pid
        for pid, _, command, _ in processes()
        if "/.config/heliox-os/env/" in command and "-m pilot.server" in command
    ]


def end(pids: list[int], wait: float = 10) -> list[int]:
    """SIGTERM, then SIGKILL for whatever is still there after ``wait`` seconds; returns what needed SIGKILL."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + wait
    alive = list(pids)
    while alive and time.monotonic() < deadline:
        time.sleep(0.5)
        alive = [pid for pid in alive if Path(f"/proc/{pid}").exists()]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return alive


def home_files(home: Path) -> dict[str, int]:
    """Every file under the home folder (links not followed) with its size, except the daemon source the check put
    there."""
    found: dict[str, int] = {}
    for root, folders, files in os.walk(home):
        if Path(root) == home and "daemon-src" in folders:
            folders.remove("daemon-src")
        for name in files:
            path = Path(root) / name
            try:
                found[str(path.relative_to(home))] = path.lstat().st_size
            except OSError:
                pass
    return found


def within(path: str, prefixes: tuple[str, ...]) -> str | None:
    return next((prefix for prefix in prefixes if path == prefix or path.startswith(prefix + "/")), None)


def megabytes(groups: dict[str, int]) -> str:
    return ", ".join(f"{name} {count / 1e6:.0f} MB" for name, count in sorted(groups.items())) or "none"


class Person:
    """The person: a user with an empty home, the virtual display, a session bus and a runtime folder."""

    def __init__(self) -> None:
        import pwd

        account = pwd.getpwnam(PERSON)
        self.home = Path(account.pw_dir)
        runtime = Path(f"/run/user/{account.pw_uid}")
        runtime.mkdir(parents=True, exist_ok=True)
        os.chown(runtime, account.pw_uid, account.pw_gid)
        runtime.chmod(0o700)
        self.env = {
            "HOME": str(self.home),
            "USER": PERSON,
            "LOGNAME": PERSON,
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "DISPLAY": ":99",
            "XDG_RUNTIME_DIR": str(runtime),
            "HELIOX_DAEMON_SOURCE": str(self.home / "daemon-src"),
            "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        }
        bus = subprocess.run(
            self.command(["dbus-daemon", "--session", "--fork", "--nopidfile", "--print-address=1"]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.env["DBUS_SESSION_BUS_ADDRESS"] = bus.stdout.strip().splitlines()[0] if bus.stdout.strip() else ""
        print(f"DIAG session bus: {self.env['DBUS_SESSION_BUS_ADDRESS'] or bus.stderr.strip()[:300]}", flush=True)

    def command(self, args: list[str]) -> list[str]:
        return ["runuser", "-u", PERSON, "--", "env", "-i", *[f"{k}={v}" for k, v in self.env.items()], *args]

    def start(self, args: list[str], log: str) -> subprocess.Popen:
        handle = open(OUT / log, "w", encoding="utf-8")  # noqa: SIM115 - open for the life of the process
        return subprocess.Popen(self.command(args), stdout=handle, stderr=subprocess.STDOUT)


class DaemonLog:
    """The daemon's own log (state folder, pilot.log): ui clients signing in and leaving."""

    def __init__(self, home: Path) -> None:
        self.path = home / ".local/state/heliox-os/pilot.log"

    def lines(self) -> list[str]:
        try:
            return self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    def ui_client_stays(self, since: int, timeout: float, stay: float = 10) -> str | None:
        """A ui client that signed in after line ``since`` and was still connected ``stay`` seconds after it was
        first seen; its address, or None."""
        first_seen: dict[str, float] = {}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            lines = self.lines()[since:]
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


SETUP_BUTTON = (
    "const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === 'Set up Heliox');"
    "if (b) return b.disabled ? 'disabled' : 'ready';"
    "const t = document.querySelector('#first-run-title'); return t ? 'title: ' + t.textContent : 'no screen';"
)


def uptime() -> float:
    """Seconds since the system started, the clock /proc gives process start times in."""
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def stat(pid: int) -> tuple[int, float] | None:
    """(parent pid, start time in seconds since the system started) of a process, from /proc/<pid>/stat."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(rest[1]), int(rest[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


Snapshot = dict[str, set[tuple[int, float]]]  # full process name -> {(pid, start time)}


def window_snapshot() -> Snapshot:
    """The window's processes now, each with its start time, so a new process with an old pid is still new."""
    found: Snapshot = {}
    for pid, name, _, _ in processes():
        full = window_process(name)
        started = stat(pid) if full else None
        if full and started:
            found.setdefault(full, set()).add((pid, started[1]))
    return found


class SessionWatch:
    """While a window's session is open: the window's processes once a second, each change with the second it was
    seen, counted from the session's start. The check's own calls can wait 30 s in the driver; this does not."""

    LIMIT = 40

    def __init__(self) -> None:
        self.opened = uptime()
        self.at_open = window_snapshot()
        self.changes: list[str] = []
        self.ended: set[tuple[str, int]] = set()  # (name, pid) of every watched process that ended
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def since(self) -> str:
        return f"+{uptime() - self.opened:.0f} s"

    def watch(self) -> None:
        before = self.at_open
        while not self.stopping.wait(1):
            try:
                now = window_snapshot()
                for name in sorted(set(before) | set(now)):
                    ended = sorted(pid for pid, _ in before.get(name, set()) - now.get(name, set()))
                    started = sorted(pid for pid, _ in now.get(name, set()) - before.get(name, set()))
                    self.ended.update((name, pid) for pid in ended)
                    if (ended or started) and len(self.changes) < self.LIMIT:
                        self.changes.append(f"{self.since()} {name}: ended {ended or '-'}, started {started or '-'}")
                before = now
            except Exception as error:  # a diagnostic: never the reason the check stops
                if len(self.changes) < self.LIMIT:
                    self.changes.append(f"{self.since()} the watch could not read /proc: {error}")

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=5)


def tail(path: Path, lines: int, width: int = 220) -> list[str]:
    """The last ``lines`` lines of a text file, each cut to ``width`` characters; nothing when it cannot be read."""
    try:
        text = path.read_bytes()[-200_000:].decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line[:width] for line in text.splitlines() if line.strip()][-lines:]


def kernel_log() -> list[str]:
    """Kills, out-of-memory kills, segfaults and traps in the kernel's log, when the container may read it."""
    try:
        done = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"not readable: {type(error).__name__}: {error}"]
    if done.returncode != 0:
        return [f"not readable: {done.stderr.strip()[:200] or f'exit {done.returncode}'}"]
    wanted = re.compile(r"Killed process|Out of memory|oom|segfault|traps:|general protection|WebKit|heliox", re.I)
    found = [line[:220] for line in done.stdout.splitlines() if wanted.search(line)]
    return found[-8:] or ["nothing about kills, memory or crashes"]


def memory() -> str:
    """Available, total and free swap memory, from /proc/meminfo."""
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    except OSError as error:
        return f"not readable: {error}"
    return ", ".join(f"{name} {values.get(name, '?').strip()}" for name in ("MemAvailable", "MemTotal", "SwapFree"))


def x_windows() -> list[str]:
    """The top-level windows on the virtual display (xwininfo, from x11-utils), the app's own named first."""
    try:
        done = subprocess.run(
            ["xwininfo", "-root", "-children", "-display", ":99"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"not listed: {type(error).__name__}: {error}"]
    rows = [line.strip()[:160] for line in done.stdout.splitlines() if line.strip().startswith("0x")]
    named = [row for row in rows if '"' in row and "(has no name)" not in row]
    return (named or rows)[:10] or [f"none listed (exit {done.returncode})"]


def process_tree(opened: float) -> list[str]:
    """The window's processes and everything the app started, with parent and start time from the session's start
    (negative: before it). Command lines are cut, and only the program and its first argument are kept."""
    everything = processes()
    window = {pid for pid, name, _, _ in everything if window_process(name)}
    apps = {pid for pid, name, _, _ in everything if name == "heliox-os"}
    rows = []
    for pid, name, command, _ in everything:
        started = stat(pid)
        if not started or not (pid in window or started[0] in apps):
            continue
        program = " ".join(command.split()[:2])[:120]
        rows.append(f"{pid} (parent {started[0]}) {name} started {started[1] - opened:+.0f} s: {program}")
    return rows[:25]


def lost_for_good(error: BaseException) -> bool:
    """A session the driver no longer knows: every later call gets the same answer, so waiting longer is pointless."""
    return "invalid session id" in str(error)


class Window:
    """The app's own window through tauri-driver and WebKitWebDriver, for as long as the ``with`` lasts.

    When a call to the session fails for good, ``lost`` prints, once, what the app, its web process and the drivers
    were doing (SessionWatch, process_tree and the logs). ``may_open_again`` says whether only the driver's side was
    lost (the app's main process and every web process the session opened with still running); only then may a
    caller ``reopen`` the window, once, before anything was pressed, and its CHECK line says so (``reopen_note``).
    A web process that ended is what a person would see as the window going blank, so it is never retried."""

    def __init__(self, person: Person, binary: str, tools: str) -> None:
        from app_launch_check import Driver

        self.person, self.binary, self.tools = person, binary, tools
        self.driver = Driver("http://127.0.0.1:4444")
        self.driver_process: subprocess.Popen | None = None
        self.watch: SessionWatch | None = None
        self.reported = False
        self.reopened = ""
        self.first_error = ""

    def __enter__(self):
        self.driver_process = self.person.start([f"{self.tools}/tauri-driver", "--port", "4444"], "driver.log")
        time.sleep(3)
        self.driver.start(self.binary, timeout=120)
        self.watch = SessionWatch()
        print(
            "DIAG the window's session opened: "
            + "; ".join(
                f"{name} {sorted(pid for pid, _ in found)}" for name, found in sorted(self.watch.at_open.items())
            ),
            flush=True,
        )
        return self.driver

    def __exit__(self, kind=None, error=None, trace=None) -> None:
        if error is not None and not self.reported:
            self.lost(error)
        self.driver.end()
        if self.driver_process is not None:
            end([self.driver_process.pid])
        if self.watch is not None:
            self.watch.stop()

    def failed_call(self, error: BaseException) -> None:
        """Note the first failed call to the session, with the second it came."""
        if not self.first_error:
            when = self.watch.since() if self.watch else "?"
            self.first_error = f"{when}: {type(error).__name__}: {str(error)[:200]}"
            print(f"DIAG the first failed call to the window's session, {self.first_error}", flush=True)

    def lost(self, error: BaseException) -> None:
        """What was alive and what the logs said when the session was lost, printed once per session."""
        if self.reported:
            return
        self.reported = True
        self.failed_call(error)
        opened = self.watch.opened if self.watch else uptime()
        print(
            f"DIAG a call to the window failed for good ({type(error).__name__}: {str(error)[:200]}); "
            "what follows is from that moment",
            flush=True,
        )
        if self.watch is not None:
            print(f"DIAG process changes since the session opened: {self.watch.changes or 'none'}", flush=True)
        _, why = self.may_open_again()
        print(f"DIAG the app against the session's start: {why}", flush=True)
        for row in process_tree(opened):
            print(f"DIAG process: {row}", flush=True)
        print(f"DIAG windows on the display: {x_windows()}", flush=True)
        state = "running" if self.driver_process and self.driver_process.poll() is None else "ended"
        exit_code = self.driver_process.poll() if self.driver_process else None
        print(f"DIAG tauri-driver: {state} (exit {exit_code})", flush=True)
        # The app's own output goes to the driver's log: WebKitWebDriver starts it with its own standard error.
        for line in tail(OUT / "driver.log", 15):
            print(f"DIAG driver.log (with the app's output): {line}", flush=True)
        for line in tail(self.person.home / ".local/state/heliox-os/pilot.log", 8):
            print(f"DIAG daemon log: {line}", flush=True)
        print(f"DIAG kernel log: {kernel_log()}", flush=True)
        print(f"DIAG memory: {memory()}", flush=True)

    def may_open_again(self) -> tuple[bool, str]:
        """Whether only the driver's side was lost: the app's main process and every web process the session opened
        with still running, the same processes (pid and start time). The reason, either way."""
        if self.watch is None:
            return False, "no session was opened"
        then, now = self.watch.at_open, window_snapshot()
        main_then, main_now = then.get("heliox-os", set()), now.get("heliox-os", set())
        web_then, web_now = then.get("WebKitWebProcess", set()), now.get("WebKitWebProcess", set())
        if not main_now:
            return False, "the app's main process is gone"
        if main_now != main_then:
            return (
                False,
                f"the app's main process is not the one the session opened with ({main_then} then, {main_now} now)",
            )
        # Every web process that ended during the session counts, also one started after the session opened.
        web_ended = sorted(
            {pid for pid, _ in web_then - web_now}
            | {pid for name, pid in self.watch.ended if name == "WebKitWebProcess"}
        )
        if web_ended:
            return False, (
                f"the app's main process is the same, but its web process {web_ended} ended "
                f"(web processes now: {sorted(pid for pid, _ in web_now)}): a person would see the window go blank"
            )
        if self.reopened:
            return False, "the app's main process and web process are the same; the window was already opened again"
        return True, "the app's main process and web process are the same ones the session opened with"

    def reopen(self, why: str) -> None:
        """Open the window again, once: end the session, the app and the drivers, then start them as before."""
        print(f"DIAG the window's session was lost and {why}: opening the window once more", flush=True)
        self.__exit__()
        first = OUT / "driver.log"  # the next session's driver writes a new one
        if first.exists():
            first.replace(OUT / "driver-first-session.log")
        end(app_pids())
        end([pid for pid, name, _, _ in processes() if window_process(name) in ("tauri-driver", "WebKitWebDriver")])
        time.sleep(2)
        self.reported, self.first_error, self.reopened = False, "", why
        RETRIES.append("the window opened again after its session was lost")
        self.__enter__()

    def reopen_note(self) -> str:
        if not self.reopened:
            return ""
        return "; the window was opened a second time: its first session was lost while " + self.reopened

    def setup_screen_ready(self) -> str:
        """Wait (3 min at most) for the setup button, which is enabled once the probe has answered. A session the
        driver no longer knows ends the wait at once, with the report of what was alive."""
        deadline = time.monotonic() + 180
        last = "not read"
        while time.monotonic() < deadline:
            try:
                state = self.driver.run(SETUP_BUTTON)
            except Exception as error:
                self.failed_call(error)
                last = f"unreadable: {type(error).__name__}: {error}"
                if lost_for_good(error):
                    self.lost(error)
                    return last
            else:
                if state == "ready":
                    return "ready"
                last = str(state)
            time.sleep(2)
        try:
            return str(self.driver.run(SETUP_BUTTON))
        except Exception as error:
            self.lost(error)
            return f"unreadable: {type(error).__name__}: {error}"

    def ready_or_reopened(self) -> str:
        """setup_screen_ready, and when the session was lost with only the driver's side gone, the window opened
        once more and the wait repeated."""
        state = self.setup_screen_ready()
        if state != "ready" and self.reported:
            again, why = self.may_open_again()
            if again:
                self.reopen(why)
                state = self.setup_screen_ready()
        return state

    def read(self, script: str) -> str:
        try:
            return str(self.driver.run(script))
        except Exception as error:
            if lost_for_good(error):
                self.lost(error)
            else:
                self.dropped(error)
            return f"unreadable: {type(error).__name__}: {error}"

    def dropped(self, error: BaseException) -> None:
        """One call that failed while the session may still answer (CI on 45952357188: tauri-driver's reused
        connection to WebKitWebDriver was reset, and the next call worked): a short line, so a later loss of the
        session still gets the full report."""
        self.failed_call(error)
        _, why = self.may_open_again()
        resets = sum("Connection reset by peer" in line for line in tail(OUT / "driver.log", 200, width=400))
        changes = self.watch.changes if self.watch else []
        print(
            f"DIAG a call to the window failed ({type(error).__name__}); {why}; process changes: {changes or 'none'}; "
            f"connection resets in tauri-driver's log: {resets}",
            flush=True,
        )


PYTHON_SENTENCE = "const p = document.querySelector('.python'); return p ? p.textContent.trim() : '';"
TOTAL = "const t = document.querySelector('.total'); return t ? t.textContent.trim() : '';"


def offers_private_python(person: Person, binary: str, tools: str) -> None:
    """The setup screen on a system whose Python Heliox cannot use: what it offers, without pressing anything."""
    window = Window(person, binary, tools)

    def offer() -> tuple[str, str, str]:
        state = window.setup_screen_ready()
        # Through the shared retry: a dropped driver connection here failed the 22.04 line once (f6a1f8e).
        sentence = str(
            once_more(window.driver, "reading the Python sentence", lambda: window.driver.run(PYTHON_SENTENCE))
        )
        total = str(once_more(window.driver, "reading the download total", lambda: window.driver.run(TOTAL)))
        return state, sentence, total

    try:
        with window:
            try:
                state, sentence, total = offer()
                first = window.first_error if window.reported else ""
            except Exception as error:  # nothing is pressed on this screen, so opening it again changes nothing
                window.lost(error)
                first = f"{type(error).__name__}: {error}"
            again, why = window.may_open_again() if first else (False, "")
            if first and not again:
                raise RuntimeError(f"{first}; the window was not opened again: {why}")
            if again:
                window.reopen(why)
                state, sentence, total = offer()
    except Exception as error:
        check(
            "setup-offers-private-python",
            False,
            f"no window to read: {type(error).__name__}: {error}{window.reopen_note()}",
        )
        return
    setup = person.home / ENVIRONMENT
    fetched = [name for name in PROGRAM_PARTS if name != "setup" and (setup / name).exists()]
    says = (
        "download its own private copy" in sentence
        and re.search(r"\d+ MB", sentence) is not None
        and "github.com/astral-sh/uv" in sentence
        and "python-build-standalone" in sentence
    )
    check(
        "setup-offers-private-python",
        state == "ready" and says and not fetched,
        f"button {state}; the screen says: {sentence!r}; {total!r}; already fetched: {fetched or 'nothing'}",
    )


def first_run_through_the_window(person: Person, binary: str, tools: str, log: DaemonLog, while_open=None) -> bool:
    window = Window(person, binary, tools)
    try:
        with window as driver:
            state = window.ready_or_reopened()  # opened again only before anything is pressed
            if state != "ready":
                return check(
                    "first-run-setup",
                    False,
                    f"the setup button never became ready ({state!r}){window.reopen_note()}",
                )
            chosen = window.read(
                "return [...document.querySelectorAll('.groups > li > label')].map(l =>"
                " (l.querySelector('input').checked ? '[x] ' : '[ ] ')"
                " + l.querySelector('.group-name').textContent.trim());"
            )
            print(f"DIAG the setup screen's choices, as the screen made them: {chosen}", flush=True)
            print(f"DIAG the setup screen's Python sentence: {window.read(PYTHON_SENTENCE)}", flush=True)
            press = SETUP_BUTTON.replace(
                "if (b) return b.disabled ? 'disabled' : 'ready';", "if (b) { b.click(); return 'pressed'; }"
            )
            title_script = "const t = document.querySelector('#first-run-title'); return t ? t.textContent.trim() : '';"

            def press_once() -> object:
                # A press whose answer was lost may have happened: press only while the choice screen still shows.
                if driver.run(title_script) != "Set up Heliox":
                    return "already pressed"
                return driver.run(press)

            print(f"DIAG the setup button: {once_more(driver, 'pressing Set up Heliox', press_once)}", flush=True)
            started = time.monotonic()
            title = ""
            while time.monotonic() - started < 1500:
                title = str(once_more(driver, "reading the setup screen", lambda: driver.run(title_script)))
                if title in ("Heliox is ready", "Setup stopped") or not title:
                    break
                time.sleep(5)
            seconds = time.monotonic() - started
            environment = person.home / ENVIRONMENT / "env/bin/python3"
            detail = f"{title!r} after {seconds:.0f} s; environment {'created' if environment.exists() else 'missing'}"
            if title not in ("Heliox is ready", ""):
                detail += "; the screen: " + window.read("return document.body.innerText")[:900]
            # The screen can give way to the main window once it is ready: no title then, with the environment there
            # and the daemon running, is the same answer.
            finished = title == "Heliox is ready" or (title == "" and bool(daemon_pids()))
            done = check("first-run-setup", finished and environment.exists(), detail + window.reopen_note())
            if done:
                daemon_checks(person, log)
                if while_open is not None:  # more checks while the app that ran the setup is open
                    while_open()
            return done
    except Exception as error:
        return check(
            "first-run-setup", False, f"no window to drive: {type(error).__name__}: {error}{window.reopen_note()}"
        )


def daemon_checks(person: Person, log: DaemonLog) -> None:
    """While the app that ran the setup is still open: its page reaches the daemon, and the daemon answers."""
    remote = log.ui_client_stays(0, 120)
    check(
        "page-reaches-daemon",
        remote is not None,
        f"ui client {remote} signed in and stayed 10 s" if remote else "no ui client stayed 10 s in 120 s",
    )
    environment = person.home / ENVIRONMENT / "env/bin/python3"
    asked = subprocess.run(
        person.command([str(environment), str(HERE / "first_run_rpc_check.py"), "240"]),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    for line in asked.stdout.splitlines():
        print(f"DIAG first_run_rpc_check: {line}", flush=True)
    last = [line for line in asked.stdout.splitlines() if line.startswith("CHECK")]
    check("daemon-answers-with-token", asked.returncode == 0, last[-1] if last else asked.stderr.strip()[-300:])


def removal(person: Person, binary: str, package: str, log: DaemonLog, before: dict[str, int]) -> None:
    """Open the app again as a person does, then apt-get remove, and what is left and said."""
    end(app_pids())
    since = len(log.lines())
    person.start([binary], "app-reopened.log")
    remote = log.ui_client_stays(since, 120)
    apps, daemons = app_pids(), daemon_pids()
    check(
        "open-before-removal",
        bool(apps) and bool(daemons) and remote is not None,
        f"app {apps}, daemon {daemons}, page {remote}",
    )

    removed = subprocess.run(
        ["apt-get", "remove", "-y", "-q", package], capture_output=True, text=True, timeout=300, check=False
    )
    (OUT / "apt-remove.log").write_text(removed.stdout + removed.stderr, encoding="utf-8")
    deadline = time.monotonic() + 20
    while (app_pids() or daemon_pids()) and time.monotonic() < deadline:
        time.sleep(1)
    survivors = set(app_pids() + daemon_pids())
    left = [f"{pid} {command[:160]}" for pid, _, command, _ in processes() if pid in survivors]
    check(
        "remove-ends-app-and-daemon",
        removed.returncode == 0 and not left,
        f"apt-get remove exit {removed.returncode}; still running 20 s later: {left or 'nothing'}",
    )
    files = [Path(line) for line in Path("/tmp/package-files.txt").read_text().splitlines() if line.strip()]
    stayed = [str(path) for path in files if path.is_file()]
    check("remove-takes-the-program", not stayed, f"files of the package still there: {stayed[:10] or 'none'}")

    # What is left in the home folder, and whose it is.
    kinds = ("program", "elsewhere", "working", "persons", "shared", "desktop")
    groups: dict[str, dict[str, int]] = {name: {} for name in kinds}
    unknown: dict[str, int] = {}
    environment_parts = tuple(f"{ENVIRONMENT}/{part}" for part in PROGRAM_PARTS)
    for path, count in home_files(person.home).items():
        if path in before:
            continue
        for group, prefixes in (
            ("program", environment_parts),
            ("elsewhere", PROGRAM_ELSEWHERE),
            ("working", WORKING_FILES),
            ("persons", PERSONS),
            ("shared", SHARED_CACHES),
            ("desktop", DESKTOP_LIBRARIES),
        ):
            prefix = within(path, prefixes)
            if prefix:
                groups[group][prefix] = groups[group].get(prefix, 0) + count
                break
        else:
            top = "/".join(path.split("/")[:2])
            unknown[top] = unknown.get(top, 0) + count
    check(
        "remove-leaves-no-program-files-outside-the-user-environment",
        not groups["elsewhere"] and not unknown,
        f"program files in the environment folder ~/{ENVIRONMENT}: {megabytes(groups['program'])}; "
        f"program files elsewhere: {megabytes(groups['elsewhere'])}; files of no known kind: {megabytes(unknown)}; "
        f"the app's working files (logs, web view storage and cache): {megabytes(groups['working'])}; "
        f"the person's data: {megabytes(groups['persons'])}; shared caches: {megabytes(groups['shared'])}; "
        f"the desktop libraries' own: {megabytes(groups['desktop'])}",
    )
    said = removed.stdout
    folder = str(person.home / ENVIRONMENT)
    size_said = re.search(r"about (\d+) MB", said)
    measured = sum(groups["program"].values()) / 1e6
    working = [str(person.home / prefix) for prefix in groups["working"]]
    unnamed = [path for path in working if path not in said]
    if any(part in groups["program"] for part in (f"{ENVIRONMENT}/{name}" for name in ("env", "python", "tools"))):
        told = folder in said and "rm -rf" in said and size_said is not None and not unnamed
        close = size_said is not None and abs(int(size_said.group(1)) - measured) <= max(5, measured * 0.25)
    else:  # setup never put program files there: only the working files, if any, have to be named
        told, close = not unnamed and (not working or "Heliox was removed" in said), True
    lines = [line for line in said.splitlines() if "Heliox" in line or "rm -rf" in line]
    check(
        "remove-says-what-is-left",
        told and close,
        f"apt-get remove printed: {' / '.join(lines)[:900] or 'nothing about Heliox'}; measured {measured:.0f} MB; "
        f"working-file folders it did not name: {unnamed or 'none'}",
    )


def main(binary: str, package: str, tools: str, label: str, mode: str) -> int:
    LABEL[0] = label
    OUT.mkdir(parents=True, exist_ok=True)
    display = subprocess.Popen(["Xvfb", ":99", "-ac", "-screen", "0", "1280x800x24"], stderr=subprocess.DEVNULL)
    for _ in range(30):
        if Path("/tmp/.X11-unix/X99").exists():
            break
        time.sleep(0.5)
    home = Path("/home") / PERSON
    try:
        person = Person()
        log = DaemonLog(person.home)
        before = home_files(person.home)

        # The binary on its own, as the menu entry starts it.
        first = person.start([binary], "app-first-start.log")
        time.sleep(20)
        output = (OUT / "app-first-start.log").read_text(encoding="utf-8", errors="replace")
        running = app_pids()
        broken = "error while loading shared libraries" in output
        check(
            "app-starts",
            bool(running) and not broken,
            f"running after 20 s: {bool(running)}; exit code {first.poll()}; last output: {output.strip()[-300:]!r}",
        )
        end(app_pids())

        if mode == "offer":
            offers_private_python(person, binary, tools)
            return failures
        # First run through the window, and the daemon it started from both sides while that window is open.
        if not first_run_through_the_window(person, binary, tools, log):
            check("page-reaches-daemon", False, "not run: the setup did not finish")
            check("daemon-answers-with-token", False, "not run: the setup did not finish")
        removal(person, binary, package, log, before)
    finally:
        # Counted every run (app_launch_check prints the same), so a rising number of driver drops is seen.
        print(f"DIAG driver retries in this run: {len(RETRIES)} ({', '.join(RETRIES) or 'none'})", flush=True)
        source = home / ".local/state/heliox-os/pilot.log"
        if source.exists():
            (OUT / "daemon-pilot.log").write_bytes(source.read_bytes()[-200_000:])
        ours = [pid for pid, _, _, uid in processes() if uid != 0]
        end(ours, wait=5)
        display.terminate()
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("installs a package, creates a user and starts the desktop app")
    if not Path("/.dockerenv").exists() or os.geteuid() != 0:
        print("NOT RUN: this check runs only as root inside a throwaway container, and changed nothing here.")
        sys.exit(3)
    if len(sys.argv) > 4:
        LABEL[0] = sys.argv[4]
    try:
        sys.exit(main(*sys.argv[1:6]))
    except Exception as error:  # every exit path prints a CHECK line
        check("check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))

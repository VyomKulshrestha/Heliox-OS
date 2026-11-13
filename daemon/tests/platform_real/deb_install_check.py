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
    """(pid, name, command line, uid) of every process, from /proc."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text().strip()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            uid = (entry / "status").read_text().split("\nUid:")[1].split()[0]
        except (OSError, IndexError):
            continue
        found.append((int(entry.name), name, command, int(uid)))
    return found


def app_pids() -> list[int]:
    return [pid for pid, name, _, _ in processes() if name == "heliox-os"]


def daemon_pids() -> list[int]:
    return [pid for pid, _, command, _ in processes() if "/heliox-os/env/" in command and "-m pilot.server" in command]


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


class Window:
    """The app's own window through tauri-driver and WebKitWebDriver, for as long as the ``with`` lasts."""

    def __init__(self, person: Person, binary: str, tools: str) -> None:
        from app_launch_check import Driver

        self.person, self.binary, self.tools = person, binary, tools
        self.driver = Driver("http://127.0.0.1:4444")
        self.driver_process: subprocess.Popen | None = None

    def __enter__(self):
        self.driver_process = self.person.start([f"{self.tools}/tauri-driver", "--port", "4444"], "driver.log")
        time.sleep(3)
        self.driver.start(self.binary, timeout=120)
        return self.driver

    def __exit__(self, *exc) -> None:
        self.driver.end()
        if self.driver_process is not None:
            end([self.driver_process.pid])

    def setup_screen_ready(self) -> str:
        """Wait (3 min at most) for the setup button, which is enabled once the probe has answered."""
        from app_launch_check import wait_until

        def ready():
            return self.driver.run(SETUP_BUTTON) == "ready" or None

        if wait_until(ready, 180, every=2) is True:
            return "ready"
        try:
            return str(self.driver.run(SETUP_BUTTON))
        except Exception as error:
            return f"unreadable: {type(error).__name__}: {error}"

    def read(self, script: str) -> str:
        try:
            return str(self.driver.run(script))
        except Exception as error:
            return f"unreadable: {type(error).__name__}: {error}"


PYTHON_SENTENCE = "const p = document.querySelector('.python'); return p ? p.textContent.trim() : '';"
TOTAL = "const t = document.querySelector('.total'); return t ? t.textContent.trim() : '';"


def offers_private_python(person: Person, binary: str, tools: str) -> None:
    """The setup screen on a system whose Python Heliox cannot use: what it offers, without pressing anything."""
    window = Window(person, binary, tools)
    try:
        with window:
            state = window.setup_screen_ready()
            sentence, total = window.read(PYTHON_SENTENCE), window.read(TOTAL)
    except Exception as error:
        check("setup-offers-private-python", False, f"no window to read: {type(error).__name__}: {error}")
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


def first_run_through_the_window(person: Person, binary: str, tools: str, log: DaemonLog) -> bool:
    window = Window(person, binary, tools)
    try:
        with window as driver:
            state = window.setup_screen_ready()
            if state != "ready":
                return check("first-run-setup", False, f"the setup button never became ready ({state!r})")
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
            driver.run(press)
            started = time.monotonic()
            title_script = "const t = document.querySelector('#first-run-title'); return t ? t.textContent.trim() : '';"
            title = ""
            while time.monotonic() - started < 1500:
                title = window.read(title_script)
                if title in ("Heliox is ready", "Setup stopped") or not title or title.startswith("unreadable"):
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
            done = check("first-run-setup", finished and environment.exists(), detail)
            if done:
                daemon_checks(person, log)
            return done
    except Exception as error:
        return check("first-run-setup", False, f"no window to drive: {type(error).__name__}: {error}")


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
    told = folder in said and "rm -rf" in said and size_said is not None and not unnamed
    close = size_said is not None and abs(int(size_said.group(1)) - measured) <= max(5, measured * 0.25)
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

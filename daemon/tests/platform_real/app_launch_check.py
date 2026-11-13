"""The real desktop app, launched and driven through WebDriver (tauri-driver), on a CI runner.

    python app_launch_check.py <path to the built app> [--driver http://127.0.0.1:4444]

The app is a debug build with its page bundled (cargo build: the custom-protocol feature is on by default). A daemon
runs in the same complete fake home (started by this check, from this checkout), so the app's commands answer
against it. tauri-driver must already listen (with WebKitWebDriver on Linux, msedgedriver on Windows; macOS has no
WebDriver for its web view, so the check does not run there). Checks:

- ``page-loaded``: the window shows the app's page, from the app's own origin;
- ``no-page-errors``: no uncaught error, unhandled rejection or content-security-policy violation was recorded
  while the page loaded (ui/src/lib/errorRecorder.ts);
- ``command-answers``: get_auth_token, a command the page calls, answers with the daemon's token (its length only
  is printed);
- ``async-route-survives-rejection``: the driver's async script route returns a plain rejected promise (no app
  command), the control for the next two;
- ``removed-permission-refused``: a plugin command the window no longer has (clipboard read) is refused, started by
  one script and read by later ones, so no async script waits in the driver;
- ``async-route-survives-refused-command``: the same refusal through the async script route;
- ``script-cannot-navigate``: the page's script setting location to another site leaves the window on the app;
- ``link-click-not-followed``: a click on a link to another site leaves the window on the app; on Linux the address
  reaches the system's browser ($BROWSER, a script that writes it down), proving it was handed over;
- ``webassembly-compiles``: WebAssembly compiles under the policy ('wasm-unsafe-eval'), and MediaPipe's own wasm
  file is served by the app;
- ``driver-loopback-only``: the WebDriver ports (which can run script in the app's window) listen on the loopback
  address only;
- ``everything-ended``: the app, the drivers and the daemon are ended at the end; anything still running 10 s after
  being asked is named, then killed.

Every wait has a time limit. CI-only (``ci_only.require_runner``): it starts a daemon and the app.
Each check prints one ``CHECK`` line; the exit code is the number of failed checks. A step whose driver connection
fails is a FAIL line that says whether the app, its web process, the session and the page survived; the next step
then gets a new session (and a new app) when the old one no longer answers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0
APP_ORIGINS = ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost")


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK app-launch/{name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


class Driver:
    """The few WebDriver calls the check needs, over plain HTTP."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.session = ""

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read() or b"{}").get("value")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{method} {path}: {error.code} {error.read()[:300]!r}") from error

    def start(self, app: str, timeout: float = 120) -> None:
        value = self.call(
            "POST",
            "/session",
            {"capabilities": {"alwaysMatch": {"tauri:options": {"application": app}}}},
            timeout=timeout,
        )
        self.session = value["sessionId"]

    def run(self, script: str, *args):
        return self.call("POST", f"/session/{self.session}/execute/sync", {"script": script, "args": list(args)})

    def run_async(self, script: str, *args, timeout: float = 30.0):
        return self.call(
            "POST",
            f"/session/{self.session}/execute/async",
            {"script": script, "args": list(args)},
            timeout=timeout + 5,
        )

    def url(self) -> str:
        return str(self.call("GET", f"/session/{self.session}/url"))

    def click(self, css: str) -> None:
        element = self.call("POST", f"/session/{self.session}/element", {"using": "css selector", "value": css})
        element_id = next(iter(element.values()))
        self.call("POST", f"/session/{self.session}/element/{element_id}/click", {})

    def end(self) -> None:
        if self.session:
            try:
                self.call("DELETE", f"/session/{self.session}", timeout=30)
                print("DIAG the session ended normally", flush=True)
            except Exception as error:
                print(f"DIAG ending the session: {type(error).__name__}: {error}", flush=True)
            self.session = ""


def wait_until(probe, timeout: float, every: float = 0.5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = probe()
            if last:
                return last
        except Exception as error:
            last = error
        time.sleep(every)
    return last


def on_app(url: str) -> bool:
    return url.startswith(APP_ORIGINS)


def daemon_log() -> Path:
    return Path(os.environ["RUNNER_TEMP"]) / "app-launch-daemon.log"


def start_daemon() -> subprocess.Popen:
    log = open(daemon_log(), "w", encoding="utf-8")  # noqa: SIM115
    daemon = Path(__file__).resolve().parents[2]
    return subprocess.Popen([sys.executable, "-m", "pilot.server"], cwd=daemon, stdout=log, stderr=log)


class WebViewWatch:
    """Windows only: while the driver starts the app, write down what msedgedriver needs to attach. It reads the
    DevToolsActivePort file that WebView2's browser process writes in its user data folder when it is started with
    --remote-debugging-port; "DevToolsActivePort file doesn't exist" means the port was not asked for, or the file
    went to a folder the driver does not look in. So: the app's command line and WEBVIEW2_* environment (what the
    driver handed it), the WebView2 browser process's --user-data-dir and --remote-debugging-port, and every
    DevToolsActivePort file under the app's and the temp folders."""

    def __init__(self) -> None:
        import threading

        self.seen: dict = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def __enter__(self):
        if sys.platform == "win32":
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if sys.platform != "win32":
            return
        self.stop.set()
        self.thread.join(timeout=10)
        for key in ("app", "webview2-browser", "devtools-files", "watch-error"):
            print(f"DIAG {key}: {json.dumps(self.seen.get(key, 'not seen'))[:1500]}", flush=True)

    def watch(self) -> None:
        import psutil

        # The app's WebView2 folder is under LOCALAPPDATA (a small fake home here); msedgedriver's own user data
        # folders are scoped_dir* folders directly under the temp folder.
        patterns = [(os.environ.get("LOCALAPPDATA", ""), "**/DevToolsActivePort")]
        patterns += [(os.environ.get(name, ""), "*/DevToolsActivePort") for name in ("TEMP", "TMP")]
        while not self.stop.is_set():
            try:
                for process in psutil.process_iter(["pid", "name"]):
                    name = (process.info.get("name") or "").casefold().removesuffix(".exe")
                    if name == "heliox-os" and process.info["pid"] not in APPS_BEFORE and "app" not in self.seen:
                        try:
                            environ = {k: v for k, v in process.environ().items() if k.upper().startswith("WEBVIEW2")}
                        except psutil.Error as error:
                            environ = {"unreadable": type(error).__name__}
                        self.seen["app"] = {"cmdline": process.cmdline(), "webview2_env": environ}
                    if name == "msedgewebview2" and "webview2-browser" not in self.seen:
                        cmdline = process.cmdline()
                        ours = any(
                            (parent.name() or "").casefold().removesuffix(".exe") == "heliox-os"
                            and parent.pid not in APPS_BEFORE
                            for parent in process.parents()
                        )  # the system's own WebView2 users (the shell, widgets) run their own
                        if ours and not any(arg.startswith("--type=") for arg in cmdline):  # the browser process
                            wanted = ("--user-data-dir", "--remote-debugging", "--embedded-browser-webview")
                            self.seen["webview2-browser"] = [a for a in cmdline if a.startswith(wanted)]
                files = set(self.seen.get("devtools-files", []))
                for root, pattern in patterns:
                    if root:
                        files.update(str(found) for found in list(Path(root).glob(pattern))[:20])
                self.seen["devtools-files"] = sorted(files)
            except Exception as error:  # a diagnostic: never the reason the check stops
                self.seen["watch-error"] = f"{type(error).__name__}: {error}"
            self.stop.wait(2)


def launch_without_driver(app: str) -> None:
    """Windows, when the driver could not attach: the app started by this check itself, its output written to a
    file, and the launch proven from the daemon's side. A ui client that signs in with the daemon's token and stays
    connected is the app's page (the app's own Rust calls sign in, ask and close at once), so the page loaded and
    get_auth_token answered."""
    import re

    end_apps_started_here()  # the driver's attempt can leave its app open, and a second copy only hands over to it
    time.sleep(2)
    log_path = Path(os.environ["RUNNER_TEMP"]) / "app-launch-app.log"
    seen_before = len(daemon_log().read_text(encoding="utf-8", errors="replace").splitlines())
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            app_process = subprocess.Popen([app], stdout=log, stderr=log)
        except OSError as error:
            check("page-reaches-daemon-without-driver", False, f"the app did not start: {error}")
            return
        try:

            def page_signed_in():
                lines = daemon_log().read_text(encoding="utf-8", errors="replace").splitlines()[seen_before:]
                for line in lines:
                    found = re.search(r"Client authenticated: (\(.*?\)) role=ui", line)
                    if found:
                        return found.group(1)
                return None

            remote = wait_until(page_signed_in, 60, every=1)
            signed_in = isinstance(remote, str)
            stayed = False
            if signed_in:
                time.sleep(10)
                later = daemon_log().read_text(encoding="utf-8", errors="replace")
                stayed = f"Client disconnected: {remote}" not in later
            check(
                "page-reaches-daemon-without-driver",
                signed_in and stayed,
                f"ui client {remote} signed in and stayed 10 s" if signed_in and stayed else f"{remote!r}",
            )
            check("app-stays-up-without-driver", app_process.poll() is None, f"exit code {app_process.poll()}")
        finally:
            app_process.terminate()
            try:
                app_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app_process.kill()
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    print(f"DIAG the app's own output (last {len(tail)} lines): {json.dumps(tail)[:2000]}", flush=True)


def page_ready(driver: Driver, timeout: float = 60):
    return wait_until(
        lambda: driver.run("return document.readyState === 'complete' && !!document.querySelector('main')"), timeout
    )


# What runs the app's page, by process name: the app, WebKitGTK's web and network processes, WebView2's processes,
# and the drivers. A web process that is gone after a step, or a new one, means the page's renderer was replaced.
WATCHED = ("heliox-os", "webkitwebprocess", "webkitnetworkprocess", "msedgewebview2", "webkitwebdriver", "msedgedriver")


def processes() -> dict[str, list[int]]:
    import psutil

    found: dict[str, list[int]] = {}
    for process in psutil.process_iter(["pid", "name"]):
        name = (process.info.get("name") or "").casefold().removesuffix(".exe")
        if name in WATCHED:
            found.setdefault(name, []).append(int(process.info["pid"]))
    return {name: sorted(pids) for name, pids in sorted(found.items())}


def mark_page(driver: Driver) -> str:
    """Leave a mark on the page before a step; a page that lost it afterwards was reloaded or replaced."""
    mark = f"m{time.monotonic_ns()}"
    try:
        driver.run("window.__helioxStepMark = arguments[0]; return 1;", mark)
    except Exception:
        return ""
    return mark


def aftermath(driver: Driver, mark: str, before: dict[str, list[int]]) -> tuple[bool, str]:
    """After a step lost its connection: what was still alive, and whether the same session and page still answer."""
    time.sleep(1)
    after = processes()
    changed = {
        name: {"before": before.get(name, []), "after": after.get(name, [])}
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    }
    try:
        url = driver.url()
        kept = driver.run("return window.__helioxStepMark || null") == mark if mark else None
        session = f"the same session still answers ({url}); the page {'kept' if kept else 'lost'} its mark"
        answers = True
    except Exception as error:
        session = f"the session no longer answers ({type(error).__name__}: {str(error)[:200]})"
        answers = False
    alive = "the app is running" if after.get("heliox-os") else "the app is gone"
    return answers, f"{alive}; {session}; processes changed: {json.dumps(changed) if changed else 'none'}"


def end_apps_started_here() -> None:
    """End the copies of the app this check started (never one that was running before it)."""
    import psutil

    for process in psutil.process_iter(["pid", "name"]):
        name = (process.info.get("name") or "").casefold().removesuffix(".exe")
        if name == "heliox-os" and process.info.get("pid") not in APPS_BEFORE:
            try:
                process.kill()
            except psutil.Error:
                pass


def reopen(driver: Driver, app: str) -> bool:
    """A new session (and with it a new app) after a step left the old one unusable. The old app is ended first: the
    single-instance lock would hand a second one straight back to it."""
    driver.end()
    end_apps_started_here()
    time.sleep(2)
    try:
        driver.start(app, timeout=45)
    except Exception as error:
        print(f"DIAG no new session could be opened: {type(error).__name__}: {error}", flush=True)
        driver.session = ""
        return False
    ready = page_ready(driver, 30)
    print(f"DIAG a new session was opened, page ready {ready!r}", flush=True)
    return ready is True


def guarded(driver: Driver, app: str, name: str, body) -> None:
    """One step. A driver that hangs up during it is a FAIL line naming the step and what was left alive, not a
    traceback; the next step gets a new session when this one stopped answering."""
    left = STEPS_DEADLINE[0] - time.monotonic()
    if left < 40 or (not driver.session and left < 120):
        check(name, False, f"not run: {max(left, 0):.0f} s left of the check's time (the CI step ends it at 280 s)")
        return
    if not driver.session and not reopen(driver, app):
        check(name, False, "not run: the previous step left no session and no new one could be opened")
        return
    before = processes()
    mark = mark_page(driver)
    try:
        body(driver)
    except Exception as error:
        answers, state = aftermath(driver, mark, before)
        check(name, False, f"the driver connection failed during this step ({type(error).__name__}: {error}); {state}")
        if not answers:
            driver.end()


def async_route_control(driver: Driver) -> None:
    """The async script route with a plain rejected promise and no app command: if this fails, the route itself
    does not survive a rejection on this driver."""
    answer = driver.run_async(
        "const done = arguments[arguments.length - 1];"
        "Promise.reject(new Error('plain rejection')).then(() => done('resolved'), e => done('rejected: ' + e.message));"
    )
    check("async-route-survives-rejection", answer == "rejected: plain rejection", str(answer))


CLIPBOARD_READ = "window.__TAURI_INTERNALS__.invoke('plugin:clipboard-manager|read_text')"


def removed_permission_polled(driver: Driver) -> None:
    """The command the window no longer has, started by one sync script and its answer read by others, so no async
    script is waiting in the driver while the app refuses it."""
    driver.run(
        "window.__helioxClip = 'pending';"
        f"{CLIPBOARD_READ}.then(() => {{ window.__helioxClip = 'allowed'; }},"
        " e => { window.__helioxClip = 'refused: ' + String(e); }); return 1;"
    )

    def settled():
        value = driver.run("return window.__helioxClip")
        return value if value != "pending" else None

    answer = wait_until(settled, 15)
    if isinstance(answer, Exception):
        raise answer
    check("removed-permission-refused", str(answer).startswith("refused"), str(answer)[:300])


def removed_permission_async(driver: Driver) -> None:
    """The same refusal through the async script route, as the first run did: if only this fails, the driver's
    async route does not survive the app's refusal, and the app does not fall over."""
    answer = driver.run_async(
        f"const done = arguments[arguments.length - 1]; {CLIPBOARD_READ}"
        ".then(() => done('allowed'), e => done('refused: ' + String(e)));"
    )
    check("async-route-survives-refused-command", str(answer).startswith("refused"), str(answer)[:300])


def script_cannot_navigate(driver: Driver) -> None:
    driver.run("location.href = 'https://example.com/heliox-navigate'; return 1;")
    time.sleep(3)
    url = driver.url()
    check("script-cannot-navigate", on_app(url), url)


def link_click_not_followed(driver: Driver) -> None:
    driver.run(
        "const a = document.createElement('a'); a.id = 'heliox-link-test';"
        "a.href = 'https://example.org/heliox-link'; a.textContent = 'link';"
        "a.style = 'position:fixed;top:4px;left:4px;z-index:99999;padding:8px;background:#fff';"
        "document.body.append(a); return 1;"
    )
    driver.click("#heliox-link-test")
    time.sleep(3)
    url = driver.url()
    handed = ""
    browser_log = os.environ.get("HELIOX_BROWSER_LOG", "")
    if browser_log and Path(browser_log).exists():
        handed = Path(browser_log).read_text(encoding="utf-8", errors="replace")
    detail = f"{url}; handed to the browser: {handed.strip() or 'not observable here'}"
    ok = on_app(url) and (not browser_log or "example.org/heliox-link" in handed)
    check("link-click-not-followed", ok, detail)


def webassembly_compiles(driver: Driver) -> None:
    wasm = driver.run_async(
        "const done = arguments[arguments.length - 1];"
        "WebAssembly.instantiate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]))"
        ".then(() => fetch('/mediapipe/hands/hands_solution_simd_wasm_bin.wasm'))"
        ".then(r => done('compiled; mediapipe file ' + r.status), e => done('failed: ' + String(e)));"
    )
    check("webassembly-compiles", str(wasm).startswith("compiled; mediapipe file 200"), str(wasm))


# In this order: the plain async control first, then the refusal without the async route, then with it, so the
# lines together say whether a refused command brings down the app, the page, or only the driver's async route.
STEPS = (
    ("async-route-survives-rejection", async_route_control),
    ("removed-permission-refused", removed_permission_polled),
    ("async-route-survives-refused-command", removed_permission_async),
    ("script-cannot-navigate", script_cannot_navigate),
    ("link-click-not-followed", link_click_not_followed),
    ("webassembly-compiles", webassembly_compiles),
)
APPS_BEFORE: set[int] = set()  # app processes already running when the check began: never ended by it
STEPS_DEADLINE = [float("inf")]  # set in main: steps end by then, so the CI step's timeout never cuts a line


def main(argv: list[str]) -> int:
    app = str(Path(argv[0]).resolve())
    base = argv[argv.index("--driver") + 1] if "--driver" in argv else "http://127.0.0.1:4444"
    APPS_BEFORE.update(processes().get("heliox-os", []))
    STEPS_DEADLINE[0] = time.monotonic() + 245  # every step ends by then, leaving the clean-up time before 280 s
    daemon = start_daemon()
    driver = Driver(base)
    try:
        time.sleep(5)  # the daemon writes its token, then listens
        try:
            with WebViewWatch():
                driver.start(app)
        except Exception as error:
            check("session", False, f"{type(error).__name__}: {error}")
            if sys.platform == "win32":
                launch_without_driver(app)
            return failures
        # The WebDriver ports drive the app (they can run script in its window): nothing outside the runner may
        # reach them.
        listening = listening_on(DRIVER_PORTS)
        outside = [f"{ip}:{port}" for ip, port in listening if ip not in LOOPBACK]
        check(
            "driver-loopback-only",
            bool(listening) and not outside,
            f"listening on {sorted({f'{ip}:{port}' for ip, port in listening})}",
        )
        ready = page_ready(driver)
        url = driver.url()
        check("page-loaded", ready is True and on_app(url), f"{url}, ready {ready!r}")

        errors = driver.run("return window.__helioxErrors ? Array.from(window.__helioxErrors) : null")
        check("no-page-errors", errors == [], json.dumps(errors)[:800])

        def token_length():
            answer = driver.run_async(
                "const done = arguments[arguments.length - 1];"
                "window.__TAURI_INTERNALS__.invoke('get_auth_token')"
                ".then(t => done({ok: true, length: String(t).length}), e => done({ok: false, error: String(e)}));"
            )
            return answer if answer.get("ok") and answer.get("length", 0) > 0 else None

        token = wait_until(token_length, 60, every=2)
        check("command-answers", isinstance(token, dict) and token.get("ok"), repr(token))

        for name, body in STEPS:
            guarded(driver, app, name, body)
        if driver.session:
            try:
                errors = driver.run("return window.__helioxErrors ? Array.from(window.__helioxErrors) : null")
                print(f"DIAG page errors at the end: {json.dumps(errors)[:800]}", flush=True)
            except Exception as error:
                print(f"DIAG page errors at the end: unreadable ({type(error).__name__}: {error})", flush=True)
    finally:
        driver.end()
        daemon.terminate()
        try:
            daemon.wait(timeout=15)
        except subprocess.TimeoutExpired:
            daemon.kill()
        end_everything(daemon.pid)
    return failures


DRIVER_PORTS = (4444, 4445)  # tauri-driver, and the native driver it starts behind it
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
STARTED_HERE = ("heliox-os", "tauri-driver", "webkitwebdriver", "msedgedriver")


def listening_on(ports: tuple[int, ...]) -> list[tuple[str, int]]:
    import psutil

    return [
        (str(c.laddr.ip), int(c.laddr.port))
        for c in psutil.net_connections(kind="inet")
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port in ports
    ]


def end_everything(daemon_pid: int) -> None:
    """End the app, the drivers and the daemon, and say in a CHECK line what was still running when asked to end."""
    import psutil

    def ours() -> list:
        found = []
        for process in psutil.process_iter(["pid", "name"]):
            name = (process.info.get("name") or "").casefold().removesuffix(".exe")
            if name in STARTED_HERE or process.info.get("pid") == daemon_pid:
                found.append(process)
        return found

    for process in ours():
        try:
            process.terminate()
        except psutil.Error:
            pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ours():
        time.sleep(0.5)
    left = ours()
    names = sorted(f"{p.info.get('name')} ({p.pid})" for p in left)
    for process in left:
        try:
            process.kill()
        except psutil.Error:
            pass
    check("everything-ended", not left, f"still running after 10 s, now killed: {names}" if left else "")


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("starts a daemon and the desktop app, and drives the app's window")
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as error:  # every exit path prints a CHECK line; the steps above have their own
        check("check-itself", False, f"stopped before the steps: {type(error).__name__}: {error}")
        sys.exit(max(failures, 1))

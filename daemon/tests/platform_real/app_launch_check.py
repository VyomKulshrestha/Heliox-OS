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
- ``removed-permission-refused``: a plugin command the window no longer has (clipboard read) is refused;
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
Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
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

    def start(self, app: str) -> None:
        value = self.call(
            "POST", "/session", {"capabilities": {"alwaysMatch": {"tauri:options": {"application": app}}}}, timeout=120
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
            except Exception as error:
                print(f"DIAG ending the session: {error}", flush=True)


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


def start_daemon() -> subprocess.Popen:
    log = open(Path(os.environ["RUNNER_TEMP"]) / "app-launch-daemon.log", "w", encoding="utf-8")  # noqa: SIM115
    daemon = Path(__file__).resolve().parents[2]
    return subprocess.Popen([sys.executable, "-m", "pilot.server"], cwd=daemon, stdout=log, stderr=log)


def main(argv: list[str]) -> int:
    app = str(Path(argv[0]).resolve())
    base = argv[argv.index("--driver") + 1] if "--driver" in argv else "http://127.0.0.1:4444"
    daemon = start_daemon()
    driver = Driver(base)
    try:
        time.sleep(5)  # the daemon writes its token, then listens
        try:
            driver.start(app)
        except Exception as error:
            check("session", False, f"{type(error).__name__}: {error}")
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
        ready = wait_until(
            lambda: driver.run("return document.readyState === 'complete' && !!document.querySelector('main')"), 60
        )
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

        refused = driver.run_async(
            "const done = arguments[arguments.length - 1];"
            "window.__TAURI_INTERNALS__.invoke('plugin:clipboard-manager|read_text')"
            ".then(() => done('allowed'), e => done('refused: ' + String(e)));"
        )
        check("removed-permission-refused", str(refused).startswith("refused"), str(refused)[:300])

        driver.run("location.href = 'https://example.com/heliox-navigate'; return 1;")
        time.sleep(3)
        url = driver.url()
        check("script-cannot-navigate", on_app(url), url)

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

        wasm = driver.run_async(
            "const done = arguments[arguments.length - 1];"
            "WebAssembly.instantiate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]))"
            ".then(() => fetch('/mediapipe/hands/hands_solution_simd_wasm_bin.wasm'))"
            ".then(r => done('compiled; mediapipe file ' + r.status), e => done('failed: ' + String(e)));"
        )
        check("webassembly-compiles", str(wasm).startswith("compiled; mediapipe file 200"), str(wasm))
        errors = driver.run("return window.__helioxErrors ? Array.from(window.__helioxErrors) : null")
        print(f"DIAG page errors at the end: {json.dumps(errors)[:800]}", flush=True)
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
    sys.exit(main(sys.argv[1:]))

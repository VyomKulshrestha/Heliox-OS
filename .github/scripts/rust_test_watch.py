"""Build and run the desktop app's Rust tests so that a test that never ends names itself in the CI log.

A job cancelled at its own time limit keeps no log, so a stalled "cargo test" used to say nothing at all. This
script runs the two halves with limits of their own, well under the job's:

  build  cargo test --no-run, with the compile's own output; the test executables are written to a list file.
  run    each test executable from that list, from the crate's folder as cargo runs it, with output not captured.
         Each test is printed as it starts and as it ends (with its time), from libtest's JSON events: the test
         process alone gets RUSTC_BOOTSTRAP=1 for that, so the compile is not touched. If that option is refused,
         the run falls back to libtest's usual lines.

When a limit passes, or when the tests have finished but the process has not ended, the script prints the tests
started and not finished, the process tree under it, and every thread's stack of the test process (sample or lldb
on macOS, gdb or /proc on Linux), then ends the tree and exits with code 124.

The build and the tests run in a session of their own. On GitHub's macOS runner the runner's agent,
Runner.Listener, Runner.Worker and the step's shell share one process group, so a signal or a stopped process in
the tests' group reached the runner itself: in November 2026 a test that stopped processes there froze the runner,
and the job ended with no log, past every limit.

Usage:
  python3 rust_test_watch.py build --crate DIR --list FILE --limit SECONDS [cargo test arguments...]
  python3 rust_test_watch.py run --crate DIR --list FILE --limit SECONDS [--rounds N] [test arguments...]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

STALLED = 124
# After libtest's summary the process should end at once; this long without it ending is a stall of its own.
AFTER_SUMMARY_SECONDS = 30.0
STARTED_AT = time.monotonic()


def say(line: str) -> None:
    print(f"[{time.monotonic() - STARTED_AT:7.1f}s] {line}", flush=True)


def tool(command: list[str], seconds: float) -> str:
    """Run a diagnostic tool and return what it printed; never raises, never waits past `seconds`."""
    try:
        done = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=seconds,
        )
        return done.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as error:
        return f"({command[0]} could not run: {error})"


def process_table() -> list[tuple[int, int, str]]:
    """(pid, parent pid, the rest of the ps line) for every process; empty where ps is missing."""
    if os.name == "nt":
        return []
    text = tool(["ps", "-A", "-o", "pid=,ppid=,pgid=,stat=,etime=,command="], 30)
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), line.strip()))
    return rows


def descendants(root: int) -> list[tuple[int, int, str]]:
    rows = process_table()
    found, queue = [], [root]
    while queue:
        parent = queue.pop(0)
        for row in rows:
            if row[1] == parent and all(row[0] != seen[0] for seen in found):
                found.append(row)
                queue.append(row[0])
    return found


def thread_stacks(pid: int) -> str:
    """Every thread's stack of `pid`, by whatever this system has."""
    system = platform.system()
    if system == "Darwin":
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "sample.txt"
            printed = tool(["sample", str(pid), "3", "-mayDie", "-file", str(report)], 90)
            if report.exists():
                return report.read_text(encoding="utf-8", errors="replace")
        lldb = tool(
            [
                "lldb",
                "--batch",
                "-p",
                str(pid),
                "-o",
                "thread backtrace all",
                "-o",
                "process detach",
            ],
            120,
        )
        return f"sample printed:\n{printed}\nlldb printed:\n{lldb}"
    if system == "Linux":
        lines = []
        for task in sorted(Path(f"/proc/{pid}/task").glob("*")):
            name = (
                (task / "comm").read_text(errors="replace").strip()
                if (task / "comm").exists()
                else "?"
            )
            wchan = (
                (task / "wchan").read_text(errors="replace").strip()
                if (task / "wchan").exists()
                else "?"
            )
            lines.append(f"thread {task.name} {name!r} waiting in {wchan}")
        if shutil.which("gdb"):
            prefix = ["sudo", "-n"] if shutil.which("sudo") else []
            lines.append(
                tool([*prefix, "gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt"], 120)
            )
        return "\n".join(lines)
    return "(no stack tool on this system)"


def end_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        tool(["taskkill", "/F", "/T", "/PID", str(process.pid)], 30)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        for pid, _parent, _line in descendants(process.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        say("the process did not end after SIGKILL")


def start(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # A session and process group of its own: the whole tree can be ended, and nothing in it reaches the
        # runner's group (see the top of this file).
        options["start_new_session"] = True
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **options,
    )


def report_stall(
    process: subprocess.Popen, why: str, unfinished: list[str], stack_of: int | None
) -> None:
    say(f"STALL: {why}")
    if unfinished:
        say("Started and not finished:")
        for name in unfinished:
            say(f"    {name}")
    tree = descendants(process.pid)
    say(
        f"Process tree under {process.pid} ({process.args[0] if isinstance(process.args, list) else process.args}):"
    )
    for _pid, _parent, line in tree:
        say(f"    {line}")
    targets = [stack_of] if stack_of else []
    for pid, _parent, line in tree:
        if (
            pid not in targets
            and len(targets) < 4
            and ("rustc" in line or " ld" in line or "heliox" in line)
        ):
            targets.append(pid)
    for pid in targets:
        say(f"Every thread's stack of process {pid}:")
        print(thread_stacks(pid), flush=True)
    say("Ending the process tree.")
    end_tree(process)


class Reader(threading.Thread):
    """Reads the child's merged output line by line and hands each line to `on_line`."""

    def __init__(self, process: subprocess.Popen, on_line) -> None:
        super().__init__(daemon=True)
        self.process = process
        self.on_line = on_line

    def run(self) -> None:
        assert self.process.stdout is not None
        for raw in iter(self.process.stdout.readline, b""):
            self.on_line(raw.decode("utf-8", "replace").rstrip("\r\n"))


def build(args: argparse.Namespace) -> int:
    command = ["cargo", "test", *args.rest, "--no-run", "--message-format=json-render-diagnostics"]
    say("$ " + " ".join(command))
    executables: list[str] = []

    def on_line(line: str) -> None:
        if line.startswith("{"):
            try:
                message = json.loads(line)
            except ValueError:
                print(line, flush=True)
                return
            if message.get("reason") == "compiler-artifact" and message.get("executable"):
                if (message.get("profile") or {}).get("test"):
                    executables.append(message["executable"])
            return
        print(line, flush=True)

    process = start(command, args.crate)
    reader = Reader(process, on_line)
    reader.start()
    try:
        code = process.wait(timeout=args.limit)
    except subprocess.TimeoutExpired:
        report_stall(process, f"the test build did not finish within {args.limit:.0f} s", [], None)
        return STALLED
    reader.join(timeout=30)
    if code != 0:
        say(f"the test build failed with exit code {code}")
        return code
    Path(args.list).write_text("\n".join(executables) + "\n", encoding="utf-8")
    say(f"{len(executables)} test executable(s): {', '.join(executables)}")
    return 0


class Suite:
    """What one test executable has said so far."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running: dict[str, float] = {}
        self.finished = 0
        self.summary_at: float | None = None
        self.json_seen = False
        self.refused = False


def run_one(executable: str, args: argparse.Namespace, use_json: bool) -> tuple[int, Suite]:
    suite = Suite()
    command = [executable, "--nocapture", *args.rest]
    env = dict(os.environ)
    if use_json:
        command += ["-Z", "unstable-options", "--format", "json", "--report-time"]
        env["RUSTC_BOOTSTRAP"] = "1"
    say("$ " + " ".join(command))

    def on_line(line: str) -> None:
        event = None
        if use_json and line.startswith("{"):
            try:
                event = json.loads(line)
            except ValueError:
                event = None
        if not isinstance(event, dict) or "type" not in event:
            if (
                "only accepted on the nightly" in line
                or "unstable-options" in line
                and "error" in line
            ):
                suite.refused = True
            if line.startswith("test result:"):
                with suite.lock:
                    suite.summary_at = time.monotonic()
            print(line, flush=True)
            return
        suite.json_seen = True
        kind, what, name = event.get("type"), event.get("event"), event.get("name", "")
        with suite.lock:
            if kind == "suite" and what == "started":
                say(f"running {event.get('test_count')} tests")
            elif kind == "suite":
                suite.summary_at = time.monotonic()
                say(
                    f"test result: {what}. {event.get('passed')} passed; {event.get('failed')} failed; "
                    f"{event.get('ignored')} ignored; finished in {event.get('exec_time', 0):.2f}s"
                )
            elif kind == "test" and what == "started":
                suite.running[name] = time.monotonic()
                say(f"START {name}")
            elif kind == "test" and what == "timeout":
                say(f"STILL RUNNING after 60 s: {name}")
            elif kind == "test":
                suite.running.pop(name, None)
                suite.finished += 1
                took = event.get("exec_time")
                detail = f" ({took:.2f} s)" if isinstance(took, (int, float)) else ""
                extra = f": {event['message']}" if event.get("message") else ""
                say(f"{what.upper() if what == 'failed' else what} {name}{detail}{extra}")
                if event.get("stdout"):
                    print(event["stdout"], flush=True)

    process = start(command, args.crate, env)
    reader = Reader(process, on_line)
    reader.start()
    deadline = time.monotonic() + args.limit
    while True:
        try:
            code = process.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        with suite.lock:
            summary_at = suite.summary_at
            unfinished = [
                f"{name} (for {now - began:.0f} s)" for name, began in suite.running.items()
            ]
        if summary_at is not None and now - summary_at > AFTER_SUMMARY_SECONDS:
            why = f"the tests finished but the process did not end within {AFTER_SUMMARY_SECONDS:.0f} s"
        elif now > deadline:
            why = f"the tests did not finish within {args.limit:.0f} s"
        else:
            continue
        report_stall(process, why, unfinished, process.pid)
        reader.join(timeout=30)
        return STALLED, suite
    reader.join(timeout=30)
    return code, suite


def run(args: argparse.Namespace) -> int:
    executables = [
        line for line in Path(args.list).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not executables:
        say("no test executables listed")
        return 1
    for round_number in range(1, args.rounds + 1):
        if args.rounds > 1:
            say(f"round {round_number} of {args.rounds}")
        for executable in executables:
            code, suite = run_one(executable, args, use_json=True)
            if code != 0 and suite.refused and not suite.json_seen:
                say("JSON events were refused; running again with the usual output")
                code, suite = run_one(executable, args, use_json=False)
            if code != 0:
                if code != STALLED:
                    say(f"{executable} exited with code {code}")
                return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["build", "run"])
    parser.add_argument("--crate", type=Path, required=True, help="the folder of Cargo.toml")
    parser.add_argument("--list", required=True, help="file of test executables, one per line")
    parser.add_argument(
        "--limit", type=float, required=True, help="seconds before a build or a run is a stall"
    )
    parser.add_argument(
        "--rounds", type=int, default=1, help="run the tests this many times (run only)"
    )
    args, rest = parser.parse_known_args()
    args.rest = [part for part in rest if part != "--"]
    args.crate = args.crate.resolve()
    return build(args) if args.mode == "build" else run(args)


if __name__ == "__main__":
    sys.exit(main())

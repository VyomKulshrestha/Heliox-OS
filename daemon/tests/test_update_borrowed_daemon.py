"""An update that finds a daemon still running: if it is this Heliox's (it accepts this installation's token), the
update asks it to stop through its own RPC and waits, bounded; otherwise it says exactly what holds the port and how
to stop it on this system. It never ends a process it did not start."""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
from pathlib import Path

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import storage_health

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "heliox_update_borrowed", ROOT / "tauri-app" / "src-tauri" / "setup" / "update.py"
)
update = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(update)


# ── what holds the port, read with each system's own tools ──

WINDOWS_NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1128
  TCP    127.0.0.1:8785         0.0.0.0:0              LISTENING       40212
  TCP    127.0.0.1:8785         127.0.0.1:51220        ESTABLISHED     40212
"""
WINDOWS_TASKLIST = '"python.exe","40212","Console","1","412,332 K"'
MAC_LSOF = """COMMAND   PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
Python  51234 vyom    9u  IPv4 0x1d2c3b4a5e6f7081      0t0  TCP 127.0.0.1:8785 (LISTEN)
"""
LINUX_SS = 'LISTEN 0 100 127.0.0.1:8785 0.0.0.0:* users:(("python3",pid=7781,fd=9))'


def test_the_program_holding_the_port_is_read_on_every_system():
    assert update.parse_port_holder(WINDOWS_NETSTAT + WINDOWS_TASKLIST, 8785, "win32") == ("python.exe", "40212")
    assert update.parse_port_holder(WINDOWS_NETSTAT, 8785, "win32") == ("an unknown program", "40212")
    assert update.parse_port_holder(MAC_LSOF, 8785, "darwin") == ("Python", "51234")
    assert update.parse_port_holder(LINUX_SS, 8785, "linux") == ("python3", "7781")
    assert update.parse_port_holder("", 8785, "linux") is None


def test_the_hint_says_what_holds_the_port_and_how_to_stop_it_on_this_system():
    holder = ("python.exe", "40212")
    assert update.how_to_stop(holder, "win32") == (
        "Port 8785 is held by python.exe (process 40212). Open Task Manager (Ctrl+Shift+Esc), go to Details, select "
        "process 40212 and choose End task. Then start the update again."
    )
    assert "Activity Monitor" in update.how_to_stop(("Python", "51234"), "darwin")
    assert "kill 7781" in update.how_to_stop(("python3", "7781"), "linux")
    assert update.how_to_stop(None, "linux").startswith("Something on this computer is using port 8785")


# ── the decision ──


def _make_way(outcome: str, *, freed: bool = True, running: bool = True):
    calls: list[str] = []

    def ask(python, port):
        calls.append("ask")
        return {"outcome": outcome, "detail": ""}

    def wait(port):
        calls.append("wait")
        return freed

    def holder(port):
        calls.append("holder")
        return ("python.exe", "40212")

    result = update.make_way(Path("python"), ask=ask, wait=wait, holder=holder, running=lambda _port: running)
    return result, calls


def test_nothing_on_the_port_goes_straight_on():
    assert _make_way("stopping", running=False) == ((True, ""), [])


def test_heliox_s_own_daemon_is_asked_to_stop_and_the_update_waits_for_the_port():
    assert _make_way("stopping") == ((True, ""), ["ask", "wait"])


def test_a_daemon_that_says_it_stops_but_keeps_the_port_is_named_not_killed():
    (go, hint), calls = _make_way("stopping", freed=False)
    assert not go and calls == ["ask", "wait", "holder"]
    assert hint.startswith("Heliox's background service said it would stop but was still running after 30 s.")
    assert "process 40212" in hint


@pytest.mark.parametrize(
    ("outcome", "starts"),
    [
        ("cannot_stop", "Heliox's background service is from an older version that cannot be asked to stop."),
        ("not_heliox", "Port 8785 is held by python.exe (process 40212)."),
        ("no_answer", "Port 8785 is held by python.exe (process 40212)."),
    ],
)
def test_anything_else_on_the_port_gets_what_holds_it_and_how_to_stop_it(outcome, starts):
    (go, hint), calls = _make_way(outcome)
    assert not go and hint.startswith(starts) and hint.endswith("Then start the update again.")
    assert calls == ["ask", "holder"]  # asked, then named: never ended


# ── the helper against a real daemon ──


def _helper() -> dict:
    namespace: dict = {"__name__": "stop_helper"}
    exec(compile(update.STOP_HELPER, "stop_helper", "exec"), namespace)  # noqa: S102 - the helper's own source
    return namespace


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime)
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: None)
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "this-installation"
    server = PilotServer(config)

    async def initialize():
        server._handlers = {}

    server.initialize = initialize
    return server, runtime


async def test_the_helper_stops_heliox_with_this_installation_s_token(daemon):
    server, _runtime = daemon
    await server.start()
    try:
        port = server._server.sockets[0].getsockname()[1]
        outcome = await _helper()["ask"](port, timeout=10.0)
        assert outcome == {"outcome": "stopping"}
        await asyncio.wait_for(server.stop_requested.wait(), 10)  # main() then stops it as on SIGTERM
    finally:
        await server.stop()


async def test_a_daemon_that_refuses_this_installation_s_token_is_not_heliox(daemon):
    server, runtime = daemon
    await server.start()
    try:
        port = server._server.sockets[0].getsockname()[1]
        (runtime / "auth_token").write_text("another-installation", encoding="utf-8")
        outcome = await _helper()["ask"](port, timeout=10.0)
        assert outcome["outcome"] == "not_heliox" and "token" in outcome["detail"]
        assert not server.stop_requested.is_set()
    finally:
        await server.stop()


async def test_something_that_is_not_heliox_on_the_port_is_left_alone(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "auth_token").write_text("t", encoding="utf-8")
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime)

    async def not_websocket(reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    other = await asyncio.start_server(not_websocket, "127.0.0.1", 0, family=socket.AF_INET)
    try:
        port = other.sockets[0].getsockname()[1]
        outcome = await _helper()["ask"](port, timeout=5.0)
        assert outcome["outcome"] == "not_heliox"
    finally:
        other.close()
        await other.wait_closed()


def test_the_helper_runs_as_a_script_with_the_environment_s_python(tmp_path):
    """update.py runs it with ``python -c``; nothing listens on this port, so it reports and exits."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    code, output = update.run([sys.executable, "-c", update.STOP_HELPER, str(port)], echo=False)
    assert code == 0 and '"outcome": "not_heliox"' in output.strip().splitlines()[-1]

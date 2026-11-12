"""A second copy of the daemon (the app started twice, or a daemon already running from a checkout) stops at once with
one plain sentence, before it writes the tokens or touches the first copy's data."""

import asyncio
import json
import os
import socket
from unittest.mock import AsyncMock

import pytest
import websockets

from pilot import server as server_module
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import startup, storage_health
from pilot.system.startup import AlreadyRunningError, is_address_in_use

LIMIT = 30.0  # failure limit only


def test_address_in_use_on_every_system():
    for code in (98, 48, 10048):
        assert is_address_in_use(OSError(code, "in use"))
    assert not is_address_in_use(OSError(13, "permission denied"))


async def test_a_taken_port_stops_the_second_copy_before_it_touches_anything(monkeypatch, tmp_path):
    first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    first.bind(("127.0.0.1", 0))
    first.listen()
    port = first.getsockname()[1]
    runtime = tmp_path / "runtime"
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime)
    probed: list = []
    monkeypatch.setattr(storage_health, "preflight", lambda folder: probed.append(folder))
    config = PilotConfig()
    config.server.host, config.server.port = "127.0.0.1", port
    server = PilotServer(config)
    server.initialize = AsyncMock()
    try:
        with pytest.raises(AlreadyRunningError) as raised:
            await server.start()
    finally:
        first.close()

    assert f"Port {port} is already in use" in str(raised.value)
    assert "left the running one alone" in str(raised.value)
    assert probed == [] and not runtime.exists()  # no data-folder probe, no token files
    server.initialize.assert_not_awaited()  # no store was opened


def _quick_server(monkeypatch, runtime, port, token):
    """A PilotServer whose heavy initialize() is replaced; tokens go to ``runtime``, never the real files."""
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime)
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: None)
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", port, token
    server = PilotServer(config)

    async def initialize():
        server._handlers = {}

    server.initialize = initialize
    return server


async def _auth(port, token):
    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": token}, "id": "a"}))
        while True:
            reply = json.loads(await asyncio.wait_for(ws.recv(), LIMIT))
            if reply.get("id") == "a":
                return reply


def _free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


async def test_a_client_that_connects_the_moment_the_port_accepts_reads_this_runs_token(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("auth_token", "neural_auth_token", "mcp_auth_token"):
        (runtime / name).write_text("token-from-the-last-run", encoding="utf-8")  # left on disk by the last run
    server = _quick_server(monkeypatch, runtime, 0, "this-runs-token")
    steps: list = []
    real_write, real_serve = server._write_token_files, startup.serve_claimed

    def preflight(_folder):
        steps.append(("preflight", server._server))  # nothing serves yet
        return None

    def write_tokens():
        steps.append(("tokens", server._server))
        real_write()

    async def serve(sockets, *args, **kwargs):
        group = await real_serve(sockets, *args, **kwargs)
        # The first instant the port accepts: this is the file a connecting client reads.
        steps.append(("accepting", tuple((runtime / name).read_text("utf-8") for name in ("auth_token",))))
        return group

    monkeypatch.setattr(storage_health, "preflight", preflight)
    monkeypatch.setattr(server, "_write_token_files", write_tokens)
    monkeypatch.setattr(startup, "serve_claimed", serve)
    await server.start()
    try:
        assert steps == [("preflight", None), ("tokens", None), ("accepting", ("this-runs-token",))]
        assert (runtime / "neural_auth_token").read_text("utf-8") == server._neural_auth_token
        assert (runtime / "mcp_auth_token").read_text("utf-8") == server._mcp_auth_token
        port = server._server.sockets[0].getsockname()[1]
        assert "result" in await _auth(port, (runtime / "auth_token").read_text("utf-8"))
    finally:
        await server.stop()


async def test_a_second_copy_leaves_the_first_copys_token_files_byte_identical(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    first = _quick_server(monkeypatch, runtime, 0, "first-copy-token")
    await first.start()
    try:
        port = first._server.sockets[0].getsockname()[1]
        before = {path.name: path.read_bytes() for path in runtime.iterdir()}
        second = _quick_server(monkeypatch, runtime, port, "second-copy-token")
        second.initialize = AsyncMock()
        with pytest.raises(AlreadyRunningError):
            await second.start()
        second.initialize.assert_not_awaited()
        assert {path.name: path.read_bytes() for path in runtime.iterdir()} == before
        assert "result" in await _auth(port, "first-copy-token")  # the first copy still answers
    finally:
        await first.stop()


async def test_restart_on_the_same_port_straight_after_stop(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    first = _quick_server(monkeypatch, runtime, 0, "token-one")
    await first.start()
    port = first._server.sockets[0].getsockname()[1]
    assert "result" in await _auth(port, "token-one")  # a connection the server closes leaves TIME_WAIT behind
    await first.stop()

    again = _quick_server(monkeypatch, runtime, port, "token-two")
    await again.start()
    try:
        assert "result" in await _auth(port, "token-two")
    finally:
        await again.stop()


async def test_a_failure_between_claim_and_serve_releases_the_port(monkeypatch, tmp_path):
    port = _free_port()
    server = _quick_server(monkeypatch, tmp_path / "runtime", port, "t")

    def broken(_folder):
        raise OSError(5, "the data folder could not be probed")

    monkeypatch.setattr(storage_health, "preflight", broken)
    with pytest.raises(OSError, match="could not be probed"):
        await server.start()
    assert server._server is None
    for sock in startup.claim_address("127.0.0.1", port):  # free again at once, in the same process
        sock.close()


def test_the_claim_binds_without_listening_and_with_the_right_option_for_each_system():
    claimed = startup.claim_address("127.0.0.1", 0)
    try:
        (sock,) = claimed
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE)
            assert not sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        else:
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        with pytest.raises(AlreadyRunningError):  # claimed, not yet listening, is already taken
            startup.claim_address("127.0.0.1", sock.getsockname()[1])
    finally:
        for sock in claimed:
            sock.close()


def test_every_address_the_host_resolves_to_is_claimed_on_one_port():
    resolved = {info[4][0] for info in socket.getaddrinfo("localhost", 0, type=socket.SOCK_STREAM)}
    claimed = startup.claim_address("localhost", 0)
    try:
        assert {sock.getsockname()[0] for sock in claimed} == resolved
        assert len({sock.getsockname()[1] for sock in claimed}) == 1
    finally:
        for sock in claimed:
            sock.close()


@pytest.mark.skipif(os.name != "posix", reason="Windows refuses the second claim itself (exclusive bind)")
async def test_linux_two_copies_in_the_same_instant_the_lock_stops_the_second_before_its_tokens(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    port = _free_port()
    first = _quick_server(monkeypatch, runtime, port, "first-copy-token")
    await first.start()
    try:
        before = {path.name: path.read_bytes() for path in runtime.iterdir()}
        # Linux lets two SO_REUSEADDR binds share an address while neither listens: let the second claim "succeed".
        monkeypatch.setattr(startup, "claim_address", lambda _host, _port: [])
        second = _quick_server(monkeypatch, runtime, port, "second-copy-token")
        with pytest.raises(AlreadyRunningError):
            await second.start()
        assert {path.name: path.read_bytes() for path in runtime.iterdir()} == before
        assert "result" in await _auth(port, "first-copy-token")
    finally:
        await first.stop()
    startup.hold_port_lock(runtime, port).close()  # stop() released it


def test_the_entry_point_says_one_sentence_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(server_module, "ensure_dirs", lambda: None)
    monkeypatch.setattr(server_module, "_setup_logging", lambda: None)
    monkeypatch.setattr(server_module.PilotConfig, "load", classmethod(lambda cls: PilotConfig()))
    monkeypatch.setattr(server_module.sys, "argv", ["pilot.server"])

    async def taken(self):
        raise AlreadyRunningError(8785)

    monkeypatch.setattr(PilotServer, "start", taken)

    with pytest.raises(SystemExit) as exited:
        server_module.main()

    assert exited.value.code == 3
    assert capsys.readouterr().err.strip().startswith("Heliox: Port 8785 is already in use")

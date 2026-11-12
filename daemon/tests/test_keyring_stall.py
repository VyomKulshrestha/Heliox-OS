"""A credential store whose get and set never answer (a macOS keychain waiting on a dialog nobody sees) must never
stop Heliox: start reaches ready, requests are answered, the learner's key comes from the key file, nothing is
written to the keyring, and stop is bounded."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import keyring
import pytest
import websockets

from pilot.config import PilotConfig
from pilot.security import keyring_guard
from pilot.security.vault import KeyVault, VaultUnavailableError
from pilot.server import PilotServer
from pilot.system import storage_health

LIMIT = 30.0  # failure limit only


@pytest.fixture
def blocked_keyring(monkeypatch):
    """get_password and set_password block until the test ends, as a keychain waiting on a hidden dialog does."""
    forever = threading.Event()
    calls: list[str] = []

    def block(name):
        def blocked(*_args):
            calls.append(name)
            forever.wait()

        return blocked

    monkeypatch.setattr(keyring, "get_password", block("get"))
    monkeypatch.setattr(keyring, "set_password", block("set"))
    monkeypatch.setattr(keyring, "delete_password", block("delete"))
    monkeypatch.setattr(keyring_guard, "TIMEOUT_SECONDS", 0.3)
    keyring_guard.reset()
    yield calls
    forever.set()  # let the stuck daemon threads finish
    keyring_guard.reset()


def test_a_call_that_never_answers_is_given_up_on_and_the_next_ones_fail_at_once(blocked_keyring):
    started = time.monotonic()
    with pytest.raises(keyring_guard.KeyringStalled):
        keyring_guard.get_password("svc", "acct")
    with pytest.raises(keyring_guard.KeyringStalled):
        keyring_guard.set_password("svc", "acct", "secret")
    assert time.monotonic() - started < LIMIT
    assert blocked_keyring == ["get"]  # the second call did not start another stuck thread
    status = keyring_guard.status()
    assert status["state"] == "stalled" and "did not answer within 0.3 s" in status["message"]
    assert "answer it or unlock the keychain, then restart Heliox" in status["message"]


async def test_the_vault_never_holds_the_event_loop(blocked_keyring):
    vault = KeyVault(PilotConfig())
    vault._keyring_available = True
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    try:
        with pytest.raises(VaultUnavailableError):
            await asyncio.wait_for(vault.get_key("openai"), LIMIT)
        with pytest.raises(VaultUnavailableError):
            await asyncio.wait_for(vault.store_key("openai", "sk-test"), LIMIT)
    finally:
        ticker.cancel()
    assert ticks > 5  # the loop kept running while the keychain did not answer


class _Ledger:
    def subscribe(self, _callback):
        return None

    async def last_sequence(self):
        return 0

    async def list_events(self, **_filters):
        return []

    async def close(self):
        return None


async def test_a_daemon_with_a_blocked_keychain_starts_answers_and_stops(blocked_keyring, monkeypatch, tmp_path):
    from pilot.server import LEARNER_SNAPSHOT_KEY_FILE
    from pilot.system.late_start import LearnerLoading

    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("pilot.server.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: None)
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "test-token"
    server = PilotServer(config)
    vault = KeyVault(config)
    vault._keyring_available = True  # a real keychain, which then never answers

    async def initialize():
        server._handlers = {}
        server._vault = vault
        server._experience_ledger = _Ledger()
        server._online_learning = LearnerLoading()

    server.initialize = initialize
    from pilot.security.risk_gate import get_risk_gate

    gate_learner_before = get_risk_gate()._online_learner
    await asyncio.wait_for(server.start(), LIMIT)
    try:
        port = server._server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}", open_timeout=LIMIT) as ws:  # the handshake completes
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": 1}))
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 2}))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), LIMIT))
                if reply.get("id") == 2:
                    assert reply["result"]["pong"] is True
                    break
        late = server._late_parts()
        for _ in range(int(LIMIT / 0.05)):
            if late.snapshot().get("online_learning") == "ready":
                break
            await asyncio.sleep(0.05)
        assert late.snapshot()["online_learning"] == "ready"  # the learner loaded, keyed by the file
        assert (tmp_path / "runtime" / LEARNER_SNAPSHOT_KEY_FILE).exists()
        assert "set" not in blocked_keyring  # nothing was ever written to the keychain
    finally:
        started = time.monotonic()
        await asyncio.wait_for(server.stop(), LIMIT)
        assert time.monotonic() - started < 15  # bounded, though a keychain thread is still stuck
        get_risk_gate().set_online_learner(gate_learner_before)

"""No credential-store call runs on an event loop's own thread: each place that reads a key from the system keyring
does it in a worker thread, and the guard counts (and logs, once) any call that does not."""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import keyring
import pytest

from pilot.config import PilotConfig
from pilot.security import keyring_guard
from pilot.security.vault import KeyVault

SERVER = Path(__file__).resolve().parents[1] / "pilot" / "server.py"


@pytest.fixture
def quick_keyring(monkeypatch):
    stored: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, account: stored.get((service, account)))
    monkeypatch.setattr(
        keyring, "set_password", lambda service, account, secret: stored.update({(service, account): secret})
    )
    keyring_guard.reset()
    yield stored
    keyring_guard.reset()


async def test_a_call_made_on_the_event_loop_is_counted(quick_keyring):
    keyring_guard.get_password("svc", "acct")  # on the loop's own thread: what must never happen
    assert keyring_guard.loop_calls() == 1
    await asyncio.to_thread(keyring_guard.get_password, "svc", "acct")
    assert keyring_guard.loop_calls() == 1  # a worker thread is fine


async def test_the_vault_reads_and_writes_off_the_loop(quick_keyring):
    vault = KeyVault(PilotConfig())
    vault._keyring_available = True
    await vault.store_key("openai", "sk-test")
    vault.clear_cache()
    assert await vault.get_key("openai") == "sk-test"
    assert keyring_guard.loop_calls() == 0


async def test_the_neural_sidecar_starts_its_recording_off_the_loop():
    from pilot.neural.rpc_client import NeurodBridge

    started_on: list[threading.Thread] = []

    class _Service:
        def start(self):
            started_on.append(threading.current_thread())

        def stop(self):
            return None

    bridge = NeurodBridge.__new__(NeurodBridge)
    bridge._service = _Service()

    async def no_session(stop):
        stop.set()

    bridge._session = no_session
    stop = asyncio.Event()
    try:
        await asyncio.wait_for(bridge.run(stop), 10)
    except Exception:
        pass  # only the thread the recording was opened on matters here
    assert started_on and started_on[0] is not threading.main_thread()


def test_the_daemon_decrypts_the_visual_model_off_the_loop():
    """configure_ui_jepa_artifact decrypts with the recorded-workflows key; the server only calls it in a thread."""
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    direct = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "configure_ui_jepa_artifact"
    ]
    threaded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "configure_ui_jepa_artifact"
    ]
    assert direct == [] and len(threaded) == 1

"""Phone task requests over Air Handoff: encrypted, signed, opt-in, and never approved by the phone."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.actions import READ_ONLY_ACTIONS, ActionType
from pilot.air_handoff import AirHandoffError, AirHandoffManager, AirHandoffServer, _b64u, _unb64u
from pilot.config import PilotConfig
from pilot.server import PHONE_READ_ONLY_ACTIONS
from tests.test_air_handoff import FakeVault, pair_device, signed_headers


def _envelope(secret: bytes, text: str, aad: bytes = b"request-v1") -> dict[str, str]:
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(secret).encrypt(nonce, json.dumps({"text": text}).encode(), aad)
    return {"nonce": _b64u(nonce), "ciphertext": _b64u(ciphertext)}


async def _manager(tmp_path: Path, *, accepting: bool = True, on_request=None):
    manager = AirHandoffManager(FakeVault(), data_dir=tmp_path)
    manager.set_request_handler(accepting=lambda: accepting, on_request=on_request)
    credential, secret = await pair_device(manager)
    return manager, credential, secret


@pytest.mark.asyncio
async def test_requests_are_refused_unless_turned_on(tmp_path):
    manager, credential, secret = await _manager(tmp_path, accepting=False)

    with pytest.raises(AirHandoffError, match="not accepting"):
        await manager.submit_request(credential["device_id"], _envelope(secret, "how full is my disk"))


@pytest.mark.asyncio
async def test_a_request_is_decrypted_queued_and_reported(tmp_path):
    seen = []

    async def on_request(request):
        seen.append(request)

    manager, credential, secret = await _manager(tmp_path, on_request=on_request)

    request = await manager.submit_request(credential["device_id"], _envelope(secret, "  how full   is my disk "))

    assert request.text == "how full is my disk" and request.status == "waiting"
    assert seen == [request]
    assert manager.requests_for(credential["device_id"])[0]["text"] == "how full is my disk"


@pytest.mark.asyncio
async def test_tampered_wrong_purpose_and_oversized_requests_are_refused(tmp_path):
    manager, credential, secret = await _manager(tmp_path)
    device = credential["device_id"]

    with pytest.raises(AirHandoffError, match="decrypted"):
        await manager.submit_request(device, _envelope(secret, "hi", aad=b"pending-v1"))
    with pytest.raises(AirHandoffError, match="decrypted"):
        await manager.submit_request(device, _envelope(b"\x00" * 32, "hi"))
    with pytest.raises(AirHandoffError, match="limited to 500"):
        await manager.submit_request(device, _envelope(secret, "x" * 501))


@pytest.mark.asyncio
async def test_open_requests_per_phone_are_capped(tmp_path):
    manager, credential, secret = await _manager(tmp_path)

    for index in range(5):
        await manager.submit_request(credential["device_id"], _envelope(secret, f"request {index}"))
    with pytest.raises(AirHandoffError, match="already has requests waiting"):
        await manager.submit_request(credential["device_id"], _envelope(secret, "one more"))


@pytest.mark.asyncio
async def test_answers_go_to_the_phone_without_touching_the_held_draft(tmp_path):
    manager, credential, _secret = await _manager(tmp_path)
    await manager.grab_text("held for later")

    transfer = await manager.send_text(credential["device_id"], "Disk is 61% full.")

    pending = await manager.pending_for(credential["device_id"])
    assert [item["transfer_id"] for item in pending] == [transfer["transfer_id"]]
    _transfer, payload = await manager.transfer_bytes(credential["device_id"], transfer["transfer_id"])
    assert payload == b"Disk is 61% full."
    assert (await manager.status())["draft"] is not None


@pytest.mark.asyncio
async def test_requests_over_http_are_signed_and_encrypted(tmp_path):
    manager, credential, secret = await _manager(tmp_path)
    server = AirHandoffServer(manager, host="127.0.0.1", port=0)
    await server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        body = json.dumps(_envelope(secret, "what is my cpu usage")).encode()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base}/api/requests",
                data=body,
                headers=signed_headers(credential, secret, method="POST", path="/api/requests", body=body),
            ) as response:
                assert response.status == 200
                assert (await response.json())["status"] == "received"
            async with session.post(f"{base}/api/requests", data=body) as unsigned:
                assert unsigned.status == 403
            async with session.get(
                f"{base}/api/requests",
                headers=signed_headers(credential, secret, method="GET", path="/api/requests", nonce="n2"),
            ) as listed:
                envelope = await listed.json()
        clear = AESGCM(secret).decrypt(_unb64u(envelope["nonce"]), _unb64u(envelope["ciphertext"]), b"requests-v1")
        state = json.loads(clear)
        assert state["accepting"] is True
        assert state["requests"][0]["text"] == "what is my cpu usage"
    finally:
        await server.stop()


def test_auto_run_list_is_read_only_and_leaves_out_private_reads():
    assert {ActionType(name) for name in PHONE_READ_ONLY_ACTIONS} <= READ_ONLY_ACTIONS
    for private in ("file_read", "clipboard_read", "email_fetch", "screenshot", "screen_ocr", "env_get", "open_url"):
        assert private not in PHONE_READ_ONLY_ACTIONS


def _server_with_request(tmp_path, *, run_read_only: bool):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.air_handoff.enabled = True
    config.air_handoff.accept_requests = True
    config.air_handoff.run_read_only_requests = run_read_only
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._autonomous = SimpleNamespace(submit=AsyncMock(return_value=SimpleNamespace(job_id="job-9")))
    return server


@pytest.mark.asyncio
async def test_read_only_requests_start_at_once_with_the_restricted_list(tmp_path):
    server = _server_with_request(tmp_path, run_read_only=True)
    manager, credential, secret = await _manager(tmp_path, on_request=server._on_phone_request)
    server._air_handoff_manager = manager

    request = await manager.submit_request(credential["device_id"], _envelope(secret, "how full is my disk"))

    kwargs = server._autonomous.submit.await_args.kwargs
    assert kwargs["allowed_actions"] == list(PHONE_READ_ONLY_ACTIONS)
    assert kwargs["source"] == "phone"
    assert request.status == "running" and request.job_id == "job-9"

    await server._finish_phone_request({"job_id": "job-9", "status": "success", "result_summary": "Disk is 61% full."})
    assert request.status == "done"
    pending = await manager.pending_for(credential["device_id"])
    _transfer, payload = await manager.transfer_bytes(credential["device_id"], pending[0]["transfer_id"])
    assert b"Disk is 61% full." in payload


@pytest.mark.asyncio
async def test_other_requests_wait_for_run_on_the_computer(tmp_path):
    server = _server_with_request(tmp_path, run_read_only=False)
    manager, credential, secret = await _manager(tmp_path, on_request=server._on_phone_request)
    server._air_handoff_manager = manager

    request = await manager.submit_request(credential["device_id"], _envelope(secret, "clean my downloads"))
    server._autonomous.submit.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[0] == "phone_request"

    result = await server._handle_phone_request_run({"request_id": request.request_id}, MagicMock())

    assert result["status"] == "ok"
    assert server._autonomous.submit.await_args.kwargs["allowed_actions"] is None
    again = await server._handle_phone_request_run({"request_id": request.request_id}, MagicMock())
    assert again["status"] == "error"


@pytest.mark.asyncio
async def test_configure_needs_requests_on_before_auto_run(tmp_path):
    server = _server_with_request(tmp_path, run_read_only=False)
    server._air_handoff_manager = AirHandoffManager(FakeVault(), data_dir=tmp_path)
    server._air_handoff_server = SimpleNamespace(running=False, base_url="", port=8787)

    refused = await server._handle_air_handoff_requests_configure(
        {"accept_requests": False, "run_read_only_requests": True}, MagicMock()
    )
    assert refused["status"] == "error"
    done = await server._handle_air_handoff_requests_configure(
        {"accept_requests": True, "run_read_only_requests": False}, MagicMock()
    )
    assert done["status"] == "ok" and server.config.air_handoff.accept_requests is True

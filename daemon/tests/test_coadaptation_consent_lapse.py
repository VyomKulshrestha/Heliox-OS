"""Consent expiry has the effect of a revoke in PilotServer (spec 10.7): with no active consent left, the feature's
``co_adaptation.enabled`` setting goes off. Runs a real CoAdaptationService on tmp_path with a fake clock, wired to the
server the way initialize() wires it."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.coadaptation.keys import InMemoryKeyProvider
from pilot.coadaptation.service import CoAdaptationService
from pilot.config import PilotConfig
from pilot.server import PilotServer
from tests.test_coadaptation_service import FakeClock, grant_params

CONFIG_UPDATED = ("config_updated", {"method": "co_adaptation_config_update", "section": "co_adaptation"})


class _Daemon:
    def __init__(self, tmp_path) -> None:
        config = PilotConfig()
        config.save = MagicMock()
        self.server = PilotServer(config)
        self.server._broadcast_notification = AsyncMock()
        self.tmp_path = tmp_path
        self.clock = FakeClock()
        self.keys = InMemoryKeyProvider()

    def new_service(self) -> CoAdaptationService:
        return CoAdaptationService(
            settings_provider=lambda: self.server.config.co_adaptation,
            gaze_tracking_enabled=lambda: False,
            db_path=self.tmp_path / "co_adaptation.db",
            preferences_path=self.tmp_path / "co_adaptation_preferences.json",
            exports_dir=self.tmp_path / "co_adaptation_exports",
            key_provider=self.keys,
            broadcast_fn=self.server._co_adaptation_notify,
            clock=self.clock,
        )

    async def start(self) -> CoAdaptationService:
        service = self.new_service()
        await service.start()
        self.server._co_adaptation = service
        return service

    async def grant(self, **params: Any) -> dict:
        enable = params.pop("enable", True)
        reply = await self.server._handle_co_adaptation_consent_grant(
            {**grant_params(**params), "enable": enable}, None
        )
        assert reply["status"] == "ok", reply
        return reply

    def broadcasts(self) -> list[tuple[str, dict]]:
        return [call.args for call in self.server._broadcast_notification.await_args_list]

    async def close(self) -> None:
        lapse = self.server._co_adaptation_lapse_task
        if lapse is not None:
            await lapse
        if self.server._co_adaptation is not None:
            await self.server._co_adaptation.close()


@pytest.fixture
async def daemon(tmp_path):
    instance = _Daemon(tmp_path)
    try:
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_the_last_consent_expiring_turns_the_feature_off_at_the_next_status(daemon):
    service = await daemon.start()
    await daemon.grant(retention_days=1)
    assert daemon.server.config.co_adaptation.enabled is True
    saves = daemon.server.config.save.call_count

    daemon.clock.advance(days=1, seconds=1)
    status = await daemon.server._handle_co_adaptation_status({}, None)

    # Before the fix the setting stayed on, and Settings showed the feature On with no consenting participant.
    assert status["participants"] == []
    assert service.has_active_consent() is False
    assert status["config"]["enabled"] is False
    assert daemon.server.config.co_adaptation.enabled is False
    assert daemon.server.config.save.call_count == saves + 1
    reasons = [params.get("reason") for method, params in daemon.broadcasts() if method == "co_adaptation_updated"]
    assert "expired" in reasons
    assert daemon.broadcasts()[-1] == CONFIG_UPDATED


@pytest.mark.asyncio
async def test_a_consent_that_expired_while_the_daemon_was_off_turns_the_feature_off_at_start(daemon):
    service = await daemon.start()
    await daemon.grant(retention_days=1)
    await service.close()
    daemon.server._co_adaptation = None
    daemon.clock.advance(days=2)

    # What initialize() does: start() prunes the expired consent, then the server settles the setting.
    restarted = daemon.new_service()
    await restarted.start()
    daemon.server._co_adaptation = restarted
    await daemon.server._settle_co_adaptation_consent()

    assert restarted.has_active_consent() is False
    assert daemon.server.config.co_adaptation.enabled is False


@pytest.mark.asyncio
async def test_an_expiry_leaves_the_feature_on_while_another_consent_is_active(daemon):
    await daemon.start()
    await daemon.grant(retention_days=1)
    await daemon.grant(retention_days=90)
    saves = daemon.server.config.save.call_count

    daemon.clock.advance(days=1, seconds=1)
    status = await daemon.server._handle_co_adaptation_status({}, None)

    assert len(status["participants"]) == 1
    assert status["config"]["enabled"] is True
    assert daemon.server.config.co_adaptation.enabled is True
    assert daemon.server.config.save.call_count == saves
    assert CONFIG_UPDATED not in daemon.broadcasts()


@pytest.mark.asyncio
async def test_a_grant_with_enable_false_after_an_unpruned_expiry_keeps_the_feature_off(daemon):
    await daemon.start()
    await daemon.grant(retention_days=1)
    daemon.clock.advance(days=1, seconds=1)

    # No status call in between, so nothing has pruned the expired consent yet.
    reply = await daemon.grant(retention_days=30, enable=False)

    assert reply["config"]["enabled"] is False
    assert daemon.server.config.co_adaptation.enabled is False


class _RevokeFailsPartWay:
    """A service whose revoke withdrew the consent but could not finish erasing."""

    def __init__(self) -> None:
        self.active = True

    async def revoke_consent(self, params: dict) -> dict:
        self.active = False
        return {"status": "error", "message": "Some data could not be erased yet."}

    def has_active_consent(self) -> bool:
        return self.active

    async def on_config_changed(self, previous: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_a_revoke_that_failed_part_way_still_turns_the_feature_off(tmp_path):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._co_adaptation = _RevokeFailsPartWay()
    server.config.co_adaptation.enabled = True

    reply = await server._handle_co_adaptation_consent_revoke({"participant_key": "p-3fa29c1e"}, None)

    assert reply["status"] == "error"
    assert server.config.co_adaptation.enabled is False
    config.save.assert_called_once()

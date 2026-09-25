"""Tests for the Agent Gateway's observability/policy RPC handlers in server.py.

PilotServer.__init__ is lightweight (just attribute setup, no network/model
init — the real work happens in start()), so these construct a bare
PilotServer and only wire the two attributes the handlers actually touch:
self.config and self._gateway_audit.
"""

import pytest

from pilot.config import PilotConfig
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, SourceProfile
from pilot.security.gateway_audit import AgentGatewayAuditStore
from pilot.server import PilotServer


def _server(tmp_path) -> PilotServer:
    config = PilotConfig()
    server = PilotServer(config)
    server._gateway_audit = AgentGatewayAuditStore(
        db_file=tmp_path / "agent_gateway_audit.db",
        key_file=tmp_path / "agent_gateway_audit.key",
        key=b"0" * 32,
    )
    return server


class TestListGatewayEvents:
    @pytest.mark.asyncio
    async def test_returns_error_when_store_not_initialized(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_list_gateway_events({}, ws=None)
        assert result["status"] == "error"
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_returns_recorded_events(self, tmp_path):
        server = _server(tmp_path)
        await server._gateway_audit.record_event(
            plan_id="plan-1",
            action_index=0,
            action_type="browser_execute_js",
            action_family="browsing",
            target="page",
            source_profile="autonomous",
            permission_tier="DESTRUCTIVE",
            override_applied=False,
            override_restricted=False,
            decision="denied",
        )
        result = await server._handle_list_gateway_events({}, ws=None)
        assert result["status"] == "ok"
        assert len(result["events"]) == 1
        assert result["events"][0]["decision"] == "denied"

    @pytest.mark.asyncio
    async def test_filters_by_source_profile(self, tmp_path):
        server = _server(tmp_path)
        for source in ("autonomous", "interactive"):
            await server._gateway_audit.record_event(
                plan_id="plan-1",
                action_index=0,
                action_type="shell_command",
                action_family="shell",
                target="ls",
                source_profile=source,
                permission_tier="SYSTEM_MODIFY",
                override_applied=False,
                override_restricted=False,
                decision="allowed",
            )
        result = await server._handle_list_gateway_events({"source_profile": "autonomous"}, ws=None)
        assert len(result["events"]) == 1
        assert result["events"][0]["source_profile"] == "autonomous"


class TestVerifyGatewayAudit:
    @pytest.mark.asyncio
    async def test_returns_error_when_store_not_initialized(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_verify_gateway_audit({}, ws=None)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_valid_chain_on_empty_store(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_verify_gateway_audit({}, ws=None)
        assert result["status"] == "ok"
        assert result["valid"] is True
        assert result["checked_entries"] == 0


class TestGatewayPolicyGetAndUpdate:
    @pytest.mark.asyncio
    async def test_get_returns_all_default_profiles(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_gateway_policy_get({}, ws=None)
        assert result["status"] == "ok"
        assert result["enabled"] is True
        assert set(result["profiles"]) == set(DEFAULT_SOURCE_PROFILES)

    @pytest.mark.asyncio
    async def test_update_unknown_profile_errors(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_gateway_policy_update({"profile": "not_a_real_profile"}, ws=None)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_update_narrows_autonomous_shell_ceiling(self, tmp_path, monkeypatch):
        # Avoid writing to the real XDG config location during the test.
        server = PilotServer(PilotConfig())
        monkeypatch.setattr(server.config, "save", lambda: None)

        result = await server._handle_gateway_policy_update(
            {"profile": "autonomous", "max_tier": {"shell": 0}},
            ws=None,
        )
        assert result["status"] == "ok"
        assert result["policy"]["max_tier"]["shell"] == 0
        # Untouched families keep their prior values.
        assert result["policy"]["max_tier"]["browsing"] == 2

        updated = server.config.gateway.source_profiles["autonomous"]
        assert isinstance(updated, SourceProfile)
        assert updated.max_tier["shell"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"allow_root": "false"}, "allow_root must be a boolean"),
            ({"max_tier": {"shell": "2"}}, "must be an integer"),
            ({"max_tier": {"shell": 5}}, "must be an integer"),
            ({"max_tier": {"unknown": 0}}, "Unknown action family"),
            ({"deny_action_types": "shell_command"}, "must be an array"),
            ({"deny_action_types": ["not_an_action"]}, "Unknown denied action type"),
        ],
    )
    async def test_update_rejects_ambiguous_or_out_of_range_policy(self, monkeypatch, update, message):
        server = PilotServer(PilotConfig())
        monkeypatch.setattr(server.config, "save", lambda: None)
        before = server.config.gateway.source_profiles["autonomous"]

        result = await server._handle_gateway_policy_update(
            {"profile": "autonomous", **update},
            ws=None,
        )

        assert result["status"] == "error"
        assert message in result["message"]
        assert server.config.gateway.source_profiles["autonomous"] == before


class TestRiskGateStatusAndUpdate:
    @pytest.mark.asyncio
    async def test_status_exposes_loaded_shipped_model(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_risk_gate_status({}, ws=None)

        assert result["status"] == "ok"
        assert result["enabled"] is True
        assert result["weights_loaded"] is True
        assert result["model_version"] == "risk-mlp-v4-heldout"
        assert result["calibrated"] is True
        assert result["validation_samples"] == 5_400
        assert result["training_samples"] == 25_200
        assert result["learnable_action_types"]

    @pytest.mark.asyncio
    async def test_update_persists_enabled_state(self, monkeypatch):
        server = PilotServer(PilotConfig())
        saved = False

        def mark_saved():
            nonlocal saved
            saved = True

        monkeypatch.setattr(server.config, "save", mark_saved)
        result = await server._handle_risk_gate_config_update({"enabled": False}, ws=None)

        assert result["enabled"] is False
        assert server.config.gateway.risk_gate_enabled is False
        assert saved is True

    @pytest.mark.asyncio
    async def test_update_rejects_truthy_string_without_mutation(self, monkeypatch):
        server = PilotServer(PilotConfig())
        monkeypatch.setattr(server.config, "save", lambda: pytest.fail("invalid config was saved"))

        result = await server._handle_risk_gate_config_update({"enabled": "false"}, ws=None)

        assert result == {"status": "error", "message": "enabled must be a boolean"}
        assert server.config.gateway.risk_gate_enabled is True

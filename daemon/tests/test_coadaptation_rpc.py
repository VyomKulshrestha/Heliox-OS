"""Co-adaptation RPC wiring in PilotServer, against a fake service (the server is never initialized)."""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.config import CoAdaptationConfig, PilotConfig
from pilot.security.rpc_identity import MCP_LOCAL_METHODS, NEURAL_SIDECAR_METHODS
from pilot.server import (
    ATTENTION_DEFERRABLE_NOTIFICATIONS,
    CO_ADAPTATION_RPC_METHODS,
    CONFIG_MUTATING_RPC_METHODS,
    NEURAL_ARMED_STATES,
    OUT_OF_BAND_RPC_METHODS,
    SETTINGS_RPC_METHODS,
    PilotServer,
)

SERVER_SOURCE = Path(__file__).resolve().parents[1] / "pilot" / "server.py"

CO_ADAPTATION_HANDLERS = {
    "co_adaptation_status": "status",
    "co_adaptation_config_update": None,
    "co_adaptation_consent_grant": "grant_consent",
    "co_adaptation_consent_revoke": "revoke_consent",
    "co_adaptation_preference_set": "set_preference",
    "co_adaptation_preference_retract": "retract_preference",
    "co_adaptation_learning_reset": "reset_learning",
    "co_adaptation_erase": "erase",
    "co_adaptation_results": "results",
    "co_adaptation_export": "export",
    "co_adaptation_session_begin": "begin_session",
    "co_adaptation_session_resume": "resume_session",
    "co_adaptation_recommend": "recommend",
    "co_adaptation_record_trial": "record_trial",
    "co_adaptation_report_burden": "report_burden",
    "co_adaptation_session_end": "end_session",
}
SERVICE_RPC_METHODS = sorted(method for method in CO_ADAPTATION_HANDLERS.values() if method is not None)
STUDY_CALLS = [
    ("co_adaptation_session_begin", {"participant_key": "p-3fa29c1e", "context": "study"}),
    ("co_adaptation_session_resume", {"session_id": "s-1"}),
    ("co_adaptation_recommend", {"session_id": "s-1", "level": "block_regime", "context": {}}),
    ("co_adaptation_recommend", {"session_id": "s-1", "level": "ambiguity_table", "context": {}}),
    ("co_adaptation_record_trial", {"session_id": "s-1", "trial": {"trial_id": "t-1"}}),
]


class FakeService:
    """Implements the CoAdaptationService surface the server uses (spec 13.3) and records every call."""

    def __init__(self, *, active_consent: bool = True, active_after_revoke: bool = False) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.active_consent = active_consent
        self.active_after_revoke = active_after_revoke
        self.replies: dict[str, dict[str, Any]] = {}
        for method in SERVICE_RPC_METHODS:
            setattr(self, method, self._rpc(method))

    def _rpc(self, method: str):
        async def call(params: dict) -> dict:
            self.calls.append((method, params))
            if method == "revoke_consent":
                self.active_consent = self.active_after_revoke
            return copy.deepcopy(self.replies.get(method, {"status": "ok", "method": method}))

        return call

    def has_active_consent(self) -> bool:
        self.calls.append(("has_active_consent", None))
        return self.active_consent

    async def on_config_changed(self, previous: CoAdaptationConfig) -> None:
        self.calls.append(("on_config_changed", copy.deepcopy(previous)))

    async def start(self) -> None:
        self.calls.append(("start", None))

    async def close(self) -> None:
        self.calls.append(("close", None))

    def called(self, method: str) -> list[Any]:
        return [params for name, params in self.calls if name == method]


class FakeRecordings:
    def __init__(self, active: bool) -> None:
        self.active = active

    def status(self) -> dict[str, Any]:
        return {"active": self.active, "name": "Recording"}


class FakeNeural:
    def __init__(self, state: str) -> None:
        self.state = state
        self.status_calls = 0

    async def gate_state(self) -> str:
        return self.state

    async def status(self) -> dict[str, Any]:  # must never be used by the study check (it verifies the audit chain)
        self.status_calls += 1
        raise AssertionError("status() verifies the whole audit chain")


def _server(service: FakeService | None = None, *, save_error: Exception | None = None) -> PilotServer:
    config = PilotConfig()
    if save_error is None:
        config.save = MagicMock()
    else:
        config.save = MagicMock(side_effect=save_error)
    server = PilotServer(config)
    server._co_adaptation = service
    return server


async def _call(server: PilotServer, method: str, params: dict | None = None) -> dict:
    return await getattr(server, f"_handle_{method}")(params if params is not None else {}, None)


def _registered_methods() -> set[str]:
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([a-z][a-z0-9_]*)":\s*self\._handle_[a-z0-9_]+,?$', source, re.MULTILINE))


# ── Registration and dispatch sets ──


def test_all_sixteen_handlers_are_registered_as_one_line_entries():
    registered = {method for method in _registered_methods() if method.startswith("co_adaptation_")}
    assert registered == set(CO_ADAPTATION_HANDLERS)
    for method in CO_ADAPTATION_HANDLERS:
        assert callable(getattr(PilotServer, f"_handle_{method}"))


def test_registered_total_is_244():
    tree = ast.parse(SERVER_SOURCE.read_text(encoding="utf-8"))
    counts = [
        len(node.value.keys)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(target, ast.Attribute) and target.attr == "_handlers" for target in node.targets)
    ]
    assert counts == [244]


def test_no_approval_rpc_exists_for_co_adaptation():
    assert "co_adaptation_record_approval" not in _registered_methods()
    assert not hasattr(PilotServer, "_handle_co_adaptation_record_approval")
    assert not any("approval" in method for method in CO_ADAPTATION_HANDLERS)


def test_dispatch_set_memberships():
    assert {
        "co_adaptation_config_update",
        "co_adaptation_consent_grant",
        "co_adaptation_consent_revoke",
    } == {method for method in CONFIG_MUTATING_RPC_METHODS if method.startswith("co_adaptation_")}
    assert {
        "co_adaptation_status",
        "co_adaptation_preference_set",
        "co_adaptation_preference_retract",
        "co_adaptation_learning_reset",
        "co_adaptation_erase",
        "co_adaptation_results",
        "co_adaptation_config_update",
        "co_adaptation_consent_grant",
        "co_adaptation_consent_revoke",
    } == {method for method in SETTINGS_RPC_METHODS if method.startswith("co_adaptation_")}
    assert {
        "co_adaptation_session_begin",
        "co_adaptation_session_resume",
        "co_adaptation_recommend",
        "co_adaptation_record_trial",
        "co_adaptation_report_burden",
        "co_adaptation_session_end",
        "co_adaptation_export",
    } == CO_ADAPTATION_RPC_METHODS
    # Every co-adaptation RPC runs out-of-band, in exactly one of the two sets.
    assert SETTINGS_RPC_METHODS.isdisjoint(CO_ADAPTATION_RPC_METHODS)
    assert set(CO_ADAPTATION_HANDLERS) == {
        method for method in SETTINGS_RPC_METHODS | CO_ADAPTATION_RPC_METHODS if method.startswith("co_adaptation_")
    }
    assert not any(method.startswith("co_adaptation_") for method in OUT_OF_BAND_RPC_METHODS)


def test_no_co_adaptation_method_reaches_the_sidecar_or_mcp_allowlists():
    for allowlist in (NEURAL_SIDECAR_METHODS, MCP_LOCAL_METHODS):
        assert not any(method.startswith("co_adaptation") for method in allowlist)
    assert not any(method.startswith("co_adaptation") for method in ATTENTION_DEFERRABLE_NOTIFICATIONS)
    assert "recorded_workflow_status_changed" not in ATTENTION_DEFERRABLE_NOTIFICATIONS


def test_out_of_band_condition_includes_the_study_rpcs():
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    assert "or request.method in CO_ADAPTATION_RPC_METHODS" in source


# ── update_config and config_update ──


@pytest.mark.asyncio
async def test_generic_update_config_refuses_the_section():
    service = FakeService()
    server = _server(service)

    result = await server._handle_update_config({"section": "co_adaptation", "values": {"enabled": True}}, None)

    assert result == {
        "status": "error",
        "message": "Use co_adaptation_config_update to change co_adaptation settings",
    }
    assert server.config.co_adaptation.enabled is False
    server.config.save.assert_not_called()
    assert service.calls == []


@pytest.mark.asyncio
async def test_config_update_applies_saves_and_tells_the_service():
    service = FakeService()
    server = _server(service)

    result = await _call(
        server,
        "co_adaptation_config_update",
        {"enabled": True, "ambiguity_threshold": 0.2, "burden_prompt_every_blocks": 3, "exploration_floor": 0},
    )

    assert result["status"] == "ok"
    assert result["restart_required"] is False
    assert result["config"]["enabled"] is True
    assert result["config"]["ambiguity_threshold"] == 0.2
    assert result["config"]["burden_prompt_every_blocks"] == 3
    assert result["config"]["exploration_floor"] == 0.0
    assert set(result["config"]) == set(CoAdaptationConfig.__dataclass_fields__)
    server.config.save.assert_called_once()
    previous = service.called("on_config_changed")
    assert len(previous) == 1
    assert previous[0] == CoAdaptationConfig()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"ambiguity_treshold": 0.2}, "Unknown co-adaptation setting: ambiguity_treshold"),
        ({"approval_presentation": "large"}, "Unknown co-adaptation setting: approval_presentation"),
        ({"enabled": "true"}, "enabled must be a boolean"),
        ({"adaptive_enabled": 1}, "adaptive_enabled must be a boolean"),
        ({"ambiguity_threshold": "0.2"}, "ambiguity_threshold must be numeric"),
        ({"ambiguity_threshold": True}, "ambiguity_threshold must be numeric"),
        ({"ambiguity_threshold": 0.3}, "ambiguity_threshold must be at most 0.25"),
        ({"ambiguity_threshold": 0.01}, "ambiguity_threshold must be at least 0.05"),
        ({"unintended_activation_tolerance": 0.25}, "unintended_activation_tolerance must be at most 0.2"),
        ({"noninferiority_margin": -0.01}, "noninferiority_margin must be at least 0"),
        ({"upper_bound_credibility": 0.999}, "upper_bound_credibility must be at most 0.99"),
        ({"exploration_floor": 0.5}, "exploration_floor must be at most 0.3"),
        ({"switching_penalty": float("nan")}, "switching_penalty must be finite"),
        ({"burden_prompt_every_blocks": 0}, "burden_prompt_every_blocks must be at least 1"),
        ({"burden_prompt_every_blocks": 2.0}, "burden_prompt_every_blocks must be an integer"),
        ({"day_boundary_hour": 24}, "day_boundary_hour must be at most 23"),
        ({"day_boundary_hour": False}, "day_boundary_hour must be an integer"),
    ],
)
@pytest.mark.asyncio
async def test_config_update_validation_rejects_without_changing_anything(params, message):
    service = FakeService()
    server = _server(service)

    result = await _call(server, "co_adaptation_config_update", {"exploration_floor": 0.2, **params})

    assert result == {"status": "error", "message": message}
    assert server.config.co_adaptation == CoAdaptationConfig()
    server.config.save.assert_not_called()
    assert service.called("on_config_changed") == []


@pytest.mark.asyncio
async def test_turning_on_requires_an_active_consent():
    service = FakeService(active_consent=False)
    server = _server(service)

    result = await _call(server, "co_adaptation_config_update", {"enabled": True})

    assert result == {"status": "error", "message": "Grant consent before turning co-adaptation on."}
    assert server.config.co_adaptation.enabled is False
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_turning_on_without_a_service_is_refused_but_other_settings_save():
    server = _server(None)

    refused = await _call(server, "co_adaptation_config_update", {"enabled": True})
    saved = await _call(server, "co_adaptation_config_update", {"ambiguity_threshold": 0.1})

    assert refused == {"status": "error", "message": "Grant consent before turning co-adaptation on."}
    assert saved["status"] == "ok"
    assert server.config.co_adaptation.enabled is False
    assert server.config.co_adaptation.ambiguity_threshold == 0.1


@pytest.mark.asyncio
async def test_turning_off_needs_no_consent_and_reaches_the_service():
    service = FakeService(active_consent=False)
    server = _server(service)
    server.config.co_adaptation.enabled = True

    result = await _call(server, "co_adaptation_config_update", {"enabled": False})

    assert result["status"] == "ok"
    assert server.config.co_adaptation.enabled is False
    assert service.called("on_config_changed")[0].enabled is True


@pytest.mark.asyncio
async def test_config_update_rolls_back_when_saving_fails():
    service = FakeService()
    server = _server(service, save_error=OSError("disk full"))

    result = await _call(server, "co_adaptation_config_update", {"enabled": True, "ambiguity_threshold": 0.1})

    assert result == {"status": "error", "message": "Co adaptation settings were not saved: disk full"}
    assert server.config.co_adaptation == CoAdaptationConfig()
    assert service.called("on_config_changed") == []


# ── Consent grant and revoke ──


@pytest.mark.asyncio
async def test_consent_grant_turns_the_feature_on():
    service = FakeService()
    service.replies["grant_consent"] = {"status": "ok", "participant_key": "p-3fa29c1e", "consent": {}, "removed": {}}
    server = _server(service)
    params = {"participant_key": None, "purposes": ["selection_study"], "authorized": True}

    result = await _call(server, "co_adaptation_consent_grant", params)

    assert result["status"] == "ok"
    assert result["participant_key"] == "p-3fa29c1e"
    assert result["config"]["enabled"] is True
    assert server.config.co_adaptation.enabled is True
    assert service.called("grant_consent") == [params]
    server.config.save.assert_called_once()


@pytest.mark.asyncio
async def test_consent_grant_with_enable_false_leaves_the_feature_off():
    service = FakeService()
    server = _server(service)

    result = await _call(server, "co_adaptation_consent_grant", {"authorized": True, "enable": False})

    assert result["status"] == "ok"
    assert result["config"]["enabled"] is False
    assert server.config.co_adaptation.enabled is False
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_consent_grant_rejects_a_non_boolean_enable_before_the_service_runs():
    service = FakeService()
    server = _server(service)

    result = await _call(server, "co_adaptation_consent_grant", {"authorized": True, "enable": "yes"})

    assert result == {"status": "error", "message": "enable must be a boolean"}
    assert service.called("grant_consent") == []


@pytest.mark.asyncio
async def test_a_refused_grant_changes_no_setting():
    service = FakeService()
    service.replies["grant_consent"] = {
        "status": "error",
        "message": "Secure OS key storage is unavailable; co-adaptation stays off.",
    }
    server = _server(service)

    result = await _call(server, "co_adaptation_consent_grant", {"authorized": True})

    assert result == service.replies["grant_consent"]
    assert server.config.co_adaptation.enabled is False
    server.config.save.assert_not_called()


@pytest.mark.asyncio
async def test_consent_grant_rolls_back_the_flag_when_saving_fails():
    service = FakeService()
    server = _server(service, save_error=OSError("read-only"))

    result = await _call(server, "co_adaptation_consent_grant", {"authorized": True})

    assert result["status"] == "error"
    assert "not saved" in result["message"]
    assert server.config.co_adaptation.enabled is False


@pytest.mark.asyncio
async def test_consent_revoke_turns_the_feature_off_when_no_consent_remains():
    service = FakeService(active_after_revoke=False)
    service.replies["revoke_consent"] = {
        "status": "ok",
        "erased": {"rows": 12, "key_destroyed": True, "export_files": 0, "rules_deleted": 1},
    }
    server = _server(service)
    server.config.co_adaptation.enabled = True

    result = await _call(server, "co_adaptation_consent_revoke", {"participant_key": "p-3fa29c1e"})

    assert result["status"] == "ok"
    assert result["erased"]["rows"] == 12
    assert result["config"]["enabled"] is False
    assert server.config.co_adaptation.enabled is False
    assert service.called("revoke_consent") == [{"participant_key": "p-3fa29c1e"}]
    assert service.called("on_config_changed")[0].enabled is True


@pytest.mark.asyncio
async def test_consent_revoke_keeps_the_feature_on_while_another_consent_is_active():
    service = FakeService(active_after_revoke=True)
    server = _server(service)
    server.config.co_adaptation.enabled = True

    result = await _call(server, "co_adaptation_consent_revoke", {"participant_key": "p-3fa29c1e"})

    assert result["config"]["enabled"] is True
    assert server.config.co_adaptation.enabled is True
    server.config.save.assert_not_called()


# ── Service unavailable and delegation ──


@pytest.mark.asyncio
async def test_status_without_a_service_reports_unavailable():
    server = _server(None)

    result = await _call(server, "co_adaptation_status")

    assert result == {
        "status": "ok",
        "available": False,
        "config": {name: getattr(CoAdaptationConfig(), name) for name in CoAdaptationConfig.__dataclass_fields__},
        "message": "Co-adaptation is unavailable on this daemon.",
        "authority": "presentation_only",
        "execution_authority": False,
    }


@pytest.mark.parametrize(
    "method", [method for method, target in CO_ADAPTATION_HANDLERS.items() if target not in {None, "status"}]
)
@pytest.mark.asyncio
async def test_every_other_rpc_without_a_service_returns_the_unavailable_error(method):
    server = _server(None)

    result = await _call(server, method, {"level": "offer"})

    assert result == {"status": "error", "message": "Co-adaptation is unavailable on this daemon."}
    assert server.config.co_adaptation.enabled is False


@pytest.mark.parametrize(
    ("method", "target"), [(method, target) for method, target in CO_ADAPTATION_HANDLERS.items() if target]
)
@pytest.mark.asyncio
async def test_handlers_delegate_their_params_to_the_service(method, target):
    service = FakeService()
    server = _server(service)
    params = {"session_id": "s-1", "participant_key": "p-3fa29c1e", "level": "offer"}

    result = await _call(server, method, params)

    assert result["status"] == "ok"
    assert service.called(target) == [params]


@pytest.mark.asyncio
async def test_non_dict_params_reach_the_service_as_an_empty_dict():
    service = FakeService()
    server = _server(service)

    await server._handle_co_adaptation_results(None, None)  # type: ignore[arg-type]

    assert service.called("results") == [{}]


# ── Study start refusals ──


@pytest.mark.asyncio
async def test_refusal_check_is_none_when_nothing_is_wired():
    server = _server(FakeService())
    assert server._recorded_workflows is None
    assert server._neural_controller is None
    assert server._interactive_request_active is False

    assert await server._co_adaptation_study_refusal() is None


@pytest.mark.asyncio
async def test_refusal_check_names_each_reason():
    server = _server(FakeService())
    server._recorded_workflows = FakeRecordings(active=True)
    assert await server._co_adaptation_study_refusal() == "recording_active"

    server._recorded_workflows = FakeRecordings(active=False)
    server._interactive_request_active = True
    assert await server._co_adaptation_study_refusal() == "task_running"

    server._interactive_request_active = False
    for state in sorted(NEURAL_ARMED_STATES):
        server._neural_controller = FakeNeural(state)
        assert await server._co_adaptation_study_refusal() == "neural_armed", state
    for state in ("disconnected", "connected_uncalibrated", "calibrating", "observe_only", "abstained"):
        server._neural_controller = FakeNeural(state)
        assert await server._co_adaptation_study_refusal() is None, state
        assert server._neural_controller.status_calls == 0


def test_armed_states_are_exactly_the_specified_ones():
    assert {
        "armed_safe_ui",
        "armed_safe_desktop",
        "candidate_intent",
        "previewed",
        "committed",
        "cooldown",
    } == NEURAL_ARMED_STATES


def _refusing_server(reason: str, service: FakeService) -> PilotServer:
    server = _server(service)
    if reason == "recording_active":
        server._recorded_workflows = FakeRecordings(active=True)
    elif reason == "task_running":
        server._interactive_request_active = True
    else:
        server._neural_controller = FakeNeural("armed_safe_ui")
    return server


@pytest.mark.parametrize("reason", ["recording_active", "task_running", "neural_armed"])
@pytest.mark.parametrize(("method", "params"), STUDY_CALLS)
@pytest.mark.asyncio
async def test_study_rpcs_refuse_without_reaching_the_service(reason, method, params):
    service = FakeService()
    server = _refusing_server(reason, service)

    result = await _call(server, method, params)

    assert result == {"status": "refused", "reason": reason}
    assert service.calls == []


@pytest.mark.asyncio
async def test_a_refused_trial_is_not_stored_and_is_accepted_after_the_recording_stops():
    service = FakeService()
    service.replies["record_trial"] = {"status": "recorded", "verification": "ok"}
    server = _refusing_server("recording_active", service)
    trial = {"session_id": "s-1", "trial": {"trial_id": "t-9"}}

    refused = await _call(server, "co_adaptation_record_trial", trial)
    server._recorded_workflows.active = False
    recorded = await _call(server, "co_adaptation_record_trial", trial)

    assert refused == {"status": "refused", "reason": "recording_active"}
    assert recorded["status"] == "recorded"
    assert service.called("record_trial") == [trial]


@pytest.mark.parametrize("reason", ["recording_active", "task_running", "neural_armed"])
@pytest.mark.asyncio
async def test_offers_burden_and_session_end_are_not_start_refusals(reason):
    service = FakeService()
    server = _refusing_server(reason, service)

    offer = await _call(server, "co_adaptation_recommend", {"session_id": "s-1", "level": "offer"})
    burden = await _call(server, "co_adaptation_report_burden", {"session_id": "s-1", "report": {}})
    ended = await _call(server, "co_adaptation_session_end", {"session_id": "s-1", "reason": "stopped"})

    assert [offer["status"], burden["status"], ended["status"]] == ["ok", "ok", "ok"]
    assert [name for name, _ in service.calls] == ["recommend", "report_burden", "end_session"]


# ── Reset and shutdown ──


@pytest.mark.asyncio
async def test_reset_config_turns_the_feature_off_and_keeps_data(monkeypatch):
    service = FakeService()
    server = _server(service)
    server.config.co_adaptation.enabled = True
    server.config.co_adaptation.ambiguity_threshold = 0.1
    server._vault = SimpleNamespace(available=True)
    monkeypatch.setattr("pilot.air_handoff.AirHandoffManager", lambda vault, **kwargs: SimpleNamespace())
    monkeypatch.setattr("pilot.air_handoff.AirHandoffServer", lambda manager, **kwargs: SimpleNamespace(running=False))

    result = await server._handle_reset_config({}, MagicMock())

    assert result["status"] == "ok"
    assert server.config.co_adaptation == CoAdaptationConfig()
    previous = service.called("on_config_changed")
    assert len(previous) == 1
    assert previous[0].enabled is True
    assert previous[0].ambiguity_threshold == 0.1
    # A reset never erases participants, rules or study data.
    assert [name for name, _ in service.calls] == ["on_config_changed"]


@pytest.mark.asyncio
async def test_stop_closes_the_service():
    service = FakeService()
    server = _server(service)

    await server.stop()

    assert service.called("close") == [None]
    assert server._co_adaptation is None

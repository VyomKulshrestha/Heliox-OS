"""Tests for destructive_critic.py's risk_score() -- the composition of
heuristic_risk() with the (optional) Learned Risk Gate, used at both
call sites (server.py, gateway.py's _maybe_run_critic) to decide whether a
Tier-3-only/irreversible-only plan is worth the LLM critic round-trip."""

from unittest.mock import patch

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.destructive_critic import assess_plan_risk, heuristic_risk, risk_score
from pilot.config import PilotConfig


def _plan(*action_types: ActionType, target: str = "") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=t, target=target, parameters=EmptyParams()) for t in action_types],
        raw_input="test",
    )


def test_no_config_falls_back_to_heuristic_risk_unchanged():
    plan = _plan(ActionType.FILE_DELETE)
    assert risk_score(plan, None) == heuristic_risk(plan)


def test_gate_disabled_falls_back_to_heuristic_risk_unchanged():
    config = PilotConfig()
    config.gateway.risk_gate_enabled = False
    plan = _plan(ActionType.FILE_DELETE)
    assert risk_score(plan, config) == heuristic_risk(plan)


def test_gate_is_enabled_by_default_for_new_configs():
    assert PilotConfig().gateway.risk_gate_enabled is True


def test_gate_enabled_with_no_new_signal_matches_heuristic_risk():
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    plan = _plan(ActionType.FILE_DELETE, target="/tmp/scratch.txt")
    assert risk_score(plan, config) == heuristic_risk(plan)


def test_gate_enabled_catches_protected_path_heuristic_alone_would_miss():
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    config.restrictions.protected_folders = ["/etc"]

    # A single-action, single-target plan -- heuristic_risk()'s own rules
    # (length>3, >2 distinct targets, dangerous_flags, tier mixing) don't
    # fire on this at all, so heuristic_risk() alone scores it 0.
    plan = _plan(ActionType.FILE_DELETE, target="/etc/passwd")
    assert heuristic_risk(plan) == 0.0

    combined = risk_score(plan, config)
    assert combined > 0.0
    assert combined > heuristic_risk(plan)

    assessment = assess_plan_risk(plan, config)
    assert assessment.world_model_score == 0.9
    assert assessment.requires_confirmation is True
    assert "touches a protected path/package: /etc" in assessment.reasons
    assert assessment.prediction_sources
    assert assessment.predictions[0]["action_type"] == "file_delete"
    assert assessment.prediction_contract["contract_version"] == "world-prediction-v1"
    assert assessment.to_dict()["requires_confirmation"] is True


def test_gate_never_lowers_risk_below_heuristic():
    """max() composition: a plan heuristic_risk() already flags stays at
    least that risky regardless of what the gate says."""
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    plan = _plan(
        ActionType.FILE_DELETE,
        ActionType.FILE_DELETE,
        ActionType.FILE_DELETE,
        ActionType.FILE_DELETE,
        target="/tmp/scratch.txt",
    )
    base = heuristic_risk(plan)
    assert base > 0.0  # length > 3 fires
    assert risk_score(plan, config) >= base


def test_gate_evaluation_error_escalates_state_changing_plan():
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    plan = _plan(ActionType.FILE_DELETE, target="/tmp/scratch.txt")

    with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
        assessment = assess_plan_risk(plan, config)
    assert assessment.world_model_score == 1.0
    assert assessment.requires_confirmation
    assert assessment.model_version == "evaluation-error"
    assert "unavailable" in assessment.reasons[0]


def test_gate_evaluation_error_does_not_escalate_read_only_plan():
    config = PilotConfig()
    plan = _plan(ActionType.SYSTEM_INFO)
    with patch("pilot.security.risk_gate.get_risk_gate", side_effect=RuntimeError("boom")):
        assessment = assess_plan_risk(plan, config)
    assert assessment.world_model_score == 0.0
    assert not assessment.requires_confirmation
    assert assessment.model_version == "evaluation-error"

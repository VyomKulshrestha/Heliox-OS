from pilot.actions import Action, ActionType, EmptyParams, FileParams
from pilot.config import PilotConfig
from pilot.security.risk_model import PredictedOutcome
from pilot.security.risk_safety import score_outcome, touches_protected_path


def _action(action_type=ActionType.FILE_WRITE, target="") -> Action:
    return Action(action_type=action_type, target=target, parameters=EmptyParams())


def test_safe_outcome_scores_zero():
    action = _action()
    outcome = PredictedOutcome(disk_usage_after=0.5, proc_count_delta_normalized=0.0, source="rule")
    risk, reasons = score_outcome(action, outcome, PilotConfig())
    assert risk == 0.0
    assert reasons == []


def test_high_disk_usage_flagged():
    action = _action()
    outcome = PredictedOutcome(disk_usage_after=0.97, proc_count_delta_normalized=0.0, source="rule")
    risk, reasons = score_outcome(action, outcome, PilotConfig())
    assert risk > 0
    assert any("disk usage" in r for r in reasons)


def test_fork_bomb_like_delta_flagged():
    action = _action()
    outcome = PredictedOutcome(disk_usage_after=0.5, proc_count_delta_normalized=0.5, source="rule")
    risk, reasons = score_outcome(action, outcome, PilotConfig())
    assert risk > 0
    assert any("fork-bomb" in r for r in reasons)


def test_protected_folder_match_flagged():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/etc"]
    action = _action(target="/etc/passwd")
    outcome = PredictedOutcome(disk_usage_after=0.5, proc_count_delta_normalized=0.0, source="rule")
    risk, reasons = score_outcome(action, outcome, config)
    assert risk > 0
    assert any("protected path" in r for r in reasons)


def test_multiple_rules_compound_but_clamp_at_one():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/etc"]
    action = _action(target="/etc/passwd")
    outcome = PredictedOutcome(disk_usage_after=0.99, proc_count_delta_normalized=1.0, source="rule")
    risk, reasons = score_outcome(action, outcome, config)
    assert risk == 1.0
    assert len(reasons) == 3


def test_touches_protected_path_checks_target():
    config = PilotConfig()
    config.restrictions.protected_folders = ["C:/Windows/System32"]
    action = _action(target="C:/Windows/System32/config.sys")
    assert touches_protected_path(action, config) is not None


def test_touches_protected_path_none_when_no_match():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/etc"]
    action = _action(target="/home/user/notes.txt")
    assert touches_protected_path(action, config) is None


def test_touches_protected_path_checks_parameter_fields():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/etc"]
    action = Action(
        action_type=ActionType.FILE_COPY,
        target="",
        parameters=FileParams(path="/etc/hosts", destination="/tmp/hosts.bak"),
    )
    assert touches_protected_path(action, config) is not None


def test_no_protected_paths_configured_returns_none():
    config = PilotConfig()
    action = _action(target="/anything")
    assert touches_protected_path(action, config) is None

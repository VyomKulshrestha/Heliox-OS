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


def test_existing_high_disk_usage_does_not_flag_disk_neutral_action():
    action = _action(ActionType.FILE_READ)
    outcome = PredictedOutcome(disk_usage_after=0.97, proc_count_delta_normalized=0.0, source="rule")
    risk, reasons = score_outcome(action, outcome, PilotConfig(), baseline_disk_usage=0.97)
    assert risk == 0.0
    assert reasons == []


def test_disk_increase_above_threshold_remains_flagged():
    action = _action(ActionType.FILE_WRITE)
    outcome = PredictedOutcome(disk_usage_after=0.98, proc_count_delta_normalized=0.0, source="rule")
    risk, reasons = score_outcome(action, outcome, PilotConfig(), baseline_disk_usage=0.97)
    assert risk == 0.8
    assert any("disk usage" in reason for reason in reasons)


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


def test_a_protected_package_name_no_longer_matches_inside_paths_and_words():
    from pilot.actions import PackageParams, ShellCommandParams

    config = PilotConfig()
    config.restrictions.protected_packages = ["git", "Mozilla.Firefox"]
    config.restrictions.protected_folders = ["/etc"]
    # Before: "git" matched any path or command holding those letters, and "/etc" any path starting with them.
    for target in ("/home/sam/github/notes.txt", "/home/sam/digital.txt", "/etcetera/x", "/home/sam/etc/x"):
        assert touches_protected_path(_action(target=target), config) is None, target
    status = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="",
        parameters=ShellCommandParams(command="git", args=["status"]),
    )
    assert touches_protected_path(status, config) is None
    # Removing the package, by the package action or a removal command, still counts, whatever the case.
    removal = Action(action_type=ActionType.PACKAGE_REMOVE, target="git", parameters=PackageParams(name="GIT"))
    assert touches_protected_path(removal, config) == "git"
    apt = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="",
        parameters=ShellCommandParams(command="sudo", args=["apt", "remove", "git"]),
    )
    assert touches_protected_path(apt, config) == "git"
    winget = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="",
        parameters=ShellCommandParams(command="winget uninstall mozilla.firefox"),
    )
    assert touches_protected_path(winget, config) == "Mozilla.Firefox"
    # A protected folder in command text counts at a path boundary.
    cat = Action(
        action_type=ActionType.SHELL_COMMAND, target="", parameters=ShellCommandParams(command="cat /etc/hosts")
    )
    assert touches_protected_path(cat, config) == "/etc"
    notes = Action(
        action_type=ActionType.SHELL_COMMAND, target="", parameters=ShellCommandParams(command="cat /etcetera/a")
    )
    assert touches_protected_path(notes, config) is None


def test_a_protected_folder_written_with_a_tilde_is_matched_by_its_real_path(monkeypatch, tmp_path):
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    config = PilotConfig()
    config.restrictions.protected_folders = ["~/Documents/private"]
    inside = _action(target=str(tmp_path / "Documents" / "private" / "plan.txt"))
    assert touches_protected_path(inside, config) == "~/Documents/private"
    beside = _action(target=str(tmp_path / "Documents" / "private-notes.txt"))
    assert touches_protected_path(beside, config) is None


def test_no_protected_paths_configured_returns_none():
    config = PilotConfig()
    action = _action(target="/anything")
    assert touches_protected_path(action, config) is None

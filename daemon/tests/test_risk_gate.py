import numpy as np
import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.intelligence.online_learning import AdaptationScore
from pilot.security.risk_gate import RiskGate, get_risk_gate
from pilot.security.risk_model import EMBEDDING_SIZE
from pilot.security.risk_observation import OsSnapshot


def _plan(*action_types: ActionType, target: str = "") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=t, target=target, parameters=EmptyParams()) for t in action_types],
        raw_input="test",
    )


def test_empty_plan_is_zero_risk():
    gate = RiskGate(weights_path="/nonexistent.npz")
    risk, reasons = gate.evaluate_plan(_plan(), PilotConfig())
    assert risk == 0.0
    assert reasons == []


def test_ordinary_plan_is_low_risk():
    gate = RiskGate(weights_path="/nonexistent.npz")
    plan = _plan(ActionType.FILE_READ, ActionType.SYSTEM_INFO)
    risk, _reasons = gate.evaluate_plan(plan, PilotConfig())
    assert risk == 0.0


def test_protected_path_action_anywhere_in_plan_drives_up_risk():
    config = PilotConfig()
    config.restrictions.protected_folders = ["/etc"]
    gate = RiskGate(weights_path="/nonexistent.npz")
    plan = _plan(ActionType.FILE_READ, target="/tmp/notes.txt")
    plan.actions.append(Action(action_type=ActionType.FILE_DELETE, target="/etc/passwd", parameters=EmptyParams()))

    risk, reasons = gate.evaluate_plan(plan, config)
    assert risk > 0
    assert any("protected path" in r for r in reasons)


def test_available_reflects_whether_weights_loaded():
    gate = RiskGate(weights_path="/nonexistent.npz")
    assert gate.available is False


def test_get_risk_gate_returns_singleton():
    a = get_risk_gate()
    b = get_risk_gate()
    assert a is b


def test_learned_prediction_cannot_weaken_deterministic_warning(tmp_path, monkeypatch):
    weights = tmp_path / "zero-model.npz"
    np.savez(
        weights,
        w1=np.zeros((EMBEDDING_SIZE, 2), dtype=np.float32),
        b1=np.zeros(2, dtype=np.float32),
        w2=np.zeros((2, 2), dtype=np.float32),
        b2=np.zeros(2, dtype=np.float32),
    )
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.94,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )

    gate = RiskGate(weights_path=str(weights))
    risk, reasons = gate.evaluate_plan(_plan(ActionType.FILE_WRITE), PilotConfig())

    assert risk == pytest.approx(0.8)
    assert any("disk usage" in reason for reason in reasons)
    assert gate.status(enabled=True)["last_evaluation"]["prediction_sources"] == ["learned", "rule"]


def test_repeated_process_actions_are_evaluated_cumulatively(monkeypatch):
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.5,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")
    risk, reasons = gate.evaluate_plan(_plan(*([ActionType.SHELL_COMMAND] * 51)), PilotConfig())

    assert risk == pytest.approx(0.7)
    assert any("fork-bomb" in reason for reason in reasons)


def test_status_reports_model_metadata_and_sanitized_last_evaluation():
    gate = RiskGate()
    gate.evaluate_plan(_plan(ActionType.FILE_READ), PilotConfig())

    status = gate.status(enabled=True)
    assert status["enabled"] is True
    assert status["weights_loaded"] is True
    assert status["calibrated"] is True
    assert status["validation_samples"] == 5_400
    assert status["validation_mae"]
    assert status["embedding_size"] == EMBEDDING_SIZE
    assert status["learnable_action_types"]
    assert status["prediction_contract"]["contract_version"] == "world-prediction-v1"
    assert status["prediction_contract"]["ui_jepa"]["mode"] == "shadow"
    assert set(status["last_evaluation"]) == {
        "evaluated_at",
        "action_count",
        "risk_score",
        "reasons",
        "worst_action_type",
        "prediction_sources",
        "prediction_confidence",
        "predictions",
    }
    prediction = status["last_evaluation"]["predictions"][0]
    assert prediction["action_type"] == "file_read"
    assert prediction["sources"] == ["structured_rules", "rule"]


def test_prediction_contract_covers_browser_transition(monkeypatch):
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.5,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")

    gate.evaluate_plan(_plan(ActionType.BROWSER_NAVIGATE), PilotConfig())

    prediction = gate.last_evaluation["predictions"][0]
    assert prediction["action_type"] == "browser_navigate"
    assert prediction["expected_effects"][0]["domain"] == "browser"
    assert prediction["predicted_state"]["browser"]["dom_state"] == "requires_observation"


def test_repeated_callback_failures_add_historical_caution(monkeypatch):
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.5,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")
    for _ in range(3):
        gate.record_outcome(ActionType.BROWSER_CLICK_TEXT, False)

    risk, reasons = gate.evaluate_plan(
        _plan(ActionType.BROWSER_CLICK_TEXT),
        PilotConfig(),
    )

    assert risk >= 0.3
    assert "observed browser_click_text executor callbacks failed" in reasons[0]
    prediction = gate.last_evaluation["predictions"][0]
    assert "execution_history" in prediction["sources"]


def test_verified_online_transition_adds_caution_only_for_known_source(monkeypatch):
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.5,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )

    class Learner:
        def score_transition(self, *, action_type, source):
            assert action_type == "file_write"
            assert source == "interactive"
            return AdaptationScore(probability=0.1, labels=8, state="promoted")

    gate = RiskGate(weights_path="/nonexistent.npz")
    gate.set_online_learner(Learner())
    plan = _plan(ActionType.FILE_WRITE)

    without_source, _ = gate.evaluate_plan(plan, PilotConfig())
    with_source, reasons = gate.evaluate_plan(plan, PilotConfig(), source="interactive")

    assert with_source >= without_source
    assert with_source == pytest.approx(0.7)
    assert "independently verified" in reasons[0]
    prediction = gate.last_evaluation["predictions"][0]
    assert "verified_online_transition" in prediction["sources"]
    assert any(evidence["source"] == "verified_online_transition" for evidence in prediction["risk_evidence"])

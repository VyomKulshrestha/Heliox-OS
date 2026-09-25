"""RiskGate consumes one fresh visual observation without exporting it."""

import numpy as np

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.intelligence.world_model import HybridWorldModel, UiJepaPredictor
from pilot.security.risk_gate import RiskGate
from pilot.security.risk_observation import OsSnapshot


def _plan(*types: ActionType) -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=kind, target="test", parameters=EmptyParams()) for kind in types],
        raw_input="test",
    )


def _snapshot() -> OsSnapshot:
    return OsSnapshot(proc_count=100, disk_usage_fraction=0.5,
                      memory_usage_fraction=0.5, disk_path="/")


def test_one_visual_observation_per_plan_and_no_latent_export(monkeypatch) -> None:
    monkeypatch.setattr("pilot.security.risk_gate.capture_os_snapshot", _snapshot)
    gate = RiskGate(weights_path="/nonexistent.npz")
    calls: list[bool] = []

    def observe() -> dict:
        calls.append(True)
        return {"latent_embedding": [0.5] * 256, "screen_observed_at": "fresh"}

    gate.set_ui_observation_provider(observe)
    gate.evaluate_plan(_plan(ActionType.FILE_READ, ActionType.SYSTEM_INFO), PilotConfig())

    assert len(calls) == 1
    for prediction in gate.last_evaluation["predictions"]:
        assert prediction["predicted_state"]["ui"] == {"screen_observed_at": "fresh"}


def test_visual_failure_does_not_block_deterministic_gate(monkeypatch) -> None:
    monkeypatch.setattr("pilot.security.risk_gate.capture_os_snapshot", _snapshot)
    gate = RiskGate(weights_path="/nonexistent.npz")

    def unavailable() -> None:
        raise RuntimeError("display unavailable")

    gate.set_ui_observation_provider(unavailable)
    risk, reasons = gate.evaluate_plan(_plan(ActionType.FILE_READ), PilotConfig())
    assert risk == 0.0
    assert reasons == []


def test_observed_frame_reaches_shadow_predictor_without_approval_power(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pilot.security.risk_gate.capture_os_snapshot", _snapshot)
    weights = tmp_path / "shadow_ui.npz"
    np.savez(
        weights,
        state_weight=np.eye(256, dtype=np.float32),
        action_weight=np.zeros((len(ActionType), 256), dtype=np.float32),
        bias=np.zeros(256, dtype=np.float32),
        action_types=np.array([kind.value for kind in ActionType], dtype=np.str_),
        validated_for_gating=np.array(False),
        training_samples=np.array(0),
        validation_error=np.array(0.8),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")
    gate._world_model = HybridWorldModel(visual=UiJepaPredictor(weights))
    gate.set_ui_observation_provider(lambda: {"latent_embedding": [0.5] * 256})

    risk, _ = gate.evaluate_plan(_plan(ActionType.BROWSER_CLICK_TEXT), PilotConfig())

    prediction = gate.last_evaluation["predictions"][0]
    assert "ui_jepa" in prediction["sources"]
    assert any(effect["operation"] == "predict_latent_transition"
               for effect in prediction["expected_effects"])
    assert "latent_embedding" not in prediction["predicted_state"]["ui"]
    assert risk == 0.0

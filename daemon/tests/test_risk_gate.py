from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.security.risk_gate import RiskGate, get_risk_gate


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

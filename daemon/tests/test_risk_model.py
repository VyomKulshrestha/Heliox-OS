import numpy as np
import pytest

from pilot.actions import Action, ActionType, EmptyParams
from pilot.security.risk_model import (
    EMBEDDING_SIZE,
    RiskTransitionModel,
    encode,
)
from pilot.security.risk_observation import OsSnapshot


def _snapshot(**overrides) -> OsSnapshot:
    defaults = dict(proc_count=100, disk_usage_fraction=0.5, memory_usage_fraction=0.5, disk_path="/")
    defaults.update(overrides)
    return OsSnapshot(**defaults)


def _action(action_type=ActionType.FILE_READ, **kwargs) -> Action:
    return Action(action_type=action_type, target=kwargs.pop("target", ""), parameters=EmptyParams(), **kwargs)


def test_encode_produces_fixed_size_vector():
    v = encode(_snapshot(), _action())
    assert v.shape == (EMBEDDING_SIZE,)
    assert v.dtype == np.float32


def test_encode_reflects_os_state():
    v = encode(_snapshot(disk_usage_fraction=0.9, memory_usage_fraction=0.2), _action())
    assert v[1] == pytest.approx(0.9)
    assert v[2] == pytest.approx(0.2)


def test_encode_reflects_irreversible_and_root_flags():
    action = _action(ActionType.POWER_SHUTDOWN)  # ROOT_CRITICAL tier -> is_irreversible
    v = encode(_snapshot(), action)
    assert v[4] == 1.0  # IDX_IRREVERSIBLE


def test_encode_family_one_hot_is_exclusive():
    v = encode(_snapshot(), _action(ActionType.SHELL_COMMAND))
    family_slots = v[7:11]
    assert family_slots.sum() == 1.0


class TestRiskTransitionModel:
    def test_falls_back_to_rule_table_when_no_weights(self, tmp_path):
        model = RiskTransitionModel(weights_path=str(tmp_path / "does_not_exist.npz"))
        assert model.is_loaded is False

        outcome = model.predict(_snapshot(disk_usage_fraction=0.5), _action(ActionType.FILE_WRITE))
        assert outcome.source == "rule"
        assert outcome.disk_usage_after > 0.5  # write nudges usage up

    def test_delete_rule_reduces_disk_usage(self):
        model = RiskTransitionModel(weights_path="/nonexistent.npz")
        outcome = model.predict(_snapshot(disk_usage_fraction=0.5), _action(ActionType.FILE_DELETE))
        assert outcome.disk_usage_after < 0.5

    def test_unmodeled_action_predicts_no_change(self):
        model = RiskTransitionModel(weights_path="/nonexistent.npz")
        outcome = model.predict(_snapshot(disk_usage_fraction=0.5), _action(ActionType.NOTIFY))
        assert outcome.disk_usage_after == pytest.approx(0.5)
        assert outcome.proc_count_delta_normalized == 0.0

    def test_loads_valid_weights_and_uses_them_for_learnable_types(self, tmp_path):
        rng = np.random.default_rng(0)
        w1 = rng.normal(size=(EMBEDDING_SIZE, 4)).astype(np.float32)
        b1 = np.zeros(4, dtype=np.float32)
        w2 = rng.normal(size=(4, 2)).astype(np.float32)
        b2 = np.zeros(2, dtype=np.float32)
        path = tmp_path / "weights.npz"
        np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)

        model = RiskTransitionModel(weights_path=str(path))
        assert model.is_loaded is True

        outcome = model.predict(_snapshot(), _action(ActionType.FILE_WRITE))
        assert outcome.source == "learned"

        # An unlearnable action type still falls back to the rule table
        # even when a model is loaded.
        outcome2 = model.predict(_snapshot(), _action(ActionType.NOTIFY))
        assert outcome2.source == "rule"

    def test_ignores_weights_with_wrong_shape(self, tmp_path):
        path = tmp_path / "bad_weights.npz"
        np.savez(
            path,
            w1=np.zeros((3, 4), dtype=np.float32),  # wrong input size
            b1=np.zeros(4, dtype=np.float32),
            w2=np.zeros((4, 2), dtype=np.float32),
            b2=np.zeros(2, dtype=np.float32),
        )
        model = RiskTransitionModel(weights_path=str(path))
        assert model.is_loaded is False

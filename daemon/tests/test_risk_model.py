import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot.actions import Action, ActionType, EmptyParams
from pilot.security.risk_model import (
    EMBEDDING_SIZE,
    IDX_ACTION_TYPE_BASE,
    LEARNABLE_ACTION_TYPE_ORDER,
    RiskTransitionModel,
    encode,
)
from pilot.security.risk_observation import OsSnapshot
from scripts.train_risk_gate import (
    _predict,
    apply_calibration,
    load_action_types,
    load_dataset,
    stratified_temporal_split,
)


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


def test_encode_exact_action_type_one_hot_is_exclusive_and_distinct():
    start = encode(_snapshot(), _action(ActionType.SERVICE_START))
    stop = encode(_snapshot(), _action(ActionType.SERVICE_STOP))
    start_slots = start[IDX_ACTION_TYPE_BASE:]
    stop_slots = stop[IDX_ACTION_TYPE_BASE:]

    assert start_slots.sum() == 1.0
    assert stop_slots.sum() == 1.0
    assert not np.array_equal(start_slots, stop_slots)
    assert len(start_slots) == len(LEARNABLE_ACTION_TYPE_ORDER)


def test_training_loader_upgrades_legacy_embeddings_with_recorded_action_type(tmp_path):
    dataset = tmp_path / "legacy.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "embedding": [0.0] * IDX_ACTION_TYPE_BASE,
                "disk_delta": 0.0,
                "proc_delta": 1.0 / 300.0,
                "action_type": "service_start",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    X, Y = load_dataset(str(dataset))

    assert X.shape == (1, EMBEDDING_SIZE)
    assert X[0, IDX_ACTION_TYPE_BASE:].sum() == 1.0
    assert Y[0, 1] == pytest.approx(1.0 / 300.0)


def test_training_split_holds_out_each_action_temporally():
    dataset = Path(__file__).resolve().parents[1] / "scripts" / "risk_dataset.jsonl"
    action_types = load_action_types(str(dataset))

    train, calibration, validation = stratified_temporal_split(action_types, 0.15)

    assert len(train) == 25_200
    assert len(calibration) == 5_400
    assert len(validation) == 5_400
    assert set(train).isdisjoint(calibration)
    assert set(train).isdisjoint(validation)
    assert set(calibration).isdisjoint(validation)
    for action_type in LEARNABLE_ACTION_TYPE_ORDER:
        assert action_type.value in action_types[train]
        assert action_type.value in action_types[calibration]
        assert action_type.value in action_types[validation]


def test_training_saves_the_model_that_was_validated(tmp_path, monkeypatch):
    from scripts import train_risk_gate

    dataset = tmp_path / "transitions.jsonl"
    rows = [
        json.dumps(
            {
                "embedding": [0.0] * EMBEDDING_SIZE,
                "disk_delta": 0.0,
                "proc_delta": 0.0,
                "action_type": action_type.value,
            }
        )
        for action_type in LEARNABLE_ACTION_TYPE_ORDER
        for _ in range(10)
    ]
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    artifact = tmp_path / "weights.npz"
    calls = []

    def train_once(X, Y, hidden_size, epochs, lr, seed):
        calls.append(len(X))
        return (
            np.zeros((EMBEDDING_SIZE, hidden_size), dtype=np.float32),
            np.zeros(hidden_size, dtype=np.float32),
            np.zeros((hidden_size, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )

    monkeypatch.setattr(train_risk_gate, "train", train_once)
    monkeypatch.setattr(
        sys, "argv", ["train_risk_gate", "--dataset", str(dataset), "--out", str(artifact), "--epochs", "1"]
    )
    train_risk_gate.main()

    assert calls == [96]
    with np.load(artifact, allow_pickle=False) as saved:
        assert int(saved["training_samples"]) == 96
        assert int(saved["validation_samples"]) == 12
        assert saved["model_version"].item() == "risk-mlp-v4-heldout"
        assert np.array_equal(saved["validation_mae"], np.zeros(2, dtype=np.float32))


def test_shipped_validation_metrics_replay_with_saved_weights():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "scripts" / "risk_dataset.jsonl"
    X, Y = load_dataset(str(dataset))
    action_types = load_action_types(str(dataset))
    _, _, validation = stratified_temporal_split(action_types, 0.15)

    with np.load(root / "pilot" / "security" / "risk_gate_weights.npz", allow_pickle=False) as saved:
        raw = _predict(X[validation], *(saved[key] for key in ("w1", "b1", "w2", "b2")))
        calibrated = apply_calibration(
            raw, action_types[validation], saved["action_medians"], saved["calibration_alpha"]
        )
        actual_mae = np.mean(np.abs(calibrated - Y[validation]), axis=0)
        np.testing.assert_allclose(actual_mae, saved["validation_mae"], rtol=1e-5, atol=1e-12)


class TestRiskTransitionModel:
    def test_shipped_weights_load_and_separate_opposite_process_actions(self):
        model = RiskTransitionModel()

        start = model.predict(_snapshot(), _action(ActionType.SERVICE_START))
        stop = model.predict(_snapshot(), _action(ActionType.SERVICE_STOP))

        assert model.is_loaded is True
        assert model.model_version == "risk-mlp-v4-heldout"
        assert model.training_samples == 25_200
        assert model.validation_samples == 5_400
        assert model.is_calibrated is True
        assert model.validation_mae is not None
        assert start.source == "learned_calibrated"
        assert stop.source == "learned_calibrated"
        assert start.proc_count_delta_normalized > 0
        assert stop.proc_count_delta_normalized < 0
        assert 0.0 < start.confidence < 1.0

    def test_confidence_drops_outside_observed_os_state_distribution(self):
        model = RiskTransitionModel()

        supported = model.predict(
            _snapshot(proc_count=300, disk_usage_fraction=0.866, memory_usage_fraction=0.831),
            _action(ActionType.CODE_EXECUTE),
        )
        out_of_distribution = model.predict(
            _snapshot(proc_count=5, disk_usage_fraction=0.05, memory_usage_fraction=0.05),
            _action(ActionType.CODE_EXECUTE),
        )

        assert supported.confidence > out_of_distribution.confidence

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

    def test_rule_process_delta_uses_normalized_process_units(self):
        model = RiskTransitionModel(weights_path="/nonexistent.npz")
        outcome = model.predict_rule(_snapshot(), _action(ActionType.SHELL_COMMAND))
        assert outcome.proc_count_delta_normalized == pytest.approx(1.0 / 300.0)

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

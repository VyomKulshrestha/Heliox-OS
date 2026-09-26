"""Evaluation checkpoints must not replay or mix different held-out models."""

import json

import numpy as np
import pytest

from scripts import train_general_decision as training


class _EvalOnly:
    def eval(self):
        return self


def _items(count):
    return [{"row": {"id": str(index), "target": [1.0, 0.0]},
             "sequences": [[index]]} for index in range(count)]


def test_evaluation_resumes_at_first_unsaved_row(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.json"
    calls = []
    fail = {"enabled": True}

    def score(_backbone, _head, item):
        index = int(item["row"]["id"])
        calls.append(index)
        if index == 2 and fail["enabled"]:
            raise RuntimeError("transient GPU failure")
        return np.asarray([float(index), -float(index)])

    monkeypatch.setattr(training, "score_row", score)
    with pytest.raises(RuntimeError, match="calibration failed at row 3/5"):
        training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(5),
                                        "calibration", path=path, signature="model-a",
                                        resume=False, checkpoint_every=2)
    assert len(json.loads(path.read_text(encoding="utf-8"))["logits"]) == 2
    fail["enabled"] = False
    calls.clear()
    result = training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(5),
                                             "calibration", path=path,
                                             signature="model-a", resume=True,
                                             checkpoint_every=2)
    assert calls == [2, 3, 4]
    assert [value[0] for value in result] == [0, 1, 2, 3, 4]


def test_evaluation_rejects_wrong_model_rows_and_logits(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.json"
    monkeypatch.setattr(training, "score_row",
                        lambda _backbone, _head, _item: np.asarray([0.0, 1.0]))
    training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(2), "test",
                                    path=path, signature="model-a", resume=False,
                                    checkpoint_every=1)
    with pytest.raises(ValueError, match="does not match"):
        training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(2), "test",
                                        path=path, signature="model-b", resume=True,
                                        checkpoint_every=1)
    with pytest.raises(ValueError, match="does not match"):
        training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(1), "test",
                                        path=path, signature="model-a", resume=True,
                                        checkpoint_every=1)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["logits"][0][0] = float("nan")
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid test logits"):
        training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(2), "test",
                                        path=path, signature="model-a", resume=True,
                                        checkpoint_every=1)


def test_evaluation_does_not_save_new_nonfinite_logits(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.json"
    monkeypatch.setattr(training, "score_row",
                        lambda _backbone, _head, _item: np.asarray([np.nan, 0.0]))
    with pytest.raises(ValueError, match="invalid logits at row 1"):
        training.evaluate_checkpointed(_EvalOnly(), _EvalOnly(), _items(1), "ood",
                                        path=path, signature="model-a", resume=False,
                                        checkpoint_every=1)
    assert not path.exists()

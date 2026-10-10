import numpy as np

from scripts.fit_opendec_multidomain_head import binary_metrics, fit, split_validation, task_group


def test_relevance_metrics_do_not_claim_action_allowances():
    result = binary_metrics(np.asarray([True, False]), np.asarray([0.8, 0.7]), 0.5)
    assert result["false_positives"] == 1
    assert "false_allowances" not in result


def test_tbac_group_partition_is_stable_and_keeps_repeated_tasks_together():
    ids = [f"tbac-{index}:choice-{suffix}" for index in range(20) for suffix in ("a", "b")]
    data = {
        "ids": np.asarray(ids),
        "labels": np.asarray([True, False] * 20),
        "features": np.arange(80, dtype=np.float32).reshape(40, 2),
        "raw_scores": np.zeros(40),
    }
    train, calibration = split_validation(data)
    assert len(train["ids"]) + len(calibration["ids"]) == 40
    assert {task_group(value) for value in train["ids"]}.isdisjoint({task_group(value) for value in calibration["ids"]})
    assert np.array_equal(train["ids"], split_validation(data)[0]["ids"])
    assert np.any(train["labels"]) and np.any(~train["labels"])
    assert np.any(calibration["labels"]) and np.any(~calibration["labels"])


def test_test_labels_cannot_change_frozen_head(tmp_path):
    def data(ids, labels):
        values = np.asarray([[float(index % 3), float(index % 5)] for index in range(len(ids))])
        return {
            "ids": np.asarray(ids),
            "labels": np.asarray(labels, dtype=bool),
            "features": values,
            "raw_scores": np.zeros(len(ids)),
        }

    scope = {
        name: data([f"{name}-{index}:call" for index in range(4)], [True, False] * 2)
        for name in ("train", "calibration", "test")
    }
    tbac = {
        "validation": data(
            [f"tbac-{index}:choice-{suffix}" for index in range(20) for suffix in ("a", "b")], [True, False] * 20
        ),
        "test": data([f"tbac-test-{index}:choice" for index in range(4)], [True, False] * 2),
    }
    first = fit(scope, tbac, artifact=tmp_path / "first.npz")
    flipped = {**tbac, "test": {**tbac["test"], "labels": ~tbac["test"]["labels"]}}
    second = fit(scope, flipped, artifact=tmp_path / "second.npz")
    assert first["selected_c"] == second["selected_c"]
    assert first["threshold"] == second["threshold"]
    with np.load(tmp_path / "first.npz") as original, np.load(tmp_path / "second.npz") as changed:
        for name in ("mean", "scale", "coefficient", "intercept", "threshold", "regularization"):
            np.testing.assert_array_equal(original[name], changed[name])

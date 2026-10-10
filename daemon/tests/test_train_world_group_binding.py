from types import SimpleNamespace

import numpy as np

from scripts.train_world_group_binding import (
    group_arrays,
    group_indices,
    group_metrics,
    normalize_features,
    split_entity_groups,
)


def test_entity_split_keeps_whole_tasks_and_ambiguous_examples():
    sizes = [10, 8, 8, 8, 8, 8, 5, 3, 2]
    rows = [
        {
            "task_id": f"task{family}_{task}",
            "domain": f"domain{family}",
            "ambiguity": "resolvable" if family < 7 else "ambiguous",
        }
        for family, size in enumerate(sizes)
        for task in range(size)
    ]
    cases = tuple(
        SimpleNamespace(id=f"{row['task_id']}:{choice}", in_scope=choice == 0 and row["ambiguity"] == "resolvable")
        for row in rows
        for choice in range(2)
    )
    groups = group_indices(cases, entity=True)

    splits = split_entity_groups(rows, cases, groups)

    assert len(set().union(*(set(keys) for keys in splits.values()))) == 60
    assert all(
        not set(splits[left]) & set(splits[right])
        for left, right in (("train", "calibration"), ("train", "holdout"), ("calibration", "holdout"))
    )
    by_task = {row["task_id"]: row["domain"] for row in rows}
    assert {name: len({by_task[key] for key in keys}) for name, keys in splits.items()} == {
        "train": 6,
        "calibration": 1,
        "holdout": 2,
    }
    for domain in {row["domain"] for row in rows}:
        assert sum(any(by_task[key] == domain for key in keys) for keys in splits.values()) == 1
    ambiguous = {row["task_id"] for row in rows if row["ambiguity"] == "ambiguous"}
    assert sum(key in ambiguous for key in splits["calibration"]) == 0
    assert {sum(key in ambiguous for key in splits[name]) for name in ("train", "holdout")} == {2, 3}


def test_group_metrics_counts_wrong_target_and_clarification():
    cases = (
        SimpleNamespace(id="one:a", in_scope=True),
        SimpleNamespace(id="one:b", in_scope=False),
        SimpleNamespace(id="two:a", in_scope=False),
        SimpleNamespace(id="two:b", in_scope=False),
    )
    groups = group_indices(cases, entity=True)

    correct = group_metrics(cases, groups, list(groups), np.array([0.9, 0.1, 0.2, 0.3]), 0.5)
    wrong = group_metrics(cases, groups, list(groups), np.array([0.1, 0.9, 0.8, 0.3]), 0.5)

    assert correct == {
        "groups": 2,
        "resolvable": 1,
        "correct_top_rank": 1,
        "exact_or_valid_selection": 2,
        "wrong_target": 0,
        "false_warning": 0,
        "false_allowance_on_ambiguous": 0,
    }
    assert wrong == {
        "groups": 2,
        "resolvable": 1,
        "correct_top_rank": 0,
        "exact_or_valid_selection": 0,
        "wrong_target": 1,
        "false_warning": 0,
        "false_allowance_on_ambiguous": 1,
    }


def test_group_arrays_and_normalization_do_not_use_transfer_rows():
    cases = (SimpleNamespace(id="one:a", in_scope=True), SimpleNamespace(id="one:b", in_scope=False))
    groups = group_indices(cases, entity=True)
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transfer = np.array([[100.0, 200.0]], dtype=np.float32)

    mean, scale, normalized, normalized_transfer = normalize_features(train, train, transfer)
    values, mask, gold = group_arrays(normalized, cases, groups, ["one"], 3)

    np.testing.assert_array_equal(mean, [2.0, 3.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0])
    assert values.shape == (1, 3, 2)
    np.testing.assert_array_equal(mask, [[True, True, False]])
    np.testing.assert_array_equal(gold, [0])
    np.testing.assert_array_equal(normalized_transfer, [[5.0, 5.0]])

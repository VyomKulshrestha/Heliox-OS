"""MisActBench splitting keeps balanced trajectory groups disjoint."""

from __future__ import annotations

from benchmarks.misactbench_general import split_cases


def _cases(group_count: int = 550) -> list[dict]:
    return [
        {
            "id": f"trajectory-{group:04d}:{int(misaligned)}",
            "group": f"trajectory-{group:04d}",
            "misaligned": misaligned,
        }
        for group in range(group_count)
        for misaligned in (False, True)
    ]


def _repeated_cases(group_count: int = 300) -> list[dict]:
    return [
        {
            "id": f"trajectory-{group:04d}:{int(misaligned)}:{step}",
            "group": f"trajectory-{group:04d}",
            "misaligned": misaligned,
        }
        for group in range(group_count)
        for misaligned in (False, True)
        for step in range(3)
    ]


def test_split_cases_scales_balanced_rows_without_group_leakage() -> None:
    limits = {"scope_train": 352, "scope_calibration": 64, "scope_test": 128}
    selected = split_cases(_cases(), seed=20260930, limits=limits)
    repeated = split_cases(list(reversed(_cases())), seed=20260930, limits=limits)

    assert {name: len(rows) for name, rows in selected.items()} == limits
    assert {name: [row["id"] for row in rows] for name, rows in selected.items()} == {
        name: [row["id"] for row in rows] for name, rows in repeated.items()
    }

    groups = {name: {row["group"] for row in rows} for name, rows in selected.items()}
    assert not (groups["scope_train"] & groups["scope_calibration"])
    assert not (groups["scope_train"] & groups["scope_test"])
    assert not (groups["scope_calibration"] & groups["scope_test"])
    for rows in selected.values():
        assert sum(not row["misaligned"] for row in rows) == len(rows) // 2
        assert sum(row["misaligned"] for row in rows) == len(rows) // 2


def test_split_cases_allows_bounded_repeats_only_inside_training_groups() -> None:
    cases = _repeated_cases()
    limits = {"scope_train": 352, "scope_calibration": 64, "scope_test": 128}
    selected = split_cases(cases, seed=20260930, limits=limits)
    repeated = split_cases(list(reversed(cases)), seed=20260930, limits=limits)

    assert {name: len(rows) for name, rows in selected.items()} == limits
    assert {name: [row["id"] for row in rows] for name, rows in selected.items()} == {
        name: [row["id"] for row in rows] for name, rows in repeated.items()
    }
    groups = {name: {row["group"] for row in rows} for name, rows in selected.items()}
    assert not (groups["scope_train"] & groups["scope_calibration"])
    assert not (groups["scope_train"] & groups["scope_test"])
    assert not (groups["scope_calibration"] & groups["scope_test"])

    train_counts: dict[tuple[str, bool], int] = {}
    for row in selected["scope_train"]:
        key = (row["group"], row["misaligned"])
        train_counts[key] = train_counts.get(key, 0) + 1
    assert max(train_counts.values()) == 2
    for rows in selected.values():
        assert sum(not row["misaligned"] for row in rows) == len(rows) // 2
        assert sum(row["misaligned"] for row in rows) == len(rows) // 2

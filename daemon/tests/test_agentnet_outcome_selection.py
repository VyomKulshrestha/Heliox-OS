"""AgentNet outcome split and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.select_agentnet_outcome_cases import (
    SEEDS,
    SOURCE_SHA256,
    SPLIT_ROWS,
    select,
)


def test_outcome_selection_is_group_disjoint_and_ood() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = json.loads((root / "agentnet_outcome_selection_v1.json").read_text())
    assert manifest["source_sha256"] == SOURCE_SHA256
    assert manifest["seeds"] == SEEDS
    assert {name: len(rows) for name, rows in manifest["splits"].items()} == SPLIT_ROWS
    ids = [case_id for name in SPLIT_ROWS for case_id in manifest["splits"][name]]
    assert ids == manifest["case_ids"]
    assert len(ids) == len({case_id.split(":", 1)[0] for case_id in ids})
    assert manifest["label_provenance"] == "model_generated_reflector_step_correct_not_human"
    assert "reflection" in manifest["excluded_fields"]
    assert "thought" in manifest["excluded_fields"]


def test_selector_balances_classes_without_reusing_tasks() -> None:
    candidates = [
        {"id": f"base-{index}:{index}", "group": f"base-{index}", "domain": "base", "correct": index % 2 == 0}
        for index in range(20)
    ] + [
        {"id": f"new-{index}:{index}", "group": f"new-{index}", "domain": "new", "correct": index % 2 == 0}
        for index in range(4)
    ]
    sizes = {"train": 4, "calibration": 2, "development": 2, "final": 2, "ood": 4}
    first = select(candidates, ood_domain="new", split_rows=sizes)
    assert first == select(candidates, ood_domain="new", split_rows=sizes)
    ids = [case_id for rows in first.values() for case_id in rows]
    assert len(ids) == len(set(ids)) == sum(sizes.values())
    assert all(case_id.startswith("new-") for case_id in first["ood"])
    assert all(not case_id.startswith("new-") for name in first if name != "ood" for case_id in first[name])
    lookup = {row["id"]: row["correct"] for row in candidates}
    assert all(sum(lookup[case_id] for case_id in rows) * 2 == len(rows) for rows in first.values())

"""Pure-data guards for the frozen released-checkpoint comparison."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.evaluate_released_openjev_general import (
    input_signature,
    load_logits_checkpoint,
    select_archived_rows,
    select_fresh_rows,
    summarize,
)


def test_archived_selection_is_pinned_without_teacher_report(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"id": "a", "kind": "noul", "options": [], "group_id": "ga"},
        {"id": "b", "kind": "score", "options": [str(i) for i in range(11)], "group_id": "gb"},
        {"id": "c", "kind": "choice", "options": ["x", "y"], "group_id": "gc"},
    ]
    manifest = {"sha256": "fixture"}
    monkeypatch.setattr("scripts.evaluate_released_openjev_general.load_split",
                        lambda *_args: (rows, manifest))
    digest = hashlib.sha256(b"a\nc").hexdigest()

    selected, actual_manifest = select_archived_rows(
        seed=1, pool=3, limit=2, expected_ids_sha256=digest)

    assert [row["id"] for row in selected] == ["a", "c"]
    assert actual_manifest == manifest
    with pytest.raises(ValueError, match="pinned row hash"):
        select_archived_rows(seed=1, pool=3, limit=2, expected_ids_sha256="wrong")


def test_fresh_selection_excludes_archived_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"id": f"row-{index}", "group_id": f"group-{index}"} for index in range(5)]
    rows.insert(2, {"id": "duplicate", "group_id": "group-1"})
    monkeypatch.setattr("scripts.evaluate_released_openjev_general.load_split",
                        lambda *_args: (rows, {"sha256": "fixture"}))
    fresh, manifest = select_fresh_rows([rows[0]], seed=4, pool=5, limit=3)
    assert [row["id"] for row in fresh] == ["row-1", "row-2", "row-3"]
    assert manifest == {"sha256": "fixture"}
    with pytest.raises(ValueError, match="group-disjoint"):
        select_fresh_rows(rows[:4], seed=4, pool=5, limit=3)


def test_checkpoint_signature_binds_inputs_and_model_limit() -> None:
    rows = [{"id": "a", "state_json": "{}", "kind": "choice", "question": "Which?",
             "options": ["one", "two"], "target": [0.0, 1.0]}]
    first = input_signature(rows, {"sha256": "source-a"}, max_length=1024)
    assert first != input_signature(rows, {"sha256": "source-b"}, max_length=1024)
    assert first != input_signature(rows, {"sha256": "source-a"}, max_length=768)
    rows[0]["question"] = "Which value?"
    assert first != input_signature(rows, {"sha256": "source-a"}, max_length=1024)


def test_summary_keeps_public_error_ids() -> None:
    rows = [{"id": "x", "source": "s", "kind": "choice", "options": ["a", "b"],
             "target": [0.0, 1.0]}]
    result = summarize(rows, [np.array([2.0, 0.0])])
    assert result["top1_accuracy"] == 0.0
    assert result["wrong_ids"] == ["x"]


def test_cached_logits_require_exact_selection(tmp_path) -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    path = tmp_path / "logits.json"
    path.write_text('{"schema_version":1,"signature":"frozen","ids":["a","b"],"logits":[[0,1],[2,3]]}',
                    encoding="utf-8")

    values = load_logits_checkpoint(path, rows, signature="frozen")

    assert np.array_equal(values, np.array([[0, 1], [2, 3]], dtype=np.float64))
    with pytest.raises(ValueError, match="selection"):
        load_logits_checkpoint(path, rows, signature="other")

"""The general benchmark must replay the same rows as archived Jev evidence."""

import hashlib

import pytest

from scripts import evaluate_world_general_qlora as probe


def test_selected_rows_rejects_selection_drift(monkeypatch):
    rows = [{"id": f"row-{index}", "kind": "noul", "options": ["no", "yes"]}
            for index in range(128)]
    monkeypatch.setattr(probe, "load_split", lambda *_args: (rows, {"sha256": "source"}))
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    reference = {"calls": 128, "selected_ids_sha256": digest,
                 "dataset_manifest": {"sha256": "source"}}
    assert probe.selected_rows(reference)[0] == rows
    reference["selected_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selection differs"):
        probe.selected_rows(reference)

"""Fresh transfer selection never reuses a prior task group."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from scripts import select_agentnet_fresh_transfer as transfer
from scripts.extract_agentnet_outcome_features import sha256_file
from tests.eol import assert_pinned_text_hash


def test_reserve_excludes_prior_groups_and_pairs_new_domains(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(transfer, "FRESH_SPLIT_ROWS", {"final": 2, "ood": 2})
    candidates = [
        {"id": f"{name}:0", "group": name, "domain": domain, "correct": correct}
        for name, domain, correct in (
            ("used", "Ubuntu", False),
            ("normal-no", "Ubuntu", False),
            ("normal-yes", "Ubuntu", True),
            ("vscode-no", "VScode", False),
            ("vscode-yes", "VScode", True),
        )
    ]
    monkeypatch.setattr(transfer, "load_candidates", lambda _source: (candidates, Counter(groups=len(candidates))))
    previous = {
        "source_sha256": transfer.SOURCE_SHA256,
        "case_ids": ["used:0"],
        "ood_domain": "VScode",
        "label_provenance": "synthetic_test",
    }
    selected = transfer.reserve(tmp_path / "unused", previous, "previous-hash")
    assert selected["previous_manifest_sha256"] == "previous-hash"
    assert selected["splits"]["final"] == ["normal-no:0", "normal-yes:0"]
    assert selected["splits"]["ood"] == ["vscode-no:0", "vscode-yes:0"]
    assert "used:0" not in selected["case_ids"]


def test_tracked_fresh_reservation_is_disjoint_from_fit_groups() -> None:
    benchmark = Path(__file__).resolve().parents[1] / "benchmarks"
    previous_path = benchmark / "agentnet_outcome_selection_v1.json"
    fresh_path = benchmark / "agentnet_fresh_transfer_selection_v1.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    pre_action = json.loads((benchmark / "agentnet_preaction_dev_v1.json").read_text(encoding="utf-8"))
    previous_groups = {case_id.split(":", 1)[0] for case_id in previous["case_ids"]}
    fresh_groups = [case_id.split(":", 1)[0] for case_id in fresh["case_ids"]]
    assert_pinned_text_hash(fresh["previous_manifest_sha256"], previous_path)
    assert previous_groups.isdisjoint(fresh_groups)
    assert len(fresh_groups) == len(set(fresh_groups)) == 256
    assert {name: len(ids) for name, ids in fresh["splits"].items()} == fresh["split_rows"]
    assert_pinned_text_hash(pre_action["fit_selection_sha256"], previous_path)
    assert_pinned_text_hash(pre_action["fresh_transfer_selection_sha256"], fresh_path)
    assert pre_action["fresh_transfer_labels_opened"] is False
    assert pre_action["promotion"] == "none"


def test_tracked_paired_transfer_uses_frozen_heads_and_exact_auc_differences() -> None:
    benchmark = Path(__file__).resolve().parents[1] / "benchmarks"
    paired = json.loads((benchmark / "agentnet_paired_fresh_transfer_v1.json").read_text(encoding="utf-8"))
    pre = json.loads((benchmark / "agentnet_preaction_dev_v1.json").read_text(encoding="utf-8"))
    post = json.loads((benchmark / "agentnet_outcome_dev_v1.json").read_text(encoding="utf-8"))
    assert_pinned_text_hash(paired["fresh_manifest_sha256"], benchmark / "agentnet_fresh_transfer_selection_v1.json")
    assert paired["training"]["pre_action"]["head_sha256"] == pre["head_sha256"]
    assert paired["training"]["post_action"]["head_sha256"] == post["head_sha256"]
    for split in ("final", "ood"):
        result = paired["results"][split]
        assert math.isclose(
            result["post_minus_pre_auc"],
            result["post_action"]["default"]["auc"] - result["pre_action"]["default"]["auc"],
        )
    assert paired["promotion"] == "none"

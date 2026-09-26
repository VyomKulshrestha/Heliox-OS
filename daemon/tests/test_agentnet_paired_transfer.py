"""Paired transfer opens fresh labels only after both frozen arms validate."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts import evaluate_agentnet_paired_transfer as paired
from scripts.extract_agentnet_outcome_features import save, sha256_file


def test_stratified_paired_auc_interval() -> None:
    labels = np.asarray([True, True, False, False])
    post = np.asarray([0.9, 0.8, 0.2, 0.1])
    pre = 1 - post
    assert paired.paired_auc_interval(labels, post, pre) == [1.0, 1.0]


def test_paired_transfer_evaluates_two_valid_frozen_views(tmp_path, monkeypatch) -> None:
    fit_ids = ["train-yes:0", "train-no:0"]
    fresh_ids = ["final-yes:0", "final-no:0", "ood-yes:0", "ood-no:0"]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({
        "task_id": case_id.split(":", 1)[0],
        "traj": [{"index": 0, "value": {"last_step_correct": "yes" in case_id}}],
    }) + "\n" for case_id in fit_ids + fresh_ids), encoding="utf-8")
    source_hash = sha256_file(source)
    monkeypatch.setattr(paired, "SOURCE_SHA256", source_hash)
    fit_manifest = tmp_path / "fit.json"
    fit_manifest.write_text(json.dumps({"source_sha256": source_hash,
                                        "case_ids": fit_ids}), encoding="utf-8")
    fit_hash = sha256_file(fit_manifest)
    fresh_manifest = tmp_path / "fresh.json"
    fresh_manifest.write_text(json.dumps({
        "source_sha256": source_hash, "previous_manifest_sha256": fit_hash,
        "label_provenance": "synthetic_test", "case_ids": fresh_ids,
        "splits": {"final": fresh_ids[:2], "ood": fresh_ids[2:]},
    }), encoding="utf-8")
    args = ["paired", "--source", str(source), "--fit-manifest", str(fit_manifest),
            "--fresh-manifest", str(fresh_manifest)]
    for short, view in (("post", "post_action"), ("pre", "pre_action")):
        features = tmp_path / f"{short}-features.npz"
        save(features, f"{short}-signature", fresh_ids,
             [np.asarray([float("yes" in case_id), 0.0]) for case_id in fresh_ids])
        feature_report = tmp_path / f"{short}-feature-report.json"
        feature_report.write_text(json.dumps({
            "source_sha256": source_hash, "manifest_sha256": sha256_file(fresh_manifest),
            "features_sha256": sha256_file(features), "signature": f"{short}-signature",
            "model": paired.MODEL, "revision": paired.REVISION,
            "input_view": view, "prompt_sha256": f"{short}-prompt",
        }), encoding="utf-8")
        artifact = tmp_path / f"{short}-head.npz"
        np.savez_compressed(artifact, mean=np.zeros(2), scale=np.ones(2),
                            coefficient=np.asarray([[4.0, 0.0]]), intercept=np.asarray([-2.0]),
                            cautious_threshold=0.5, manifest_sha256=fit_hash,
                            feature_sha256=f"{short}-fit-feature")
        fit_report = tmp_path / f"{short}-fit-report.json"
        fit_report.write_text(json.dumps({
            "source_sha256": source_hash, "manifest_sha256": fit_hash,
            "feature_sha256": f"{short}-fit-feature", "artifact_sha256": sha256_file(artifact),
            "input_view": view, "final_and_ood_labels_opened": False,
        }), encoding="utf-8")
        freeze = tmp_path / f"{short}-freeze.json"
        freeze.write_text(json.dumps({
            "fit_selection_sha256": fit_hash, "features_sha256": f"{short}-fit-feature",
            "head_sha256": sha256_file(artifact), "promotion": "none",
        }), encoding="utf-8")
        for name, path in (("features", features), ("feature-report", feature_report),
                           ("fit-report", fit_report), ("artifact", artifact),
                           ("freeze-report", freeze)):
            args.extend((f"--{short}-{name}", str(path)))
    report = tmp_path / "paired.json"
    monkeypatch.setattr(sys, "argv", [*args, "--report", str(report)])
    paired.main()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["fresh_task_groups"] == 4
    assert result["results"]["final"]["pre_action"]["default"]["auc"] == 1.0
    assert result["results"]["ood"]["post_action"]["default"]["false_allowances"] == 0
    assert result["promotion"] == "none"
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    frozen["head_sha256"] = "tampered"
    freeze.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(paired, "labels_for", lambda *_args: pytest.fail("labels opened early"))
    monkeypatch.setattr(sys, "argv", [*args, "--report", str(tmp_path / "invalid.json")])
    with pytest.raises(ValueError, match="frozen provenance"):
        paired.main()

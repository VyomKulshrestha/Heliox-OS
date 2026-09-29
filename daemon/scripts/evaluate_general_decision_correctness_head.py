"""Fit a research-only correctness scorer and test it on untouched groups.

The frozen decision model and action-execution path are not changed. All rows
come from licensed public Open-Jev data; no TypeSafe Jev calls or outputs are
used. Fit and evaluation partitions must be outside the released model's
training mixture and every previously exposed group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.general_decision import (  # noqa: E402
    DATASET_REVISION,
    correctness_features,
    correctness_metrics,
    fit_correctness_scorer,
    load_split,
    metrics,
    paired_aurc_bootstrap_delta,
    paired_mean_bootstrap_delta,
    predict_correctness,
    verify_disjoint,
)
from scripts.evaluate_released_openjev_general import (  # noqa: E402
    load_excluded_group_ids,
    select_archived_rows,
    select_fresh_rows,
)
from scripts.score_general_decision import score_requests, verify_artifact  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402

PROTOCOL = ROOT / "benchmarks/openjev_correctness_head_v2.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Protocol paths must stay inside the benchmark root")
    return path


def checked_json(relative: str, expected_sha256: str) -> dict:
    path = repo_file(relative)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Pinned protocol checksum mismatch: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Pinned protocol is not an object: {relative}")
    return value


def prior_groups(protocol: dict) -> set[str]:
    """Reconstruct previous exposure groups from checksum-pinned protocols."""
    base = checked_json(protocol["base_protocol_path"], protocol["base_protocol_sha256"])
    groups = load_excluded_group_ids(ROOT, base["excluded_group_manifest"])
    for descriptor in protocol["prior_comparison_protocols"]:
        comparison = checked_json(descriptor["path"], descriptor["sha256"])
        source_descriptor = comparison["source_selection_protocol"]
        source = checked_json(source_descriptor["path"], source_descriptor["sha256"])
        archived, _ = select_archived_rows(
            seed=source["archived_seed"], pool=source["archived_pool"],
            limit=source["archived_rows"],
            expected_ids_sha256=source["archived_ids_sha256"],
        )
        exclusions = load_excluded_group_ids(ROOT, source.get("excluded_group_manifest"))
        fresh, _ = select_fresh_rows(
            archived, split=source.get("fresh_split", "ood"),
            seed=source["fresh_seed"], pool=source["fresh_pool"],
            limit=source["fresh_rows"], excluded_group_ids=exclusions,
        )
        groups.update(row["group_id"] for row in archived + fresh)
    return groups


def digest_lines(values) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def validate_split_policy(protocol: dict) -> None:
    """Do not fit or report on a split used to train the released model."""
    training_splits = protocol.get("base_training_splits")
    fit_split = protocol.get("fit_split")
    test_split = protocol.get("test_split")
    if (not isinstance(training_splits, list) or not training_splits
            or any(not isinstance(split, str) or not split for split in training_splits)
            or not isinstance(fit_split, str) or not isinstance(test_split, str)
            or fit_split == test_split
            or fit_split in training_splits or test_split in training_splits):
        raise ValueError("Fit and evaluation splits must be distinct from the base training mixture")


def select_unseen_group_rows(rows: list[dict], excluded_groups: set[str], limit: int) -> list[dict]:
    """Select in source order while excluding groups already used or selected."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Selection limit must be a positive integer")
    seen = set(excluded_groups)
    selected = []
    for row in rows:
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Selection rows must carry a nonempty group ID")
        if group_id in seen:
            continue
        selected.append(row)
        seen.add(group_id)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"Not enough previously unseen groups: selected {len(selected)} of {limit}")
    return selected


def select_partitions(protocol: dict) -> tuple[list[dict], list[dict], dict, dict]:
    """Recreate fresh calibration/evaluation rows from frozen split identities."""
    validate_split_policy(protocol)
    legacy = checked_json(
        protocol["legacy_diagnostic_protocol"]["path"],
        protocol["legacy_diagnostic_protocol"]["sha256"],
    )
    groups = prior_groups(legacy)
    groups_sha256 = digest_lines(sorted(groups))
    if (len(groups) != protocol["prior_excluded_group_count"]
            or groups_sha256 != protocol["prior_excluded_group_ids_sha256"]):
        raise ValueError("Reconstructed prior exposure groups differ from the frozen protocol")
    candidate_training = checked_json(
        protocol["candidate_training_protocol"]["path"],
        protocol["candidate_training_protocol"]["sha256"],
    )
    if (candidate_training.get("source", {}).get("revision") != DATASET_REVISION
            or candidate_training.get("training", {}).get("warm_start_from_released_openjev") is not True):
        raise ValueError("Candidate training provenance is not the pinned public Open-Jev protocol")
    candidate_groups = load_excluded_group_ids(ROOT, protocol["candidate_exposure_manifest"])
    if (len(candidate_groups) != protocol["candidate_exposure_group_count"]
            or digest_lines(sorted(candidate_groups))
            != protocol["candidate_exposure_group_ids_sha256"]
            or not candidate_groups <= groups):
        raise ValueError("Candidate training/calibration/evaluation exposure is not fully excluded")

    # Rebuild and exclude the earlier train-split diagnostic. Those rows were
    # already seen by the released checkpoint and must not be reused for tuning.
    diagnostic = protocol["legacy_diagnostic_selection"]
    diagnostic_rows, diagnostic_manifest = load_split(
        diagnostic["split"], diagnostic["pool"], diagnostic["seed"])
    if diagnostic_manifest["sha256"] != protocol["dataset_sha256_by_split"][diagnostic["split"]]:
        raise ValueError("Pinned legacy diagnostic split checksum changed")
    diagnostic_rows = select_unseen_group_rows(
        diagnostic_rows, groups, diagnostic["selected_rows"])
    if (digest_lines(row["id"] for row in diagnostic_rows) != diagnostic["ids_sha256"]
            or digest_lines(sorted({row["group_id"] for row in diagnostic_rows}))
            != diagnostic["group_ids_sha256"]):
        raise ValueError("Legacy diagnostic exposure differs from the frozen selection")
    groups.update(row["group_id"] for row in diagnostic_rows)
    combined_hash = digest_lines(sorted(groups))
    if (len(groups) != protocol["excluded_before_new_selection_group_count"]
            or combined_hash != protocol["excluded_before_new_selection_group_ids_sha256"]):
        raise ValueError("Legacy exposure groups differ from the frozen protocol")

    selected_partitions = {}
    manifests = {}
    for name in ("fit", "test"):
        split = protocol[f"{name}_split"]
        rows, manifest = load_split(split, protocol[f"{name}_pool"], protocol[f"{name}_seed"])
        if manifest["sha256"] != protocol["dataset_sha256_by_split"][split]:
            raise ValueError(f"Pinned {split} split checksum changed")
        selected = select_unseen_group_rows(rows, groups, protocol[f"{name}_rows"])
        if (digest_lines(row["id"] for row in selected) != protocol[f"{name}_ids_sha256"]
                or digest_lines(sorted({row["group_id"] for row in selected}))
                != protocol[f"{name}_group_ids_sha256"]):
            raise ValueError(f"Frozen {name} selection differs from the recorded IDs")
        groups.update(row["group_id"] for row in selected)
        selected_partitions[name] = selected
        manifests[name] = manifest
    verify_disjoint({"correctness_fit": selected_partitions["fit"],
                     "correctness_test": selected_partitions["test"]})
    return selected_partitions["fit"], selected_partitions["test"], manifests["fit"], manifests["test"]


def option_probabilities(row: dict, prediction: dict) -> np.ndarray:
    answer = prediction.get("answer", {})
    if row["kind"] == "noul":
        yes = answer.get("yes_probability")
        values = np.asarray([1.0 - float(yes), float(yes)], dtype=np.float64)
    else:
        probabilities = answer.get("probabilities", {})
        values = np.asarray([probabilities.get(option) for option in row["options"]], dtype=np.float64)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.isclose(values.sum(), 1.0, atol=1e-5)):
        raise ValueError("Candidate returned an invalid option distribution")
    return values


def load_or_score(rows: list[dict], tokenizer, candidate_report: dict, artifact_dir: Path,
                  cache_path: Path, *, resume: bool) -> list[dict]:
    encoded, overlong = encode_rows(
        tokenizer, rows, candidate_report["max_length"],
        prompt_style=candidate_report.get("prompt_style", "candidate_v1"),
    )
    eligible = [item["row"] for item in encoded]
    if not eligible or len(eligible) < 32:
        raise ValueError("Too few tokenizable rows remain for an informative comparison")
    ids = [row["id"] for row in eligible]
    signature = hashlib.sha256(json.dumps({
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": candidate_report["dataset_sha256"],
        "adapter_sha256": candidate_report["adapter_sha256"],
        "head_sha256": candidate_report["head_sha256"],
        "ids": ids,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    predictions = []
    if cache_path.exists():
        if not resume:
            raise FileExistsError(f"Prediction cache exists; pass --resume to reuse it: {cache_path}")
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if (saved.get("signature") != signature or saved.get("ids") != ids
                or not isinstance(saved.get("predictions"), list)
                or len(saved["predictions"]) > len(ids)):
            raise ValueError("Prediction cache does not match the frozen rows and model")
        predictions = saved["predictions"]
    start = len(predictions)

    def save(index: int, prediction: dict) -> None:
        if start + index != len(predictions):
            raise ValueError("Predictions arrived out of order")
        predictions.append(prediction)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"signature": signature, "ids": ids,
                                         "predictions": predictions}, allow_nan=False),
                             encoding="utf-8")
        os.replace(temporary, cache_path)
        if len(predictions) % 16 == 0 or len(predictions) == len(eligible):
            print(json.dumps({"event": "gpu_scoring", "completed": len(predictions),
                              "total": len(eligible)}), flush=True)

    if start < len(eligible):
        payloads = [{"state": json.loads(row["state_json"]), "kind": row["kind"],
                     "question": row["question"], "options": row["options"]}
                    for row in eligible[start:]]
        score_requests(payloads, artifact_dir, repo_file(candidate_report["report_path"]),
                       on_prediction=save, max_decisions=len(eligible))
    if len(predictions) != len(eligible):
        raise ValueError("Candidate prediction count does not match eligible rows")
    return [{"row": row, "prediction": prediction}
            for row, prediction in zip(eligible, predictions, strict=True)], overlong


def correctness_labels(scored: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    raw_confidence = []
    labels = []
    for item in scored:
        row = item["row"]
        distribution = option_probabilities(row, item["prediction"])
        features.append(correctness_features(distribution))
        raw_confidence.append(float(distribution.max()))
        labels.append(int(np.argmax(distribution)) == int(np.argmax(row["target"])))
    return np.stack(features), np.asarray(raw_confidence), np.asarray(labels, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.protocol.resolve() != PROTOCOL.resolve():
        parser.error("This evaluator only accepts the frozen v2 protocol")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != 2
            or protocol.get("dataset_revision") != DATASET_REVISION):
        raise ValueError("Protocol dataset revision differs from the loader")
    validate_split_policy(protocol)
    if (protocol.get("research_only") is not True
            or protocol.get("execution_authority") is not False
            or protocol.get("uses_typesafe_jev_outputs") is not False
            or protocol.get("promotion") != "none"):
        raise ValueError("Correctness-head evaluation must remain research-only")

    output_dir = repo_file(protocol["output_dir"])
    report_path = output_dir / "report.json"
    if report_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite the research report: {report_path}")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (previous.get("purpose") != "research_only_posthoc_correctness_scoring_v2"
                or previous.get("candidate_adapter_sha256") != protocol["candidate_adapter_sha256"]
                or previous.get("candidate_head_sha256") != protocol["candidate_head_sha256"]
                or previous.get("selection", {}).get("fit_ids_sha256") is None
                or previous.get("selection", {}).get("test_ids_sha256") is None):
            raise ValueError("Existing report is not a matching resumable correctness experiment")
    candidate_report_path = repo_file(protocol["candidate_report_path"])
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (sha256_file(candidate_report_path) != protocol["candidate_report_sha256"]
            or candidate_report.get("initial_checkpoint") != protocol["candidate_initial_checkpoint"]):
        raise ValueError("Frozen candidate report or Open-Jev warm-start provenance is invalid")
    artifact_dir = repo_file(protocol["candidate_artifact_dir"])
    verified_report, _ = verify_artifact(artifact_dir, candidate_report_path)
    if (verified_report["adapter_sha256"] != protocol["candidate_adapter_sha256"]
            or verified_report["head_sha256"] != protocol["candidate_head_sha256"]
            or verified_report.get("initial_checkpoint") != protocol["candidate_initial_checkpoint"]):
        raise ValueError("Frozen candidate artifact checksums differ from the protocol")
    baseline_report_path = repo_file(protocol["baseline_report_path"])
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if (sha256_file(baseline_report_path) != protocol["baseline_report_sha256"]
            or baseline_report.get("research_only") is not True
            or baseline_report.get("execution_authority") is not False
            or baseline_report.get("source_license") != "Apache-2.0"):
        raise ValueError("Frozen released Open-Jev baseline provenance is invalid")
    baseline_artifact_dir = repo_file(protocol["baseline_artifact_dir"])
    verified_baseline, _ = verify_artifact(baseline_artifact_dir, baseline_report_path)
    if (verified_baseline.get("initial_checkpoint") != protocol["candidate_initial_checkpoint"]
            or verified_baseline.get("adapter_sha256") != protocol["baseline_adapter_sha256"]
            or verified_baseline.get("head_sha256") != protocol["baseline_head_sha256"]):
        raise ValueError("Released Open-Jev baseline artifact differs from the frozen protocol")

    fit_rows, test_rows, fit_manifest, test_manifest = select_partitions(protocol)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(candidate_report["model"], revision=candidate_report["revision"],
                          local_files_only=True), local_files_only=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scored, fit_overlong = load_or_score(
        fit_rows, tokenizer, {**candidate_report, "dataset_sha256": fit_manifest["sha256"],
                              "report_path": protocol["candidate_report_path"]},
        artifact_dir, output_dir / "fit_predictions.json", resume=args.resume,
    )
    test_scored, test_overlong = load_or_score(
        test_rows, tokenizer, {**candidate_report, "dataset_sha256": test_manifest["sha256"],
                               "report_path": protocol["candidate_report_path"]},
        artifact_dir, output_dir / "test_predictions.json", resume=args.resume,
    )
    verify_disjoint({"scored_correctness_fit": [item["row"] for item in scored],
                     "scored_correctness_test": [item["row"] for item in test_scored]})
    baseline_scored, baseline_overlong = load_or_score(
        test_rows, tokenizer, {**baseline_report, "dataset_sha256": test_manifest["sha256"],
                               "report_path": protocol["baseline_report_path"]},
        baseline_artifact_dir, output_dir / "baseline_test_predictions.json", resume=args.resume,
    )
    baseline_by_id = {item["row"]["id"]: item["prediction"] for item in baseline_scored}
    common_test = [item for item in test_scored if item["row"]["id"] in baseline_by_id]
    common_rows = [item["row"] for item in common_test]
    candidate_logits = [np.log(np.clip(option_probabilities(
        item["row"], item["prediction"]), 1e-12, 1.0)) for item in common_test]
    baseline_logits = [np.log(np.clip(option_probabilities(
        item["row"], baseline_by_id[item["row"]["id"]]), 1e-12, 1.0)) for item in common_test]
    if len(common_rows) < 32:
        raise ValueError("Too few common held-out rows for a released-model comparison")
    candidate_decisions = metrics(common_rows, candidate_logits, 1.0)
    baseline_decisions = metrics(common_rows, baseline_logits, 1.0)
    candidate_correct = []
    baseline_correct = []
    candidate_nll = []
    baseline_nll = []
    candidate_brier = []
    baseline_brier = []
    for item in common_test:
        row = item["row"]
        target = np.asarray(row["target"], dtype=np.float64)
        candidate_distribution = option_probabilities(row, item["prediction"])
        baseline_distribution = option_probabilities(row, baseline_by_id[row["id"]])
        gold = int(np.argmax(target))
        candidate_correct.append(int(np.argmax(candidate_distribution)) == gold)
        baseline_correct.append(int(np.argmax(baseline_distribution)) == gold)
        candidate_nll.append(float(-target @ np.log(np.clip(candidate_distribution, 1e-12, 1.0))))
        baseline_nll.append(float(-target @ np.log(np.clip(baseline_distribution, 1e-12, 1.0))))
        candidate_brier.append(float(np.mean((candidate_distribution - target) ** 2)))
        baseline_brier.append(float(np.mean((baseline_distribution - target) ** 2)))
    metric_bootstrap = {
        "top1_accuracy": paired_mean_bootstrap_delta(
            np.asarray(candidate_correct, dtype=float).tolist(),
            np.asarray(baseline_correct, dtype=float).tolist(),
            replicates=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
        ),
        "mean_nll": paired_mean_bootstrap_delta(
            candidate_nll, baseline_nll,
            replicates=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
        ),
        "mean_brier": paired_mean_bootstrap_delta(
            candidate_brier, baseline_brier,
            replicates=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
        ),
    }
    decision_delta = {
        "top1_accuracy": candidate_decisions["top1_accuracy"] - baseline_decisions["top1_accuracy"],
        "mean_nll": candidate_decisions["mean_nll"] - baseline_decisions["mean_nll"],
        "mean_brier": candidate_decisions["mean_brier"] - baseline_decisions["mean_brier"],
    }

    fit_features, _, fit_labels = correctness_labels(scored)
    test_features, raw_scores, test_labels = correctness_labels(test_scored)
    scorer = fit_correctness_scorer(fit_features, fit_labels.tolist())
    learned_scores = predict_correctness(test_features, scorer)
    raw_metrics = correctness_metrics(
        raw_scores.tolist(), test_labels.tolist(), selection_score="maximum_candidate_probability")
    learned_metrics = correctness_metrics(learned_scores.tolist(), test_labels.tolist())
    by_kind = {}
    for kind in sorted({item["row"]["kind"] for item in test_scored}):
        mask = np.asarray([item["row"]["kind"] == kind for item in test_scored])
        by_kind[kind] = {
            "rows": int(mask.sum()),
            "raw_max_probability": correctness_metrics(
                raw_scores[mask].tolist(), test_labels[mask].tolist(),
                selection_score="maximum_candidate_probability"),
            "learned_correctness_probability": correctness_metrics(
                learned_scores[mask].tolist(), test_labels[mask].tolist()),
        }
    bootstrap = paired_aurc_bootstrap_delta(
        raw_scores.tolist(), learned_scores.tolist(), test_labels.tolist(),
        replicates=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
    )
    report = {
        "schema_version": 2,
        "purpose": "research_only_posthoc_correctness_scoring_v2",
        "dataset_revision": DATASET_REVISION,
        "protocol_sha256": sha256_file(args.protocol),
        "dataset_sha256_by_split": {
            protocol["fit_split"]: fit_manifest["sha256"],
            protocol["test_split"]: test_manifest["sha256"],
        },
        "candidate_model": candidate_report["model"],
        "candidate_revision": candidate_report["revision"],
        "candidate_adapter_sha256": candidate_report["adapter_sha256"],
        "candidate_head_sha256": candidate_report["head_sha256"],
        "released_openjev_head_sha256": baseline_report["head_sha256"],
        "uses_typesafe_jev_outputs": False,
        "research_only": True,
        "execution_authority": False,
        "selection": {
            "fit_split": protocol["fit_split"], "fit_seed": protocol["fit_seed"],
            "test_split": protocol["test_split"], "test_seed": protocol["test_seed"],
            "prior_excluded_group_count": protocol["excluded_before_new_selection_group_count"],
            "fit_rows_selected": len(fit_rows), "test_rows_selected": len(test_rows),
            "fit_rows_scored": len(scored), "test_rows_scored": len(test_scored),
            "fit_overlength_skipped": fit_overlong,
            "test_overlength_skipped": test_overlong,
            "fit_ids_sha256": hashlib.sha256("\n".join(
                item["row"]["id"] for item in scored).encode()).hexdigest(),
            "test_ids_sha256": hashlib.sha256("\n".join(
                item["row"]["id"] for item in test_scored).encode()).hexdigest(),
            "fit_group_count": len({item["row"]["group_id"] for item in scored}),
            "test_group_count": len({item["row"]["group_id"] for item in test_scored}),
            "fit_group_ids_sha256": hashlib.sha256("\n".join(sorted(
                {item["row"]["group_id"] for item in scored})).encode()).hexdigest(),
            "test_group_ids_sha256": hashlib.sha256("\n".join(sorted(
                {item["row"]["group_id"] for item in test_scored})).encode()).hexdigest(),
        },
        "correctness_scorer": {
            "features": ["top_probability", "top_two_margin", "normalized_entropy",
                         "log_option_count"],
            "fit_rows": len(scored), "fit_accuracy": float(fit_labels.mean()),
            "l2": scorer["l2"], "mean": scorer["mean"].tolist(),
            "scale": scorer["scale"].tolist(), "weights": scorer["weights"].tolist(),
            "bias": scorer["bias"],
        },
        "heldout_evaluation": {
            "top1_accuracy": float(test_labels.mean()),
            "same_row_model_comparison": {
                "rows": len(common_rows),
                "released_openjev": baseline_decisions,
                "heliox_rps_candidate": candidate_decisions,
                "candidate_minus_released": decision_delta,
                "paired_bootstrap_delta": metric_bootstrap,
                "released_baseline_rows_scored": len(baseline_scored),
                "released_baseline_overlength_skipped": baseline_overlong,
            },
            "raw_max_probability": raw_metrics,
            "learned_correctness_probability": learned_metrics,
            "by_kind": by_kind,
            "paired_bootstrap_delta_aurc": bootstrap,
            "aurc_delta_learned_minus_raw": (
                learned_metrics["selective_prediction"]["aurc"]
                - raw_metrics["selective_prediction"]["aurc"]),
        },
        "limitations": [
            "This is a small public-data research experiment, not proprietary Jev parity.",
            "AURC and calibration metrics are descriptive on one untouched public selection; no risk guarantee is claimed.",
            "The correctness scorer does not change the selected action, approvals, or execution authority.",
            "The public Open-Jev labels do not establish success on live Heliox desktop tasks.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"fit_rows": len(scored), "test_rows": len(test_scored),
                      "raw_aurc": raw_metrics["selective_prediction"]["aurc"],
                      "learned_aurc": learned_metrics["selective_prediction"]["aurc"],
                      "raw_brier": raw_metrics["brier"], "learned_brier": learned_metrics["brier"],
                      "candidate_accuracy": candidate_decisions["top1_accuracy"],
                      "baseline_accuracy": baseline_decisions["top1_accuracy"],
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

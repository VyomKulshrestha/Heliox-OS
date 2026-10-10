"""Compare a frozen candidate with released Open-Jev on pinned fresh public rows.

The candidate is evaluated only on rows that also have verified released-model
logits. Overlength prompts are excluded without truncation, and no labels from
this selection are used for training or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.general_decision import DATASET_REVISION, grouped_mean_bootstrap_delta, metrics
from scripts.evaluate_general_decision_artifact import (
    prediction_metrics,
    retain_tokenizable_rows,
)
from scripts.evaluate_released_opendec_general import (
    OPENJEV_REPO,
    OPENJEV_REVISION,
    input_signature,
    load_excluded_group_ids,
    select_archived_rows,
    select_fresh_rows,
    verify_fresh_selection,
)
from scripts.score_general_decision import score_requests, verify_artifact
from scripts.train_general_decision import encode_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/opendec_scale_1024_rps_fresh_protocol_v3.json"


def resolve_repo_path(value: str, *, allowed_root: Path, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must stay inside the repository")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"{label} path is outside its allowed directory")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_rows(rows: list[dict], saved: dict, *, signature: str) -> list[np.ndarray]:
    """Require baseline logits to match the exact ordered selection and labels."""
    ids = [row["id"] for row in rows]
    raw = saved.get("logits")
    if (
        saved.get("schema_version") != 1
        or saved.get("signature") != signature
        or saved.get("ids") != ids
        or not isinstance(raw, list)
        or len(raw) != len(rows)
    ):
        raise ValueError("Released baseline logits do not match the frozen fresh selection")
    logits = [np.asarray(values, dtype=np.float64) for values in raw]
    if any(
        values.shape != (len(row["target"]),) or not np.isfinite(values).all()
        for row, values in zip(rows, logits, strict=True)
    ):
        raise ValueError("Released baseline logits contain invalid values")
    return logits


def load_partial(path: Path, *, signature: str, ids: list[str], resume: bool) -> list[dict]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"Fresh replication checkpoint already exists: {path}; pass --resume")
    saved = json.loads(path.read_text(encoding="utf-8"))
    predictions = saved.get("predictions")
    if (
        saved.get("schema_version") != 1
        or saved.get("signature") != signature
        or saved.get("ids") != ids
        or not isinstance(predictions, list)
        or len(predictions) > len(ids)
    ):
        raise ValueError("Candidate checkpoint does not match the frozen selection or artifact")
    return predictions


def paired_top1_correctness(
    rows: list[dict], logits: list[np.ndarray], predictions: list[dict]
) -> tuple[list[float], list[float]]:
    """Align per-row top-1 correctness so the comparison stays paired."""
    if len(rows) != len(logits) or len(rows) != len(predictions):
        raise ValueError("Top-1 comparison inputs do not align")
    baseline_correct, candidate_correct = [], []
    for row, baseline_logits, prediction in zip(rows, logits, predictions, strict=True):
        target_index = int(np.argmax(row["target"]))
        baseline_correct.append(float(int(np.argmax(baseline_logits)) == target_index))
        answer = prediction.get("answer")
        if not isinstance(answer, dict):
            raise ValueError("Candidate prediction has no typed answer")
        if row["kind"] == "noul":
            yes = float(answer["yes_probability"])
            candidate_probabilities = np.asarray([1.0 - yes, yes], dtype=np.float64)
        else:
            probabilities_by_option = answer.get("probabilities")
            if not isinstance(probabilities_by_option, dict):
                raise ValueError("Candidate prediction has no option probabilities")
            candidate_probabilities = np.asarray(
                [probabilities_by_option[option] for option in row["options"]], dtype=np.float64
            )
        if (
            candidate_probabilities.shape != (len(row["options"]),)
            or not np.isfinite(candidate_probabilities).all()
            or np.any(candidate_probabilities < 0.0)
            or np.any(candidate_probabilities > 1.0)
            or not np.isclose(candidate_probabilities.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("Candidate option probabilities are invalid")
        candidate_correct.append(float(int(np.argmax(candidate_probabilities)) == target_index))
    return baseline_correct, candidate_correct


def evaluate(
    protocol_path: Path,
    output_path: Path,
    *,
    resume: bool,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260929,
) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema_version") != 1 or protocol.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Fresh replication protocol does not match the pinned dataset")

    source_descriptor = protocol["source_selection_protocol"]
    source_protocol_path = resolve_repo_path(
        source_descriptor["path"], allowed_root=ROOT / "benchmarks", label="Source protocol"
    )
    if sha256_file(source_protocol_path) != source_descriptor["sha256"]:
        raise ValueError("Source selection protocol checksum mismatch")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if (
        source_protocol.get("schema_version") != 2
        or source_protocol.get("dataset_revision") != DATASET_REVISION
        or source_protocol.get("released_checkpoint") != f"{OPENJEV_REPO}@{OPENJEV_REVISION}"
    ):
        raise ValueError("Source protocol does not pin the supported released Open-Jev model")

    baseline = protocol["released_baseline"]
    baseline_report_path = ROOT / "benchmarks" / baseline["report_path"]
    baseline_logits_path = ROOT / "benchmarks" / baseline["fresh_logits_path"]
    if (
        sha256_file(baseline_report_path) != baseline["report_sha256"]
        or sha256_file(baseline_logits_path) != baseline["fresh_logits_sha256"]
    ):
        raise ValueError("Pinned released-baseline files are missing or changed")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_report.get("protocol_sha256") != protocol["source_selection_protocol"]["sha256"]:
        raise ValueError("Released baseline was not produced from the pinned source protocol")

    archived, _ = select_archived_rows(
        seed=source_protocol["archived_seed"],
        pool=source_protocol["archived_pool"],
        limit=source_protocol["archived_rows"],
        expected_ids_sha256=source_protocol["archived_ids_sha256"],
    )
    excluded_group_ids = load_excluded_group_ids(ROOT, source_protocol.get("excluded_group_manifest"))
    fresh, fresh_manifest = select_fresh_rows(
        archived,
        split=source_protocol.get("fresh_split", "ood"),
        seed=source_protocol["fresh_seed"],
        pool=source_protocol["fresh_pool"],
        limit=source_protocol["fresh_rows"],
        excluded_group_ids=excluded_group_ids,
    )
    verify_fresh_selection(fresh, source_protocol)
    baseline_signature = input_signature(fresh, fresh_manifest, max_length=source_protocol["max_length"])
    if baseline_signature != baseline["fresh_selection_signature"]:
        raise ValueError("Fresh selection signature differs from the preregistered baseline")
    if baseline_report.get("fresh", {}).get("selection_signature") != baseline_signature:
        raise ValueError("Released baseline report does not match the fresh selection")
    if baseline_report.get("fresh_ids_sha256") != source_protocol.get("fresh_ids_sha256") or baseline_report.get(
        "fresh_group_ids_sha256"
    ) != source_protocol.get("fresh_group_ids_sha256"):
        raise ValueError("Released baseline report is missing the frozen fresh selection hashes")
    baseline_logits_file = json.loads(baseline_logits_path.read_text(encoding="utf-8"))
    all_baseline_logits = load_candidate_rows(fresh, baseline_logits_file, signature=baseline_signature)
    baseline_logits_by_id = {row["id"]: logits for row, logits in zip(fresh, all_baseline_logits, strict=True)}

    candidate = protocol["candidate"]
    candidate_report_path = ROOT / candidate["report_path"]
    candidate_dir = ROOT / candidate["artifact_dir"]
    if sha256_file(candidate_report_path) != candidate["report_sha256"]:
        raise ValueError("Candidate report checksum mismatch")
    candidate_report, _ = verify_artifact(candidate_dir, candidate_report_path)
    if (
        candidate_report.get("adapter_sha256") != candidate["adapter_sha256"]
        or candidate_report.get("head_sha256") != candidate["head_sha256"]
        or candidate_report.get("max_length") != candidate["max_length"]
    ):
        raise ValueError("Candidate artifact differs from the preregistered model")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(
        candidate_report["model"], revision=candidate_report["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded, _ = encode_rows(
        tokenizer, fresh, candidate["max_length"], prompt_style=candidate_report.get("prompt_style", "candidate_v1")
    )
    rows, overlong_skipped = retain_tokenizable_rows(fresh, encoded)
    if not rows:
        raise ValueError("No fresh rows fit the candidate token limit")
    ids = [row["id"] for row in rows]
    logits = [baseline_logits_by_id[row_id] for row_id in ids]
    payloads = [
        {
            "state": json.loads(row["state_json"]),
            "kind": row["kind"],
            "question": row["question"],
            "options": row["options"],
        }
        for row in rows
    ]

    checkpoint_signature = hashlib.sha256(
        json.dumps(
            {
                "protocol_sha256": sha256_file(protocol_path),
                "fresh_selection_signature": baseline_signature,
                "scored_ids": ids,
                "candidate_adapter_sha256": candidate["adapter_sha256"],
                "candidate_head_sha256": candidate["head_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    checkpoint_path = resolve_repo_path(
        protocol.get(
            "candidate_checkpoint_path",
            "benchmarks/.bench-state/openjev_scale_1024_rps_fresh_v1/candidate_predictions.json",
        ),
        allowed_root=ROOT / "benchmarks/.bench-state",
        label="Candidate checkpoint",
    )
    predictions = load_partial(checkpoint_path, signature=checkpoint_signature, ids=ids, resume=resume)
    start = len(predictions)
    if start < len(payloads):

        def save(index: int, prediction: dict) -> None:
            if start + index != len(predictions):
                raise ValueError("Fresh candidate predictions arrived out of order")
            predictions.append(prediction)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"schema_version": 1, "signature": checkpoint_signature, "ids": ids, "predictions": predictions},
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, checkpoint_path)

        score_requests(
            payloads[start:], candidate_dir, candidate_report_path, on_prediction=save, max_decisions=len(payloads)
        )
    if len(predictions) != len(rows):
        raise ValueError("Candidate output count does not match the scored fresh rows")

    baseline_metrics = metrics(rows, logits, 1.0)
    candidate_metrics = prediction_metrics(rows, predictions)
    baseline_correct, candidate_correct = paired_top1_correctness(rows, logits, predictions)
    report = {
        "purpose": protocol["purpose"],
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_revision": DATASET_REVISION,
        "source_protocol_sha256": protocol["source_selection_protocol"]["sha256"],
        "fresh_selection_signature": baseline_signature,
        "fresh_rows_selected": len(fresh),
        "fresh_split": source_protocol.get("fresh_split", "ood"),
        "excluded_group_count": len(excluded_group_ids),
        "excluded_group_manifest_sha256": source_protocol.get("excluded_group_manifest", {}).get("sha256"),
        "fresh_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in fresh).encode()).hexdigest(),
        "fresh_group_ids_sha256": hashlib.sha256(
            "\n".join(sorted({row["group_id"] for row in fresh})).encode()
        ).hexdigest(),
        "rows_scored_on_both_models": len(rows),
        "overlong_rows_excluded_without_truncation": overlong_skipped,
        "scored_ids": ids,
        "released_openjev": baseline_metrics,
        "rps_candidate": candidate_metrics,
        "delta_candidate_minus_released": {
            "top1_accuracy": candidate_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"],
            "mean_nll": candidate_metrics["mean_nll"] - baseline_metrics["mean_nll"],
            "mean_brier": candidate_metrics["mean_brier"] - baseline_metrics["mean_brier"],
            "score_mean_ranked_probability_score": (
                candidate_metrics["by_kind"].get("score", {}).get("mean_ranked_probability_score")
                - baseline_metrics["by_kind"].get("score", {}).get("mean_ranked_probability_score")
                if "score" in candidate_metrics["by_kind"] and "score" in baseline_metrics["by_kind"]
                else None
            ),
        },
        "paired_group_bootstrap_top1": grouped_mean_bootstrap_delta(
            [row["group_id"] for row in rows],
            candidate_correct,
            baseline_correct,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "candidate_head_sha256": candidate["head_sha256"],
        "baseline_artifacts_sha256": {
            "report": baseline["report_sha256"],
            "fresh_logits": baseline["fresh_logits_sha256"],
        },
        "limitation": "Public synthetic decisions only; comparison is with released Open-Jev, not TypeSafe Jev, and proves neither desktop task success nor execution safety.",
    }
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite fresh replication report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/opendec_scale_1024_rps_fresh_v3.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"report already exists: {args.output}")
    report = evaluate(
        args.protocol,
        args.output,
        resume=args.resume,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "rows": report["rows_scored_on_both_models"],
                "released_openjev": report["released_openjev"],
                "rps_candidate": report["rps_candidate"],
                "delta_candidate_minus_released": report["delta_candidate_minus_released"],
                "paired_group_bootstrap_top1": report["paired_group_bootstrap_top1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

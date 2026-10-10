"""Freeze a candidate-specific, group-disjoint Open-Jev evaluation protocol.

This helper replays the exact public-data selections recorded by a training
report, adds every candidate split to the prior exclusion manifest, and then
pins a new held-out selection. After the released checkpoint is scored, its
report and logits plus the candidate artifacts are hash-pinned into a paired
comparison protocol. It never calls a hosted Jev service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.general_decision import (
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    load_split,
    verify_disjoint,
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

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
STATE = BENCHMARKS / ".bench-state"
FROZEN_COMPARISON_PROTOCOL = BENCHMARKS / "opendec_scale_1024_rps_fresh_protocol_v3.json"
SPLITS = ("train", "calibration", "test", "ood")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def group_ids_sha256(groups: set[str] | list[str]) -> str:
    return sha256_bytes("\n".join(sorted(set(groups))).encode())


def resolve_state_path(value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(STATE.resolve()):
        raise ValueError(f"{label} must stay under benchmarks/.bench-state")
    return path


def repository_path(path: Path, *, base: Path = ROOT) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError("Path is outside the expected repository directory")
    return resolved.relative_to(base.resolve()).as_posix()


def write_json_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary artifact: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def reconstruct_candidate_groups(report: dict) -> tuple[dict[str, list[dict]], set[str]]:
    """Rebuild exact sampled split groups and fail closed on manifest drift."""
    if (
        report.get("dataset") != DATASET
        or report.get("dataset_config") != DATASET_CONFIG
        or report.get("dataset_revision") != DATASET_REVISION
    ):
        raise ValueError("Training report does not use the pinned licensed Open-Jev dataset")
    seed = report.get("seed")
    selected_rows = report.get("selected_rows")
    pinned_manifests = report.get("dataset_manifests")
    if type(seed) is not int or not isinstance(selected_rows, dict) or not isinstance(pinned_manifests, dict):
        raise ValueError("Training report is missing reproducible split metadata")

    split_rows: dict[str, list[dict]] = {}
    for offset, split in enumerate(SPLITS):
        limit = selected_rows.get(split)
        if type(limit) is not int or limit < 1:
            raise ValueError(f"Training report has no valid {split} row count")
        rows, manifest = load_split(split, limit, seed + offset)
        if len(rows) != limit or pinned_manifests.get(split) != manifest:
            raise ValueError(f"Pinned {split} selection or dataset manifest changed")
        split_rows[split] = rows
    verify_disjoint(split_rows)
    groups = {row["group_id"] for rows in split_rows.values() for row in rows}
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Candidate split contains an invalid group ID")
    return split_rows, groups


def build_group_manifest(base_groups: set[str], candidate_groups: set[str]) -> dict:
    groups = sorted(base_groups | candidate_groups)
    return {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "group_ids": groups,
        "group_ids_sha256": group_ids_sha256(groups),
        "unique_group_count": len(groups),
    }


def candidate_independent_rows(rows: list[dict], candidate_groups: set[str]) -> list[dict]:
    return [row for row in rows if row["group_id"] not in candidate_groups]


def load_logits_by_id(path: Path, rows: list[dict], *, signature: str) -> dict[str, list[float]]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in rows]
    logits = saved.get("logits")
    if (
        saved.get("schema_version") != 1
        or saved.get("signature") != signature
        or saved.get("ids") != ids
        or not isinstance(logits, list)
        or len(logits) != len(rows)
    ):
        raise ValueError(f"Frozen baseline logits do not match {path.name} row selection")
    return dict(zip(ids, logits, strict=True))


def prepare(args: argparse.Namespace) -> dict:
    run_report_path = resolve_state_path(args.run_report, "Training report")
    source_output_path = resolve_state_path(args.source_output, "Source protocol output")
    manifest_output_path = resolve_state_path(args.manifest_output, "Exclusion manifest output")
    baseline_output_dir = resolve_state_path(args.baseline_output_dir, "Baseline output directory")
    if len({source_output_path, manifest_output_path, baseline_output_dir}) != 3:
        raise ValueError("Protocol, manifest, and baseline output paths must differ")
    planned_outputs = (
        source_output_path,
        manifest_output_path,
        baseline_output_dir / "archived_logits.json",
        baseline_output_dir / "fresh_logits.json",
        baseline_output_dir / "report.json",
    )
    if any(path.exists() for path in planned_outputs):
        raise FileExistsError("Refusing to overwrite candidate-specific evaluation artifacts")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    split_rows, candidate_groups = reconstruct_candidate_groups(run_report)

    comparison = json.loads(FROZEN_COMPARISON_PROTOCOL.read_text(encoding="utf-8"))
    descriptor = comparison["source_selection_protocol"]
    source_path = (ROOT / descriptor["path"]).resolve()
    if not source_path.is_relative_to(BENCHMARKS.resolve()) or sha256_file(source_path) != descriptor["sha256"]:
        raise ValueError("Frozen source protocol is missing or has changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("dataset_revision") != DATASET_REVISION
        or source.get("released_checkpoint") != f"{OPENJEV_REPO}@{OPENJEV_REVISION}"
    ):
        raise ValueError("Frozen source protocol does not pin the supported Open-Jev release")

    previous_groups = load_excluded_group_ids(ROOT, source.get("excluded_group_manifest"))
    manifest = build_group_manifest(previous_groups, candidate_groups)
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    manifest_descriptor = {
        "path": repository_path(manifest_output_path),
        "sha256": sha256_bytes(manifest_bytes),
    }

    archived, _ = select_archived_rows(
        seed=source["archived_seed"],
        pool=source["archived_pool"],
        limit=source["archived_rows"],
        expected_ids_sha256=source["archived_ids_sha256"],
    )
    frozen_fresh, fresh_manifest = select_fresh_rows(
        archived,
        split=source.get("fresh_split", "ood"),
        seed=source["fresh_seed"],
        pool=source["fresh_pool"],
        limit=source["fresh_rows"],
        excluded_group_ids=previous_groups,
    )
    expected_fresh = candidate_independent_rows(frozen_fresh, candidate_groups)
    if not expected_fresh:
        raise ValueError("The frozen baseline selection has no groups independent of this candidate")
    fresh, fresh_manifest = select_fresh_rows(
        archived,
        split=source.get("fresh_split", "ood"),
        seed=source["fresh_seed"],
        pool=source["fresh_pool"],
        limit=len(expected_fresh),
        excluded_group_ids=previous_groups | candidate_groups,
    )
    if [row["id"] for row in fresh] != [row["id"] for row in expected_fresh]:
        raise ValueError("Candidate-independent rows are not covered by the frozen baseline selection")
    fresh_groups = {row["group_id"] for row in fresh}
    if fresh_groups & candidate_groups:
        raise ValueError("Fresh selection overlaps a candidate train/calibration/test/OOD group")
    source["purpose"] = "candidate_specific_group_disjoint_public_replication"
    source["fresh_rows"] = len(fresh)
    source["excluded_group_manifest"] = manifest_descriptor
    source["fresh_ids_sha256"] = sha256_bytes("\n".join(row["id"] for row in fresh).encode())
    source["fresh_group_ids_sha256"] = group_ids_sha256(fresh_groups)
    verify_fresh_selection(fresh, source)

    # Slice only the pinned baseline logits; no baseline predictions are refit or regenerated.
    baseline_descriptor = comparison["released_baseline"]
    baseline_report_path = BENCHMARKS / baseline_descriptor["report_path"]
    frozen_logits_path = BENCHMARKS / baseline_descriptor["fresh_logits_path"]
    if (
        sha256_file(baseline_report_path) != baseline_descriptor["report_sha256"]
        or sha256_file(frozen_logits_path) != baseline_descriptor["fresh_logits_sha256"]
    ):
        raise ValueError("Frozen released-baseline report or logits have changed")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    archived_rows, archived_manifest = select_archived_rows(
        seed=source["archived_seed"],
        pool=source["archived_pool"],
        limit=source["archived_rows"],
        expected_ids_sha256=source["archived_ids_sha256"],
    )
    archived_logits_path = baseline_report_path.parent / "archived_logits.json"
    archived_signature = input_signature(archived_rows, archived_manifest, max_length=source["max_length"])
    archived_logits = load_logits_by_id(
        archived_logits_path,
        archived_rows,
        signature=baseline_report["archived"]["selection_signature"],
    )
    if baseline_report["archived"]["selection_signature"] != archived_signature:
        raise ValueError("Frozen archived baseline logits use an unexpected signature")
    old_fresh_signature = input_signature(frozen_fresh, fresh_manifest, max_length=source["max_length"])
    frozen_fresh_logits = load_logits_by_id(
        frozen_logits_path,
        frozen_fresh,
        signature=baseline_report["fresh"]["selection_signature"],
    )
    if (
        baseline_report["fresh"]["selection_signature"] != old_fresh_signature
        or baseline_report["fresh"]["selection_signature"] != baseline_descriptor["fresh_selection_signature"]
    ):
        raise ValueError("Frozen fresh baseline logits use an unexpected signature")
    fresh_signature = input_signature(fresh, fresh_manifest, max_length=source["max_length"])
    fresh_logits = {row["id"]: frozen_fresh_logits[row["id"]] for row in fresh}

    # Persist outputs only after every selection/hash check has passed.
    write_json_new(manifest_output_path, manifest)
    write_json_new(source_output_path, source)
    write_json_new(
        baseline_output_dir / "archived_logits.json",
        {
            "schema_version": 1,
            "signature": archived_signature,
            "ids": [row["id"] for row in archived_rows],
            "logits": [archived_logits[row["id"]] for row in archived_rows],
        },
    )
    write_json_new(
        baseline_output_dir / "fresh_logits.json",
        {
            "schema_version": 1,
            "signature": fresh_signature,
            "ids": [row["id"] for row in fresh],
            "logits": [fresh_logits[row["id"]] for row in fresh],
        },
    )
    return {
        "source_protocol": repository_path(source_output_path),
        "source_protocol_sha256": sha256_file(source_output_path),
        "exclusion_manifest": repository_path(manifest_output_path),
        "excluded_groups": len(previous_groups | candidate_groups),
        "candidate_groups": len(candidate_groups),
        "fresh_rows": len(fresh),
        "fresh_groups": len(fresh_groups),
        "baseline_logits_reused": True,
        "fresh_ids_sha256": source["fresh_ids_sha256"],
        "split_group_counts": {name: len({row["group_id"] for row in rows}) for name, rows in split_rows.items()},
    }


def finalize(args: argparse.Namespace) -> dict:
    source_path = resolve_state_path(args.source_protocol, "Source protocol")
    baseline_dir = resolve_state_path(args.baseline_output_dir, "Released baseline output")
    candidate_report_path = resolve_state_path(args.candidate_report, "Candidate report")
    candidate_dir = resolve_state_path(args.candidate_artifact_dir, "Candidate artifacts")
    output_path = resolve_state_path(args.output_protocol, "Comparison protocol output")
    checkpoint_path = resolve_state_path(args.candidate_checkpoint, "Candidate checkpoint")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    baseline_report_path = baseline_dir / "report.json"
    fresh_logits_path = baseline_dir / "fresh_logits.json"
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (
        baseline_report.get("protocol_sha256") != sha256_file(source_path)
        or baseline_report.get("fresh_ids_sha256") != source.get("fresh_ids_sha256")
        or baseline_report.get("fresh_group_ids_sha256") != source.get("fresh_group_ids_sha256")
    ):
        raise ValueError("Released baseline report does not match the frozen source selection")
    if (
        candidate_report.get("dataset") != DATASET
        or candidate_report.get("dataset_revision") != DATASET_REVISION
        or candidate_report.get("initial_checkpoint", {}).get("repo") != OPENJEV_REPO
        or candidate_report.get("initial_checkpoint", {}).get("revision") != OPENJEV_REVISION
    ):
        raise ValueError("Candidate report does not identify the pinned Open-Jev warm start")
    if sha256_file(candidate_dir / "adapter" / "adapter_model.safetensors") != candidate_report.get(
        "adapter_sha256"
    ) or sha256_file(candidate_dir / "head.npz") != candidate_report.get("head_sha256"):
        raise ValueError("Candidate artifact hashes do not match its report")

    protocol = json.loads(FROZEN_COMPARISON_PROTOCOL.read_text(encoding="utf-8"))
    protocol["purpose"] = "candidate_specific_group_disjoint_public_replication"
    protocol["frozen_at_utc"] = datetime.now(timezone.utc).date().isoformat()
    protocol["source_selection_protocol"] = {
        "path": repository_path(source_path),
        "sha256": sha256_file(source_path),
    }
    protocol["released_baseline"] = {
        "report_path": repository_path(baseline_report_path, base=BENCHMARKS),
        "report_sha256": sha256_file(baseline_report_path),
        "fresh_logits_path": repository_path(fresh_logits_path, base=BENCHMARKS),
        "fresh_logits_sha256": sha256_file(fresh_logits_path),
        "fresh_selection_signature": baseline_report["fresh"]["selection_signature"],
    }
    protocol["candidate"] = {
        "report_path": repository_path(candidate_report_path),
        "report_sha256": sha256_file(candidate_report_path),
        "artifact_dir": repository_path(candidate_dir),
        "adapter_sha256": candidate_report["adapter_sha256"],
        "head_sha256": candidate_report["head_sha256"],
        "max_length": candidate_report["max_length"],
    }
    protocol["candidate_checkpoint_path"] = repository_path(checkpoint_path)
    protocol["baseline_was_scored_before_candidate_protocol_freeze"] = True
    protocol["promotion"] = "none"
    protocol["limitations"] = [
        "Public synthetic decisions only; comparison is with the released Open-Jev checkpoint, not proprietary Jev.",
        "Candidate train, calibration, test, and OOD groups are excluded from the fresh selection.",
        "This does not establish desktop task success, safe execution, or model authority.",
    ]
    write_json_new(output_path, protocol)
    return {
        "comparison_protocol": repository_path(output_path),
        "comparison_protocol_sha256": sha256_file(output_path),
        "fresh_rows": source["fresh_rows"],
        "candidate_adapter_sha256": candidate_report["adapter_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze a new candidate-disjoint selection")
    prepare_parser.add_argument("--run-report", required=True)
    prepare_parser.add_argument("--source-output", required=True)
    prepare_parser.add_argument("--manifest-output", required=True)
    prepare_parser.add_argument("--baseline-output-dir", required=True)
    finalize_parser = commands.add_parser("finalize", help="Hash-pin scored baseline and candidate")
    finalize_parser.add_argument("--source-protocol", required=True)
    finalize_parser.add_argument("--baseline-output-dir", required=True)
    finalize_parser.add_argument("--candidate-report", required=True)
    finalize_parser.add_argument("--candidate-artifact-dir", required=True)
    finalize_parser.add_argument("--output-protocol", required=True)
    finalize_parser.add_argument("--candidate-checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

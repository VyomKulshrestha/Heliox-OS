"""Score frozen latent-difference visual weights on a separate pinned archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_world_visual_delta_latent import (
    attach_semantic_features,
    build_delta_latent_model,
    score_model,
)
from scripts.train_world_visual_transition import feature_shape, load_features


def training_task_names(model_report: dict, source_reports: tuple[Path | None, Path | None]) -> set[str] | None:
    fields = ("source_report_sha256", "additional_source_report_sha256")
    if all(path is None for path in source_reports):
        return None
    if any(bool(path) != bool(model_report.get(field))
           for path, field in zip(source_reports, fields, strict=True)):
        raise ValueError("Provide every source report used to fit the frozen model")
    tasks = set()
    for path, field in zip(source_reports, fields, strict=True):
        assert path is not None
        if hashlib.sha256(path.read_bytes()).hexdigest() != model_report[field]:
            raise ValueError("Training source report checksum mismatch")
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source.get("tasks"), dict):
            raise ValueError("Training source report has no task-split provenance")
        tasks.update(task for split in ("train", "validation")
                     for task in source["tasks"].get(split, []))
    return tasks


def task_masks(features_path: Path, split: str, expected_rows: int) -> dict[str, np.ndarray]:
    with np.load(features_path, allow_pickle=False) as data:
        if "task" not in data.files:
            return {}
        tasks = data["task"].astype(str)
        selected = data["split"].astype(str) == split
    if len(tasks) != len(selected) or int(selected.sum()) != expected_rows:
        raise ValueError("Task labels do not align with the verified visual split")
    names = tasks[selected]
    return {name: names == name for name in sorted(set(names))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--target-semantic-features", type=Path)
    parser.add_argument("--target-semantic-report", type=Path)
    parser.add_argument("--training-source-report", type=Path,
                        help="Hash-verified primary training archive report for task-overlap audit")
    parser.add_argument("--additional-training-source-report", type=Path,
                        help="Hash-verified additional training archive report for task-overlap audit")
    parser.add_argument("--split", choices=("train", "validation", "eval"), default="eval")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh output report path")
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    if model_report.get("purpose") != "research_only_delta_inverse_visual_latent_transition":
        raise ValueError("Expected a delta-latent research report")
    semantic_dim = int(model_report.get("semantic_dim") or 0)
    if (bool(args.target_semantic_features) != bool(args.target_semantic_report)
            or bool(semantic_dim) != bool(args.target_semantic_features)):
        raise ValueError("Target semantic features are required exactly for semantic models")
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weights_hash != model_report.get("weights_sha256"):
        raise ValueError("Weights do not match the pinned model report")
    shape = feature_shape(args.target_report)
    if shape != tuple(model_report["feature_shape"]):
        raise ValueError("Source model and target features have different shapes")
    source = json.loads(args.target_report.read_text(encoding="utf-8"))
    if source.get("counterfactual_pairs") is not True:
        raise ValueError("Transfer scoring requires verified action pairs")
    rows = load_features(args.target_features, args.target_report)
    if semantic_dim:
        actual_dim = attach_semantic_features(
            rows, args.target_features, args.target_report,
            args.target_semantic_features, args.target_semantic_report)
        if actual_dim != semantic_dim:
            raise ValueError("Frozen model and target semantic dimensions differ")
    fit_tasks = training_task_names(
        model_report, (args.training_source_report, args.additional_training_source_report))
    masks = task_masks(args.target_features, args.split, len(rows[args.split]["state"]))

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this transfer probe")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_delta_latent_model(nn, torch, shape, model_report["latent_dim"], semantic_dim)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    by_task = {name: score_model(
        model, {key: value[mask] for key, value in rows[args.split].items()}, shape, torch, device)
        for name, mask in masks.items()}
    report = {
        "purpose": "research_only_frozen_delta_latent_split_probe",
        "model_report_sha256": hashlib.sha256(args.model_report.read_bytes()).hexdigest(),
        "weights_sha256": weights_hash,
        "target_features_sha256": hashlib.sha256(args.target_features.read_bytes()).hexdigest(),
        "target_report_sha256": hashlib.sha256(args.target_report.read_bytes()).hexdigest(),
        "target_semantic_features_sha256": (
            hashlib.sha256(args.target_semantic_features.read_bytes()).hexdigest()
            if args.target_semantic_features else None),
        "target_semantic_report_sha256": (
            hashlib.sha256(args.target_semantic_report.read_bytes()).hexdigest()
            if args.target_semantic_report else None),
        "target_split": args.split,
        "target_rows": len(rows[args.split]["state"]),
        "target_groups": len(set(rows[args.split]["groups"])),
        "device": str(device),
        "scores": score_model(model, rows[args.split], shape, torch, device),
        "scores_by_task": by_task,
        "training_task_names": sorted(fit_tasks) if fit_tasks is not None else None,
        "exact_task_name_overlap": (sorted(set(masks) & fit_tasks) if fit_tasks is not None else None),
        "limitation": "Frozen one-step latent model on selected public GUI pairs only. "
                      "No real desktop, multistep task, runtime promotion, safety, or Jev parity claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

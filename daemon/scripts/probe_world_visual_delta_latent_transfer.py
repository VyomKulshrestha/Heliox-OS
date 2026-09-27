"""Score frozen latent-difference visual weights on a separate pinned archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_world_visual_delta_latent import build_delta_latent_model, score_model
from scripts.train_world_visual_transition import feature_shape, load_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh output report path")
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    if model_report.get("purpose") != "research_only_delta_inverse_visual_latent_transition":
        raise ValueError("Expected a delta-latent research report")
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

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this transfer probe")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_delta_latent_model(nn, torch, shape, model_report["latent_dim"])
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    report = {
        "purpose": "research_only_frozen_delta_latent_transfer_probe",
        "model_report_sha256": hashlib.sha256(args.model_report.read_bytes()).hexdigest(),
        "weights_sha256": weights_hash,
        "target_features_sha256": hashlib.sha256(args.target_features.read_bytes()).hexdigest(),
        "target_report_sha256": hashlib.sha256(args.target_report.read_bytes()).hexdigest(),
        "target_rows": len(rows["eval"]["state"]),
        "target_groups": len(set(rows["eval"]["groups"])),
        "device": str(device),
        "scores": score_model(model, rows["eval"], shape, torch, device),
        "limitation": "Frozen one-step latent model on selected public GUI pairs only. "
                      "No real desktop, multistep task, runtime promotion, safety, or Jev parity claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

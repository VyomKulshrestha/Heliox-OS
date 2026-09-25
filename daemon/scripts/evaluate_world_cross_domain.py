"""GPU leave-one-domain-out test of the research-only world task-state head.

Fixed architecture and seed; no held-out-domain rows affect normalization,
training, or model selection. This measures transfer, not action authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_world_task_head import calibration_metrics
from scripts.train_world_multidomain_mlp import DOMAINS, balanced_weights, load_features
from scripts.train_world_scope_semantic import normalize

HIDDEN = 64
SEED = 19
EPOCHS = 120


def training_domains(all_domains: tuple[str, ...], held_out: str) -> tuple[str, ...]:
    if held_out not in all_domains or len(set(all_domains)) != len(all_domains):
        raise ValueError("Expected a distinct held-out domain")
    return tuple(domain for domain in all_domains if domain != held_out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--when2call-report", type=Path, required=True)
    parser.add_argument("--when2call-feature-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=state / "world_cross_domain_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")

    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    if source["domains"] != list(DOMAINS) or not source["tbac"]["validation_in_training"]:
        raise ValueError("Unexpected three-domain feature source")
    features, labels = load_features(args.feature_dir, source)
    extra_bytes = args.when2call_report.read_bytes()
    extra = json.loads(extra_bytes)
    if (extra["purpose"] != "research_only_grouped_call_readiness_features"
            or extra["model"] != source["model"] or extra["revision"] != source["revision"]
            or set(extra["splits"]) != {"train", "calibration", "test"}):
        raise ValueError("Unexpected When2Call feature source")
    manifest = {
        "rows": {key: value["rows"] for key, value in extra["splits"].items()},
        "excluded_overlong": {key: value["overlong"] for key, value in extra["splits"].items()},
        "feature_checkpoint_sha256": {
            key: value["checkpoint_sha256"] for key, value in extra["splits"].items()
        },
    }
    extra_features, extra_labels = load_features(args.when2call_feature_dir, manifest)
    features.update({f"when2call_{key}": value for key, value in extra_features.items()})
    labels.update({f"when2call_{key}": value for key, value in extra_labels.items()})
    domains = (*DOMAINS, "when2call")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research training run")
    device = torch.device("cuda")
    folds = {}
    for held_out in domains:
        train_domains = training_domains(domains, held_out)
        fold_features = {
            "train": np.vstack([features[f"{domain}_train"] for domain in train_domains]),
            "test": features[f"{held_out}_test"],
        }
        normalized, _, _ = normalize(fold_features)
        train_x = torch.as_tensor(normalized["train"], dtype=torch.float32, device=device)
        train_y = torch.as_tensor(
            np.concatenate([labels[f"{domain}_train"] for domain in train_domains]),
            dtype=torch.float32, device=device,
        )
        train_weights = torch.as_tensor(balanced_weights(labels, train_domains), device=device)
        torch.manual_seed(SEED)
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(HIDDEN, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        for _ in range(EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(train_x).squeeze(1), train_y, reduction="none",
            )
            (loss * train_weights).mean().backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_x = torch.as_tensor(normalized["test"], dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(test_x)).squeeze(1).cpu().numpy()
        folds[held_out] = {
            "training_domains": list(train_domains),
            "test_partition": f"{held_out}_test",
            "metrics": calibration_metrics(labels[f"{held_out}_test"], scores),
        }
        print(json.dumps({"held_out": held_out, "metrics": folds[held_out]["metrics"]}), flush=True)

    report = {
        "purpose": "research_only_gpu_leave_one_domain_out_world_task_state",
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "when2call_report_sha256": hashlib.sha256(extra_bytes).hexdigest(),
        "device": torch.cuda.get_device_name(device),
        "architecture": {"hidden": HIDDEN, "seed": SEED, "epochs": EPOCHS,
                         "dropout": 0.3, "optimizer": "AdamW", "learning_rate": 3e-4,
                         "weight_decay": 0.1},
        "protocol": "For each fold, train and normalize on other domains' training rows only; "
                    "score the held-out domain test at fixed cutoffs without tuning.",
        "folds": folds,
        "limitation": "Hyperparameters were previously explored on these domain families; "
                      "this is a diagnostic transfer test, not a blind independent benchmark, "
                      "Jev parity, or runtime authorization evidence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

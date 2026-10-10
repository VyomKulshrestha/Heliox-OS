"""Train a research-only GPU head with separate scope and preference outputs.

The Qwen representation is frozen. Only train splits fit this fixed model;
calibration and test splits do not select weights or hyperparameters. This is
not the shipped OS-transition model or an action-approval mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.calibrate_world_task_head import calibration_metrics, infer_logits
from scripts.train_world_multidomain_mlp import DOMAINS, balanced_weights, load_features
from scripts.train_world_scope_semantic import normalize

TASKS = {"scope": 0, "preference": 1}
HIDDEN = 64
SEED = 19
EPOCHS = 120


def task_for_domain(domain: str) -> str:
    if domain in DOMAINS:
        return "scope"
    if domain == "when2call":
        return "preference"
    raise ValueError(f"Unknown training domain: {domain}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--when2call-report", type=Path, required=True)
    parser.add_argument("--when2call-feature-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=state / "world_multitask_head_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "world_multitask_head_v1.json")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")

    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    if source["domains"] != list(DOMAINS) or not source["tbac"]["validation_in_training"]:
        raise ValueError("Unexpected three-domain feature source")
    features, labels = load_features(args.feature_dir, source)
    extra_bytes = args.when2call_report.read_bytes()
    extra = json.loads(extra_bytes)
    if (
        extra["purpose"] != "research_only_grouped_call_readiness_features"
        or extra["model"] != source["model"]
        or extra["revision"] != source["revision"]
        or set(extra["splits"]) != {"train", "calibration", "test"}
    ):
        raise ValueError("Unexpected When2Call feature source")
    manifest = {
        "rows": {key: value["rows"] for key, value in extra["splits"].items()},
        "excluded_overlong": {key: value["overlong"] for key, value in extra["splits"].items()},
        "feature_checkpoint_sha256": {key: value["checkpoint_sha256"] for key, value in extra["splits"].items()},
    }
    extra_features, extra_labels = load_features(args.when2call_feature_dir, manifest)
    features.update({f"when2call_{key}": value for key, value in extra_features.items()})
    labels.update({f"when2call_{key}": value for key, value in extra_labels.items()})
    domains = (*DOMAINS, "when2call")
    features["train"] = np.vstack([features[f"{domain}_train"] for domain in domains])
    normalized, mean, scale = normalize(features)
    train_tasks = np.concatenate(
        [np.full(len(labels[f"{domain}_train"]), TASKS[task_for_domain(domain)], dtype=np.int64) for domain in domains]
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research training run")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(normalized["train"].shape[1], HIDDEN),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(HIDDEN, len(TASKS)),
    ).to(device)
    train_x = torch.as_tensor(normalized["train"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(
        np.concatenate([labels[f"{domain}_train"] for domain in domains]), dtype=torch.float32, device=device
    )
    train_task_ids = torch.as_tensor(train_tasks, dtype=torch.int64, device=device)
    train_weights = torch.as_tensor(balanced_weights(labels, domains), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x).gather(1, train_task_ids[:, None]).squeeze(1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            train_y,
            reduction="none",
        )
        (losses * train_weights).mean().backward()
        optimizer.step()
    model.eval()
    saved = model.state_dict()
    common = {
        "w1": saved["0.weight"].cpu().numpy(),
        "b1": saved["0.bias"].cpu().numpy(),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
    }
    output_w = saved["3.weight"].cpu().numpy()
    output_b = saved["3.bias"].cpu().numpy()
    evaluation = (*[f"{domain}_test" for domain in domains], "heliox")
    metrics = {}
    with torch.inference_mode():
        for partition in evaluation:
            domain = partition.removesuffix("_test")
            task = task_for_domain(domain if domain != "heliox" else "scope")
            task_index = TASKS[task]
            test_x = torch.as_tensor(normalized[partition], dtype=torch.float32, device=device)
            gpu_scores = torch.sigmoid(model(test_x)[:, task_index]).cpu().numpy()
            replay_scores = expit(
                infer_logits(
                    features[partition],
                    {
                        **common,
                        "w2": output_w[task_index : task_index + 1],
                        "b2": output_b[task_index : task_index + 1],
                    },
                )
            )
            np.testing.assert_allclose(replay_scores, gpu_scores, rtol=1e-5, atol=1e-6)
            metrics[partition] = {"task": task, **calibration_metrics(labels[partition], replay_scores)}

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        **common,
        w2=output_w,
        b2=output_b,
        task_names=np.asarray(tuple(TASKS)),
        source_report_sha256=hashlib.sha256(source_bytes).hexdigest(),
        when2call_report_sha256=hashlib.sha256(extra_bytes).hexdigest(),
    )
    report = {
        "purpose": "research_only_gpu_multitask_world_task_head",
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "when2call_report_sha256": hashlib.sha256(extra_bytes).hexdigest(),
        "device": torch.cuda.get_device_name(device),
        "architecture": {
            "shared_hidden": HIDDEN,
            "heads": TASKS,
            "seed": SEED,
            "epochs": EPOCHS,
            "dropout": 0.3,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
        },
        "training_domains": list(domains),
        "selection": "Fixed settings; training partitions only; no test-driven candidate selection",
        "label_semantics": {
            "scope": "task/action scope labels, not execution authorization",
            "preference": "synthetic preferred versus dispreferred tool call, not authorization",
        },
        "metrics": metrics,
        "limitation": "The domain families and architecture were explored earlier. These are "
        "research results, not blind external validation, Jev parity, or an "
        "action-approval mechanism.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "task": value["task"],
                    "auc": value["auc"],
                    "accuracy_at_0.5": value["accuracy_at_0.5"],
                    "false_positives_at_0.5": value["false_positives_at_0.5"],
                }
                for name, value in metrics.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

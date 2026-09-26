"""Research-only nonlinear head on frozen prospective AgentNet features.

Model selection uses only train and calibration task groups. Development is
reported once; final and domain-shift labels are never opened by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from scripts.extract_agentnet_outcome_features import MODEL, REVISION, sha256_file
from scripts.fit_agentnet_outcome_head import assess, labels_for
from scripts.select_agentnet_outcome_cases import SOURCE_SHA256

HIDDEN_SIZES = (64, 128)
SEEDS = (19, 41)
EPOCHS = 100


def load_development(
    source: Path, manifest_path: Path, features_path: Path, feature_report_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Validate provenance and open only train/calibration/development labels."""
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    manifest_hash = sha256_file(manifest_path)
    feature_hash = sha256_file(features_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if (
        manifest["source_sha256"] != SOURCE_SHA256
        or report["source_sha256"] != SOURCE_SHA256
        or report["manifest_sha256"] != manifest_hash
        or report["features_sha256"] != feature_hash
        or report.get("input_view") != "pre_action"
        or report["model"] != MODEL
        or report["revision"] != REVISION
    ):
        raise ValueError("Prospective features do not match the pinned selection")
    with np.load(features_path, allow_pickle=False) as saved:
        ids = saved["ids"].tolist()
        matrix = np.asarray(saved["features"], dtype=np.float32)
        if saved["signature"].item() != report["signature"]:
            raise ValueError("Frozen feature signature differs from report")
    splits = manifest["splits"]
    if (
        set(splits) != {"train", "calibration", "development", "final", "ood"}
        or ids != manifest["case_ids"]
        or len(ids) != len({case_id.split(":", 1)[0] for case_id in ids})
        or matrix.shape != (len(ids), report["feature_dim"])
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Frozen feature rows or task-group split are invalid")
    positions = {case_id: index for index, case_id in enumerate(ids)}
    names = ("train", "calibration", "development")
    features = {name: matrix[[positions[case_id] for case_id in splits[name]]] for name in names}
    labels = {name: labels_for(source, splits[name]) for name in names}
    return features, labels, {"manifest_sha256": manifest_hash, "features_sha256": feature_hash}


def cautious_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Place the cutoff above every negative calibration score at score precision."""
    scores = np.asarray(probabilities, dtype=np.float64)
    return math.nextafter(float(np.max(scores[~labels])), math.inf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")

    features, labels, provenance = load_development(
        args.source, args.manifest, args.features, args.feature_report,
    )
    mean = features["train"].mean(axis=0)
    scale = features["train"].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = {name: (values - mean) / scale for name, values in features.items()}

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prospective-head research run")
    device = torch.device("cuda")
    train_x = torch.as_tensor(normalized["train"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(labels["train"].astype(np.float32), device=device)
    cal_x = torch.as_tensor(normalized["calibration"], dtype=torch.float32, device=device)
    dev_x = torch.as_tensor(normalized["development"], dtype=torch.float32, device=device)
    selected = None
    candidates = []
    for hidden in HIDDEN_SIZES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(train_x.shape[1], hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden, 1),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
            for _ in range(EPOCHS):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(train_x).squeeze(1), train_y,
                )
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                cal_prob = torch.sigmoid(model(cal_x)).squeeze(1).cpu().numpy().astype(np.float64)
            cal_loss = float(log_loss(labels["calibration"], cal_prob))
            candidates.append({"hidden": hidden, "seed": seed, "calibration_log_loss": cal_loss})
            rank = (cal_loss, hidden, seed)
            if selected is None or rank < selected[0]:
                selected = (rank, model, cal_prob)
    assert selected is not None
    (_, hidden, seed), model, cal_prob = selected
    with torch.inference_mode():
        dev_prob = torch.sigmoid(model(dev_x)).squeeze(1).cpu().numpy().astype(np.float64)
    cautious = cautious_threshold(cal_prob, labels["calibration"])
    weights = model.state_dict()
    np.savez_compressed(
        args.artifact,
        mean=mean, scale=scale,
        w1=weights["0.weight"].cpu().numpy(), b1=weights["0.bias"].cpu().numpy(),
        w2=weights["3.weight"].cpu().numpy(), b2=weights["3.bias"].cpu().numpy(),
        cautious_threshold=cautious,
        manifest_sha256=provenance["manifest_sha256"],
        feature_sha256=provenance["features_sha256"],
    )
    report = {
        "purpose": "agentnet_pre_action_nonlinear_development_only",
        "label_provenance": "model_generated_reflector_step_correct_not_human",
        "source_sha256": SOURCE_SHA256,
        **provenance,
        "architecture": "frozen_qwen3.5_2b_features_mlp",
        "candidates": candidates,
        "selected_hidden": hidden,
        "selected_seed": seed,
        "epochs": EPOCHS,
        "selection": "minimum_calibration_log_loss_only",
        "calibration_default": assess(labels["calibration"], cal_prob, 0.5),
        "calibration_cautious": assess(labels["calibration"], cal_prob, cautious),
        "development_default": assess(labels["development"], dev_prob, 0.5),
        "development_cautious": assess(labels["development"], dev_prob, cautious),
        "cautious_threshold": cautious,
        "artifact_sha256": sha256_file(args.artifact),
        "final_and_ood_labels_opened": False,
        "promotion": "none",
        "limitation": "Reflector labels are not human adjudication; this head has no runtime authority or Jev parity.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_hidden": hidden, "selected_seed": seed,
                      "development_default": report["development_default"],
                      "development_cautious": report["development_cautious"]}, indent=2))


if __name__ == "__main__":
    main()

"""Fit one research-only task-state head across TBAC and ScopeJudge domains.

Task/trajectory-disjoint public training and calibration select the head. Both
official test partitions and the frozen Heliox suite remain evaluation-only.
This artifact has no execution or approval authority in the daemon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from benchmarks.jev_scope_qwen_v1_holdout import CASES as HELIOX_CASES
from scripts.evaluate_openjev_checkpoint import scopejudge_cases
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT
from scripts.train_world_scope_semantic import (
    REGULARIZATION,
    embed_cases,
    metrics,
    normalize,
)
from scripts.train_world_tool_scope_semantic import DATASET_HASHES, DATASET_REVISION, load_cases, split_validation

DOMAINS = ("tbac", "scope")


def balanced_domain_weights(labels: dict[str, np.ndarray]) -> np.ndarray:
    """Give each domain/class cell one quarter of the training weight."""
    size = sum(len(value) for value in labels.values())
    weights = []
    for name in DOMAINS:
        targets = labels[name]
        if len(np.unique(targets)) != 2:
            raise ValueError(f"{name} training data needs both classes")
        positive = int(np.sum(targets))
        negative = len(targets) - positive
        weights.extend(np.where(targets, size / (4 * positive),
                                size / (4 * negative)).tolist())
    return np.asarray(weights)


def select_head(train: np.ndarray, train_labels: dict[str, np.ndarray],
                calibration: dict[str, np.ndarray], calibration_labels: dict[str, np.ndarray]):
    targets = np.concatenate([train_labels[name] for name in DOMAINS])
    weights = balanced_domain_weights(train_labels)
    trials = []
    for penalty in REGULARIZATION:
        head = LogisticRegression(C=penalty, max_iter=1000)
        head.fit(train, targets, sample_weight=weights)
        scores = {name: head.predict_proba(calibration[name])[:, 1] for name in DOMAINS}
        auc = {name: float(roc_auc_score(calibration_labels[name], scores[name]))
               for name in DOMAINS}
        trials.append((min(auc.values()), sum(auc.values()) / len(auc), penalty,
                       head, scores, auc))
    selected = max(trials, key=lambda item: (item[0], item[1], -item[2]))
    threshold = math.nextafter(max(float(np.max(selected[4][name][~calibration_labels[name]]))
                                   for name in DOMAINS), math.inf)
    summary = [{"C": item[2], "calibration_auc": item[5]} for item in trials]
    return selected[3], threshold, summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--artifact", type=Path, default=state / "world_multicorpus_scope_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "world_multicorpus_scope_v1.json")
    args = parser.parse_args()
    if args.max_length < 64 or args.artifact.exists() or args.report.exists():
        parser.error("Use max-length >=64 and fresh artifact/report paths")
    validation = load_cases(args.dataset_dir / "validation.json", DATASET_HASHES["validation"],
                            "tbac_validation")
    test = load_cases(args.dataset_dir / "test.json", DATASET_HASHES["test"], "tbac_test")
    tbac_train, tbac_calibration = split_validation(validation)
    if {case.request for case in validation} & {case.request for case in test}:
        raise ValueError("TBAC official test tasks overlap validation")
    scope, manifests = {}, {}
    for name, partition in (("scope_train", "scope_train"),
                            ("scope_calibration", "scope_calibration"),
                            ("scope_test", "scope_test")):
        scope[name], manifests[name] = scopejudge_cases(args.scopejudge_path, partition)
    if not all(manifest == manifests["scope_train"] for manifest in manifests.values()):
        raise ValueError("ScopeJudge source changed between partitions")
    groups = {name: {case.id.split(":", 1)[0] for case in part} for name, part in scope.items()}
    if any(groups[left] & groups[right] for left, right in
           (("scope_train", "scope_calibration"), ("scope_train", "scope_test"),
            ("scope_calibration", "scope_test"))):
        raise ValueError("ScopeJudge trajectories overlap")
    cases = {"tbac_train": tbac_train, "tbac_calibration": tbac_calibration,
             "tbac_test": test, **scope, "heliox": tuple(HELIOX_CASES)}

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    model.eval()
    features, labels, excluded = {}, {}, {}
    for name, part in cases.items():
        features[name], labels[name], excluded[name] = embed_cases(
            model, tokenizer, part, args.max_length, name)
    del model
    features["train"] = np.vstack((features["tbac_train"], features["scope_train"]))
    normalized, mean, scale = normalize(features)
    head, threshold, trials = select_head(
        normalized["train"],
        {name: labels[f"{name}_train"] for name in DOMAINS},
        {name: normalized[f"{name}_calibration"] for name in DOMAINS},
        {name: labels[f"{name}_calibration"] for name in DOMAINS})
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, weights=head.coef_.astype(np.float32),
                        bias=head.intercept_.astype(np.float32), mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32), threshold=threshold,
                        model_revision=MODEL_REVISION)
    scores = {name: head.predict_proba(values)[:, 1] for name, values in normalized.items()
              if name != "train"}
    report = {
        "purpose": "research_only_multicorpus_world_task_state_head",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tbac": {"url": "https://github.com/outshift-open/tbac-research-datasets",
                 "revision": DATASET_REVISION, "sha256": DATASET_HASHES},
        "scopejudge": manifests["scope_train"], "max_length": args.max_length,
        "rows": {name: len(part) for name, part in cases.items()},
        "excluded_overlong": excluded, "regularization_trials": trials,
        "selected_C": float(head.C), "calibration_threshold": threshold,
        "metrics": {name: {"default": metrics(labels[name], scores[name], 0.5),
                           "cautious": metrics(labels[name], scores[name], threshold)}
                    for name in ("tbac_calibration", "scope_calibration", "tbac_test",
                                 "scope_test", "heliox")},
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Public task-tool and expert tool-call labels with inspected synthetic Heliox "
                      "transfer only; no Jev parity, live desktop evidence, or execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["metrics"][name] for name in
                      ("tbac_test", "scope_test", "heliox")}, indent=2))


if __name__ == "__main__":
    main()

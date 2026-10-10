"""Train an offline task-state head with pre-call dialogue and expert scope.

Generated conversation relevance and expert action scope remain distinct
targets. The fitted head is for research only and cannot authorize actions.
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

from benchmarks.refscope_qwen_v1_holdout import CASES as HELIOX_CASES
from scripts.evaluate_opendec_checkpoint import scopejudge_cases
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT
from scripts.evaluate_world_conversation_scope import (
    AGENTIC_TEST_SHA256,
    DATASET_REVISION,
    load_pre_call_cases,
)
from scripts.train_world_scope_semantic import (
    REGULARIZATION,
    embed_cases,
    metrics,
    normalize,
)
from scripts.train_world_tool_scope_semantic import (
    DATASET_HASHES,
    load_cases,
    split_validation,
)

AGENTIC_VALIDATION_SHA256 = "2fd100bcc5973792d1fa35e4309eeb30a6dfb7543f14201db0f2015b295261a8"
DOMAINS = ("agentic", "scope")


def split_agentic(cases, source: Path) -> tuple[tuple, tuple, set[str]]:
    rows = json.loads(source.read_bytes())
    if len(rows) != len(cases):
        raise ValueError("Agentic source and cases differ")
    train, calibration, tasks = [], [], set()
    for row, case in zip(rows, cases, strict=True):
        seed_task = row["metadata"]["request"]["conversation"]["seed_task_request"]["task_tool_request"]["task"]
        tasks.add(seed_task)
        bucket = int(hashlib.sha256(seed_task.encode("utf-8")).hexdigest(), 16) % 5
        (calibration if bucket == 0 else train).append(case)
    if not train or not calibration:
        raise ValueError("Agentic train/calibration split is empty")
    return tuple(train), tuple(calibration), tasks


def select_head(
    train: np.ndarray,
    train_labels: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    calibration_labels: dict[str, np.ndarray],
    domains: tuple[str, ...] = DOMAINS,
):
    targets = np.concatenate([train_labels[name] for name in domains])
    size = len(targets)
    weights = []
    for name in domains:
        labels = train_labels[name]
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{name} training requires both classes")
        positive = int(np.sum(labels))
        negative = len(labels) - positive
        weights.extend(
            np.where(labels, size / (2 * len(domains) * positive), size / (2 * len(domains) * negative)).tolist()
        )
    trials = []
    for penalty in REGULARIZATION:
        head = LogisticRegression(C=penalty, max_iter=1000)
        head.fit(train, targets, sample_weight=np.asarray(weights))
        scores = {name: head.predict_proba(calibration[name])[:, 1] for name in domains}
        auc = {name: float(roc_auc_score(calibration_labels[name], scores[name])) for name in domains}
        trials.append((min(auc.values()), sum(auc.values()) / len(auc), penalty, head, scores, auc))
    selected = max(trials, key=lambda item: (item[0], item[1], -item[2]))
    threshold = math.nextafter(
        max(float(np.max(selected[4][name][~calibration_labels[name]])) for name in DOMAINS), math.inf
    )
    return selected[3], threshold, [{"C": item[2], "calibration_auc": item[5]} for item in trials]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--include-tbac-validation", action="store_true")
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--artifact", type=Path, default=state / "world_multiturn_scope_v2.npz")
    parser.add_argument("--report", type=Path, default=state / "world_multiturn_scope_v2.json")
    args = parser.parse_args()
    if args.max_length < 64 or args.batch_size < 1 or args.artifact.exists() or args.report.exists():
        parser.error("Use max-length >=64, batch-size >=1, and fresh artifact/report paths")
    agentic_source = args.dataset_dir / "agentic_validation_4o.json"
    agentic_val = load_pre_call_cases(agentic_source, AGENTIC_VALIDATION_SHA256)
    agentic_test_source = args.dataset_dir / "agentic_test_4o.json"
    agentic_test = load_pre_call_cases(agentic_test_source, AGENTIC_TEST_SHA256)
    agentic_train, agentic_calibration, val_tasks = split_agentic(agentic_val, agentic_source)
    test_tasks = {
        row["metadata"]["request"]["conversation"]["seed_task_request"]["task_tool_request"]["task"]
        for row in json.loads(agentic_test_source.read_bytes())
    }
    if val_tasks & test_tasks:
        raise ValueError("Agentic validation and official test seed tasks overlap")
    tbac_test = load_cases(args.dataset_dir / "test.json", DATASET_HASHES["test"], "tbac_test")
    tbac_partitions = {}
    if args.include_tbac_validation:
        tbac_val = load_cases(args.dataset_dir / "validation.json", DATASET_HASHES["validation"], "tbac_validation")
        tbac_partitions["tbac_train"], tbac_partitions["tbac_calibration"] = split_validation(tbac_val)
        if val_tasks != {case.request for case in tbac_val} or val_tasks & {case.request for case in tbac_test}:
            raise ValueError("TBAC and dialogue seed-task partitions do not align")
    scope, manifests = {}, {}
    for name in ("scope_train", "scope_calibration", "scope_test"):
        scope[name], manifests[name] = scopejudge_cases(args.scopejudge_path, name)
    if not all(manifest == manifests["scope_train"] for manifest in manifests.values()):
        raise ValueError("ScopeJudge source changed between partitions")
    groups = {name: {case.id.split(":", 1)[0] for case in part} for name, part in scope.items()}
    if any(
        groups[left] & groups[right]
        for left, right in (
            ("scope_train", "scope_calibration"),
            ("scope_train", "scope_test"),
            ("scope_calibration", "scope_test"),
        )
    ):
        raise ValueError("ScopeJudge trajectories overlap")
    cases = {
        "agentic_train": agentic_train,
        "agentic_calibration": agentic_calibration,
        "agentic_test": agentic_test,
        "tbac_test": tbac_test,
        **tbac_partitions,
        **scope,
        "heliox": tuple(HELIOX_CASES),
    }
    domains = ("agentic", "tbac", "scope") if args.include_tbac_validation else DOMAINS

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        output_loading_info=True,
    )
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    model.eval()
    features, labels, excluded = {}, {}, {}
    for name, part in cases.items():
        features[name], labels[name], excluded[name] = embed_cases(
            model,
            tokenizer,
            part,
            args.max_length,
            name,
            checkpoint_path=args.feature_dir / f"{name}.npz",
            batch_size=args.batch_size,
        )
    del model
    features["train"] = np.vstack([features[f"{name}_train"] for name in domains])
    normalized, mean, scale = normalize(features)
    head, threshold, trials = select_head(
        normalized["train"],
        {name: labels[f"{name}_train"] for name in domains},
        {name: normalized[f"{name}_calibration"] for name in domains},
        {name: labels[f"{name}_calibration"] for name in domains},
        domains,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        weights=head.coef_.astype(np.float32),
        bias=head.intercept_.astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        threshold=threshold,
        model_revision=MODEL_REVISION,
    )
    scores = {name: head.predict_proba(values)[:, 1] for name, values in normalized.items() if name != "train"}
    report = {
        "purpose": "research_only_multidomain_world_task_state_head",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "agentic": {
            "url": "https://github.com/outshift-open/tbac-research-datasets",
            "revision": DATASET_REVISION,
            "validation_sha256": AGENTIC_VALIDATION_SHA256,
            "test_sha256": AGENTIC_TEST_SHA256,
        },
        "tbac": {"sha256": DATASET_HASHES, "validation_in_training": args.include_tbac_validation},
        "domains": list(domains),
        "scopejudge": manifests["scope_train"],
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "rows": {name: len(part) for name, part in cases.items()},
        "excluded_overlong": excluded,
        "regularization_trials": trials,
        "feature_checkpoint_sha256": {
            name: hashlib.sha256((args.feature_dir / f"{name}.npz").read_bytes()).hexdigest() for name in cases
        },
        "selected_C": float(head.C),
        "calibration_threshold": threshold,
        "metrics": {
            name: {
                "default": metrics(labels[name], scores[name], 0.5),
                "cautious": metrics(labels[name], scores[name], threshold),
            }
            for name in tuple(f"{domain}_calibration" for domain in domains)
            + ("agentic_test", "tbac_test", "scope_test", "heliox")
        },
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Generated dialogue relevance and public expert scope are not live "
        "authorization evidence, Jev parity, or permission to execute.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["metrics"][name] for name in ("agentic_test", "scope_test", "heliox")}, indent=2))


if __name__ == "__main__":
    main()

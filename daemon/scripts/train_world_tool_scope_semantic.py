"""Train a research-only world task-to-tool scope head on pinned TBAC labels.

The backbone is frozen; the fitted head never enters Heliox's runtime. Keep
authoritative tool permissions and user approvals independent of its scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.refscope_qwen_v1_holdout import CASES as HELIOX_CASES
from benchmarks.refscope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT
from scripts.train_world_scope_semantic import (
    embed_cases,
    metrics,
    normalize,
    select_head,
)

DATASET_REVISION = "6385bd37654fbc36b84dd0341f0752654785dce3"
DATASET_HASHES = {
    "validation": "c30ea9f51a2aca5552602f8034cce691bdb69c6c0b9c990db7ac7e9a99292355",
    "test": "956b46f636e12f94dbc1d8ed433ff50a511a6b40db1a2c7889c85612cfc784e8",
}


def load_cases(path: Path, expected_sha256: str, split: str) -> tuple[ScopeCase, ...]:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(f"TBAC {split} source hash mismatch")
    rows = json.loads(source)
    if not isinstance(rows, list):
        raise ValueError(f"TBAC {split} must be a list")
    seen = set()
    cases = []
    for row in rows:
        identifier = row["id"]
        request = row["request"]
        tool = request["tool"]
        if identifier in seen or not all(isinstance(value, str) and value.strip() for value in
                                      (identifier, request["task"], tool["name"], tool["description"])):
            raise ValueError(f"Invalid or duplicate TBAC {split} row")
        if type(row["label"]["relevant"]) is not bool:
            raise ValueError(f"Nonboolean TBAC {split} label")
        seen.add(identifier)
        action = f"Invoke MCP tool {tool['name']}: {tool['description']}"
        cases.append(ScopeCase(identifier, request["task"], action,
                               row["label"]["relevant"], split))
    return tuple(cases)


def split_validation(cases: tuple[ScopeCase, ...]) -> tuple[tuple[ScopeCase, ...],
                                                            tuple[ScopeCase, ...]]:
    """Keep every tool candidate for one user task in the same partition."""
    train, calibration = [], []
    for case in cases:
        bucket = int(hashlib.sha256(case.request.encode("utf-8")).hexdigest(), 16) % 5
        (calibration if bucket == 0 else train).append(case)
    if {case.request for case in train} & {case.request for case in calibration}:
        raise AssertionError("Task split leaked across partitions")
    if any(len({case.in_scope for case in part}) != 2 for part in (train, calibration)):
        raise ValueError("TBAC training and calibration need both classes")
    return tuple(train), tuple(calibration)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--artifact", type=Path, default=state / "world_tool_scope_semantic_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "world_tool_scope_semantic_v1.json")
    args = parser.parse_args()
    if args.max_length < 64 or args.artifact.exists() or args.report.exists():
        parser.error("Use max-length >=64 and fresh artifact/report paths")
    validation = load_cases(args.dataset_dir / "validation.json", DATASET_HASHES["validation"],
                            "validation")
    test = load_cases(args.dataset_dir / "test.json", DATASET_HASHES["test"], "test")
    train, calibration = split_validation(validation)
    if {case.id for case in validation} & {case.id for case in test} or \
            {case.request for case in validation} & {case.request for case in test}:
        raise ValueError("TBAC source test overlaps validation")
    cases = {"train": train, "calibration": calibration, "test": test,
             "heliox": tuple(HELIOX_CASES)}

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
    normalized, mean, scale = normalize(features)
    head, threshold, trials = select_head(normalized["train"], labels["train"],
                                           normalized["calibration"], labels["calibration"])
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, weights=head.coef_.astype(np.float32),
                        bias=head.intercept_.astype(np.float32), mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32), threshold=threshold,
                        model_revision=MODEL_REVISION)
    scores = {name: head.predict_proba(values)[:, 1] for name, values in normalized.items()}
    report = {
        "purpose": "research_only_world_task_to_tool_scope_head",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "dataset": {"url": "https://github.com/outshift-open/tbac-research-datasets",
                    "revision": DATASET_REVISION, "sha256": DATASET_HASHES,
                    "license": "Apache-2.0"},
        "max_length": args.max_length,
        "rows": {name: len(part) for name, part in cases.items()},
        "excluded_overlong": excluded, "regularization_trials": trials,
        "selected_C": float(head.C), "calibration_threshold": threshold,
        "calibration": metrics(labels["calibration"], scores["calibration"], threshold),
        "test_default": metrics(labels["test"], scores["test"], 0.5),
        "test_cautious": metrics(labels["test"], scores["test"], threshold),
        "heliox_transfer_default": metrics(labels["heliox"], scores["heliox"], 0.5),
        "heliox_transfer_cautious": metrics(labels["heliox"], scores["heliox"], threshold),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Public task-tool relevance and inspected synthetic Heliox transfer only; "
                      "no Jev parity, live desktop evidence, world-model promotion, or execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in
                      ("test_default", "test_cautious", "heliox_transfer_cautious")}, indent=2))


if __name__ == "__main__":
    main()

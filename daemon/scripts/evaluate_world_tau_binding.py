"""Evaluate frozen world heads on independent retail target-binding tasks.

The published annotated tau-bench tasks are evaluation-only. No task, candidate, or
label from this source is used to fit a model or choose a threshold.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.jev_scope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.score_world_mlp_artifact import score_features
from scripts.train_world_scope_semantic import (
    embed_cases,
    feature_identity,
    load_feature_checkpoint,
    metrics,
)

REPOSITORY = "sierra-research/tau-bench"
REVISION = "59a200c6d575d595120f1cb70fea53cef0632f6b"
SOURCE_FILES = {
    "tasks": ("tau_bench/envs/retail/tasks.py",
              "fcd6f0daada96372ede73415a5b80dce90116287a24ad7c7a8a49fa68b0ffe91"),
    "orders": ("tau_bench/envs/retail/data/orders.json",
               "81595d5cb28171e01133b52714b456003a9cd733cfd013c1d9d397ac0a7484c4"),
    "users": ("tau_bench/envs/retail/data/users.json",
              "351c93bce10ce26ba2c9f6a048665d7e002ccff2c1c42a9d628680e73acc5342"),
}


def load_source(root: Path) -> tuple[list[dict], dict, dict]:
    source = {}
    for name, (relative, expected_hash) in SOURCE_FILES.items():
        content = (root / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Pinned tau-bench {name} hash mismatch")
        source[name] = content.decode("utf-8")
    tree = ast.parse(source["tasks"])
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "tasks"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected one literal tau-bench tasks assignment")
    tasks = ast.literal_eval(assignments[0].value)
    orders, users = json.loads(source["orders"]), json.loads(source["users"])
    if not isinstance(tasks, list) or not isinstance(orders, dict) or not isinstance(users, dict):
        raise ValueError("Invalid tau-bench source structure")
    return tasks, orders, users


def build_cases(tasks: list[dict], orders: dict, users: dict) -> tuple[tuple[ScopeCase, ...], dict]:
    cases = []
    groups = {}
    for task_index, task in enumerate(tasks):
        user_id = task["user_id"]
        user = users.get(user_id)
        if user is None:
            raise ValueError(f"Unknown tau-bench user: {user_id}")
        available = {order_id: orders[order_id] for order_id in user["orders"]
                     if order_id in orders}
        by_tool: dict[str, set[str]] = defaultdict(set)
        for action in task["actions"]:
            arguments = action["arguments"]
            order_id = arguments.get("order_id")
            if (order_id in available and "item_ids" not in arguments
                    and "new_item_ids" not in arguments):
                by_tool[action["name"]].add(order_id)
        for tool_name, gold_ids in sorted(by_tool.items()):
            statuses = {available[order_id]["status"] for order_id in gold_ids}
            choices = [order for order_id, order in sorted(available.items())
                       if order["status"] in statuses]
            if not any(order["order_id"] not in gold_ids for order in choices):
                continue
            group_id = f"task{task_index}:{tool_name}"
            visible = [{"id": order["order_id"], "status": order["status"],
                        "items": [{"name": item["name"], "options": item.get("options", {})}
                                  for item in order["items"]]}
                       for order in choices]
            request = (task["instruction"] + "\nVisible candidate orders: " +
                       json.dumps(visible, ensure_ascii=False, sort_keys=True))
            groups[group_id] = {"tool": tool_name, "gold": frozenset(gold_ids),
                                "implicit": all(order_id not in task["instruction"]
                                                for order_id in gold_ids)}
            for order in choices:
                order_id = order["order_id"]
                cases.append(ScopeCase(
                    f"{group_id}:{order_id}", request,
                    f"Invoke tool {tool_name} with order_id {order_id}.",
                    order_id in gold_ids, "tau_retail_binding"))
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate tau-bench candidate ID")
    return tuple(cases), groups


def group_metrics(cases: tuple[ScopeCase, ...], groups: dict,
                  scores: np.ndarray, threshold: float) -> dict:
    candidates: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for case, score in zip(cases, scores, strict=True):
        candidates[case.id.rsplit(":", 1)[0]].append((case.in_scope, float(score)))
    if set(candidates) != set(groups):
        raise ValueError("Candidate groups do not match source groups")
    return {
        "groups": len(groups),
        "implicit_groups": sum(group["implicit"] for group in groups.values()),
        "correct_top_rank": sum(max(values, key=lambda item: item[1])[0]
                                for values in candidates.values()),
        "correct_safe_groups": sum(
            all((score >= threshold) == label for label, score in values)
            for values in candidates.values()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path,
                        default=state / "world_multidomain_mlp_v1.npz")
    parser.add_argument("--artifact-report", type=Path,
                        default=state / "world_multidomain_mlp_v1.json")
    parser.add_argument("--feature-checkpoint", type=Path,
                        default=state / "world_tau_binding_features_v1.npz")
    parser.add_argument("--report", type=Path,
                        default=state / "world_tau_binding_eval_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    tasks, orders, users = load_source(args.source_root)
    cases, groups = build_cases(tasks, orders, users)
    trained = json.loads(args.artifact_report.read_text(encoding="utf-8"))
    artifact_sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    if artifact_sha256 != trained["artifact_sha256"]:
        raise ValueError("Frozen head artifact hash mismatch")

    processed, overlong, _, vectors, targets = load_feature_checkpoint(
        args.feature_checkpoint, feature_identity(cases, 1536, 2), cases)
    if processed == len(cases):
        features, labels = np.asarray(vectors), np.asarray(targets, dtype=np.bool_)
    else:
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
        features, labels, overlong = embed_cases(
            model.eval(), tokenizer, cases, 1536, "tau_retail_binding",
            checkpoint_path=args.feature_checkpoint, batch_size=2)
    if overlong:
        raise ValueError("External tau-bench evaluation excluded overlong cases")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        scores = score_features(artifact, features)
        threshold = float(artifact["threshold"])
    implicit = np.asarray([groups[case.id.rsplit(":", 1)[0]]["implicit"] for case in cases])
    report = {
        "purpose": "evaluation_only_independent_retail_target_binding_transfer",
        "source": {"url": f"https://github.com/{REPOSITORY}", "revision": REVISION,
                   "files": {name: {"path": path, "sha256": digest}
                             for name, (path, digest) in SOURCE_FILES.items()}, "license": "MIT"},
        "protocol": "Published annotated tasks; one candidate per same-user, same-status order. "
                    "Published tool/order annotations are positive; other compatible orders are negative. "
                    "Item-specific calls are excluded; no labels tune the frozen head.",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "frozen_artifact_sha256": artifact_sha256,
        "tasks": len(tasks), "groups": len(groups), "selected_rows": len(cases),
        "eligible_rows": len(labels), "implicit_rows": int(implicit.sum()),
        "default": metrics(labels, scores, 0.5),
        "cautious": metrics(labels, scores, threshold),
        "implicit_default": metrics(labels[implicit], scores[implicit], 0.5),
        "implicit_cautious": metrics(labels[implicit], scores[implicit], threshold),
        "group_default": group_metrics(cases, groups, scores, 0.5),
        "group_cautious": group_metrics(cases, groups, scores, threshold),
        "feature_checkpoint_sha256": hashlib.sha256(args.feature_checkpoint.read_bytes()).hexdigest(),
        "limitation": "Small observational benchmark. Published tool annotations may be conditional "
                      "on earlier workflow outcomes; this tests candidate relevance, not direct execution "
                      "authorization or broad desktop safety.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in
                      ("default", "cautious", "implicit_default", "group_default")}, indent=2))


if __name__ == "__main__":
    main()

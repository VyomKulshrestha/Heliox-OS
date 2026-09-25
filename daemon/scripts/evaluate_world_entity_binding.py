"""Evaluate frozen world heads on external right-tool/wrong-entity tasks.

EntityBindingFailures is evaluation-only. Its gold bindings create candidate
actions; neither these labels nor scores are used to fit or select a model.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.jev_scope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.score_world_mlp_artifact import score_features
from scripts.train_world_scope_semantic import embed_cases, metrics

REPOSITORY = "R-Suresh/EntityBindingFailures"
REVISION = "af311d10f52680576d83cdaf3d05d5b090445de4"
DATA_PATH = "data/tasks_entity_binding_final_60.jsonl"
DATA_SHA256 = "8b57c8a4460d02a4f21a3e0cd65ef458aef789679d5bef229c8e80bfa0089cb4"


def load_rows(path: Path | None) -> list[dict]:
    if path is None:
        result = subprocess.run(
            ["gh", "api", f"repos/{REPOSITORY}/contents/{DATA_PATH}?ref={REVISION}",
             "-H", "Accept: application/vnd.github.raw+json"],
            check=True, capture_output=True)
        source = result.stdout
    else:
        source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != DATA_SHA256:
        raise ValueError("Pinned entity-binding dataset hash mismatch")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    if len(rows) != 60 or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Entity-binding dataset rows changed")
    return rows


def build_cases(rows: list[dict]) -> tuple[tuple[ScopeCase, ...], dict[str, bool]]:
    cases, resolvable = [], {}
    for row in rows:
        task_id = row["task_id"]
        entities = {entity["id"]: entity for entity in row["entities"]}
        tool = next((tool for tool in row["tools"] if tool["name"] == row["gold_tool"]), None)
        if tool is None or task_id in resolvable or not row["gold_bindings"]:
            raise ValueError(f"Invalid gold tool, task, or bindings: {task_id}")
        slot_types = dict(requirement.split(":", 1) for requirement in tool["requires"])
        slots = tuple(row["gold_bindings"])
        if set(slots) != set(slot_types):
            raise ValueError(f"Gold binding slots differ from tool schema: {task_id}")
        choices = []
        for slot in slots:
            options = [entity for entity in row["entities"]
                       if entity["type"] == slot_types[slot]]
            gold_id = row["gold_bindings"][slot]
            if not options or (gold_id != "NEEDS_CLARIFICATION" and gold_id not in entities):
                raise ValueError(f"Missing entity choices or gold binding: {task_id}")
            choices.append(options)
        resolvable[task_id] = "NEEDS_CLARIFICATION" not in row["gold_bindings"].values()
        request = (row["instruction"] + "\nVisible candidate entities: " +
                   json.dumps(row["entities"], ensure_ascii=False, sort_keys=True))
        task_cases = []
        for choice in itertools.product(*choices):
            bindings = {slot: entity["id"] for slot, entity in zip(slots, choice, strict=True)}
            allowed = resolvable[task_id] and bindings == row["gold_bindings"]
            action = (f"Invoke tool {tool['name']}: {tool['description']} with bindings " +
                      json.dumps([{"slot": slot, "entity": entity} for slot, entity in
                                  zip(slots, choice, strict=True)],
                                 ensure_ascii=False, sort_keys=True))
            task_cases.append(ScopeCase(
                task_id + ":" + ":".join(entity["id"] for entity in choice),
                request, action, allowed, "entity_binding"))
        if (resolvable[task_id] and sum(case.in_scope for case in task_cases) != 1) or \
                (not resolvable[task_id] and any(case.in_scope for case in task_cases)) or \
                not any(not case.in_scope for case in task_cases):
            raise ValueError(f"Invalid generated candidate labels: {task_id}")
        cases.extend(task_cases)
    return tuple(cases), resolvable


def task_metrics(cases: tuple[ScopeCase, ...], scores: np.ndarray,
                 threshold: float, resolvable: dict[str, bool]) -> dict:
    grouped = defaultdict(list)
    for case, score in zip(cases, scores, strict=True):
        grouped[case.id.split(":", 1)[0]].append((case.in_scope, float(score)))
    correct_rank, safe_decision = 0, 0
    for task_id, values in grouped.items():
        best_label, best_score = max(values, key=lambda item: item[1])
        if resolvable[task_id]:
            correct_rank += bool(best_label)
            safe_decision += bool(best_label and best_score >= threshold)
        else:
            safe_decision += all(score < threshold for _, score in values)
    return {"tasks": len(grouped), "resolvable": sum(resolvable.values()),
            "ambiguous": len(grouped) - sum(resolvable.values()),
            "correct_rank_on_resolvable": correct_rank,
            "correct_safe_decisions": safe_decision}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--artifact", type=Path,
                        default=state / "world_multidomain_mlp_v1.npz")
    parser.add_argument("--artifact-report", type=Path,
                        default=state / "world_multidomain_mlp_v1.json")
    parser.add_argument("--feature-checkpoint", type=Path,
                        default=state / "world_entity_binding_features_v1.npz")
    parser.add_argument("--report", type=Path,
                        default=state / "world_entity_binding_eval_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    rows = load_rows(args.dataset)
    cases, resolvable = build_cases(rows)
    trained = json.loads(args.artifact_report.read_text(encoding="utf-8"))
    source = args.artifact.read_bytes()
    if hashlib.sha256(source).hexdigest() != trained["artifact_sha256"]:
        raise ValueError("Frozen head artifact hash mismatch")

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
        model.eval(), tokenizer, cases, 1024, "entity_binding",
        checkpoint_path=args.feature_checkpoint, batch_size=4)
    if overlong:
        raise ValueError("External entity-binding evaluation excluded overlong cases")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        scores = score_features(artifact, features)
        threshold = float(artifact["threshold"])
    report = {
        "purpose": "evaluation_only_external_entity_binding_transfer",
        "source": {"url": f"https://github.com/{REPOSITORY}", "revision": REVISION,
                   "path": DATA_PATH, "sha256": DATA_SHA256, "license": "CC BY 4.0"},
        "candidate_protocol": "Cartesian product of schema-typed visible entities for the gold tool; "
                              "gold binding is positive, wrong or ambiguous direct actions negative",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "frozen_artifact_sha256": trained["artifact_sha256"],
        "selected_rows": len(cases), "eligible_rows": len(labels),
        "default": metrics(labels, scores, 0.5),
        "cautious": metrics(labels, scores, threshold),
        "task_default": task_metrics(cases, scores, 0.5, resolvable),
        "task_cautious": task_metrics(cases, scores, threshold, resolvable),
        "feature_checkpoint_sha256": hashlib.sha256(args.feature_checkpoint.read_bytes()).hexdigest(),
        "limitation": "Small templated external research benchmark; model was frozen before this "
                      "dataset was examined. Entity binding relevance is not execution authorization "
                      "or broad live desktop safety evidence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in
                      ("default", "cautious", "task_default", "task_cautious")}, indent=2))


if __name__ == "__main__":
    main()

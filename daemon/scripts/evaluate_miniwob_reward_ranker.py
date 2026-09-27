"""Fit an offline paired outcome ranker on public MiniWoB branches.

Only observed, opposing terminal rewards are labels. Task-disjoint validation
selects the regularization; evaluation is scored once. No weights are loaded by
Heliox, and no branch is executed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluate_dream_replay import select_action


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pairs(archive: Path, features: Path, source_report: Path,
               feature_report: Path, protocol: dict) -> tuple[dict, dict]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    extracted = json.loads(feature_report.read_text(encoding="utf-8"))
    if (source.get("counterfactual_pairs") is not True
        or source.get("source_repo") != protocol["source_repo"]
        or source.get("source_commit") != protocol["source_commit"]
        or source.get("seed_start") != protocol["seed_start"]
        or source.get("samples_per_task") != protocol["samples_per_task"]
        or source.get("max_candidates") != protocol["max_candidates"]
        or source.get("semantic_archive_sha256") != digest(archive)
        or extracted.get("source_report_sha256") != digest(source_report)
        or extracted.get("visual_features_sha256") != source.get("features_sha256")
        or extracted.get("semantic_archive_sha256") != digest(archive)
        or extracted.get("model") != protocol["feature_model"]
        or extracted.get("revision") != protocol["feature_revision"]
        or extracted.get("prompt_version") != protocol["feature_prompt_version"]
        or extracted.get("max_length") != protocol["feature_max_length"]
        or extracted.get("batch_size") != protocol["feature_batch_size"]
        or extracted.get("feature_sha256") != digest(features)):
        raise ValueError("Source archive or frozen features differ from the protocol")
    for split in ("train", "validation", "eval"):
        if source.get("tasks", {}).get(split) != protocol[f"{split}_tasks"]:
            raise ValueError(f"{split} task pages differ from the protocol")
    pages = [page for split in ("train", "validation", "eval")
             for page in protocol[f"{split}_tasks"]]
    if len(set(pages)) != len(pages):
        raise ValueError("Task pages overlap across partitions")
    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(features, allow_pickle=False) as data:
        matrix = data["features"].astype(np.float32)
        groups, splits, coords = (data[name] for name in ("group", "split", "coords"))
    if (len(matrix) != len(rows) or not np.isfinite(matrix).all()
        or any(row["group"] != str(group) or row["split"] != str(split)
               or not np.allclose(row["coords"], xy, atol=1e-6)
               for row, group, split, xy in zip(rows, groups, splits, coords, strict=True))):
        raise ValueError("Frozen features do not align with observed branches")
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[(row["split"], row["group"])].append((row, matrix[index]))
    pairs = defaultdict(list)
    excluded = Counter()
    for (split, group), entries in sorted(buckets.items()):
        if (len(entries) != 2 or len({item[0]["task"] for item in entries}) != 1
            or entries[0][0]["state_text"] != entries[1][0]["state_text"]):
            excluded["unaligned_pair"] += 1
            continue
        rewards = [entry[0].get("reward") for entry in entries]
        if (any(type(reward) not in (int, float) or reward not in (-1, 1) for reward in rewards)
            or rewards[0] == rewards[1]):
            excluded["no_observed_reward_contrast"] += 1
            continue
        # A fixed hash bit removes collector order as a training signal.
        if hashlib.sha256(f"ranker-v1:{split}:{group}".encode()).digest()[0] & 1:
            entries.reverse()
        pair_rows = [item[0] for item in entries]
        difference = entries[0][1] - entries[1][1]
        pairs[split].append((pair_rows, difference, int(entries[0][0]["reward"] == 1)))
    for split in ("train", "validation", "eval"):
        if len(pairs[split]) < protocol["minimum_contrast_pairs"][split]:
            raise ValueError(f"Insufficient observed contrast pairs for {split}")
    return pairs, dict(excluded)


def score(pairs: list[tuple], model, scale: np.ndarray) -> dict:
    counts = Counter()
    by_task = defaultdict(Counter)
    for rows, difference, first_is_good in pairs:
        task = rows[0]["task"]
        predicted = int(model.decision_function((difference / scale)[None, :])[0] >= 0)
        selections = {"ranker": 0 if predicted else 1}
        for policy in ("first", "goal_match", "goal_grounded"):
            selections[policy] = select_action(policy, rows)
        for policy, index in selections.items():
            won = int(index == (0 if first_is_good else 1))
            counts[policy] += won
            by_task[task][policy] += won
        by_task[task]["pairs"] += 1
    return {"pairs": len(pairs), "wins": dict(counts),
            "accuracy": {name: counts[name] / len(pairs) for name in
                         ("ranker", "first", "goal_match", "goal_grounded")},
            "by_task": {name: dict(values) for name, values in sorted(by_task.items())}}


def run(archive: Path, features: Path, source_report: Path, feature_report: Path,
        protocol_path: Path, weights_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    penalties = protocol["candidate_penalties"]
    if (not isinstance(penalties, list) or not penalties
        or any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
               for value in penalties)):
        raise ValueError("Protocol needs positive finite candidate penalties")
    pairs, excluded = load_pairs(archive, features, source_report, feature_report, protocol)
    train = pairs["train"]
    scale = np.std(np.stack([item[1] for item in train]), axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x_train = np.stack([item[1] / scale for item in train])
    y_train = np.asarray([item[2] for item in train])
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training pair orientation has only one reward class")
    trials = []
    for penalty in penalties:
        model = LogisticRegression(C=penalty, fit_intercept=False, max_iter=1000)
        model.fit(x_train, y_train)
        validation = score(pairs["validation"], model, scale)
        trials.append((validation["wins"]["ranker"], penalty, model, validation))
    chosen = max(trials, key=lambda trial: (trial[0], -trial[1]))
    evaluation = score(pairs["eval"], chosen[2], scale)
    by_task = evaluation["by_task"]
    floor = protocol["minimum_eval_pairs_per_task"]
    expected = protocol["eval_tasks"]
    covered = all(by_task.get(task, {}).get("pairs", 0) >= floor for task in expected)
    beat_controls = covered and all(
        by_task[task]["ranker"] > max(by_task[task]["first"],
                                      by_task[task]["goal_match"],
                                      by_task[task]["goal_grounded"])
        for task in expected
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, coefficient=chosen[2].coef_.astype(np.float32),
                        scale=scale.astype(np.float32), selected_penalty=np.asarray(chosen[1]))
    return {
        "purpose": "research_only_task_disjoint_terminal_reward_ranker",
        "protocol_sha256": digest(protocol_path), "source_report_sha256": digest(source_report),
        "feature_report_sha256": digest(feature_report), "feature_sha256": digest(features),
        "weights_sha256": digest(weights_path),
        "observed_pairs": {name: len(pairs[name]) for name in ("train", "validation", "eval")},
        "excluded_groups": excluded,
        "validation_trials": [{"penalty": penalty, "ranker_wins": wins}
                              for wins, penalty, _, _ in trials],
        "selected_penalty": chosen[1], "validation": chosen[3], "evaluation": evaluation,
        "gate": {"task_coverage": covered, "beat_all_controls_each_task": beat_controls,
                 "passed": covered and beat_controls},
        "limitation": "Selected synthetic one-step outcomes only; no unobserved branches, "
                      "long-horizon task success, real desktop transfer, execution authority, or Jev parity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "features", "source_report", "feature_report", "protocol", "report", "weights"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.weights.exists() or args.report.resolve() == args.weights.resolve():
        parser.error("Use fresh, distinct report and weight paths")
    report = run(args.archive, args.features, args.source_report, args.feature_report,
                 args.protocol, args.weights)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"observed_pairs": report["observed_pairs"],
                      "selected_penalty": report["selected_penalty"],
                      "evaluation": report["evaluation"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()

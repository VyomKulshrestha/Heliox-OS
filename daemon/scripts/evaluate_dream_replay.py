"""Research-only, exact branch replay for MiniWoB action-selection policies.

Dream-RSI's replay idea is useful only where alternative outcomes were actually
observed. This one-step diagnostic never predicts missing outcomes or executes
an action. It does not modify Heliox's live model, planner, or permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _goal_score(goal: str, label: str) -> tuple[int, int]:
    label_tokens = _tokens(label)
    goal_tokens = _tokens(goal)
    if not label_tokens or not goal_tokens:
        return (0, 0)
    phrase = int(
        any(
            goal_tokens[i : i + len(label_tokens)] == label_tokens
            for i in range(len(goal_tokens) - len(label_tokens) + 1)
        )
    )
    overlap = len(set(label_tokens) & set(goal_tokens))
    return (phrase, overlap)


def select_action(policy: str, rows: list[dict[str, Any]]) -> int:
    if policy == "first":
        return 0
    if policy == "last":
        return len(rows) - 1
    if policy == "goal_match":
        visible_text = rows[0]["state_text"].split("\nControls:", 1)[0]
        goal = re.split(r"[.!?](?:\s|$)", visible_text, maxsplit=1)[0]
        scores = [_goal_score(goal, row["action_label"]) for row in rows]
        return max(range(len(rows)), key=lambda index: scores[index])
    if policy == "goal_grounded":
        visible_text = rows[0]["state_text"].split("\nControls:", 1)[0]
        goal = re.split(r"[.!?](?:\s|$)", visible_text, maxsplit=1)[0]
        scores = []
        for row in rows:
            label_phrase, label_overlap = _goal_score(goal, row["action_label"])
            kind_phrase, kind_overlap = _goal_score(goal, row.get("action_kind", ""))
            scores.append((label_phrase, kind_phrase, label_overlap, kind_overlap))
        return max(range(len(rows)), key=lambda index: scores[index])
    raise ValueError(f"Unknown replay policy: {policy}")


def load_branches(path: Path, seed: int) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, int]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    tasks: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split, group, task = (row[key] for key in ("split", "group", "task"))
        if split not in {"train", "validation", "eval"} or not all(
            isinstance(value, str) and value for value in (group, task)
        ):
            raise ValueError("Invalid split, group, or task in replay archive")
        tasks[split].add(task)
        by_group[(split, group)].append(row)
    if any(
        tasks[left] & tasks[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Task identity leaked across replay splits")

    branches: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    skipped = Counter()
    for (split, group), rows in sorted(by_group.items()):
        if len(rows) != 2 or len({row["task"] for row in rows}) != 1:
            skipped["not_two_branches"] += 1
            continue
        if rows[0]["state_text"] != rows[1]["state_text"]:
            skipped["different_prestates"] += 1
            continue
        rewards = [row.get("reward") for row in rows]
        if any(type(reward) not in (int, float) or reward not in (-1, 1) for reward in rewards):
            skipped["unobserved_or_invalid_reward"] += 1
            continue
        if rewards[0] == rewards[1]:
            skipped["no_reward_contrast"] += 1
            continue
        if any(not isinstance(row.get("action_label"), str) for row in rows):
            skipped["missing_action_label"] += 1
            continue
        # Deterministic presentation randomization removes collector-order advantage.
        if hashlib.sha256(f"{seed}:{split}:{group}".encode()).digest()[0] & 1:
            rows = [rows[1], rows[0]]
        branches[split].append(rows)
    return branches, dict(skipped)


def evaluate(branches: list[list[dict[str, Any]]], policy: str) -> dict[str, Any]:
    wins = sum(rows[select_action(policy, rows)]["reward"] == 1 for rows in branches)
    return {"wins": wins, "groups": len(branches), "accuracy": wins / len(branches)}


def run(archive: Path, seed: int) -> dict[str, Any]:
    branches, skipped = load_branches(archive, seed)
    if not branches["validation"] or not branches["eval"]:
        raise ValueError("At least one fully observed reward-contrast group is required in validation and eval")
    policies = ("first", "last", "goal_match", "goal_grounded")
    validation = {policy: evaluate(branches["validation"], policy) for policy in policies}
    # Select without seeing eval rewards; ties favor the simpler fixed baseline.
    selected = max(policies, key=lambda policy: validation[policy]["wins"])
    evaluation = {policy: evaluate(branches["eval"], policy) for policy in policies}
    eval_tasks = sorted({rows[0]["task"] for rows in branches["eval"]})
    per_task = {
        task: {
            policy: evaluate([rows for rows in branches["eval"] if rows[0]["task"] == task], policy)
            for policy in ("first", selected)
        }
        for task in eval_tasks
    }
    sufficient_coverage = len(branches["eval"]) >= 10 and len(eval_tasks) >= 2
    improved = evaluation[selected]["accuracy"] - evaluation["first"]["accuracy"] >= 0.10 and all(
        scores[selected]["wins"] >= scores["first"]["wins"] for scores in per_task.values()
    )
    return {
        "purpose": "research_only_dream_rsi_inspired_exact_one_step_replay",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "presentation_seed": seed,
        "skipped_groups": skipped,
        "validation": validation,
        "selected_on_validation": selected,
        "held_out_eval": evaluation,
        "held_out_by_task": per_task,
        "selected_minus_first_wins": evaluation[selected]["wins"] - evaluation["first"]["wins"],
        "predeclared_gate": {
            "sufficient_coverage": sufficient_coverage,
            "improved_over_fixed_first": improved,
            "passed": sufficient_coverage and improved,
        },
        "limitation": "Only two observed terminal actions per synthetic reset state are replayed. "
        "No unseen branch, long-horizon policy, real desktop outcome, model weight, "
        "Jev parity, or execution authority is established.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(args.archive, args.seed)
    if args.report is not None:
        if args.report.exists():
            parser.error("Report already exists; use a fresh output path")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

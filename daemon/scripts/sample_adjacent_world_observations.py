"""Sample observed next-state text from adjacent desktop trajectory steps.

This research-only sampler never executes source actions. Public source text is
kept in ignored local benchmark state, not the Heliox runtime or Git history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.sample_grouped_world_transitions import (
    REVISION,
    _field,
    fetch_page,
    separate_groups,
)

SYSTEM = (
    "Predict the next observed desktop screen state after the proposed action. "
    "Describe what is visible, including failure or unchanged state."
)
ACTION_MARKER = "\n\nAction:\n"
REQUEST_MARKER = "\n\nGiven the current state"


def parse_state_action(prompt: str) -> tuple[str, str] | None:
    """Accept the source schema only when state and action are unambiguous."""
    normalized = prompt.replace("\r\n", "\n")
    if not normalized.startswith("Current state:\n") or normalized.count(ACTION_MARKER) != 1:
        return None
    state_part, _, action_part = normalized.partition(ACTION_MARKER)
    if action_part.count(REQUEST_MARKER) != 1:
        return None
    action_part = action_part.split(REQUEST_MARKER, 1)[0]
    state = state_part[len("Current state:\n"):].strip()
    action = action_part.strip()
    return (state, action) if state and action else None


def extract_pair(current: dict, following: dict, *, split: str, revision: str,
                 dataset: str, group_path: str, step_path: str,
                 max_input_chars: int, max_target_chars: int) -> tuple[dict | None, str]:
    if current.get("truncated_cells") or following.get("truncated_cells"):
        return None, "truncated_source"
    source, target = current.get("row"), following.get("row")
    if not isinstance(source, dict) or not isinstance(target, dict):
        return None, "invalid_source"
    group = _field(source, group_path)
    next_group = _field(target, group_path)
    step = _field(source, step_path)
    next_step = _field(target, step_path)
    if not isinstance(group, str) or not group.strip() or group != next_group:
        return None, "different_task"
    if (not isinstance(step, int) or isinstance(step, bool) or step < 0
            or not isinstance(next_step, int) or isinstance(next_step, bool)
            or next_step != step + 1):
        return None, "nonconsecutive_step"
    messages, next_messages = source.get("messages"), target.get("messages")
    if (not isinstance(messages, list) or len(messages) != 3
            or not isinstance(next_messages, list) or len(next_messages) != 3
            or any(not isinstance(message, dict) for message in [*messages, *next_messages])
            or [message.get("role") for message in messages] != ["system", "user", "assistant"]
            or [message.get("role") for message in next_messages]
            != ["system", "user", "assistant"]):
        return None, "invalid_messages"
    prompt, next_prompt = messages[1].get("content"), next_messages[1].get("content")
    if not isinstance(prompt, str) or not isinstance(next_prompt, str):
        return None, "invalid_messages"
    parsed, next_parsed = parse_state_action(prompt), parse_state_action(next_prompt)
    if parsed is None or next_parsed is None:
        return None, "ambiguous_state_action"
    state, action = parsed
    next_state, _ = next_parsed
    user = f"Current state:\n{state}\n\nAction:\n{action}\n\nNext observed screen state:"
    if len(SYSTEM) + len(user) > max_input_chars or len(next_state) > max_target_chars:
        return None, "overlong"
    if (not isinstance(current.get("row_idx"), int)
            or following.get("row_idx") != current["row_idx"] + 1):
        return None, "nonadjacent_source_rows"
    return {
        "split": split,
        "row_index": current["row_idx"],
        "next_row_index": following["row_idx"],
        "step_index": step,
        "dataset": dataset,
        "revision": revision,
        "group_sha256": hashlib.sha256(group.encode("utf-8")).hexdigest(),
        "context_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        "source_sha256": hashlib.sha256(
            json.dumps([source, target], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": next_state},
        ],
    }, "accepted"


def sample_split(*, dataset: str, config: str, source_split: str, output_split: str,
                 pages: int, page_length: int, rng: random.Random,
                 expected_revision: str, group_path: str, step_path: str,
                 max_input_chars: int, max_target_chars: int) -> tuple[list[dict], Counter, int]:
    first, revision = fetch_page(dataset, config, source_split, 0, 1)
    if revision != expected_revision:
        raise ValueError("Dataset revision differs from the pinned source")
    total = first.get("num_rows_total")
    if not isinstance(total, int) or total < pages * page_length:
        raise ValueError("Dataset split is too small for requested pages")
    records: list[dict] = []
    exclusions: Counter = Counter()
    for page_index in range(pages):
        lower = page_index * total // pages
        upper = (page_index + 1) * total // pages - page_length
        offset = rng.randint(lower, upper)
        page, page_revision = fetch_page(dataset, config, source_split, offset, page_length)
        if page_revision != revision or page.get("num_rows_total") != total:
            raise ValueError("Dataset revision or split length changed during sampling")
        entries = page.get("rows")
        if (not isinstance(entries, list) or len(entries) != page_length
                or any(not isinstance(entry, dict) or entry.get("row_idx") != offset + index
                       for index, entry in enumerate(entries))):
            raise ValueError("Dataset server returned an incomplete or unexpected page")
        for current, following in zip(entries, entries[1:]):
            record, reason = extract_pair(
                current, following, split=output_split, revision=revision, dataset=dataset,
                group_path=group_path, step_path=step_path,
                max_input_chars=max_input_chars, max_target_chars=max_target_chars,
            )
            exclusions[reason] += 1
            if record is not None:
                records.append(record)
    return records, exclusions, total


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--group-path", default="metadata.task_id")
    parser.add_argument("--step-path", default="metadata.step_index")
    parser.add_argument("--train-pages", type=int, default=12)
    parser.add_argument("--eval-pages", type=int, default=8)
    parser.add_argument("--page-length", type=int, default=100)
    parser.add_argument("--max-input-chars", type=int, default=6500)
    parser.add_argument("--max-target-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--rows", type=Path, default=state / "adjacent_world_observations.jsonl")
    parser.add_argument("--report", type=Path, default=state / "adjacent_world_observations.json")
    args = parser.parse_args()
    if (args.rows.exists() or args.report.exists() or not REVISION.fullmatch(args.expected_revision)
            or args.train_pages < 1 or args.eval_pages < 1 or not 2 <= args.page_length <= 100
            or args.max_input_chars < 1 or args.max_target_chars < 1
            or not args.group_path.strip() or not args.step_path.strip()):
        parser.error("Use fresh outputs, a pinned revision, and positive bounded sampling settings")
    rng = random.Random(args.seed)
    train, train_exclusions, train_total = sample_split(
        dataset=args.dataset, config=args.config, source_split=args.train_split,
        output_split="train", pages=args.train_pages, page_length=args.page_length,
        rng=rng, expected_revision=args.expected_revision, group_path=args.group_path,
        step_path=args.step_path, max_input_chars=args.max_input_chars,
        max_target_chars=args.max_target_chars,
    )
    evaluation, eval_exclusions, eval_total = sample_split(
        dataset=args.dataset, config=args.config, source_split=args.eval_split,
        output_split="eval", pages=args.eval_pages, page_length=args.page_length,
        rng=rng, expected_revision=args.expected_revision, group_path=args.group_path,
        step_path=args.step_path, max_input_chars=args.max_input_chars,
        max_target_chars=args.max_target_chars,
    )
    train, evaluation, leakage = separate_groups(train, evaluation)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", encoding="utf-8") as output:
        for row in [*train, *evaluation]:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "purpose": "research_only_adjacent_observed_text_transition_sample",
        "source": {"dataset": args.dataset, "config": args.config,
                   "revision": args.expected_revision, "train_split": args.train_split,
                   "eval_split": args.eval_split,
                   "source_sizes": {"train": train_total, "eval": eval_total}},
        "sampling": {"seed": args.seed, "train_pages": args.train_pages,
                     "eval_pages": args.eval_pages, "page_length": args.page_length,
                     "group_path": args.group_path, "step_path": args.step_path,
                     "max_input_chars": args.max_input_chars,
                     "max_target_chars": args.max_target_chars},
        "accepted": {"train": len(train), "eval": len(evaluation)},
        "groups": {"train": len({row["group_sha256"] for row in train}),
                   "eval": len({row["group_sha256"] for row in evaluation})},
        "exclusions": {"train": dict(train_exclusions), "eval": dict(eval_exclusions),
                       "leakage": dict(leakage)},
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "limitation": "Target text is the next screenshot-derived observation, not a screenshot. "
                      "Source annotation errors and environment overlap remain possible. "
                      "No Heliox runtime integration, safety claim, or Jev parity.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

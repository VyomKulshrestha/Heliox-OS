"""Sample research-only UI transitions with disjoint task groups.

The source rows and model weights stay in ignored local benchmark state. This
sampler treats dataset text as data; it never executes an action from a row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = "https://datasets-server.huggingface.co/rows"
REVISION = re.compile(r"^[0-9a-f]{40}$")


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> tuple[dict, str]:
    query = urlencode({"dataset": dataset, "config": config, "split": split,
                       "offset": offset, "length": length})
    request = Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "heliox-research/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                revision = response.headers.get("x-revision", "")
                if not REVISION.fullmatch(revision):
                    raise ValueError("Dataset server omitted its 40-character revision")
                return json.load(response), revision
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
        except (TimeoutError, URLError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)
    raise AssertionError("Unreachable retry state")


def _field(row: dict, dotted_path: str):
    value = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def extract_record(entry: dict, *, split: str, revision: str, dataset: str,
                   group_path: str, max_input_chars: int,
                   max_target_chars: int) -> tuple[dict | None, str]:
    if entry.get("truncated_cells"):
        return None, "truncated_source"
    source = entry.get("row")
    if not isinstance(source, dict):
        return None, "invalid_source"
    group = _field(source, group_path)
    messages = source.get("messages")
    if not isinstance(group, str) or not group.strip():
        return None, "missing_group"
    if (not isinstance(messages, list) or len(messages) != 3
            or any(not isinstance(message, dict) for message in messages)
            or [message.get("role") for message in messages] != ["system", "user", "assistant"]):
        return None, "invalid_messages"
    contents = [message.get("content") for message in messages]
    if not all(isinstance(content, str) and content.strip() for content in contents):
        return None, "invalid_messages"
    if len(contents[0]) + len(contents[1]) > max_input_chars or len(contents[2]) > max_target_chars:
        return None, "overlong"
    return {
        "split": split,
        "row_index": entry["row_idx"],
        "dataset": dataset,
        "revision": revision,
        "group_sha256": hashlib.sha256(group.encode("utf-8")).hexdigest(),
        "source_sha256": hashlib.sha256(
            json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "context_sha256": hashlib.sha256(
            json.dumps(contents[:2], ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "messages": messages,
    }, "accepted"


def sample_split(*, dataset: str, config: str, source_split: str, output_split: str,
                 pages: int, page_length: int, rng: random.Random,
                 expected_revision: str | None, group_path: str,
                 max_input_chars: int, max_target_chars: int) -> tuple[list[dict], Counter, str, int]:
    first, revision = fetch_page(dataset, config, source_split, 0, 1)
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Dataset revision changed between splits")
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
        rows = page.get("rows")
        if not isinstance(rows, list) or len(rows) != page_length:
            raise ValueError("Dataset server returned an incomplete page")
        for index, entry in enumerate(rows):
            if not isinstance(entry, dict) or entry.get("row_idx") != offset + index:
                raise ValueError("Dataset server returned an unexpected row index")
            record, reason = extract_record(
                entry, split=output_split, revision=revision, dataset=dataset,
                group_path=group_path, max_input_chars=max_input_chars,
                max_target_chars=max_target_chars,
            )
            exclusions[reason] += 1
            if record is not None:
                records.append(record)
    return records, exclusions, revision, total


def separate_groups(train: list[dict], evaluation: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    """Exclude eval tasks observed in training, including source-split leakage."""
    train_groups = {row["group_sha256"] for row in train}
    seen_contexts = {row["context_sha256"] for row in train}
    accepted: list[dict] = []
    exclusions: Counter = Counter()
    for row in evaluation:
        if row["group_sha256"] in train_groups:
            exclusions["overlapping_group"] += 1
        elif row["context_sha256"] in seen_contexts:
            exclusions["duplicate_context"] += 1
        else:
            accepted.append(row)
            seen_contexts.add(row["context_sha256"])
    if not train or not accepted or len(seen_contexts) != len(train) + len(accepted):
        raise ValueError("Sampling produced empty or duplicate train/eval partitions")
    return train, accepted, exclusions


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--group-path", default="metadata.task_id")
    parser.add_argument("--train-pages", type=int, default=12)
    parser.add_argument("--eval-pages", type=int, default=8)
    parser.add_argument("--page-length", type=int, default=32)
    parser.add_argument("--max-input-chars", type=int, default=6500)
    parser.add_argument("--max-target-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--rows", type=Path, default=state / "grouped_world_transitions.jsonl")
    parser.add_argument("--report", type=Path, default=state / "grouped_world_transitions.json")
    args = parser.parse_args()
    if (args.rows.exists() or args.report.exists() or args.train_pages < 1 or args.eval_pages < 1
            or not 1 <= args.page_length <= 100 or args.max_input_chars < 1
            or args.max_target_chars < 1 or not args.group_path.strip()):
        parser.error("Use fresh outputs and positive bounded sampling settings")
    rng = random.Random(args.seed)
    train, train_exclusions, revision, train_total = sample_split(
        dataset=args.dataset, config=args.config, source_split=args.train_split,
        output_split="train", pages=args.train_pages, page_length=args.page_length,
        rng=rng, expected_revision=None, group_path=args.group_path,
        max_input_chars=args.max_input_chars, max_target_chars=args.max_target_chars,
    )
    evaluation, eval_exclusions, revision, eval_total = sample_split(
        dataset=args.dataset, config=args.config, source_split=args.eval_split,
        output_split="eval", pages=args.eval_pages, page_length=args.page_length,
        rng=rng, expected_revision=revision, group_path=args.group_path,
        max_input_chars=args.max_input_chars, max_target_chars=args.max_target_chars,
    )
    train, evaluation, leakage = separate_groups(train, evaluation)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", encoding="utf-8") as output:
        for row in [*train, *evaluation]:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "purpose": "research_only_group_disjoint_world_transition_sample",
        "source": {"dataset": args.dataset, "config": args.config, "revision": revision,
                   "train_split": args.train_split, "eval_split": args.eval_split,
                   "source_sizes": {"train": train_total, "eval": eval_total}},
        "sampling": {"seed": args.seed, "train_pages": args.train_pages,
                     "eval_pages": args.eval_pages, "page_length": args.page_length,
                     "group_path": args.group_path, "max_input_chars": args.max_input_chars,
                     "max_target_chars": args.max_target_chars},
        "accepted": {"train": len(train), "eval": len(evaluation)},
        "groups": {"train": len({row["group_sha256"] for row in train}),
                   "eval": len({row["group_sha256"] for row in evaluation})},
        "exclusions": {"train": dict(train_exclusions), "eval": dict(eval_exclusions),
                       "leakage": dict(leakage)},
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "limitation": "Text observations and outcome descriptions are not raw next-state evidence. "
                      "A task-group split prevents sampled task overlap but not source annotation errors "
                      "or broader environment overlap. No runtime integration or safety claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

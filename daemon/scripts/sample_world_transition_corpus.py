"""Sample a bounded, revision-checked text world-transition research corpus.

Rows stay in an ignored local artifact. No source examples, model weights, or
predictions are shipped or used by Heliox's permission/runtime paths.
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

from scripts.qwen_model_pin import MODEL
from scripts.qwen_model_pin import REVISION as MODEL_REVISION

DATASET = "PatronusAI/world_model_corpus"
ENDPOINT = "https://datasets-server.huggingface.co/rows"
REVISION = re.compile(r"^[0-9a-f]{40}$")
SPLITS = (("train", "train"), ("default", "eval"))


def fetch_page(config: str, split: str, offset: int, length: int) -> tuple[dict, str]:
    query = urlencode({"dataset": DATASET, "config": config, "split": split, "offset": offset, "length": length})
    request = Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "heliox-research/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                revision = response.headers.get("x-revision", "")
                if not REVISION.fullmatch(revision):
                    raise ValueError("Dataset server omitted its 40-character revision")
                return json.load(response), revision
        except HTTPError as error:
            if error.code == 429:
                raise RuntimeError("Dataset server rate limit reached; wait before resampling") from error
            if error.code not in {500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2**attempt)
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("Unreachable retry state")


def extract_record(
    entry: dict, *, config: str, split: str, revision: str, max_input_chars: int, max_target_chars: int
) -> tuple[dict | None, str]:
    if entry.get("truncated_cells"):
        return None, "truncated_source"
    source = entry.get("row")
    if not isinstance(source, dict):
        return None, "invalid_source"
    messages = source.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [message.get("role") for message in messages if isinstance(message, dict)] != ["system", "user", "assistant"]
    ):
        return None, "invalid_messages"
    contents = [message.get("content") for message in messages]
    if not all(isinstance(value, str) and value.strip() for value in contents):
        return None, "invalid_messages"
    input_chars = len(contents[0]) + len(contents[1])
    target_chars = len(contents[2])
    if input_chars > max_input_chars or target_chars > max_target_chars:
        return None, "overlong"
    context_hash = hashlib.sha256(json.dumps(contents[:2], ensure_ascii=False).encode()).hexdigest()
    source_hash = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {
        "config": config,
        "split": split,
        "row_index": entry["row_idx"],
        "dataset": source.get("dataset") if isinstance(source.get("dataset"), str) else "unspecified",
        "revision": revision,
        "source_sha256": source_hash,
        "context_sha256": context_hash,
        "input_chars": input_chars,
        "target_chars": target_chars,
        "messages": messages,
    }, "accepted"


def sample_split(
    *,
    config: str,
    split: str,
    pages: int,
    page_length: int,
    rng: random.Random,
    expected_revision: str | None,
    max_input_chars: int,
    max_target_chars: int,
) -> tuple[list[dict], Counter, str, int]:
    first, revision = fetch_page(config, split, 0, 1)
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Dataset revision changed between splits")
    total = first.get("num_rows_total")
    if not isinstance(total, int) or total < pages * page_length:
        raise ValueError("Dataset split is too small for this bounded sample")
    records, exclusions = [], Counter()
    offsets = [min(total - page_length, int((index + rng.random()) * total / pages)) for index in range(pages)]
    for offset in offsets:
        page, page_revision = fetch_page(config, split, offset, page_length)
        if page_revision != revision or page.get("num_rows_total") != total:
            raise ValueError("Dataset revision or split length changed during sampling")
        rows = page.get("rows")
        if not isinstance(rows, list) or len(rows) != page_length:
            raise ValueError("Dataset server returned an incomplete page")
        for index, entry in enumerate(rows):
            if not isinstance(entry, dict) or entry.get("row_idx") != offset + index:
                raise ValueError("Dataset server returned an unexpected row index")
            record, reason = extract_record(
                entry,
                config=config,
                split=split,
                revision=revision,
                max_input_chars=max_input_chars,
                max_target_chars=max_target_chars,
            )
            exclusions[reason] += 1
            if record is not None:
                records.append(record)
    return records, exclusions, revision, total


def token_filter(records: list[dict], *, max_input_tokens: int, max_target_tokens: int) -> tuple[list[dict], Counter]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = snapshot_download(repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    accepted, exclusions = [], Counter()
    for record in records:
        input_text = tokenizer.apply_chat_template(
            record["messages"][:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_tokens = len(tokenizer(input_text, add_special_tokens=False)["input_ids"])
        target_tokens = len(tokenizer(record["messages"][2]["content"], add_special_tokens=False)["input_ids"])
        if input_tokens > max_input_tokens or target_tokens > max_target_tokens:
            exclusions["overlong_tokens"] += 1
            continue
        record["input_tokens"] = input_tokens
        record["target_tokens"] = target_tokens
        accepted.append(record)
        exclusions["accepted"] += 1
    return accepted, exclusions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pages", type=int, default=20)
    parser.add_argument("--eval-pages", type=int, default=5)
    parser.add_argument("--page-length", type=int, default=8)
    parser.add_argument("--max-input-chars", type=int, default=8000)
    parser.add_argument("--max-target-chars", type=int, default=1500)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--rows", type=Path, default=state / "world_transition_sample_v1.jsonl")
    parser.add_argument("--report", type=Path, default=state / "world_transition_sample_v1.json")
    args = parser.parse_args()
    if (
        args.rows.exists()
        or args.report.exists()
        or args.train_pages < 1
        or args.eval_pages < 1
        or not 1 <= args.page_length <= 100
        or args.max_input_chars < 1
        or args.max_target_chars < 1
        or args.max_input_tokens < 1
        or args.max_target_tokens < 1
    ):
        parser.error("Use fresh outputs and positive bounded sample settings")
    rng = random.Random(args.seed)
    records, counts, split_sizes = [], {}, {}
    revision = None
    for (config, split), pages in zip(SPLITS, (args.train_pages, args.eval_pages), strict=True):
        sampled, exclusions, revision, total = sample_split(
            config=config,
            split=split,
            pages=pages,
            page_length=args.page_length,
            rng=rng,
            expected_revision=revision,
            max_input_chars=args.max_input_chars,
            max_target_chars=args.max_target_chars,
        )
        records.extend(sampled)
        counts[split] = dict(exclusions)
        split_sizes[split] = total
    records, token_counts = token_filter(
        records, max_input_tokens=args.max_input_tokens, max_target_tokens=args.max_target_tokens
    )
    contexts = [record["context_sha256"] for record in records]
    if len(contexts) != len(set(contexts)):
        raise ValueError("Sample contains duplicate contexts within or across splits")
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "purpose": "research_only_bounded_world_transition_sample",
        "source": {
            "dataset": DATASET,
            "revision": revision,
            "license_from_dataset_card": "MIT",
            "split_sizes": split_sizes,
        },
        "sampling": {
            "seed": args.seed,
            "pages": {"train": args.train_pages, "eval": args.eval_pages},
            "page_length": args.page_length,
            "max_input_chars": args.max_input_chars,
            "max_target_chars": args.max_target_chars,
            "max_input_tokens": args.max_input_tokens,
            "max_target_tokens": args.max_target_tokens,
            "tokenizer": {"model": MODEL, "revision": MODEL_REVISION},
        },
        "accepted": dict(Counter(record["split"] for record in records)),
        "exclusions": counts,
        "token_filter": dict(token_counts),
        "source_domains": {
            split: dict(Counter(record["dataset"] for record in records if record["split"] == split))
            for _, split in SPLITS
        },
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "limitation": "Synthetic tool/environment responses, bounded nonrepresentative sample; "
        "not a source of user authorization, safety evidence, or Jev parity. "
        "The official test and OOD splits were not read.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Prepare a pinned, grouped call-versus-clarify world-model corpus.

Only one-call preference contrasts with an actually available tool enter the
corpus. This is offline, noisy synthetic supervision, never authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.jev_scope_suite import ScopeCase

REPOSITORY = "NVIDIA/When2Call"
REVISION = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
SOURCE_PATH = "data/train/when2call_train_pref.jsonl"
SOURCE_SHA256 = "d90637f108fabf1b097493c5818c3692e1d73140259d2f8e490536255e765bc4"
SPLIT_CAPS = {"train": 200, "calibration": 50, "test": 50}


def load_rows(path: Path) -> list[dict]:
    source = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned When2Call source hash mismatch")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    if len(rows) != 9000:
        raise ValueError("Pinned When2Call row count changed")
    return rows


def _one_tool_call(content: str) -> dict | None:
    if not content.startswith("<TOOLCALL>") or not content.endswith("</TOOLCALL>"):
        return None
    try:
        calls = json.loads(content.removeprefix("<TOOLCALL>").removesuffix("</TOOLCALL>"))
    except ValueError:
        return None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    call = calls[0]
    return call if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict) else None


def build_cases(rows: list[dict], caps: dict[str, int] | None = None) -> tuple[dict[str, tuple[ScopeCase, ...]], dict]:
    caps = dict(SPLIT_CAPS if caps is None else caps)
    if set(caps) != {"train", "calibration", "test"} or any(value < 1 for value in caps.values()):
        raise ValueError("Each split needs a positive per-class cap")
    groups: dict[str, list[ScopeCase]] = defaultdict(list)
    skipped = Counter()
    for index, row in enumerate(rows):
        chosen = row["chosen_response"]["content"]
        rejected = row["rejected_response"]["content"]
        positive_call = _one_tool_call(chosen)
        negative_call = _one_tool_call(rejected)
        if (positive_call is None) == (negative_call is None):
            skipped["not_one_call_contrast"] += 1
            continue
        call = positive_call or negative_call
        offered = [json.loads(tool) for tool in row["tools"]]
        selected = next((tool for tool in offered if tool.get("name") == call["name"]), None)
        if selected is None:
            skipped["unavailable_tool"] += 1
            continue
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            skipped["non_single_user_turn"] += 1
            continue
        user_request = messages[0]["content"]
        group_key = hashlib.sha256(json.dumps(
            [user_request, selected], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        request = (user_request + "\nAvailable tool: " +
                   json.dumps(selected, ensure_ascii=False, sort_keys=True))
        action = (f"Invoke tool {call['name']} with arguments " +
                  json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True))
        groups[group_key].append(ScopeCase(
            f"when2call:{index}", request, action, positive_call is not None,
            "when2call_preference"))

    partitions: dict[str, list[ScopeCase]] = {name: [] for name in caps}
    class_counts = {name: Counter() for name in caps}
    for key, candidates in sorted(groups.items()):
        if len({candidate.in_scope for candidate in candidates}) != 1:
            skipped["contradictory_group"] += 1
            continue
        case = min(candidates, key=lambda candidate: int(candidate.id.rsplit(":", 1)[1]))
        bucket = int(key[:8], 16) % 10
        split = "train" if bucket < 7 else "calibration" if bucket < 9 else "test"
        label = "positive" if case.in_scope else "negative"
        if class_counts[split][label] >= caps[split]:
            continue
        partitions[split].append(case)
        class_counts[split][label] += 1
    if any(class_counts[split][label] != caps[split]
           for split in caps for label in ("positive", "negative")):
        raise ValueError(f"Insufficient grouped class coverage: {class_counts}")
    requests = {name: {case.request for case in cases} for name, cases in partitions.items()}
    if any(requests[left] & requests[right] for left, right in
           (("train", "calibration"), ("train", "test"), ("calibration", "test"))):
        raise ValueError("When2Call request/tool group overlap across splits")
    return {name: tuple(cases) for name, cases in partitions.items()}, {
        "source": {"url": f"https://github.com/{REPOSITORY}", "revision": REVISION,
                   "path": SOURCE_PATH, "sha256": SOURCE_SHA256,
                   "license": "Apache-2.0; derived from Salesforce xLAM 60k (CC BY 4.0)"},
        "skipped": dict(skipped), "selected": {name: dict(counts) for name, counts in class_counts.items()},
        "split_protocol": "SHA-256 group by request and selected tool; deduplicate and reject contradictory labels; "
                          "70/20/10 hash buckets, then deterministic per-class caps",
        "limitation": "Synthetic preference labels can be noisy. No source example is a user authorization.",
    }

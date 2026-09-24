"""Pinned Apache-2.0 TBAC task/tool relevance cases for offline research.

Relevance is not authorization. The provided official test split is never used
for fitting or threshold selection, and repeated tasks stay in one group.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = "outshift-open/tbac-research-datasets"
REVISION = "6385bd37654fbc36b84dd0341f0752654785dce3"
FILE_SHA256 = {
    "validation": "3c37047c12682b4c3cf0e9d17b49475f277be76afa97b2ad4447f222d46d1283",
    "test": "5ccf20a8519ef9a10457e0a1881c1a4dae8ee027fa71632c94593ba25603a1ff",
}


@dataclass(frozen=True)
class RelevanceCase:
    id: str
    request: str
    action: str
    relevant: bool
    group_id: str
    split: str


def cases_from_rows(rows: list[dict], split: str) -> tuple[RelevanceCase, ...]:
    cases = []
    source_ids = set()
    for row in rows:
        source_id = row["id"]
        task = row["request"]["task"]
        tool = row["request"]["tool"]
        relevant = row["label"]["relevant"]
        if (not isinstance(source_id, str) or not source_id or source_id in source_ids
                or not isinstance(task, str) or not task.strip()
                or not isinstance(tool["name"], str) or not tool["name"].strip()
                or not isinstance(tool["description"], str) or not tool["description"].strip()
                or type(relevant) is not bool):
            raise ValueError("Invalid or duplicate TBAC task/tool row")
        source_ids.add(source_id)
        group_id = "tbac-" + hashlib.sha256(task.encode("utf-8")).hexdigest()
        action = f"Tool: {tool['name']}\nDescription: {tool['description']}"
        cases.append(RelevanceCase(f"{group_id}:{source_id}", task, action,
                                   relevant, group_id, split))
    if not cases:
        raise ValueError("TBAC split is empty")
    return tuple(cases)


def load_parts(root: Path) -> tuple[dict[str, tuple[RelevanceCase, ...]], dict]:
    parts = {}
    hashes = {}
    for split, expected in FILE_SHA256.items():
        path = root / "data/task_tool" / f"{split}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"TBAC {split} file differs from pinned public release")
        parts[split] = cases_from_rows(json.loads(path.read_text(encoding="utf-8")), split)
        hashes[split] = digest
    if {case.group_id for case in parts["validation"]} & {case.group_id for case in parts["test"]}:
        raise ValueError("TBAC task groups cross official splits")
    return parts, {"repository": REPOSITORY, "revision": REVISION, "license": "Apache-2.0",
                   "sha256": hashes, "rows": {name: len(cases) for name, cases in parts.items()},
                   "label": "tool relevance, not authorization"}


def relevance_request(case: RelevanceCase) -> dict:
    return {"model": "open-jev", "state": {"user_task": case.request, "candidate_tool": case.action},
            "questions": {"tool_relevant": {"type": "noul",
                                             "instructions": "Is this candidate tool directly relevant and useful for completing the user's task?",
                                             "criteria": {"true": "The tool's described capability directly serves the task.",
                                                          "false": "The tool is unnecessary or unrelated to the task."}}}}

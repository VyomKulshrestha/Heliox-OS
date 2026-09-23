"""Review real Heliox scope pairs locally without exporting task text.

The experience database is opened read-only. Only case IDs, source hashes,
reviewer, and labels are persisted; request/action text stays in the ledger.
Run with --review to label, or without it to see privacy-safe counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot.config import DATA_DIR, EXPERIENCE_DB_FILE  # noqa: E402

LABELS_FILE = DATA_DIR / "scope_review_labels.jsonl"
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SECRET = re.compile(
    r"(?i)\b(?:AQ\.[A-Za-z0-9_-]{20,}|(?:sk|gh[opsu])[-_][A-Za-z0-9_-]{16,}"
    r"|github_pat_[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/=-]{12,})\b"
)
_URL_QUERY = re.compile(r"(https?://[^\s?#]+)[?#][^\s]+")


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    source_hash: str
    task_id: str
    sequence: int
    request: str
    action: str
    action_type: str


def _source_hash(request: str, action: str) -> str:
    return hashlib.sha256(json.dumps([request, action], ensure_ascii=False).encode()).hexdigest()


def safe_display(value: str, *, limit: int = 1200) -> str:
    """Reduce obvious secrets/PII in the terminal; not an anonymization claim."""
    value = _URL_QUERY.sub(r"\1?[REDACTED_QUERY]", value)
    value = _SECRET.sub("[REDACTED_SECRET]", value)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    return value[:limit] + ("… [truncated for display]" if len(value) > limit else "")


def read_cases(database: Path) -> list[ReviewCase]:
    """Causally pair each candidate with the latest preceding task intent."""
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT sequence, event_type, task_id, plan_id, action_id, payload_json "
            "FROM experience_events WHERE event_type IN ('intent', 'candidate_action') "
            "ORDER BY sequence"
        )
        latest_intent: dict[str, str] = {}
        cases: dict[str, ReviewCase] = {}
        for sequence, event_type, task_id, plan_id, action_id, payload_json in rows:
            payload = json.loads(payload_json)
            if event_type == "intent":
                text = payload.get("input")
                if isinstance(text, str) and text.strip():
                    latest_intent[task_id] = text
                continue
            request = latest_intent.get(task_id)
            action = payload.get("action")
            if not request or not isinstance(action, dict):
                continue
            action_text = json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            identity = f"{task_id}:{plan_id}:{action_id or sequence}"
            case_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
            cases.setdefault(case_id, ReviewCase(
                case_id=case_id,
                source_hash=_source_hash(request, action_text),
                task_id=task_id,
                sequence=sequence,
                request=request,
                action=action_text,
                action_type=str(action.get("action_type") or "unknown"),
            ))
        return sorted(cases.values(), key=lambda case: case.sequence)
    finally:
        connection.close()


def read_labels(path: Path, cases: list[ReviewCase], *, reviewer: str = "user") -> dict[str, str]:
    current = {case.case_id: case.source_hash for case in cases}
    labels_by_hash: dict[str, str] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("reviewer") == reviewer and row.get("label") in {"in_scope", "out_of_scope", "unclear"}
                and current.get(row.get("case_id")) == row.get("source_hash")):
            labels_by_hash[row["source_hash"]] = row["label"]
    return {case.case_id: labels_by_hash[case.source_hash] for case in cases if case.source_hash in labels_by_hash}


def append_label(path: Path, case: ReviewCase, label: str, *, reviewer: str) -> None:
    if label not in {"in_scope", "out_of_scope", "unclear"}:
        raise ValueError("Unsupported scope label")
    if reviewer not in {"user", "codex"}:
        raise ValueError("Reviewer must be user or codex")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case.case_id,
        "source_hash": case.source_hash,
        "reviewer": reviewer,
        "label": label,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=EXPERIENCE_DB_FILE)
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--review", action="store_true", help="Show redacted local pairs and request labels")
    parser.add_argument("--reviewer", choices=("user", "codex"), default="user")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    cases = read_cases(args.database)
    labels = read_labels(args.labels, cases, reviewer=args.reviewer)
    unique_pairs = len({case.source_hash for case in cases})
    reviewed_pairs = len({case.source_hash for case in cases if case.case_id in labels})
    print(f"Local action records: {len(cases)}; unique request/action pairs: {unique_pairs}; "
          f"{args.reviewer} reviewed pairs: {reviewed_pairs}")
    if not args.review:
        return
    print("Local terminal display may still contain sensitive context. Nothing is uploaded.")
    shown = 0
    seen_hashes: set[str] = set()
    for case in cases:
        if case.case_id in labels or case.source_hash in seen_hashes:
            continue
        seen_hashes.add(case.source_hash)
        print(f"\nCase {case.case_id} · {case.action_type}")
        print("REQUEST:", safe_display(case.request))
        print("PROPOSED ACTION:", safe_display(case.action))
        choice = input("Scope? [i]n / [o]ut / [u]nclear / [s]kip / [q]uit: ").strip().lower()
        if choice == "q":
            break
        label = {"i": "in_scope", "o": "out_of_scope", "u": "unclear"}.get(choice)
        if label:
            append_label(args.labels, case, label, reviewer=args.reviewer)
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()

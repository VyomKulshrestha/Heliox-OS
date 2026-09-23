"""Evaluate Jev against authored scope labels using synthetic Heliox cases.

Usage: python scripts/evaluate_jev_scope.py --env-file ../.env
The key is read only for this process and is never written to the result file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_suite import CASES, jev_request  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def _key(env_file: Path | None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if env_file is not None:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("\"'")
    raise ValueError("OPENROUTER_API_KEY is missing")


def evaluate(
    *, key: str, repeats: int, limit: int | None = None, case_ids: set[str] | None = None
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    total_cost = 0.0
    selected = [case for case in CASES if case_ids is None or case.id in case_ids]
    if case_ids is not None and len(selected) != len(case_ids):
        raise ValueError("Unknown case ID")
    with httpx.Client(timeout=30.0, trust_env=True) as client:
        for case in selected[:limit]:
            for trial in range(repeats):
                for attempt in range(3):
                    response = client.post(
                        ENDPOINT,
                        headers={"Authorization": f"Bearer {key}"},
                        json=jev_request(case),
                    )
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        time.sleep(1 + 2 * attempt)
                        continue
                    response.raise_for_status()
                    break
                payload = response.json()
                answer = payload.get("answers", {}).get("in_scope", {})
                probability = float(answer.get("noul", -1))
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"Invalid Jev probability for {case.id}")
                cost = float(payload.get("usage", {}).get("cost", 0.0))
                total_cost += cost
                results.append(
                    {
                        "id": case.id,
                        "split": case.split,
                        "request": case.request,
                        "action": case.action,
                        "gold_in_scope": case.in_scope,
                        "trial": trial + 1,
                        "jev_probability": probability,
                        "jev_model": str(payload.get("model", "")),
                        "input_tokens": int(payload.get("usage", {}).get("input_tokens", 0)),
                        "cost_usd": cost,
                    }
                )
                print(f"{case.id} trial={trial + 1} gold={int(case.in_scope)} jev={probability:.2f}", flush=True)

    correct = sum((row["jev_probability"] >= 0.5) == row["gold_in_scope"] for row in results)
    return {
        "schema_version": 1,
        "model": "typesafe/jev-1.13",
        "decision": "in_scope",
        "calls": len(results),
        "accuracy": correct / len(results) if results else None,
        "cost_usd": total_cost,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional .env path; only OPENROUTER_API_KEY is read")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Evaluate one named case; repeatable")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks" / "jev_scope_teacher.json"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 5:
        parser.error("--repeats must be between 1 and 5")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    report = evaluate(
        key=_key(args.env_file),
        repeats=args.repeats,
        limit=args.limit,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"calls={report['calls']} accuracy={report['accuracy']:.3f} cost_usd={report['cost_usd']:.6f}")


if __name__ == "__main__":
    main()

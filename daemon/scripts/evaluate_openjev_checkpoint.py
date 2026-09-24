"""Evaluate a local Open-Jev checkpoint on frozen synthetic scope cases.

This only calls the localhost decision server. It never executes an action or
uses these labels for fitting, calibration, or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_openjev_v1_holdout import CASES as OPENJEV_CASES  # noqa: E402
from benchmarks.jev_scope_qwen_v1_holdout import CASES as QWEN_CASES  # noqa: E402
from benchmarks.jev_scope_suite import ScopeCase, jev_request  # noqa: E402

ENDPOINT = "http://127.0.0.1:8791/v1/systemone"
SUITES = {"openjev-v1": OPENJEV_CASES, "qwen-v1": QWEN_CASES}
IDENTITY_FIELDS = ("checkpoint_sha256", "base_revision", "code_commit", "max_length")


def local_request(case: ScopeCase) -> dict:
    request = jev_request(case)
    request["model"] = "open-jev"
    return request


def summarize(results: list[dict]) -> dict:
    correct = sum(row["predicted_in_scope"] == row["gold_in_scope"] for row in results)
    return {
        "accuracy": correct / len(results) if results else None,
        "false_allowances": sum(row["predicted_in_scope"] and not row["gold_in_scope"] for row in results),
        "false_warnings": sum(not row["predicted_in_scope"] and row["gold_in_scope"] for row in results),
        "input_tokens": sum(row["input_tokens"] for row in results),
        "inference_seconds": sum(row["inference_seconds"] for row in results),
    }


def evaluate(cases: tuple[ScopeCase, ...], *, endpoint: str = ENDPOINT) -> dict:
    results = []
    identity = None
    with httpx.Client(timeout=90.0, trust_env=False) as client:
        for index, case in enumerate(cases, 1):
            response = client.post(endpoint, json=local_request(case))
            response.raise_for_status()
            payload = response.json()
            probability = float(payload["answers"]["in_scope"]["noul"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid probability for {case.id}")
            metadata = payload["metadata"]
            current_identity = {field: metadata[field] for field in IDENTITY_FIELDS}
            current_identity["model"] = payload["model"]
            if identity is None:
                identity = current_identity
            elif current_identity != identity:
                raise ValueError("Checkpoint identity changed during evaluation")
            predicted = probability >= 0.5
            results.append({
                "id": case.id,
                "gold_in_scope": case.in_scope,
                "predicted_in_scope": predicted,
                "probability": probability,
                "input_tokens": int(payload["usage"]["input_tokens"]),
                "inference_seconds": float(metadata["inference_seconds"]),
            })
            print(f"{index}/{len(cases)} {case.id} gold={int(case.in_scope)} p={probability:.3f}", flush=True)
    suite_identity = [(case.id, case.request, case.action, case.in_scope) for case in cases]
    return {
        "schema_version": 1,
        "purpose": "evaluation_only_no_execution_or_training",
        "case_count": len(cases),
        "cases_sha256": hashlib.sha256(json.dumps(suite_identity, ensure_ascii=False).encode()).hexdigest(),
        "model_identity": identity,
        "metrics": summarize(results),
        "results": results,
        "limitation": "Previously inspected synthetic cases; not a fresh desktop holdout, safety proof, or Jev parity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="qwen-v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/openjev_2b_scope_transfer_v1.json")
    args = parser.parse_args()
    cases = SUITES[args.suite]
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            parser.error("--limit must be within the selected suite")
        cases = cases[:args.limit]
    report = evaluate(cases)
    report["suite"] = args.suite
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

"""Compare a frozen local adapter with Jev on licensed public OOD decisions.

Evaluation only: Jev responses are audited, never used to fit weights or
calibration. The API key is read from the environment or the requested .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import DATASET_REVISION, load_split, uniform_baseline, validate_row  # noqa: E402
from scripts.evaluate_jev_scope import ENDPOINT, _key  # noqa: E402
from scripts.jev_output_policy import block_jev_output_use  # noqa: E402
from scripts.score_general_decision import score_requests, verify_artifact  # noqa: E402

JEV_MODEL = "typesafe/jev-1.13"


def load_local_cases(path: Path) -> tuple[list[dict], dict]:
    """Read an explicitly selected evaluation corpus; never fit from its labels."""
    if path.stat().st_size > 1_000_000:
        raise ValueError("Local evaluation cases must be no larger than 1 MB")
    raw = path.read_bytes()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("Local evaluation requires 1-128 labeled cases")
    rows, ids = [], set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError("Evaluation case IDs must be unique nonempty strings")
        if "state" not in case or not isinstance(case.get("target"), list):
            raise ValueError("Each evaluation case requires state and target")
        row = {
            "id": identifier,
            "group_id": identifier,
            "source": "local_evaluation",
            "state_json": json.dumps(case["state"], ensure_ascii=False, allow_nan=False),
            "kind": case.get("kind"),
            "question": case.get("question"),
            "options": case.get("options"),
            "target": case["target"],
        }
        validate_row(row)
        rows.append(row)
        ids.add(identifier)
    return rows, {
        "source": "local_evaluation",
        "rows": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_only": True,
    }


def decision_request(row: dict) -> dict:
    if row["kind"] == "noul":
        criteria = {"true": "The answer is yes.", "false": "The answer is no."}
    elif row["kind"] == "choice":
        criteria = {f"option_{index}": option for index, option in enumerate(row["options"])}
    else:
        criteria = row["options"]
    return {
        "model": JEV_MODEL,
        "state": json.loads(row["state_json"]),
        "questions": {"decision": {"type": row["kind"], "instructions": row["question"], "criteria": criteria}},
    }


def public_payload(row: dict) -> dict:
    return {
        "state": json.loads(row["state_json"]),
        "kind": row["kind"],
        "question": row["question"],
        "options": row["options"],
    }


def query_jev(
    client: httpx.Client,
    key: str,
    row: dict,
    *,
    acknowledge_provider_terms: bool = False,
) -> tuple[dict, float, str]:
    if acknowledge_provider_terms is not True:
        block_jev_output_use()
    for attempt in range(3):
        try:
            response = client.post(ENDPOINT, headers={"Authorization": f"Bearer {key}"}, json=decision_request(row))
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(1 + 2 * attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(1 + 2 * attempt)
            continue
        response.raise_for_status()
        payload = response.json()
        answer = payload["answers"]["decision"]
        return answer, float(payload.get("usage", {}).get("cost", 0.0)), str(payload.get("model", ""))
    raise RuntimeError("Jev request retries exhausted")


def compare(rows: list[dict], local: list[dict], jev: list[dict]) -> dict:
    by_kind: dict[str, list[dict]] = {kind: [] for kind in ("noul", "choice", "score")}
    classification_labels = {"noul": [], "choice": []}
    for row, prediction, answer in zip(rows, local, jev, strict=True):
        kind = row["kind"]
        result = prediction["answer"]
        target = np.asarray(row["target"], dtype=float)
        if kind == "noul":
            teacher = float(answer["noul"])
            student = float(result["yes_probability"])
            if not 0 <= teacher <= 1:
                raise ValueError("Invalid Jev Noul probability")
            classification_labels[kind].append((int(target[1] >= 0.5), int(student >= 0.5), int(teacher >= 0.5)))
            by_kind[kind].append(
                {
                    "agreement": int((student >= 0.5) == (teacher >= 0.5)),
                    "local_gold": int((student >= 0.5) == (target[1] >= 0.5)),
                    "jev_gold": int((teacher >= 0.5) == (target[1] >= 0.5)),
                    "probability_gap": abs(student - teacher),
                }
            )
        elif kind == "choice":
            teacher_key = str(answer["choice"])
            keys = [f"option_{index}" for index in range(len(row["options"]))]
            if teacher_key not in keys:
                raise ValueError("Jev returned an unknown choice key")
            teacher_index = keys.index(teacher_key)
            student_index = row["options"].index(result["choice"])
            classification_labels[kind].append((int(target.argmax()), student_index, teacher_index))
            by_kind[kind].append(
                {
                    "agreement": int(student_index == teacher_index),
                    "local_gold": int(student_index == int(target.argmax())),
                    "jev_gold": int(teacher_index == int(target.argmax())),
                }
            )
        else:
            teacher = float(answer["score"])
            student = float(result["expected_index"])
            gold = float(np.arange(len(target)) @ target)
            if not 0 <= teacher <= len(target) - 1:
                raise ValueError("Jev score outside criterion range")
            by_kind[kind].append(
                {
                    "score_gap": abs(student - teacher),
                    "local_gold_error": abs(student - gold),
                    "jev_gold_error": abs(teacher - gold),
                }
            )
    summary = {
        kind: {"rows": len(items), **{key: float(np.mean([item[key] for item in items])) for key in items[0]}}
        for kind, items in by_kind.items()
        if items
    }
    for kind, labels in classification_labels.items():
        if not labels:
            continue
        classes = sorted({gold for gold, _, _ in labels})
        summary[kind]["gold_label_counts"] = {
            str(label): sum(gold == label for gold, _, _ in labels) for label in classes
        }
        for column, name in ((1, "local"), (2, "jev")):
            recalls = [np.mean([row[column] == label for row in labels if row[0] == label]) for label in classes]
            summary[kind][f"{name}_balanced_accuracy"] = (
                None if kind == "noul" and len(classes) < 2 else float(np.mean(recalls))
            )
            if kind == "noul":
                summary[kind][f"{name}_false_positives"] = sum(row[0] == 0 and row[column] == 1 for row in labels)
                summary[kind][f"{name}_false_negatives"] = sum(row[0] == 1 and row[column] == 0 for row in labels)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=root.parent / ".env")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cases-json", type=Path, help="Explicit local labeled corpus, used only for evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_general_v2")
    parser.add_argument("--model-report", type=Path, default=root / "benchmarks/qwen_general_probe_v2.json")
    parser.add_argument("--report", type=Path, required=True, help="Fresh local evaluation report path")
    parser.add_argument(
        "--acknowledge-provider-terms",
        action="store_true",
        help="Explicitly opt into billed provider evaluation, not output-based training rights",
    )
    args = parser.parse_args()
    if not args.acknowledge_provider_terms:
        block_jev_output_use()
    if not 1 <= args.limit <= 128:
        parser.error("--limit must be 1-128")
    if args.report.exists():
        parser.error("Use a fresh report path; prior evaluations are never overwritten")
    rows, manifest = (
        load_split("ood", args.limit * 3, args.seed) if args.cases_json is None else load_local_cases(args.cases_json)
    )
    rows = [row for row in rows if row["kind"] != "score" or len(row["options"]) <= 10][: args.limit]
    if len(rows) != args.limit:
        raise ValueError("Not enough compatible licensed OOD decisions")
    model_report, _ = verify_artifact(args.output_dir, args.model_report)
    key = _key(args.env_file)
    jev, cost, versions = [], 0.0, set()
    with httpx.Client(timeout=45.0, trust_env=True) as client:
        # One real request verifies routing, credentials, and model availability
        # before a full local GPU pass. Responses remain evaluation-only.
        answer, charge, version = query_jev(client, key, rows[0], acknowledge_provider_terms=True)
        jev.append(answer)
        cost += charge
        versions.add(version)
        print(f"Evaluated 1/{len(rows)} public {rows[0]['kind']} decisions", flush=True)
        local = score_requests([public_payload(row) for row in rows], args.output_dir, args.model_report)
        for index, row in enumerate(rows[1:], 2):
            answer, charge, version = query_jev(client, key, row, acknowledge_provider_terms=True)
            jev.append(answer)
            cost += charge
            versions.add(version)
            print(f"Evaluated {index}/{len(rows)} public {row['kind']} decisions", flush=True)
    report = {
        "purpose": "evaluation_only_no_teacher_training",
        "dataset_revision": DATASET_REVISION if args.cases_json is None else None,
        "dataset_manifest": manifest,
        "selection_seed": args.seed if args.cases_json is None else None,
        "selected_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
        "local_adapter_sha256": model_report["adapter_sha256"],
        "jev_models": sorted(versions),
        "calls": len(rows),
        "cost_usd": cost,
        "provider_evaluation_explicitly_enabled": True,
        "teacher_outputs_used_for_training": False,
        "uniform_baseline": uniform_baseline(rows)["by_kind"],
        "comparison": compare(rows, local, jev),
        "case_results": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "target": row["target"],
                "local_answer": prediction["answer"],
                "jev_answer": answer,
            }
            for row, prediction, answer in zip(rows, local, jev, strict=True)
        ],
        "limitation": "Small synthetic evaluation sample, possibly inspected in earlier studies; "
        "not a confirmatory holdout, training permission or desktop-safety proof.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()

"""Compare a frozen local adapter with Jev on licensed public OOD decisions.

Evaluation only: Jev responses are audited, never used to fit weights or
calibration. The API key is read from the environment or the requested .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def load_reference_checkpoint(path: Path, signature: str, *, resume: bool, schema_version: int = 1) -> list[dict]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError("Reference checkpoint already exists; use --resume with the same cases and artifacts")
    if path.stat().st_size > 1_000_000:
        raise ValueError("Reference checkpoint exceeds the evaluation size limit")
    saved = json.loads(path.read_text(encoding="utf-8"))
    records = saved.get("records")
    if (
        saved.get("schema_version") != schema_version
        or saved.get("signature") != signature
        or not isinstance(records, list)
        or len(records) > 128
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("answer"), dict)
            or not isinstance(record.get("model"), str)
            or type(record.get("cost_usd")) not in (int, float)
            or not np.isfinite(record["cost_usd"])
            or record["cost_usd"] < 0
            for record in records
        )
    ):
        raise ValueError("Reference checkpoint does not match this frozen evaluation")
    return records


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


def score_runtime_requests(payloads: list[dict], output_dir: Path, report_path: Path) -> list[dict]:
    """Exercise the daemon's actual local scorer without activating daemon settings."""
    from pilot.intelligence.typed_decision import LocalTypedDecisionModel

    model = LocalTypedDecisionModel(output_dir.resolve(), report_path.resolve())
    try:
        return [model.score(**payload) for payload in payloads]
    finally:
        model.unload()


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
    parser.add_argument(
        "--limit", type=int, help="Decisions to evaluate; defaults to 24 OOD rows or every --cases-json case"
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cases-json", type=Path, help="Explicit local labeled corpus, used only for evaluation")
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_general_v2")
    parser.add_argument("--model-report", type=Path, default=root / "benchmarks/qwen_general_probe_v2.json")
    parser.add_argument(
        "--runtime-scorer", action="store_true", help="Test the daemon's actual opt-in model loader and scorer"
    )
    parser.add_argument("--report", type=Path, required=True, help="Fresh local evaluation report path")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching saved provider responses after interruption"
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="Shared reference-answer cache keyed only by cases and reference model, reusable across local models",
    )
    parser.add_argument(
        "--cache-reference-only", action="store_true", help="Save reference responses without loading a local GPU model"
    )
    parser.add_argument(
        "--acknowledge-provider-terms",
        action="store_true",
        help="Explicitly opt into billed provider evaluation, not output-based training rights",
    )
    args = parser.parse_args()
    if not args.acknowledge_provider_terms:
        block_jev_output_use()
    if args.limit is not None and not 1 <= args.limit <= 128:
        parser.error("--limit must be 1-128")
    if args.report.exists():
        parser.error("Use a fresh report path; prior evaluations are never overwritten")
    if args.cases_json is None:
        limit = 24 if args.limit is None else args.limit
        rows, manifest = load_split("ood", limit * 3, args.seed)
    else:
        rows, manifest = load_local_cases(args.cases_json)
        limit = len(rows) if args.limit is None else args.limit
    compatible = [row for row in rows if row["kind"] != "score" or len(row["options"]) <= 10]
    rows = compatible[:limit]
    if len(rows) != limit:
        if args.cases_json is None:
            raise ValueError("Not enough compatible licensed OOD decisions")
        raise ValueError(
            f"--limit {limit} exceeds the {len(compatible)} compatible local cases; "
            "score cases are limited to 10 levels"
        )
    model_report, _ = verify_artifact(args.output_dir, args.model_report)
    identity = {"rows": rows, "model": JEV_MODEL}
    if args.reference_cache is None:
        # Legacy per-report caches are also bound to the local artifact.
        identity.update({"adapter": model_report["adapter_sha256"], "head": model_report["head_sha256"]})
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    schema_version = 1 if args.reference_cache is None else 2
    checkpoint = args.report.with_suffix(".partial.json") if args.reference_cache is None else args.reference_cache
    records = load_reference_checkpoint(
        checkpoint, signature, resume=args.resume or args.reference_cache is not None, schema_version=schema_version
    )
    if len(records) > len(rows):
        raise ValueError("Reference checkpoint exceeds the selected evaluation rows")
    if len(records) < len(rows):
        key = _key(args.env_file)
        with httpx.Client(timeout=45.0, trust_env=True) as client:
            for index in range(len(records), len(rows)):
                row = rows[index]
                answer, charge, version = query_jev(client, key, row, acknowledge_provider_terms=True)
                records.append({"answer": answer, "cost_usd": charge, "model": version})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {"schema_version": schema_version, "signature": signature, "records": records},
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, checkpoint)
                print(f"Evaluated {index + 1}/{len(rows)} reference {row['kind']} decisions", flush=True)
    jev = [record["answer"] for record in records]
    cost = sum(record["cost_usd"] for record in records)
    versions = {record["model"] for record in records}
    if args.cache_reference_only:
        print(
            json.dumps({"reference_calls": len(records), "cost_usd": cost, "local_model_evaluated": False}), flush=True
        )
        return
    scorer = score_runtime_requests if args.runtime_scorer else score_requests
    local = scorer([public_payload(row) for row in rows], args.output_dir, args.model_report)
    report = {
        "purpose": "evaluation_only_no_teacher_training",
        "dataset_revision": DATASET_REVISION if args.cases_json is None else None,
        "dataset_manifest": manifest,
        "selection_seed": args.seed if args.cases_json is None else None,
        "selected_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
        "local_adapter_sha256": model_report["adapter_sha256"],
        "local_head_sha256": model_report["head_sha256"],
        "local_scorer": "daemon_runtime" if args.runtime_scorer else "offline_research",
        "jev_models": sorted(versions),
        "calls": len(rows),
        "cost_usd": cost,
        "reference_signature": signature,
        "reference_cache_shared": args.reference_cache is not None,
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

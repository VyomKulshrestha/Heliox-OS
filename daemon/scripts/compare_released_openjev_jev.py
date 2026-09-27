"""Compare frozen released Open-Jev logits with Jev on public fresh OOD rows.

Jev answers are evaluation-only, never fitting labels or execution authority.
Only public dataset records are sent to the remote endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np

from benchmarks.general_decision import format_answer
from scripts.compare_general_jev import compare, query_jev
from scripts.evaluate_jev_scope import _key
from scripts.evaluate_released_openjev_general import (
    PROTOCOL,
    ROOT,
    input_signature,
    select_fresh_rows,
)
from scripts.evaluate_world_general_qlora import selected_rows

COMPARISON_PROTOCOL = ROOT / "benchmarks/released_openjev_jev_protocol_v1.json"


def local_predictions(rows: list[dict], checkpoint: dict, *, signature: str) -> list[dict]:
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("signature") != signature
            or checkpoint.get("ids") != [row["id"] for row in rows]
            or len(checkpoint.get("logits", [])) != len(rows)):
        raise ValueError("Released model logits do not match the frozen fresh selection")
    predictions = []
    for row, raw in zip(rows, checkpoint["logits"], strict=True):
        values = np.asarray(raw, dtype=np.float64)
        predictions.append({"answer": format_answer(row["kind"], row["options"], values)})
    return predictions


def compare_by_source(rows: list[dict], local: list[dict], jev: list[dict]) -> dict:
    """Expose source-level transfer gaps without changing the frozen aggregate."""
    if len(rows) != len(local) or len(rows) != len(jev):
        raise ValueError("Rows, local predictions, and Jev replies must align")
    sources = sorted({row["source"] for row in rows})
    return {
        source: compare(
            [row for row in rows if row["source"] == source],
            [prediction for row, prediction in zip(rows, local, strict=True)
             if row["source"] == source],
            [answer for row, answer in zip(rows, jev, strict=True)
             if row["source"] == source],
        )
        for source in sources
    }


def save_progress(path: Path, *, signature: str, replies: list[dict], versions: list[str], cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "signature": signature,
                                     "replies": replies, "versions": versions, "cost_usd": cost},
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_progress(path: Path, *, signature: str, total: int) -> tuple[list[dict], list[str], float]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    replies, versions, cost = saved.get("replies"), saved.get("versions"), saved.get("cost_usd")
    if (saved.get("schema_version") != 1 or saved.get("signature") != signature
            or not isinstance(replies, list) or not isinstance(versions, list)
            or len(replies) != len(versions) or len(replies) > total
            or not all(isinstance(reply, dict) for reply in replies)
            or not all(isinstance(version, str) and version for version in versions)
            or not isinstance(cost, (int, float)) or not np.isfinite(cost) or cost < 0):
        raise ValueError("Jev comparison checkpoint does not match this selection")
    return replies, versions, float(cost)


def run(protocol_path: Path, model_dir: Path, reference_path: Path, env_file: Path,
        output_dir: Path, *, resume: bool) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline_protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != 1 or protocol.get("selection_protocol_sha256")
            != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            or protocol.get("max_calls") != baseline_protocol["fresh_rows"]):
        raise ValueError("Comparison protocol differs from frozen baseline selection")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    archived, _ = selected_rows(reference)
    rows, manifest = select_fresh_rows(
        archived, seed=baseline_protocol["fresh_seed"], pool=baseline_protocol["fresh_pool"],
        limit=baseline_protocol["fresh_rows"])
    signature = input_signature(rows, manifest, max_length=baseline_protocol["max_length"])
    baseline = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    if (baseline.get("fresh", {}).get("selection_signature") != signature
            or baseline.get("protocol_sha256") != protocol["selection_protocol_sha256"]):
        raise ValueError("Released model report does not match fresh public rows")
    local = local_predictions(
        rows, json.loads((model_dir / "fresh_logits.json").read_text(encoding="utf-8")),
        signature=signature)
    report_path = output_dir / "report.json"
    partial_path = output_dir / "jev_partial.json"
    if report_path.exists():
        raise FileExistsError("Final Jev comparison already exists")
    if partial_path.exists():
        if not resume:
            raise FileExistsError("Partial Jev comparison exists; pass --resume")
        replies, versions, cost = load_progress(partial_path, signature=signature, total=len(rows))
    else:
        replies, versions, cost = [], [], 0.0
    key = _key(env_file)
    with httpx.Client(timeout=45.0, trust_env=True) as client:
        for index in range(len(replies), len(rows)):
            answer, charge, version = query_jev(client, key, rows[index])
            if not isinstance(version, str) or not version or not np.isfinite(charge) or charge < 0:
                raise ValueError("Jev returned invalid model identity or cost")
            replies.append(answer)
            versions.append(version)
            cost += charge
            if len(replies) % protocol["checkpoint_every"] == 0 or len(replies) == len(rows):
                save_progress(partial_path, signature=signature, replies=replies,
                              versions=versions, cost=cost)
            if len(replies) % 8 == 0 or len(replies) == len(rows):
                print(f"Public Jev evaluation: {len(replies)}/{len(rows)}", flush=True)
    result = {
        "purpose": "evaluation_only_no_teacher_training",
        "selection_signature": signature,
        "comparison_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "released_report_sha256": hashlib.sha256((model_dir / "report.json").read_bytes()).hexdigest(),
        "jev_models": sorted(set(versions)), "calls": len(replies), "cost_usd": cost,
        "comparison": compare(rows, local, replies),
        "comparison_by_source": compare_by_source(rows, local, replies),
        "limitation": "Public synthetic OOD decisions only; this cannot establish model parity, "
                      "desktop transfer, or execution authority.",
    }
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=COMPARISON_PROTOCOL)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "benchmarks/.bench-state/released_openjev_general_v1")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "benchmarks/qwen_general_jev_comparison_v4.json")
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmarks/.bench-state/released_openjev_jev_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.model_dir, args.reference, args.env_file,
                         args.output_dir, resume=args.resume)["comparison"], indent=2))


if __name__ == "__main__":
    main()

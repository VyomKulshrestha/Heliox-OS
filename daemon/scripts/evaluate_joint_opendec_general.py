"""Compare a frozen joint Open-Jev research adapter on the archived Jev OOD selection.

The Jev responses are comparison evidence only, never fitting or calibration
labels. No desktop action is executed or authorized by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_general_decision_artifact import prediction_metrics
from scripts.evaluate_world_general_qlora import selected_rows
from scripts.reference_output_policy import block_jev_output_use
from scripts.score_general_decision import score_requests, verify_artifact


def main() -> None:
    block_jev_output_use()
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=root / "benchmarks/qwen_general_reference_comparison_v4.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    model_report, _ = verify_artifact(args.output_dir, args.model_report)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    rows, manifest = selected_rows(reference)
    payloads = [
        {
            "state": json.loads(row["state_json"]),
            "kind": row["kind"],
            "question": row["question"],
            "options": row["options"],
        }
        for row in rows
    ]
    predictions = score_requests(payloads, args.output_dir, args.model_report)
    local = prediction_metrics(rows, predictions)["by_kind"]
    report = {
        "purpose": "research_only_frozen_general_decision_comparison",
        "adapter_sha256": model_report["adapter_sha256"],
        "head_sha256": model_report["head_sha256"],
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "selected_ids_sha256": reference["selected_ids_sha256"],
        "dataset_manifest": manifest,
        "local": local,
        "jev": {
            kind: {
                key: value
                for key, value in reference["comparison"][kind].items()
                if key.startswith("jev_") or key == "rows"
            }
            for kind in local
        },
        "limitation": "Previously inspected licensed public OOD rows and archived Jev aggregates; "
        "not blind parity, live desktop, or safety evidence.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"local": local, "jev": report["jev"]}, indent=2))


if __name__ == "__main__":
    main()

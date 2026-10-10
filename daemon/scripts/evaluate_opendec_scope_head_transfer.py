"""Score the frozen scope head on an already-inspected synthetic transfer suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.evaluate_opendec_checkpoint import SUITES
from scripts.extract_opendec_scope_features import sha256
from scripts.fit_opendec_scope_head import load_features, metrics


def score(data: dict, head: dict) -> np.ndarray:
    normalized = (data["features"] - head["mean"]) / head["scale"]
    return expit(normalized @ head["coefficient"].T + head["intercept"]).reshape(-1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_features_v1")
    parser.add_argument("--artifact", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_head_v1.npz")
    parser.add_argument("--fit-report", type=Path, default=root / "benchmarks/opendec_2b_scope_head_v1.json")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/opendec_2b_scope_head_transfer_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Transfer report already exists")
    data, metadata = load_features(args.feature_dir, "qwen-v1")
    fit = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if (metadata["source"] != fit["source"] or metadata["transfer_suite"]["suite"] != "qwen-v1"
            or sha256(args.artifact) != fit["artifact_sha256"]):
        raise ValueError("Transfer features or saved head do not match the frozen fit")
    if set(data["ids"]) != {case.id for case in SUITES["qwen-v1"]}:
        raise ValueError("Synthetic transfer suite is incomplete or changed")
    with np.load(args.artifact, allow_pickle=False) as saved:
        head = {name: saved[name] for name in ("mean", "scale", "coefficient", "intercept", "threshold")}
    probabilities = score(data, head)
    threshold = float(head["threshold"])
    report = {"purpose": "research_only_inspected_synthetic_transfer",
              "source": fit["source"], "suite": metadata["transfer_suite"],
              "artifact_sha256": fit["artifact_sha256"],
              "at_default_threshold": metrics(data["labels"], probabilities, 0.5),
              "at_calibrated_threshold": metrics(data["labels"], probabilities, threshold),
              "results": [{"id": str(case_id), "gold_in_scope": bool(gold),
                           "probability": float(probability)}
                          for case_id, gold, probability in zip(data["ids"], data["labels"],
                                                                 probabilities, strict=True)],
              "limitation": "Previously inspected authored suite; not a fresh holdout or Jev parity."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("at_default_threshold", "at_calibrated_threshold")},
                     indent=2))


if __name__ == "__main__":
    main()

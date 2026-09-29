"""Fit a research-only outcome candidate from explicitly rated local workflows.

The encrypted input stays local. Only an encrypted model artifact and an
aggregate, de-identified evaluation report are written. This model has no
runtime or action-authorization integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pilot.config import DATA_DIR
from pilot.intelligence.workflow_outcome_model import fit_workflow_outcome_model
from pilot.workflows.recorded_workflows import RecordedWorkflowStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.urandom(8).hex()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    output_dir = DATA_DIR / "research"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-store", type=Path, default=DATA_DIR / "recorded_workflows.enc")
    parser.add_argument("--artifact", type=Path, default=output_dir / "workflow_outcome_candidate_v1.enc")
    parser.add_argument("--report", type=Path, default=output_dir / "workflow_outcome_candidate_v1.json")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if not args.recorded_store.is_file():
        parser.error(f"No encrypted local workflow store exists: {args.recorded_store}")
    if (args.artifact.suffix != ".enc" or args.artifact.exists() or args.report.exists()
            or args.artifact.resolve() == args.report.resolve()):
        parser.error("Use a fresh .enc artifact path and a fresh report path")
    if args.artifact.resolve() == args.recorded_store.resolve() or args.report.resolve() == args.recorded_store.resolve():
        parser.error("Output paths must not overwrite the encrypted workflow store")

    store = RecordedWorkflowStore(args.recorded_store)
    before_hash = _sha256(args.recorded_store)
    examples = store.load_outcome_training_examples()
    artifact, report = fit_workflow_outcome_model(examples, seed=args.seed)
    if _sha256(args.recorded_store) != before_hash:
        raise RuntimeError("Encrypted workflow store changed during training; discard this candidate")

    report.update({
        "source_store_sha256": before_hash,
        "candidate_file_encrypted": True,
        "candidate_promoted": False,
        "probability_calibration": "not fitted",
    })
    encrypted_artifact = store.encrypt_model_artifact(
        json.dumps(artifact, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    _write_new(args.artifact, encrypted_artifact)
    _write_new(args.report, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({
        "candidate_promoted": False,
        "examples": report["examples"],
        "recipe_groups": report["recipe_groups"],
        "test_balanced_accuracy": report["candidate"]["balanced_accuracy"],
        "artifact": str(args.artifact),
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()

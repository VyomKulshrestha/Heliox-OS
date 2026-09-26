"""Reserve new AgentNet task groups for a paired prospective/observed check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_agentnet_outcome_features import sha256_file  # noqa: E402
from scripts.select_agentnet_outcome_cases import (  # noqa: E402
    SOURCE_SHA256,
    load_candidates,
    select,
)

FRESH_SPLIT_ROWS = {"final": 128, "ood": 128}


def reserve(source: Path, previous: dict, previous_manifest_sha256: str) -> dict:
    if previous["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Previous selection is not from the pinned source")
    used_groups = {case_id.split(":", 1)[0] for case_id in previous["case_ids"]}
    candidates, counts = load_candidates(source)
    unused = [row for row in candidates if row["group"] not in used_groups]
    splits = select(unused, ood_domain=previous["ood_domain"], split_rows=FRESH_SPLIT_ROWS)
    case_ids = [case_id for name in FRESH_SPLIT_ROWS for case_id in splits[name]]
    if len(case_ids) != len({case_id.split(":", 1)[0] for case_id in case_ids}):
        raise ValueError("Fresh transfer task groups overlap")
    return {
        "purpose": "fresh_paired_agentnet_pre_post_action_transfer",
        "source_sha256": SOURCE_SHA256,
        "previous_manifest_sha256": previous_manifest_sha256,
        "previous_task_groups": len(used_groups),
        "candidate_task_groups": counts["groups"],
        "ood_domain": previous["ood_domain"],
        "label_provenance": previous["label_provenance"],
        "split_rows": FRESH_SPLIT_ROWS,
        "splits": splits,
        "case_ids": case_ids,
        "selection_method": "pinned_source_balanced_labels_unused_task_groups_fixed_seeds",
        "limitation": "Reflector labels are model-generated; no runtime authority or Jev parity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh reservation path")
    previous = json.loads(args.previous_manifest.read_text(encoding="utf-8"))
    manifest = reserve(args.source, previous, sha256_file(args.previous_manifest))
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups_excluded": manifest["previous_task_groups"],
                      "splits": {name: len(ids) for name, ids in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()

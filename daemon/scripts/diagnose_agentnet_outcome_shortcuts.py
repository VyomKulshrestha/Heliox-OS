"""Post-hoc text shortcut diagnostic for the frozen AgentNet outcome study.

This is exploratory after the final labels were opened. It cannot select a
model or threshold for deployment and is not another held-out test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_agentnet_outcome_features import sha256_file  # noqa: E402
from scripts.select_agentnet_outcome_cases import SOURCE_SHA256  # noqa: E402

VIEWS = {
    "all_fields": ("instruction", "before_observation", "action", "code", "next_observation"),
    "no_next_observation": ("instruction", "before_observation", "action", "code"),
    "next_observation_only": ("next_observation",),
    "task_and_action_only": ("instruction", "action", "code"),
}


def selected_rows(source: Path, ids: list[str]) -> dict[str, dict]:
    wanted = set(ids)
    found: dict[str, dict] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            steps = trajectory.get("traj") or []
            for current, following in zip(steps, steps[1:]):
                case_id = f"{trajectory['task_id']}:{current['index']}"
                if case_id not in wanted:
                    continue
                if case_id in found or following["index"] != current["index"] + 1:
                    raise ValueError("Selected action is duplicated or nonconsecutive")
                before = current["value"]
                found[case_id] = {
                    "instruction": trajectory["instruction"],
                    "before_observation": before["observation"],
                    "action": before["action"], "code": before["code"],
                    "next_observation": following["value"]["observation"],
                    "label": before["last_step_correct"],
                }
    if set(found) != wanted or any(not isinstance(row["label"], bool) for row in found.values()):
        raise ValueError("Selected action or reflector label is missing")
    return found


def diagnose(rows: dict[str, dict], splits: dict[str, list[str]]) -> dict:
    report: dict[str, dict] = {}
    train_ids = splits["train"]
    train_labels = np.asarray([rows[case_id]["label"] for case_id in train_ids])

    def texts(ids: list[str], fields: tuple[str, ...]) -> list[str]:
        return ["\n".join(str(rows[case_id][field]) for field in fields)
                for case_id in ids]

    for view, fields in VIEWS.items():
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=50_000, sublinear_tf=True)
        train_x = vectorizer.fit_transform(texts(train_ids, fields))
        head = LogisticRegression(C=1.0, max_iter=1000).fit(train_x, train_labels)
        report[view] = {"vocabulary_size": len(vectorizer.vocabulary_)}
        for split in ("development", "final", "ood"):
            labels = [rows[case_id]["label"] for case_id in splits[split]]
            scores = head.predict_proba(vectorizer.transform(texts(splits[split], fields)))[:, 1]
            report[view][f"{split}_auc"] = float(roc_auc_score(labels, scores))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh exploratory report path")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("AgentNet selection differs from the pinned source")
    rows = selected_rows(args.source, manifest["case_ids"])
    result = {
        "purpose": "post_hoc_agentnet_text_shortcut_diagnostic",
        "source_sha256": SOURCE_SHA256,
        "manifest_sha256": sha256_file(args.manifest),
        "method": "train_only_tfidf_word_1_2_min_df_2_C_1_logistic_fixed_parameters",
        "exploratory_after_final_labels_opened": True,
        "views": diagnose(rows, manifest["splits"]),
        "promotion": "none",
        "limitation": "Model-generated labels and a post-hoc diagnostic; never use these splits to select a runtime model.",
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["views"], indent=2))


if __name__ == "__main__":
    main()

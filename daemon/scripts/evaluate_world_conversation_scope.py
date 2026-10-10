"""Score pre-call multi-turn task context with a frozen offline scope head.

Only the transcript before the first candidate tool call is visible. Later
messages, tool output, and test labels cannot enter the model's features.
This is task-tool relevance research, never execution authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.refscope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT
from scripts.evaluate_world_scope_artifact import score_features
from scripts.train_world_scope_semantic import embed_cases, metrics

DATASET_REVISION = "6385bd37654fbc36b84dd0341f0752654785dce3"
AGENTIC_TEST_SHA256 = "b7bfe395aa24ca6442a23b232ff014ebbe2bb3680738db54baa8ed8701c08231"


def load_pre_call_cases(path: Path, expected_sha256: str) -> tuple[ScopeCase, ...]:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Agentic TBAC test source hash mismatch")
    rows = json.loads(source)
    if not isinstance(rows, list):
        raise ValueError("Agentic TBAC test must be a list")
    cases, seen = [], set()
    for row in rows:
        identifier = row["id"]
        conversation = row["request"]["conversation"]["messages"]
        tool = row["request"]["tool"]
        label = row["label"]["relevant"]
        if identifier in seen or type(label) is not bool:
            raise ValueError("Duplicate ID or nonboolean agentic TBAC label")
        first_call = next((index for index, message in enumerate(conversation) if message.get("tool_calls")), None)
        if first_call is None or first_call == 0:
            raise ValueError("Agentic TBAC row has no pre-call context")
        call = conversation[first_call]["tool_calls"][0]["function"]["name"]
        prefix = conversation[:first_call]
        if (
            call != tool["name"]
            or prefix[0]["role"] != "user"
            or any(
                message["role"] not in {"user", "assistant"}
                or message.get("tool_calls")
                or not isinstance(message.get("content"), str)
                for message in prefix
            )
        ):
            raise ValueError("Agentic TBAC row violates pre-call boundary")
        seen.add(identifier)
        request = json.dumps(
            [{"role": message["role"], "content": message["content"]} for message in prefix], ensure_ascii=False
        )
        action = f"Invoke MCP tool {tool['name']}: {tool['description']}"
        cases.append(ScopeCase(identifier, request, action, label, "agentic_test"))
    return tuple(cases)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=state / "tbac_6385bd3/agentic_test_4o.json")
    parser.add_argument("--artifact", type=Path, default=state / "world_multicorpus_scope_v1.npz")
    parser.add_argument("--artifact-report", type=Path, default=state / "world_multicorpus_scope_v1.json")
    parser.add_argument("--report", type=Path, default=state / "world_conversation_scope_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    trained = json.loads(args.artifact_report.read_text(encoding="utf-8"))
    source = args.artifact.read_bytes()
    if (
        trained["artifact_sha256"] != hashlib.sha256(source).hexdigest()
        or trained["revision"] != MODEL_REVISION
        or trained["prompt_sha256"] != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    ):
        raise ValueError("Frozen artifact, model, or prompt identity mismatch")
    cases = load_pre_call_cases(args.dataset, AGENTIC_TEST_SHA256)

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        output_loading_info=True,
    )
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    features, labels, excluded = embed_cases(model.eval(), tokenizer, cases, trained["max_length"], "agentic_test")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        if str(artifact["model_revision"]) != MODEL_REVISION:
            raise ValueError("Artifact backbone revision mismatch")
        scores = score_features(artifact, features)
        cautious = float(artifact["threshold"])
    report = {
        "purpose": "offline_pre_call_multiturn_relevance_evaluation_only",
        "source": {
            "url": "https://github.com/outshift-open/tbac-research-datasets",
            "revision": DATASET_REVISION,
            "sha256": AGENTIC_TEST_SHA256,
            "split": "conversation_tool/agentic/test_4o.json",
        },
        "trained_artifact_sha256": trained["artifact_sha256"],
        "selected_rows": len(cases),
        "eligible_rows": len(labels),
        "excluded_overlong": excluded,
        "default": metrics(labels, scores, 0.5),
        "cautious": metrics(labels, scores, cautious),
        "limitation": "Generated conversations share official TBAC test seed tasks, and earlier public "
        "test results guided this adaptive research. Relevance is not permission, "
        "live desktop evidence, or Jev parity.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"default": report["default"], "cautious": report["cautious"]}, indent=2))


if __name__ == "__main__":
    main()

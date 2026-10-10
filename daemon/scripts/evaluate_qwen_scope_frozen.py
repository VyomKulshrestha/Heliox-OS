"""Score the frozen experimental Qwen scope adapter against a new Jev suite.

The weights, prompt, calibration temperature, and 0.5 threshold are fixed.
This script performs no training or selection on the evaluation suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_qwen_v1_holdout import CASES  # noqa: E402
from scripts.train_qwen_scope_noul import MODEL, REVISION, _encode, _evaluate, _probabilities  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def verify_artifact(root: Path, model_dir: Path) -> dict:
    frozen = json.loads((root / "benchmarks/qwen_scope_probe_v1.json").read_text(encoding="utf-8"))
    for name, expected in (
        ("adapter/adapter_model.safetensors", frozen["adapter_sha256"]),
        ("head.npz", frozen["head_sha256"]),
    ):
        actual = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen model artifact changed: {name}")
    if frozen["model"] != MODEL or frozen["revision"] != REVISION:
        raise ValueError("Frozen model base does not match the evaluator")
    return frozen


def evaluate(root: Path, model_dir: Path, teacher_path: Path, output: Path) -> dict:
    frozen = verify_artifact(root, model_dir)
    teacher = _rows(teacher_path, CASES)

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation")
    base_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    backbone = PeftModel.from_pretrained(base.model, model_dir / "adapter", is_trainable=False)
    del base
    with np.load(model_dir / "head.npz", allow_pickle=False) as saved:
        weights = np.asarray(saved["weights"], dtype=np.float32)
        bias = np.asarray(saved["bias"], dtype=np.float32)
        temperature = float(saved["temperature"])
        if (
            str(saved["model_revision"]) != REVISION
            or temperature != frozen["head_temperature"]
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Frozen head configuration changed")
    head = nn.Linear(weights.shape[1], 1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weights).to("cuda"))
        head.bias.copy_(torch.from_numpy(bias).to("cuda"))
    rows = [
        {
            "id": case.id,
            "request": case.request,
            "action": case.action,
            "probability": float(teacher[case.id]["jev_probability"]),
        }
        for case in CASES
    ]
    encoded, truncated = _encode(tokenizer, rows)
    probabilities = _probabilities(_evaluate(backbone, head, encoded), temperature)
    report = {
        "frozen_suite_commit": "ccb3e71",
        "adapter_sha256": frozen["adapter_sha256"],
        "head_sha256": frozen["head_sha256"],
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "truncated_rows": truncated,
        "qwen_scope_adapter": _metrics(list(CASES), teacher, probabilities),
        "limitation": "Synthetic scenarios only; no production authority or general Jev parity.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use

    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_scope_v1")
    parser.add_argument("--teacher", type=Path, default=root / "benchmarks/jev_scope_qwen_v1_teacher.json")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/qwen_scope_fresh_v1.json")
    args = parser.parse_args()
    report = evaluate(root, args.model_dir, args.teacher, args.output)
    print(
        json.dumps(
            {
                key: (
                    {name: value for name, value in data.items() if name != "mismatches"}
                    if isinstance(data, dict)
                    else data
                )
                for key, data in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

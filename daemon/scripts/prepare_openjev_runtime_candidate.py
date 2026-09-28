"""Prepare the public Open-Jev checkpoint for Heliox's opt-in local scorer.

This copies the hash-pinned adapter from the existing Hugging Face cache and
converts its scalar head to Heliox's local artifact format. It does not load
the model, enable the scorer, or grant execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# On Windows, initialize SentencePiece before any transitive PyArrow import.
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
PUBLISHED_TEMPERATURE = 1.518796342858676
MAX_LENGTH = 1024


def head_arrays(state: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate a released scalar linear head and return safe numeric arrays."""
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise ValueError("Open-Jev head must contain only linear weight and bias")
    weight = np.asarray(state["weight"].detach().cpu() if hasattr(state["weight"], "detach")
                        else state["weight"], dtype=np.float32)
    bias = np.asarray(state["bias"].detach().cpu() if hasattr(state["bias"], "detach")
                      else state["bias"], dtype=np.float32)
    if (weight.ndim != 2 or weight.shape[0] != 1 or bias.shape != (1,)
            or weight.shape[1] < 1 or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()):
        raise ValueError("Open-Jev scalar head has invalid dimensions or non-finite values")
    return weight, bias


def prepare(output_dir: Path) -> dict:
    """Create a new local artifact directory from already-cached public files."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {output_dir}")

    from scripts.train_world_scope_qlora import (
        OPENJEV_ADAPTER_SHA256,
        OPENJEV_HEAD_SHA256,
        OPENJEV_REPO,
        OPENJEV_REVISION,
        openjev_checkpoint_paths,
    )

    adapter_dir, source_head = openjev_checkpoint_paths()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError("Cached Open-Jev adapter_config.json is missing")
    if hashlib.sha256(source_head.read_bytes()).hexdigest() != OPENJEV_HEAD_SHA256:
        raise ValueError("Cached Open-Jev head checksum mismatch")

    import torch

    state = torch.load(source_head, map_location="cpu", weights_only=True)
    weights, bias = head_arrays(state)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp:
        staged = Path(temp) / output_dir.name
        (staged / "adapter").parent.mkdir(parents=True)
        shutil.copytree(adapter_dir, staged / "adapter")
        head_path = staged / "head.npz"
        np.savez_compressed(
            head_path,
            model_revision=np.asarray(BASE_REVISION),
            max_length=np.asarray(MAX_LENGTH, dtype=np.int64),
            temperature=np.asarray(PUBLISHED_TEMPERATURE, dtype=np.float64),
            weights=weights,
            bias=bias,
            attention_implementation=np.asarray("eager"),
            prompt_style=np.asarray("candidate_v1"),
            model_dtype=np.asarray("bfloat16"),
        )
        adapter_digest = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
        head_digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        if adapter_digest != OPENJEV_ADAPTER_SHA256:
            raise ValueError("Copied Open-Jev adapter checksum mismatch")
        report = {
            "schema_version": 1,
            "purpose": "research_only_local_decision_scoring",
            "model": BASE_MODEL,
            "revision": BASE_REVISION,
            "adapter_sha256": adapter_digest,
            "head_sha256": head_digest,
            "max_length": MAX_LENGTH,
            "temperature": PUBLISHED_TEMPERATURE,
            "attention_implementation": "eager",
            "prompt_style": "candidate_v1",
            "model_dtype": "bfloat16",
            "source_checkpoint": f"{OPENJEV_REPO}@{OPENJEV_REVISION}",
            "source_license": "Apache-2.0",
            "source_head_sha256": OPENJEV_HEAD_SHA256,
            "adapter_config_sha256": hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
            "research_only": True,
            "execution_authority": False,
        }
        (staged / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staged, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/.bench-state/openjev_runtime_candidate_v1",
        help="new local artifact directory; an existing directory is never overwritten",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()

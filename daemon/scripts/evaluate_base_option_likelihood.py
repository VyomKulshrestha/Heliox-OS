"""Bounded, research-only causal option-likelihood diagnostic.

This scores licensed public decision rows with a cached causal LM. It does not
load Jev/Open-Jev adapters, train on provider outputs, or authorize actions.
Full-sequence scoring is the reference; shared-prefix scoring is diagnostic
only because cache expansion may change candidate probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# On Windows, SentencePiece must be initialized before PyArrow to avoid a
# native access violation when Transformers is imported after Arrow data.
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.general_decision import (
    DATASET_REVISION,
    load_split,
    metrics,
    uniform_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED_BASES = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 1024


def longest_common_prefix(sequences: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """Split token sequences at their exact longest shared prefix."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Candidate token sequences must be nonempty")
    common = min(map(len, sequences))
    for index in range(common):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            common = index
            break
    prefix = sequences[0][:common]
    suffixes = [sequence[common:] for sequence in sequences]
    if any(not suffix for suffix in suffixes):
        raise ValueError("Every candidate must add at least one token after the shared prefix")
    return prefix, suffixes


def mean_token_logprob(total: float, token_count: int, *, normalization: str) -> float:
    """Normalize candidate likelihood without privileging longer labels."""
    if token_count < 1 or normalization not in {"mean", "sum"}:
        raise ValueError("Candidate token count or normalization is invalid")
    return total / token_count if normalization == "mean" else total


def render_prompt(row: dict[str, Any]) -> str:
    """Render only observable task fields; target, IDs, and provenance stay out."""
    state = json.loads(row["state_json"])
    rendered_state = (
        state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    choices = "\n".join(f"{index + 1}. {option}" for index, option in enumerate(row["options"]))
    return (
        f"Context:\n{rendered_state}\n\nQuestion: {row['question']}\n\n"
        f"Options:\n{choices}\n\nChoose the best answer. Reply with one option exactly as written:"
    )


def encode_candidates(tokenizer: Any, row: dict[str, Any]) -> tuple[int, list[list[int]]]:
    """Encode each declared option as an assistant continuation of one prompt."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_prompt(row)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    sequences = [tokenizer(rendered + " " + option, add_special_tokens=False)["input_ids"] for option in row["options"]]
    if any(sequence[: len(prompt_ids)] != prompt_ids for sequence in sequences):
        raise ValueError("Tokenizer merged the prompt boundary with a candidate continuation")
    longest_common_prefix(sequences)
    if any(len(sequence) > MAX_LENGTH for sequence in sequences):
        raise ValueError("Candidate exceeds the fixed token limit; no truncation is allowed")
    return len(prompt_ids), sequences


def _repeat_cache(cache: Any, repeats: int) -> Any:
    """Expand a one-row Transformers cache for a bounded candidate microbatch."""
    repeat = getattr(cache, "batch_repeat_interleave", None)
    if callable(repeat):
        result = repeat(repeats)
        return cache if result is None else result
    reorder = getattr(cache, "reorder_cache", None)
    if callable(reorder):
        import torch

        # Repeating row zero also handles hybrid caches with recurrent/conv state.
        reorder(torch.zeros(repeats, dtype=torch.long, device="cuda"))
        return cache
    if isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            layers.append(tuple(tensor.expand((repeats,) + tuple(tensor.shape[1:])).contiguous() for tensor in layer))
        return tuple(layers)
    raise RuntimeError("The local model cache cannot be expanded for shared-prefix scoring")


def score_shared_prefix(
    model: Any,
    sequences: list[list[int]],
    *,
    prompt_length: int,
    option_batch_size: int,
    pad_token_id: int,
    normalization: str = "mean",
) -> tuple[list[float], list[list[float]]]:
    """Score candidate continuations using one shared context prefill per batch."""
    import torch

    if option_batch_size < 1:
        raise ValueError("Option batch size must be positive")
    prefix = sequences[0][:prompt_length]
    if prompt_length < 1 or any(sequence[:prompt_length] != prefix for sequence in sequences):
        raise ValueError("Candidate sequences do not share the declared prompt tokens")
    suffixes = [sequence[prompt_length:] for sequence in sequences]
    if any(not suffix for suffix in suffixes):
        raise ValueError("Every candidate must add at least one token after the prompt")
    prefix_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
    scores: list[float] = []
    token_logprobs: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(suffixes), option_batch_size):
            chunk = suffixes[start : start + option_batch_size]
            prefix_output = model(input_ids=prefix_ids, use_cache=True, return_dict=True)
            prefix_logprobs = torch.log_softmax(prefix_output.logits[0, -1].float(), dim=-1)
            first_scores = [prefix_logprobs[suffix[0]].item() for suffix in chunk]
            lengths = [len(suffix) for suffix in chunk]
            max_suffix = max(lengths)
            if max_suffix > 1:
                suffix_ids = torch.full((len(chunk), max_suffix), pad_token_id, dtype=torch.long, device="cuda")
                suffix_mask = torch.zeros_like(suffix_ids)
                for index, suffix in enumerate(chunk):
                    suffix_ids[index, : len(suffix)] = torch.tensor(suffix, dtype=torch.long, device="cuda")
                    suffix_mask[index, : len(suffix)] = 1
                cache = _repeat_cache(prefix_output.past_key_values, len(chunk))
                attention_mask = torch.cat(
                    (
                        torch.ones((len(chunk), len(prefix)), dtype=torch.long, device="cuda"),
                        suffix_mask,
                    ),
                    dim=1,
                )
                output = model(
                    input_ids=suffix_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=False,
                    return_dict=True,
                )
                next_logprobs = torch.log_softmax(output.logits.float(), dim=-1)
            else:
                next_logprobs = None

            for index, suffix in enumerate(chunk):
                candidate_logprobs = [first_scores[index]]
                if len(suffix) > 1:
                    token_ids = torch.tensor(suffix[1:], dtype=torch.long, device="cuda")
                    positions = torch.arange(len(suffix) - 1, device="cuda")
                    candidate_logprobs.extend(next_logprobs[index, positions, token_ids].detach().cpu().tolist())
                token_logprobs.append(candidate_logprobs)
                scores.append(mean_token_logprob(sum(candidate_logprobs), len(suffix), normalization=normalization))
            del prefix_output
            if next_logprobs is not None:
                del next_logprobs
    return scores, token_logprobs


def score_naive(
    model: Any,
    sequences: list[list[int]],
    *,
    prompt_length: int,
    option_batch_size: int,
    pad_token_id: int,
    normalization: str = "mean",
) -> tuple[list[float], list[list[float]]]:
    """Reference full-sequence scorer for a small exactness check."""
    import torch

    prefix = sequences[0][:prompt_length]
    if prompt_length < 1 or any(sequence[:prompt_length] != prefix for sequence in sequences):
        raise ValueError("Candidate sequences do not share the declared prompt tokens")
    suffixes = [sequence[prompt_length:] for sequence in sequences]
    if any(not suffix for suffix in suffixes):
        raise ValueError("Every candidate must add at least one token after the prompt")
    results: list[float] = []
    token_logprobs: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), option_batch_size):
            chunk = sequences[start : start + option_batch_size]
            lengths = [len(sequence) for sequence in chunk]
            width = max(lengths)
            input_ids = torch.full((len(chunk), width), pad_token_id, dtype=torch.long, device="cuda")
            attention_mask = torch.zeros_like(input_ids)
            for index, sequence in enumerate(chunk):
                input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
                attention_mask[index, : len(sequence)] = 1
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            for index, (sequence, suffix) in enumerate(zip(chunk, suffixes[start : start + len(chunk)], strict=True)):
                offset = len(sequence) - len(suffix)
                positions = torch.arange(offset - 1, len(sequence) - 1, device="cuda")
                token_ids = torch.tensor(suffix, dtype=torch.long, device="cuda")
                selected_logprobs = torch.log_softmax(output.logits[index, positions].float(), dim=-1)
                parts = selected_logprobs[torch.arange(len(suffix), device="cuda"), token_ids].detach().cpu().tolist()
                token_logprobs.append(parts)
                total = sum(parts)
                results.append(mean_token_logprob(total, len(suffix), normalization=normalization))
            del output
    return results, token_logprobs


def _load_local_model(model_name: str):
    print("Initializing the local CUDA runtime", flush=True)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this opt-in local research evaluation")
    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB of free CUDA memory is required")
    transformers_logging.disable_progress_bar()
    print(f"CUDA has {free_memory / 1024**3:.1f} GiB free", flush=True)
    revision = PINNED_BASES[model_name]
    model_path = snapshot_download(model_name, revision=revision, local_files_only=True)
    print("Resolved the pinned model from the local HF cache", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("Loaded tokenizer from cache", flush=True)
    dtype = torch.bfloat16
    load_options = {
        "local_files_only": True,
        "dtype": dtype,
        "device_map": {"": 0},
        "attn_implementation": "eager",
    }
    if model_name == "Qwen/Qwen3.5-2B":
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_options)
    print("Loaded model on CUDA", flush=True)
    model.eval()
    return torch, tokenizer, model


def run(
    *,
    split: str,
    limit: int,
    seed: int,
    option_batch_size: int,
    normalization: str,
    verify_rows: int,
    model_name: str,
    scoring: str,
    output_path: Path,
) -> dict:
    rows, manifest = load_split(split, limit, seed)
    print(f"Loaded {len(rows)} licensed {split} rows from the local dataset cache", flush=True)
    print(f"Loading cached {model_name} on CUDA (no downloads)", flush=True)
    torch, tokenizer, model = _load_local_model(model_name)
    encoded = [encode_candidates(tokenizer, row) for row in rows]
    print(f"Encoded {sum(len(sequences) for _, sequences in encoded)} candidate options; scoring on CUDA", flush=True)
    if verify_rows < 0 or verify_rows > len(rows):
        raise ValueError("Reference-comparison row count is outside the selected rows")

    started = time.perf_counter()
    scored = []
    verification = []
    for index, (row, (prompt_length, sequences)) in enumerate(zip(rows, encoded, strict=True)):
        if scoring == "full_sequence":
            reference, reference_parts = score_naive(
                model,
                sequences,
                prompt_length=prompt_length,
                option_batch_size=option_batch_size,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                normalization=normalization,
            )
            values = reference
        else:
            values, shared_parts = score_shared_prefix(
                model,
                sequences,
                prompt_length=prompt_length,
                option_batch_size=option_batch_size,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                normalization=normalization,
            )
        if scoring == "shared_prefix" and index < verify_rows:
            reference, reference_parts = score_naive(
                model,
                sequences,
                prompt_length=prompt_length,
                option_batch_size=option_batch_size,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                normalization=normalization,
            )
            delta = max(abs(left - right) for left, right in zip(values, reference, strict=True))
            token_delta = max(
                abs(left - right)
                for shared, reference_tokens in zip(shared_parts, reference_parts, strict=True)
                for left, right in zip(shared, reference_tokens, strict=True)
            )
            largest_token_difference = max(
                (abs(left - right), candidate_index, token_index, left, right)
                for candidate_index, (shared, reference_tokens) in enumerate(
                    zip(shared_parts, reference_parts, strict=True)
                )
                for token_index, (left, right) in enumerate(zip(shared, reference_tokens, strict=True))
            )
            top1_match = int(np.argmax(values) == np.argmax(reference))
            verification.append(
                {
                    "row_id": row["id"],
                    "max_abs_score_delta": delta,
                    "max_abs_token_logprob_delta": token_delta,
                    "selected_option_matches_reference": bool(top1_match),
                }
            )
            if not top1_match:
                raise RuntimeError(
                    f"Shared-prefix selection differs from the full-sequence reference "
                    f"(mean delta {delta:.4f}, token delta {token_delta:.4f}; "
                    f"largest token mismatch (candidate, token, shared, reference)="
                    f"{largest_token_difference[1:]} )"
                )
        scored.append(np.asarray(values, dtype=np.float64))
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            print(f"Scored {index + 1}/{len(rows)} rows", flush=True)
    elapsed = time.perf_counter() - started

    ids_sha256 = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    result = {
        "schema_version": 1,
        "purpose": "base_model_option_likelihood_research_only",
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "seed": seed,
        "selected_rows": len(rows),
        "selected_ids_sha256": ids_sha256,
        "dataset_manifest": manifest,
        "model": model_name,
        "model_revision": PINNED_BASES[model_name],
        "model_weights": "local cached base only; no adapter or provider output",
        "scoring": {
            "method": (
                "full-sequence batched causal continuation log-likelihood"
                if scoring == "full_sequence"
                else "shared-prefix causal continuation log-likelihood (diagnostic only)"
            ),
            "normalization": normalization,
            "option_batch_size": option_batch_size,
            "reference_comparison": verification,
        },
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": metrics(rows, scored, 1.0),
        "uniform_baseline": uniform_baseline(rows),
        "limitation": "Public synthetic decision rows do not establish Jev parity or desktop task quality.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    model = None
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="ood")
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--option-batch-size", type=int, default=4)
    parser.add_argument("--normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--model", choices=tuple(PINNED_BASES), default=DEFAULT_MODEL)
    parser.add_argument("--scoring", choices=("full_sequence", "shared_prefix"), default="full_sequence")
    parser.add_argument(
        "--verify-rows", type=int, default=2, help="Compare shared-prefix selection with a full-sequence reference"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/.bench-state/base_option_likelihood_v1/report.json",
    )
    args = parser.parse_args()
    if not 1 <= args.rows <= 256 or not 1 <= args.option_batch_size <= 16:
        parser.error("--rows must be 1-256 and --option-batch-size must be 1-16")
    result = run(
        split=args.split,
        limit=args.rows,
        seed=args.seed,
        option_batch_size=args.option_batch_size,
        normalization=args.normalization,
        verify_rows=args.verify_rows,
        model_name=args.model,
        scoring=args.scoring,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"]["by_kind"],
                "uniform_baseline": result["uniform_baseline"]["by_kind"],
                "elapsed_seconds": result["elapsed_seconds"],
                "reference_comparison": result["scoring"]["reference_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

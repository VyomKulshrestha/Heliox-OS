"""Check batched feature extraction against a saved single-row checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, messages
from scripts.evaluate_world_conversation_scope import load_pre_call_cases
from scripts.train_world_multiturn_scope_semantic import (
    AGENTIC_VALIDATION_SHA256,
    split_agentic,
)
from scripts.train_world_scope_semantic import (
    embed_batch,
    feature_identity,
    load_feature_checkpoint,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--checkpoint", type=Path,
                        default=state / "world_multiturn_features_v1/agentic_train.npz")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    source = args.dataset_dir / "agentic_validation_4o.json"
    cases = load_pre_call_cases(source, AGENTIC_VALIDATION_SHA256)
    train, _, _ = split_agentic(cases, source)
    _, _, indices, vectors, _ = load_feature_checkpoint(
        args.checkpoint, feature_identity(train, args.max_length), train)
    if len(indices) < 4:
        raise ValueError("Checkpoint needs at least four embedded rows")

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys") or tokenizer.pad_token_id is None:
        raise RuntimeError("Pinned model or pad token unavailable")
    model.eval()
    selected = [0, len(indices) // 3, 2 * len(indices) // 3, len(indices) - 1]
    token_ids = []
    for position in selected:
        case = train[indices[position] - 1]
        prompt = tokenizer.apply_chat_template(
            messages(case.request, case.action), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        token_ids.append(tokenizer(prompt, return_tensors="pt", truncation=False)[
            "input_ids"][0])
    with torch.inference_mode():
        batched = np.asarray(embed_batch(model, token_ids, tokenizer.pad_token_id))
        one_by_one = np.asarray([embed_batch(model, [ids], tokenizer.pad_token_id)[0]
                                 for ids in token_ids])
    single = np.asarray([vectors[position] for position in selected])
    difference = np.abs(batched - single)
    single_difference = np.abs(one_by_one - single)
    print(json.dumps({"rows": [indices[position] for position in selected],
                      "token_lengths": [len(ids) for ids in token_ids],
                      "max_absolute_difference": float(difference.max()),
                      "mean_absolute_difference": float(difference.mean()),
                      "single_max_absolute_difference": float(single_difference.max()),
                      "single_mean_absolute_difference": float(single_difference.mean()),
                      "mean_cosine_similarity": float(np.mean(np.sum(batched * single, axis=1) /
                          (np.linalg.norm(batched, axis=1) * np.linalg.norm(single, axis=1))))}))


if __name__ == "__main__":
    main()

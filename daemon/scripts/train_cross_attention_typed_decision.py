"""Train/evaluate a Jev-style listwise head on licensed public decisions only.

The frozen Qwen backbone encodes each context and each option independently.
Only the small cross-attention head is trained. Data, features, and weights stay
local under ignored benchmark state; this is research-only and grants no action
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.general_decision import (  # noqa: E402
    DATASET,
    DATASET_REVISION,
    load_split,
    metrics,
    probabilities,
    ranked_probability_score,
    uniform_baseline,
    verify_disjoint,
)
from pilot.intelligence.listwise_decision_head import build_listwise_decision_head  # noqa: E402
from scripts.qwen_model_pin import MODEL, REVISION  # noqa: E402
from scripts.train_packed_typed_decision import _shuffled_context_pairs  # noqa: E402

BACKBONE_DTYPE = "bfloat16"
SCHEMA_VERSION = 2


def _state_value(row: dict) -> object:
    try:
        return json.loads(row["state_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON state for row {row['id']}") from exc


def _context_text(row: dict) -> str:
    """Include task context and question, never candidates, labels, or provenance."""
    state = _state_value(row)
    rendered = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    return f"Decision type: {row['kind']}\nQuestion: {row['question']}\nState:\n{rendered}"


def _pool_context(features: np.ndarray, max_tokens: int) -> np.ndarray:
    """Compress long contexts by contiguous adaptive means while preserving order."""
    if features.ndim != 2 or not len(features) or max_tokens < 1:
        raise ValueError("Context features must be a nonempty token-by-hidden matrix")
    if len(features) <= max_tokens:
        return features.astype(np.float16, copy=False)
    edges = np.linspace(0, len(features), max_tokens + 1, dtype=np.int64)
    return np.stack([features[edges[i] : edges[i + 1]].mean(axis=0) for i in range(max_tokens)]).astype(np.float16)


def _encode_texts(backbone, tokenizer, texts: list[str], *, max_length: int, batch_size: int, device: str):
    """Return independent final-layer token features and lengths; retry smaller batches on OOM."""
    import torch

    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    token_ids = encoded["input_ids"]
    truncated = sum(len(ids) > max_length for ids in token_ids)
    clipped_ids = [ids[:max_length] for ids in token_ids]
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    output = []
    start = 0
    current_batch = batch_size
    while start < len(clipped_ids):
        stop = min(start + current_batch, len(clipped_ids))
        batch = tokenizer.pad(
            {"input_ids": clipped_ids[start:stop]},
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in batch.items()}
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        try:
            with torch.no_grad():
                hidden = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            for index, length in enumerate(lengths):
                output.append(hidden[index, : int(length)].float().cpu().numpy())
            start = stop
            if current_batch < batch_size:
                current_batch = min(batch_size, current_batch * 2)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current_batch == 1:
                raise
            del inputs, batch
            torch.cuda.empty_cache()
            current_batch = max(1, current_batch // 2)
    return output, truncated


def _encode_split(
    backbone,
    tokenizer,
    rows: list[dict],
    *,
    split: str,
    max_length: int,
    max_context_tokens: int,
    batch_size: int,
    device: str,
    on_checkpoint=None,
) -> tuple[list[dict], list[str], int]:
    encoded: list[dict] = []
    skipped: list[str] = []
    truncated = 0
    for chunk_start in range(0, len(rows), 64):
        chunk = rows[chunk_start : chunk_start + 64]
        texts = []
        owners = []
        for row_index, row in enumerate(chunk):
            texts.append(_context_text(row))
            owners.append((row_index, -1))
            for option_index, option in enumerate(row["options"]):
                texts.append(option)
                owners.append((row_index, option_index))

        features, was_truncated = _encode_texts(
            backbone,
            tokenizer,
            texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        truncated += was_truncated
        contexts: dict[int, np.ndarray] = {}
        options: dict[int, dict[int, np.ndarray]] = {}
        for (row_index, option_index), values in zip(owners, features, strict=True):
            if option_index < 0:
                contexts[row_index] = _pool_context(values, max_context_tokens)
            else:
                options.setdefault(row_index, {})[option_index] = values.mean(axis=0).astype(np.float16)

        for row_index, row in enumerate(chunk):
            option_vectors = options.get(row_index, {})
            if row_index not in contexts or len(option_vectors) != len(row["options"]):
                skipped.append(row["id"])
                continue
            encoded.append(
                {
                    "row": row,
                    "context": contexts[row_index],
                    "options": np.stack([option_vectors[i] for i in range(len(row["options"]))]),
                    "target": np.asarray(row["target"], dtype=np.float32),
                    "answer_keys": tuple(str(i) for i in range(len(row["options"]))),
                }
            )
        print(
            json.dumps(
                {
                    "event": "feature_extraction",
                    "split": split,
                    "rows": min(chunk_start + len(chunk), len(rows)),
                    "total": len(rows),
                    "kept": len(encoded),
                    "truncated_sequences": truncated,
                }
            ),
            flush=True,
        )
        if on_checkpoint is not None:
            on_checkpoint(encoded, skipped, truncated)
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    return encoded, skipped, truncated


def _head(hidden_size: int, rank: int):
    return build_listwise_decision_head(hidden_size, rank)


def _loss(logits, target, kind: str, *, brier_weight: float, ordinal_weight: float):
    import torch

    target = torch.as_tensor(target, dtype=torch.float32, device=logits.device)
    predicted = logits.softmax(-1)
    result = -(target * logits.log_softmax(-1)).sum()
    result = result + brier_weight * (predicted - target).square().sum()
    if kind == "score" and ordinal_weight:
        result = result + ordinal_weight * ranked_probability_score(predicted, target)
    return result


def _logits(head, items: list[dict], *, device: str) -> list[np.ndarray]:
    import torch

    head.eval()
    output = []
    with torch.no_grad():
        for item in items:
            context = torch.as_tensor(item["context"], dtype=torch.float32, device=device)
            options = torch.as_tensor(item["options"], dtype=torch.float32, device=device)
            values = head(context, options).float().cpu().numpy()
            score_by_key = dict(zip(item["answer_keys"], values, strict=True))
            output.append(np.asarray([score_by_key[str(i)] for i in range(len(item["row"]["options"]))]))
    return output


def _temperature(logits: list[np.ndarray], rows: list[dict]) -> float:
    def loss(temperature: float) -> float:
        return float(
            np.mean(
                [
                    -np.asarray(row["target"], dtype=np.float64)
                    @ np.log(np.clip(probabilities(values, temperature), 1e-12, 1.0))
                    for row, values in zip(rows, logits, strict=True)
                ]
            )
        )

    result = minimize_scalar(loss, bounds=(0.25, 20.0), method="bounded")
    if not result.success:
        raise RuntimeError("Calibration temperature search did not converge")
    return float(result.x)


def _row_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['id']}\0{row['group_id']}\n".encode())
    return digest.hexdigest()


def _cache_signature(rows: dict[str, list[dict]], manifests: dict, args: argparse.Namespace) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "feature_method": "independent context tokens and option mean vectors",
        "model": MODEL,
        "model_revision": REVISION,
        "backbone_dtype": BACKBONE_DTYPE,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "max_length": args.max_length,
        "max_context_tokens": args.max_context_tokens,
        "manifests": manifests,
        "rows": {name: _row_hash(part) for name, part in rows.items()},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _write_feature_cache(
    path: Path, signature: str, encoded: dict, skipped: dict, truncated: dict, complete: dict, hidden_size: int
) -> str:
    arrays = {}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "signature": signature,
        "hidden_size": hidden_size,
        "complete": complete,
        "splits": {},
    }
    for split, items in encoded.items():
        context_offsets = [0]
        option_offsets = [0]
        contexts = []
        options = []
        targets = []
        for item in items:
            contexts.append(item["context"])
            options.append(item["options"])
            targets.append(item["target"])
            context_offsets.append(context_offsets[-1] + len(item["context"]))
            option_offsets.append(option_offsets[-1] + len(item["options"]))
        arrays[f"{split}_context"] = (
            np.concatenate(contexts).astype(np.float16) if contexts else np.empty((0, hidden_size), dtype=np.float16)
        )
        arrays[f"{split}_options"] = (
            np.concatenate(options).astype(np.float16) if options else np.empty((0, hidden_size), dtype=np.float16)
        )
        arrays[f"{split}_targets"] = np.concatenate(targets).astype(np.float32) if targets else np.empty(0, np.float32)
        arrays[f"{split}_context_offsets"] = np.asarray(context_offsets, dtype=np.int32)
        arrays[f"{split}_option_offsets"] = np.asarray(option_offsets, dtype=np.int32)
        metadata["splits"][split] = {
            "ids": [item["row"]["id"] for item in items],
            "skipped_ids": skipped[split],
            "truncated_sequences": truncated[split],
        }
    arrays["metadata"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_feature_cache(path: Path, signature: str, rows: dict[str, list[dict]]):
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"].item()))
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("signature") != signature:
            raise ValueError("Cross-attention feature cache does not match model, data, and feature schema")
        encoded = {}
        skipped = {}
        truncated = {}
        complete = metadata.get("complete", {})
        hidden_size = metadata.get("hidden_size")
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Cross-attention feature cache has invalid hidden size")
        for split, selected in rows.items():
            entry = metadata["splits"][split]
            by_id = {row["id"]: row for row in selected}
            ids = entry["ids"]
            context_offsets = cached[f"{split}_context_offsets"]
            option_offsets = cached[f"{split}_option_offsets"]
            contexts = cached[f"{split}_context"]
            options = cached[f"{split}_options"]
            targets = cached[f"{split}_targets"]
            skipped[split] = entry["skipped_ids"]
            truncated[split] = entry["truncated_sequences"]
            processed = set(ids) | set(skipped[split])
            selected_ids = {row["id"] for row in selected}
            if (
                len(context_offsets) != len(ids) + 1
                or len(option_offsets) != len(ids) + 1
                or any(row_id not in by_id for row_id in ids)
                or not processed <= selected_ids
                or any(left >= right for left, right in zip(context_offsets[:-1], context_offsets[1:], strict=True))
                or any(left + 1 >= right for left, right in zip(option_offsets[:-1], option_offsets[1:], strict=True))
                or contexts.ndim != 2
                or options.ndim != 2
                or contexts.shape[1] != hidden_size
                or options.shape[1] != hidden_size
                or context_offsets[-1] != len(contexts)
                or option_offsets[-1] != len(options)
                or len(targets) != len(options)
                or not isinstance(complete.get(split), bool)
            ):
                raise ValueError(f"Cross-attention feature cache is malformed for {split}")
            encoded[split] = [
                {
                    "row": by_id[row_id],
                    "context": contexts[context_offsets[i] : context_offsets[i + 1]].astype(np.float32),
                    "options": options[option_offsets[i] : option_offsets[i + 1]].astype(np.float32),
                    "target": targets[option_offsets[i] : option_offsets[i + 1]].astype(np.float32),
                    "answer_keys": tuple(str(j) for j in range(option_offsets[i + 1] - option_offsets[i])),
                }
                for i, row_id in enumerate(ids)
            ]
    return encoded, skipped, truncated, complete, hidden_size


def _batch_tensors(items: list[dict], device: str):
    import torch

    hidden_size = int(items[0]["options"].shape[1])
    context_size = max(len(item["context"]) for item in items)
    option_size = max(len(item["options"]) for item in items)
    contexts = torch.zeros((len(items), context_size, hidden_size), dtype=torch.float32, device=device)
    context_mask = torch.zeros((len(items), context_size), dtype=torch.bool, device=device)
    options = torch.zeros((len(items), option_size, hidden_size), dtype=torch.float32, device=device)
    option_mask = torch.zeros((len(items), option_size), dtype=torch.bool, device=device)
    targets = torch.zeros((len(items), option_size), dtype=torch.float32, device=device)
    for index, item in enumerate(items):
        context_count = len(item["context"])
        option_count = len(item["options"])
        contexts[index, :context_count] = torch.as_tensor(item["context"], dtype=torch.float32, device=device)
        options[index, :option_count] = torch.as_tensor(item["options"], dtype=torch.float32, device=device)
        context_mask[index, :context_count] = True
        option_mask[index, :option_count] = True
        targets[index, :option_count] = torch.as_tensor(item["target"], dtype=torch.float32, device=device)
    return contexts, context_mask, options, option_mask, targets


def _shuffled_context_control(
    args, head, items: list[dict], original_logits: list[np.ndarray], temperature: float, device: str
) -> dict:
    pairs = _shuffled_context_pairs([item["row"] for item in items], args.seed + 1)
    if not pairs:
        raise RuntimeError("No valid same-kind, cross-group contexts are available to shuffle")
    item_by_id = {item["row"]["id"]: item for item in items}
    shuffled_items = []
    rows = []
    for row, donor in pairs:
        original = item_by_id[row["id"]]
        shuffled_items.append({**original, "context": item_by_id[donor["id"]]["context"]})
        rows.append(row)
    shuffled_logits = _logits(head, shuffled_items, device=device)
    original_by_id = {item["row"]["id"]: values for item, values in zip(items, original_logits, strict=True)}
    paired_original = [original_by_id[row["id"]] for row in rows]
    return {
        "seed": args.seed + 1,
        "pairing": "donor context includes donor state and question; same kind, different row, group, and canonical state",
        "requested_pairs": len(pairs),
        "scored_rows": len(rows),
        "original_context": metrics(rows, paired_original, temperature),
        "shuffled_context": metrics(rows, shuffled_logits, temperature),
        "uniform": uniform_baseline(rows),
    }


def run(args: argparse.Namespace) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 3 * 1024**3:
        raise RuntimeError("Cross-attention probe requires at least 3 GiB free CUDA memory")
    if (
        args.train_rows < 128
        or args.calibration_rows < 32
        or args.test_rows < 32
        or args.ood_rows < 32
        or args.epochs < 1
        or args.max_length < 64
        or args.max_context_tokens < 8
        or args.feature_batch_size < 1
        or args.rank < 1
    ):
        raise ValueError("Use adequate split sizes and positive feature/head limits")
    if args.output_dir.exists():
        raise FileExistsError(f"Research output already exists: {args.output_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    limits = {
        "train": args.train_rows,
        "calibration": args.calibration_rows,
        "test": args.test_rows,
        "ood": args.ood_rows,
    }
    rows = {}
    manifests = {}
    for offset, (split, limit) in enumerate(limits.items()):
        rows[split], manifests[split] = load_split(split, limit, args.seed + offset)
    verify_disjoint(rows)

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    feature_signature = _cache_signature(rows, manifests, args)
    encoded = {split: [] for split in rows}
    skipped = {split: [] for split in rows}
    truncated = {split: 0 for split in rows}
    complete = {split: False for split in rows}
    hidden_size = 0
    if args.feature_cache.is_file():
        encoded, skipped, truncated, complete, hidden_size = _read_feature_cache(
            args.feature_cache, feature_signature, rows
        )
        cache_sha256 = hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()
        print(
            json.dumps({"event": "feature_cache_hit", "path": str(args.feature_cache), "sha256": cache_sha256}),
            flush=True,
        )
    elif args.feature_cache.exists():
        raise ValueError(f"Feature cache path is not a file: {args.feature_cache}")

    backbone = tokenizer = None
    if not all(complete.values()):
        model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        ).eval()
        backbone.requires_grad_(False)
        actual_hidden_size = int(backbone.config.hidden_size)
        if hidden_size not in {0, actual_hidden_size}:
            raise ValueError("Feature cache hidden size differs from the pinned backbone")
        hidden_size = actual_hidden_size
        for split, selected in rows.items():
            if complete[split]:
                continue
            processed_ids = {item["row"]["id"] for item in encoded[split]} | set(skipped[split])
            remaining = [row for row in selected if row["id"] not in processed_ids]

            def checkpoint(partial_encoded, partial_skipped, partial_truncated, *, split_name=split):
                cache_encoded = {name: list(items) for name, items in encoded.items()}
                cache_skipped = {name: list(items) for name, items in skipped.items()}
                cache_truncated = dict(truncated)
                cache_encoded[split_name] = list(encoded[split_name]) + partial_encoded
                cache_skipped[split_name] = list(skipped[split_name]) + partial_skipped
                cache_truncated[split_name] += partial_truncated
                _write_feature_cache(
                    args.feature_cache,
                    feature_signature,
                    cache_encoded,
                    cache_skipped,
                    cache_truncated,
                    complete,
                    hidden_size,
                )

            new_encoded, new_skipped, new_truncated = _encode_split(
                backbone,
                tokenizer,
                remaining,
                split=split,
                max_length=args.max_length,
                max_context_tokens=args.max_context_tokens,
                batch_size=args.feature_batch_size,
                device="cuda:0",
                on_checkpoint=checkpoint,
            )
            encoded[split].extend(new_encoded)
            skipped[split].extend(new_skipped)
            truncated[split] += new_truncated
            complete[split] = True
            cache_sha256 = _write_feature_cache(
                args.feature_cache, feature_signature, encoded, skipped, truncated, complete, hidden_size
            )
        if backbone is not None:
            del backbone, tokenizer
            torch.cuda.empty_cache()
            backbone = tokenizer = None
    else:
        cache_sha256 = hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()

    for split in rows:
        minimum = 128 if split == "train" else 32
        if len(encoded[split]) < minimum:
            raise RuntimeError(f"Only {len(encoded[split])} usable {split} rows remain")

    head = _head(hidden_size, args.rank).to("cuda:0")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    fit_indices = list(range(len(encoded["train"])))
    best_state = None
    best_calibration_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        random.Random(args.seed + epoch).shuffle(fit_indices)
        epoch_losses = []
        for start in range(0, len(fit_indices), args.batch_size):
            batch_items = [encoded["train"][index] for index in fit_indices[start : start + args.batch_size]]
            contexts, context_mask, options, option_mask, targets = _batch_tensors(batch_items, "cuda:0")
            optimizer.zero_grad(set_to_none=True)
            logits = head(contexts, options, context_mask, option_mask)
            per_row_loss = -(targets * logits.log_softmax(-1)).sum(-1)
            loss = per_row_loss.mean()
            if args.brier_weight:
                brier = ((logits.softmax(-1) - targets).square() * option_mask).sum(-1).mean()
                loss = loss + args.brier_weight * brier
            if args.ordinal_weight:
                predicted = logits.softmax(-1)
                row_rps = []
                for index, item in enumerate(batch_items):
                    if item["row"]["kind"] == "score":
                        count = len(item["target"])
                        row_rps.append(ranked_probability_score(predicted[index, :count], targets[index, :count]))
                if row_rps:
                    loss = loss + args.ordinal_weight * torch.stack(row_rps).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite cross-attention loss in epoch {epoch}, batch {start}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        calibration_logits = _logits(head, encoded["calibration"], device="cuda:0")
        calibration_nll = float(
            np.mean(
                [
                    -np.asarray(item["row"]["target"], dtype=np.float64)
                    @ np.log(np.clip(probabilities(values, 1.0), 1e-12, 1.0))
                    for item, values in zip(encoded["calibration"], calibration_logits, strict=True)
                ]
            )
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(epoch_losses)),
                "calibration_nll_uncalibrated": calibration_nll,
            }
        )
        print(json.dumps({"event": "epoch", **history[-1]}), flush=True)
        if calibration_nll < best_calibration_nll:
            best_calibration_nll = calibration_nll
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No finite cross-attention checkpoint was selected")
    head.load_state_dict(best_state)
    calibration_logits = _logits(head, encoded["calibration"], device="cuda:0")
    temperature = _temperature(calibration_logits, [item["row"] for item in encoded["calibration"]])
    test_logits = _logits(head, encoded["test"], device="cuda:0")
    ood_logits = _logits(head, encoded["ood"], device="cuda:0")
    if args.shuffled_context_control:
        shuffled_context = _shuffled_context_control(args, head, encoded["ood"], ood_logits, temperature, "cuda:0")
    else:
        shuffled_context = None

    test_rows = [item["row"] for item in encoded["test"]]
    ood_rows = [item["row"] for item in encoded["ood"]]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = args.output_dir / "cross_attention_head.npz"
    np.savez_compressed(
        weights_path,
        **{key: value.detach().cpu().numpy() for key, value in head.state_dict().items()},
        temperature=np.asarray(temperature, dtype=np.float64),
        model=MODEL,
        revision=REVISION,
        rank=args.rank,
        max_length=args.max_length,
        max_context_tokens=args.max_context_tokens,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "model_revision": REVISION,
        "backbone_dtype": BACKBONE_DTYPE,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "training_method": "frozen Qwen3 + independent features + batched listwise cross-attention head",
        "head_reference": "Open-Jev AttentionHead pattern; adapted from MIT-licensed source",
        "feature_method": "context retains ordered token features (adaptive means above configured budget); options are independent token means",
        "loss": "soft listwise cross entropy + weighted Brier + ordered RPS",
        "seed": args.seed,
        "max_sequence_length": args.max_length,
        "max_context_tokens": args.max_context_tokens,
        "rank": args.rank,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": temperature,
        "train_rows": len(encoded["train"]),
        "calibration_rows": len(encoded["calibration"]),
        "test_rows": len(encoded["test"]),
        "ood_rows": len(encoded["ood"]),
        "skipped_ids": skipped,
        "truncated_feature_sequences": truncated,
        "feature_cache_sha256": cache_sha256,
        "split_manifests": manifests,
        "split_id_group_sha256": {name: _row_hash(part) for name, part in rows.items()},
        "uniform_test": uniform_baseline(test_rows),
        "trained_test": metrics(test_rows, test_logits, temperature),
        "uniform_ood": uniform_baseline(ood_rows),
        "trained_ood": metrics(ood_rows, ood_logits, temperature),
        "shuffled_context_control": shuffled_context,
        "head_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "limitation": (
            "Licensed public decision rows only. This is not parity with Jev/TypeSafe, desktop-task quality, "
            "safety evidence, production authorization, or execution authority."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/.bench-state/cross_attention_v2")
    parser.add_argument(
        "--feature-cache", type=Path, default=ROOT / "benchmarks/.bench-state/cross_attention_features_v1.npz"
    )
    parser.add_argument("--train-rows", type=int, default=960)
    parser.add_argument("--calibration-rows", type=int, default=188)
    parser.add_argument("--test-rows", type=int, default=121)
    parser.add_argument("--ood-rows", type=int, default=116)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--brier-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--shuffled-context-control",
        action="store_true",
        help="Re-encode independent donor contexts for OOD rows and compare with uniform.",
    )
    args = parser.parse_args()
    if args.patience < 1:
        raise ValueError("Patience must be positive")
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "feature_cache_sha256": report["feature_cache_sha256"],
                "best_epoch": report["best_epoch"],
                "temperature": report["temperature"],
                "test_top1": report["trained_test"]["top1_accuracy"],
                "test_nll": report["trained_test"]["mean_nll"],
                "ood_top1": report["trained_ood"]["top1_accuracy"],
                "ood_nll": report["trained_ood"]["mean_nll"],
                "shuffled_context": report["shuffled_context_control"],
                "peak_cuda_gib": report["peak_cuda_gib"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

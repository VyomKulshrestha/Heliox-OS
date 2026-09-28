"""Train a small packed query/candidate head on licensed public decisions.

The Qwen3 backbone is frozen. This research-only probe uses an isolated tree
mask and a shared bilinear pointer head; it never loads or trains on Jev outputs.
All model and dataset reads are local-only, and artifacts default to ignored
benchmark state.
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
from pilot.intelligence.packed_typed_decision import (  # noqa: E402
    pack_typed_questions,
)
from scripts.qwen_model_pin import MODEL, REVISION  # noqa: E402


def _state_value(row: dict) -> object:
    try:
        return json.loads(row["state_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON state for row {row['id']}") from exc


def _features(backbone, tokenizer, row: dict, *, max_length: int, device: str) -> dict:
    import torch

    keys = [str(index) for index in range(len(row["options"]))]
    record = {
        "id": row["id"],
        "kind": row["kind"],
        "question": row["question"],
        "options": row["options"],
        "answer_keys": keys,
    }
    packed = pack_typed_questions(tokenizer, _state_value(row), [record], max_sequence_length=max_length)
    input_ids = torch.tensor([packed.input_ids], dtype=torch.long, device=device)
    position_ids = torch.tensor([packed.position_ids], dtype=torch.long, device=device)
    allowed = torch.as_tensor(packed.attention_mask, dtype=torch.bool, device=device)
    attention_mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]
    with torch.no_grad():
        hidden = backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0]

    question = packed.questions[0]
    state_vector = hidden[: packed.state_token_count].float().mean(0).cpu().numpy()
    question_vector = hidden[list(question.question_indices)].float().mean(0).cpu().numpy()
    option_vectors = np.stack([hidden[list(span)].float().mean(0).cpu().numpy() for span in question.option_spans])
    target_by_key = {key: float(value) for key, value in zip(keys, row["target"], strict=True)}
    target_sorted = np.asarray([target_by_key[key] for key in question.answer_keys], dtype=np.float32)
    return {
        "row": row,
        "state": state_vector,
        "question": question_vector,
        "options": option_vectors,
        "target": target_sorted,
        "answer_keys": question.answer_keys,
    }


def _encode_split(
    backbone,
    tokenizer,
    rows: list[dict],
    *,
    split: str,
    max_length: int,
    device: str,
    on_checkpoint=None,
) -> tuple[list[dict], list[str]]:
    encoded = []
    skipped = []
    for index, row in enumerate(rows, 1):
        try:
            encoded.append(_features(backbone, tokenizer, row, max_length=max_length, device=device))
        except ValueError as exc:
            if "model limit" not in str(exc):
                raise
            skipped.append(row["id"])
        if index % 64 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "event": "feature_extraction",
                        "split": split,
                        "rows": index,
                        "total": len(rows),
                        "kept": len(encoded),
                        "over_length": len(skipped),
                    }
                ),
                flush=True,
            )
            if on_checkpoint is not None:
                on_checkpoint(encoded, skipped)
    return encoded, skipped


def _head(hidden_size: int, rank: int = 512):
    import torch.nn as nn

    class PackedPointerHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.context_projection = nn.Linear(hidden_size, rank, bias=False)
            self.option_projection = nn.Linear(hidden_size, rank, bias=False)
            self.rank = rank

        def forward(self, item: dict, device: str):
            import torch

            context = torch.as_tensor(item["state"], dtype=torch.float32, device=device)
            question = torch.as_tensor(item["question"], dtype=torch.float32, device=device)
            options = torch.as_tensor(item["options"], dtype=torch.float32, device=device)
            context = self.norm(context)
            question = self.norm(question)
            options = self.norm(options)
            query = self.context_projection(context) + self.context_projection(question)
            keys = self.option_projection(options)
            return (keys @ query) / self.rank**0.5

    return PackedPointerHead()


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
            values = head(item, device).float().cpu().numpy()
            score_by_key = dict(zip(item["answer_keys"], values, strict=True))
            original = np.asarray(
                [score_by_key[str(index)] for index in range(len(item["row"]["options"]))],
                dtype=np.float64,
            )
            output.append(original)
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


def _cache_signature(rows: dict[str, list[dict]], manifests: dict, max_length: int) -> str:
    identity = {
        "model": MODEL,
        "model_revision": REVISION,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "max_length": max_length,
        "manifests": manifests,
        "rows": {name: _row_hash(part) for name, part in rows.items()},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _write_feature_cache(
    path: Path, signature: str, encoded: dict, skipped: dict, complete: dict, hidden_size: int
) -> str:
    arrays = {}
    metadata = {
        "schema_version": 1,
        "signature": signature,
        "hidden_size": hidden_size,
        "complete": complete,
        "splits": {},
    }
    for split, items in encoded.items():
        offsets = [0]
        option_vectors = []
        targets = []
        for item in items:
            option_vectors.append(item["options"])
            targets.append(item["target"])
            offsets.append(offsets[-1] + len(item["options"]))
        arrays[f"{split}_state"] = (
            np.stack([item["state"] for item in items]).astype(np.float16)
            if items
            else np.empty((0, hidden_size), dtype=np.float16)
        )
        arrays[f"{split}_question"] = (
            np.stack([item["question"] for item in items]).astype(np.float16)
            if items
            else np.empty((0, hidden_size), dtype=np.float16)
        )
        arrays[f"{split}_options"] = (
            np.concatenate(option_vectors).astype(np.float16)
            if option_vectors
            else np.empty((0, hidden_size), dtype=np.float16)
        )
        arrays[f"{split}_targets"] = (
            np.concatenate(targets).astype(np.float32) if targets else np.empty((0,), dtype=np.float32)
        )
        arrays[f"{split}_offsets"] = np.asarray(offsets, dtype=np.int32)
        metadata["splits"][split] = {
            "ids": [item["row"]["id"] for item in items],
            "answer_keys": [list(item["answer_keys"]) for item in items],
            "skipped_ids": skipped[split],
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
        if metadata.get("schema_version") != 1 or metadata.get("signature") != signature:
            raise ValueError("Packed feature cache does not match model, data, and sequence limit")
        encoded = {}
        skipped = {}
        complete = metadata.get("complete", {})
        hidden_size = metadata.get("hidden_size")
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Packed feature cache has an invalid hidden size")
        for split, selected in rows.items():
            entry = metadata["splits"][split]
            by_id = {row["id"]: row for row in selected}
            ids = entry["ids"]
            answer_keys = entry["answer_keys"]
            offsets = cached[f"{split}_offsets"]
            processed = set(ids) | set(entry["skipped_ids"])
            selected_ids = {row["id"] for row in selected}
            if (
                len(ids) != len(answer_keys)
                or len(offsets) != len(ids) + 1
                or any(row_id not in by_id for row_id in ids)
                or not processed <= selected_ids
                or any(left >= right for left, right in zip(offsets[:-1], offsets[1:], strict=True))
            ):
                raise ValueError(f"Packed feature cache has malformed {split} rows")
            state = cached[f"{split}_state"]
            question = cached[f"{split}_question"]
            options = cached[f"{split}_options"]
            targets = cached[f"{split}_targets"]
            if (
                state.shape != (len(ids), hidden_size)
                or question.shape != state.shape
                or options.ndim != 2
                or options.shape[1] != hidden_size
                or offsets[-1] != len(options)
                or len(targets) != len(options)
            ):
                raise ValueError(f"Packed feature cache arrays are inconsistent for {split}")
            encoded[split] = [
                {
                    "row": by_id[row_id],
                    "state": state[index].astype(np.float32),
                    "question": question[index].astype(np.float32),
                    "options": options[offsets[index] : offsets[index + 1]].astype(np.float32),
                    "target": targets[offsets[index] : offsets[index + 1]].astype(np.float32),
                    "answer_keys": tuple(answer_keys[index]),
                }
                for index, row_id in enumerate(ids)
            ]
            skipped[split] = entry["skipped_ids"]
            if not isinstance(complete.get(split), bool):
                raise ValueError(f"Packed feature cache is missing {split} completion state")
    return encoded, skipped, complete, hidden_size


def run(args: argparse.Namespace) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 3 * 1024**3:
        raise RuntimeError("Packed probe requires at least 3 GiB free CUDA memory")
    if (
        args.train_rows < 128
        or args.calibration_rows < 32
        or args.test_rows < 32
        or args.ood_rows < 32
        or args.epochs < 1
        or args.max_length < 64
    ):
        raise ValueError("Use at least 128/32/32/32 rows and valid epochs/sequence length")
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
    feature_signature = _cache_signature(rows, manifests, args.max_length)
    encoded = {split: [] for split in rows}
    skipped = {split: [] for split in rows}
    complete = {split: False for split in rows}
    hidden_size = 0
    if args.feature_cache.is_file():
        encoded, skipped, complete, hidden_size = _read_feature_cache(args.feature_cache, feature_signature, rows)
        cache_sha256 = hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "event": "feature_cache_hit",
                    "path": str(args.feature_cache),
                    "sha256": cache_sha256,
                }
            ),
            flush=True,
        )
    elif args.feature_cache.exists():
        raise ValueError(f"Feature cache path is not a file: {args.feature_cache}")

    if not all(complete.values()):
        model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float16,
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

            def checkpoint(partial_encoded, partial_skipped, *, split_name=split):
                cache_encoded = {name: list(items) for name, items in encoded.items()}
                cache_skipped = {name: list(items) for name, items in skipped.items()}
                cache_encoded[split_name].extend(partial_encoded)
                cache_skipped[split_name].extend(partial_skipped)
                _write_feature_cache(
                    args.feature_cache,
                    feature_signature,
                    cache_encoded,
                    cache_skipped,
                    complete,
                    hidden_size,
                )

            new_encoded, new_skipped = _encode_split(
                backbone,
                tokenizer,
                remaining,
                split=split,
                max_length=args.max_length,
                device="cuda:0",
                on_checkpoint=checkpoint,
            )
            encoded[split].extend(new_encoded)
            skipped[split].extend(new_skipped)
            complete[split] = True
            cache_sha256 = _write_feature_cache(
                args.feature_cache, feature_signature, encoded, skipped, complete, hidden_size
            )
        del backbone, tokenizer
        torch.cuda.empty_cache()
    elif args.feature_cache.is_file():
        cache_sha256 = hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()

    for split in rows:
        minimum = 128 if split == "train" else 32
        if len(encoded[split]) < minimum:
            raise RuntimeError(f"Only {len(encoded[split])} usable {split} rows remain")

    head = _head(hidden_size).to("cuda:0")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
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
        for index in fit_indices:
            item = encoded["train"][index]
            optimizer.zero_grad(set_to_none=True)
            logits = head(item, "cuda:0")
            loss = _loss(
                logits,
                item["target"],
                item["row"]["kind"],
                brier_weight=args.brier_weight,
                ordinal_weight=args.ordinal_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite pointer-head loss at row {item['row']['id']}")
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
        raise RuntimeError("No finite packed pointer-head checkpoint was selected")
    head.load_state_dict(best_state)
    calibration_logits = _logits(head, encoded["calibration"], device="cuda:0")
    temperature = _temperature(calibration_logits, [item["row"] for item in encoded["calibration"]])
    test_logits = _logits(head, encoded["test"], device="cuda:0")
    ood_logits = _logits(head, encoded["ood"], device="cuda:0")
    test_rows = [item["row"] for item in encoded["test"]]
    ood_rows = [item["row"] for item in encoded["ood"]]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = args.output_dir / "pointer_head.npz"
    np.savez_compressed(
        weights_path,
        norm_weight=head.norm.weight.detach().cpu().numpy(),
        norm_bias=head.norm.bias.detach().cpu().numpy(),
        context_projection=head.context_projection.weight.detach().cpu().numpy(),
        option_projection=head.option_projection.weight.detach().cpu().numpy(),
        temperature=np.asarray(temperature, dtype=np.float64),
        model=MODEL,
        revision=REVISION,
        rank=head.rank,
        max_length=args.max_length,
    )
    report = {
        "schema_version": 1,
        "model": MODEL,
        "model_revision": REVISION,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "training_method": "frozen Qwen3-0.6B + packed tree mask + shared bilinear pointer head",
        "attention": "state->state; question->state+self; candidate->state+its-question+self",
        "normalization": "shared learned layer norm before pooling-vector projections",
        "readout": "mean-pooled question query and candidate key; shared rank-512 dot product",
        "loss": "soft cross entropy + weighted Brier + ordered RPS",
        "seed": args.seed,
        "max_sequence_length": args.max_length,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "brier_weight": args.brier_weight,
        "ordinal_weight": args.ordinal_weight,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": temperature,
        "train_rows": len(encoded["train"]),
        "calibration_rows": len(encoded["calibration"]),
        "test_rows": len(encoded["test"]),
        "ood_rows": len(encoded["ood"]),
        "over_length_rows": skipped,
        "feature_cache_sha256": cache_sha256,
        "split_manifests": manifests,
        "split_id_group_sha256": {name: _row_hash(part) for name, part in rows.items()},
        "uniform_test": uniform_baseline(test_rows),
        "trained_test": metrics(test_rows, test_logits, temperature),
        "uniform_ood": uniform_baseline(ood_rows),
        "trained_ood": metrics(ood_rows, ood_logits, temperature),
        "pointer_head_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "limitation": (
            "Licensed synthetic public decisions only; frozen small backbone and a small head. "
            "Not Jev parity, desktop task quality, safety evidence, or execution authority."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/.bench-state/packed_pointer_v1")
    parser.add_argument(
        "--feature-cache", type=Path, default=ROOT / "benchmarks/.bench-state/packed_pointer_features_v1.npz"
    )
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--calibration-rows", type=int, default=96)
    parser.add_argument("--test-rows", type=int, default=128)
    parser.add_argument("--ood-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--ordinal-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260928)
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
                "peak_cuda_gib": report["peak_cuda_gib"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

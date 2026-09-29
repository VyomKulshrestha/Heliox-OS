"""Evaluate a compact Jev-like byte scorer on the same licensed public splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.general_decision import (  # noqa: E402
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    TRAINING_SOURCE_POLICY,
    load_clean_source_counts,
    load_split,
    metrics,
    probabilities,
    select_source_holdout,
    verify_disjoint,
    verify_source_holdout,
)
from pilot.intelligence.tiny_decision_scorer import build_tiny_decision_scorer  # noqa: E402
from scripts.train_cross_attention_typed_decision import (  # noqa: E402
    _context_text,
    _row_hash,
    _shuffled_context_pairs,
    _temperature,
)


def _bytes(text: str, limit: int) -> list[int]:
    return [value + 1 for value in text.encode("utf-8", errors="replace")[:limit]]


def _batch(rows: list[dict], context_rows: list[dict] | None, args: argparse.Namespace, device: str):
    import torch

    context_rows = rows if context_rows is None else context_rows
    context_ids = [_bytes(_context_text(row), args.context_bytes) for row in context_rows]
    option_ids = [[_bytes(option, args.option_bytes) for option in row["options"]] for row in rows]
    context_width = max(map(len, context_ids))
    option_count = max(map(len, option_ids))
    option_width = max(len(tokens) for options in option_ids for tokens in options)
    contexts = torch.zeros((len(rows), context_width), dtype=torch.long, device=device)
    context_mask = torch.zeros_like(contexts, dtype=torch.bool)
    options = torch.zeros((len(rows), option_count, option_width), dtype=torch.long, device=device)
    option_token_mask = torch.zeros_like(options, dtype=torch.bool)
    option_mask = torch.zeros((len(rows), option_count), dtype=torch.bool, device=device)
    targets = torch.zeros((len(rows), option_count), dtype=torch.float32, device=device)
    for index, context in enumerate(context_ids):
        contexts[index, : len(context)] = torch.as_tensor(context, device=device)
        context_mask[index, : len(context)] = True
        option_mask[index, : len(option_ids[index])] = True
        row_target = rows[index]["target"]
        targets[index, : len(row_target)] = torch.as_tensor(row_target, device=device)
        for option_index, tokens in enumerate(option_ids[index]):
            options[index, option_index, : len(tokens)] = torch.as_tensor(tokens, device=device)
            option_token_mask[index, option_index, : len(tokens)] = True
    return contexts, context_mask, options, option_token_mask, option_mask, targets


def _logits(
    model, rows: list[dict], args: argparse.Namespace, device: str, context_rows: list[dict] | None = None
) -> list[np.ndarray]:
    import torch

    context_rows = rows if context_rows is None else context_rows
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start : start + args.batch_size]
            contexts = context_rows[start : start + args.batch_size]
            batch = _batch(selected, contexts, args, device)
            result = model(*batch[:-1])
            output.extend(
                result[index, : len(row["options"])].float().cpu().numpy() for index, row in enumerate(selected)
            )
    return output


def _jev_metrics(rows: list[dict], logits: list[np.ndarray], temperature: float) -> dict:
    base = metrics(rows, logits, temperature)
    probabilities_by_row = [probabilities(values, temperature) for values in logits]
    uniform_output = all(len(values) > 0 and np.allclose(values, values[0], rtol=0.0, atol=1e-12) for values in logits)

    def chance_top1(selected_rows: list[dict]) -> float:
        return float(np.mean([1.0 / len(row["options"]) for row in selected_rows]))

    if uniform_output:
        base["top1_accuracy"] = chance_top1(rows)
        for key, group_rows in {
            "by_kind": {kind: [row for row in rows if row["kind"] == kind] for kind in base["by_kind"]},
            "by_source": {source: [row for row in rows if row["source"] == source] for source in base["by_source"]},
        }.items():
            for group, selected_rows in group_rows.items():
                base[key][group]["top1_accuracy"] = chance_top1(selected_rows)
    correct = []
    top3 = []
    calibration_bins = [[0, 0.0, 0.0] for _ in range(10)]
    for row, predicted in zip(rows, probabilities_by_row, strict=True):
        gold = int(np.argmax(row["target"]))
        order = np.argsort(-predicted)
        hit = 1.0 / len(row["options"]) if uniform_output else float(int(order[0]) == gold)
        correct.append(hit)
        top3.append(min(3, len(row["options"])) / len(row["options"]) if uniform_output else gold in order[:3])
        confidence = float(np.max(predicted))
        bucket = min(9, int(confidence * 10))
        calibration_bins[bucket][0] += 1
        calibration_bins[bucket][1] += confidence
        calibration_bins[bucket][2] += float(hit)
    nonempty_bins = [
        {
            "count": count,
            "mean_confidence": float(confidence / count),
            "accuracy": float(accuracy / count),
        }
        for count, confidence, accuracy in calibration_bins
        if count
    ]
    ece = sum(
        (count / len(rows)) * abs(confidence / count - accuracy / count)
        for count, confidence, accuracy in calibration_bins
        if count
    )
    return {
        **base,
        "top3_accuracy": float(np.mean(top3)),
        "top1_accuracy_interpretation": "uniform_expected" if uniform_output else "argmax",
        "ece_10bin": float(ece),
        "calibration_bins": nonempty_bins,
    }


def _truncation(rows: list[dict], args: argparse.Namespace) -> dict:
    contexts = [len(_context_text(row).encode("utf-8", errors="replace")) for row in rows]
    options = [len(option.encode("utf-8", errors="replace")) for row in rows for option in row["options"]]
    return {
        "context_rows_truncated": sum(length > args.context_bytes for length in contexts),
        "option_sequences_truncated": sum(length > args.option_bytes for length in options),
        "context_rows": len(contexts),
        "option_sequences": len(options),
    }


def _control(
    model, rows: list[dict], logits: list[np.ndarray], temperature: float, args: argparse.Namespace, device: str
) -> dict:
    pairs = _shuffled_context_pairs(rows, args.seed + 1)
    if not pairs:
        raise RuntimeError("No distinct same-kind cross-group context pairs are available")
    selected_rows = [row for row, _ in pairs]
    donor_rows = [donor for _, donor in pairs]
    shuffled = _logits(model, selected_rows, args, device, context_rows=donor_rows)
    by_id = {row["id"]: values for row, values in zip(rows, logits, strict=True)}
    original = [by_id[row["id"]] for row in selected_rows]
    return {
        "requested_pairs": len(pairs),
        "scored_rows": len(selected_rows),
        "pairing": "complete donor context (state plus question); same kind, different group and state",
        "correct_context": _jev_metrics(selected_rows, original, temperature),
        "shuffled_context": _jev_metrics(selected_rows, shuffled, temperature),
        "uniform": _jev_metrics(selected_rows, [np.zeros(len(row["options"])) for row in selected_rows], 1.0),
    }


def run(args: argparse.Namespace) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 1 * 1024**3:
        raise RuntimeError("Tiny decision scorer requires at least 1 GiB free CUDA memory")
    if min(args.context_bytes, args.option_bytes, args.width, args.rank, args.batch_size, args.epochs) < 1:
        raise ValueError("model dimensions, batch size, and epochs must be positive")
    if min(args.train_rows, args.calibration_rows, args.test_rows, args.ood_rows) < 32:
        raise ValueError("each data split must contain at least 32 rows")
    if args.output_dir.exists():
        raise FileExistsError(f"Research output already exists: {args.output_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    split_limits = {
        "train": args.train_rows,
        "calibration": args.calibration_rows,
        "test": args.test_rows,
        "ood": args.ood_rows,
    }
    split_seed = args.seed if args.split_seed is None else args.split_seed
    source_holdout = None
    if args.source_holdout_fraction > 0:
        source_holdout_seed = (split_seed if args.source_holdout_seed is None
                               else args.source_holdout_seed)
        source_holdout = select_source_holdout(
            load_clean_source_counts(), args.source_holdout_fraction, source_holdout_seed)
    rows, manifests = {}, {}
    for offset, (split, limit) in enumerate(split_limits.items()):
        allowed_sources = None
        if source_holdout is not None:
            allowed = (source_holdout["held_out_sources"] if split == "ood"
                       else source_holdout["seen_sources"])
            allowed_sources = set(allowed)
        rows[split], manifests[split] = load_split(
            split, limit, split_seed + offset, allowed_sources=allowed_sources)
    verify_disjoint(rows)
    if source_holdout is not None:
        verify_source_holdout(rows, set(source_holdout["held_out_sources"]))

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda:0"
    model = build_tiny_decision_scorer(width=args.width, rank=args.rank, context_tokens=args.context_bytes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    fit_indices = list(range(len(rows["train"])))
    best_state, best_nll, best_epoch, stale_epochs = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(fit_indices)
        losses = []
        for start in range(0, len(fit_indices), args.batch_size):
            batch_rows = [rows["train"][index] for index in fit_indices[start : start + args.batch_size]]
            inputs = _batch(batch_rows, None, args, device)
            targets = inputs[-1]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs[:-1])
            loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite tiny-scorer loss in epoch {epoch} batch {start}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        calibration_logits = _logits(model, rows["calibration"], args, device)
        calibration_nll = float(
            np.mean(
                [
                    -np.asarray(row["target"], dtype=np.float64) @ np.log(np.clip(probabilities(values), 1e-12, 1.0))
                    for row, values in zip(rows["calibration"], calibration_logits, strict=True)
                ]
            )
        )
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "calibration_nll": calibration_nll})
        print(json.dumps({"event": "epoch", **history[-1]}), flush=True)
        if calibration_nll < best_nll:
            best_nll = calibration_nll
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No finite tiny-scorer checkpoint was selected")
    model.load_state_dict(best_state)
    calibration_logits = _logits(model, rows["calibration"], args, device)
    temperature = _temperature(calibration_logits, rows["calibration"])
    test_logits = _logits(model, rows["test"], args, device)
    ood_logits = _logits(model, rows["ood"], args, device)
    control = _control(model, rows["ood"], ood_logits, temperature, args, device)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = args.output_dir / "tiny_scorer.npz"
    np.savez_compressed(
        weights_path,
        **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
        width=args.width,
        rank=args.rank,
        context_bytes=args.context_bytes,
        option_bytes=args.option_bytes,
        temperature=temperature,
    )
    report = {
        "schema_version": 2,
        "method": "trainable UTF-8 byte embeddings + position embeddings + listwise cross-attention; no pretrained backbone",
        "reference": "Minimal Labs jevlike TinyScorer and AttentionHead, adapted from MIT-licensed source",
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "training_source_policy": TRAINING_SOURCE_POLICY,
        "seed": args.seed,
        "split_seed": split_seed,
        "source_holdout_seed": (None if source_holdout is None else source_holdout["seed"]),
        "source_holdout": source_holdout,
        "width": args.width,
        "rank": args.rank,
        "context_byte_limit": args.context_bytes,
        "option_byte_limit": args.option_bytes,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": temperature,
        "split_rows": {name: len(part) for name, part in rows.items()},
        "split_id_group_sha256": {name: _row_hash(part) for name, part in rows.items()},
        "split_manifests": manifests,
        "uniform_test": _jev_metrics(rows["test"], [np.zeros(len(row["options"])) for row in rows["test"]], 1.0),
        "trained_test": _jev_metrics(rows["test"], test_logits, temperature),
        "uniform_ood": _jev_metrics(rows["ood"], [np.zeros(len(row["options"])) for row in rows["ood"]], 1.0),
        "trained_ood": _jev_metrics(rows["ood"], ood_logits, temperature),
        "feature_truncation": {name: _truncation(part, args) for name, part in rows.items()},
        "shuffled_context_control": control,
        "head_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitation": (
            "Licensed public decision rows only. This small research scorer is not Jev/TypeSafe parity, "
            "desktop-task quality, safety evidence, or execution authority."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/.bench-state/jevlike_tiny_v1")
    parser.add_argument("--train-rows", type=int, default=960)
    parser.add_argument("--calibration-rows", type=int, default=188)
    parser.add_argument("--test-rows", type=int, default=121)
    parser.add_argument("--ood-rows", type=int, default=116)
    parser.add_argument("--context-bytes", type=int, default=1024)
    parser.add_argument("--option-bytes", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Fix dataset sampling independently of model initialization; defaults to --seed.",
    )
    parser.add_argument("--source-holdout-fraction", type=float, default=0.2,
                        help="Fraction of shared licensed source categories reserved for OOD; 0 disables")
    parser.add_argument("--source-holdout-seed", type=int,
                        help="Choose held-out categories independently of row-sampling seed")
    args = parser.parse_args()
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if not math.isfinite(args.source_holdout_fraction) or not 0 <= args.source_holdout_fraction < 1:
        raise ValueError("source holdout fraction must be in [0, 1)")
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "best_epoch": report["best_epoch"],
                "test_nll": report["trained_test"]["mean_nll"],
                "ood_nll": report["trained_ood"]["mean_nll"],
                "shuffled_nll": report["shuffled_context_control"]["shuffled_context"]["mean_nll"],
                "peak_cuda_gib": report["peak_cuda_gib"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

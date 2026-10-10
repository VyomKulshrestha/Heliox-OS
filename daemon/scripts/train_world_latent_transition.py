"""Research-only action-conditioned latent next-state probe.

The input is adjacent, observed desktop-state text. No weights produced here
are loaded by Heliox: these text features are not its on-device UI latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

from scripts.train_world_transition_lora import load_rows

STATE = "Current state:\n"
ACTION = "\n\nAction:\n"
NEXT = "\n\nNext observed screen state:"
WORDS = re.compile(r"\w+", re.UNICODE)


def parse_row(row: dict) -> tuple[str, str, str]:
    prompt = row["messages"][1]["content"]
    if (
        not prompt.startswith(STATE)
        or prompt.count(ACTION) != 1
        or prompt.count(NEXT) != 1
        or not prompt.endswith(NEXT)
    ):
        raise ValueError("Unexpected state/action prompt format")
    state, action = prompt[len(STATE) : -len(NEXT)].split(ACTION)
    target = row["messages"][2]["content"]
    if not state.strip() or not action.strip() or not target.strip():
        raise ValueError("State, action, and observed target must be nonempty")
    return state, action, target


def text_vector(text: str, dimension: int) -> np.ndarray:
    """Fixed, shared text space; no holdout vocabulary is fitted or leaked."""
    result = np.zeros(dimension, dtype=np.float32)
    words = WORDS.findall(text.casefold())
    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dimension
        result[index] += 1.0
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def encode_rows(rows: list[dict], dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = [parse_row(row) for row in rows]
    return tuple(np.stack([text_vector(triple[index], dimension) for triple in triples]) for index in range(3))


def task_group_split(rows: list[dict], *, seed: int, validation_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    groups = sorted({row["group_sha256"] for row in rows})
    if len(groups) < 2:
        raise ValueError("At least two independent training tasks are required")
    random.Random(seed).shuffle(groups)
    validation = set(groups[: max(1, round(len(groups) * validation_fraction))])
    return (
        [row for row in rows if row["group_sha256"] not in validation],
        [row for row in rows if row["group_sha256"] in validation],
    )


def score_predictions(prediction: np.ndarray, state: np.ndarray, target: np.ndarray) -> dict:
    """Compare with unchanged-state baseline on the exact same held-out rows."""

    def metrics(candidate: np.ndarray) -> dict:
        normalized = candidate / np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-8)
        similarity = normalized @ target.T
        return {
            "mean_target_cosine": float(np.diag(similarity).mean()),
            "retrieval_at_1": float((similarity.argmax(axis=1) == np.arange(len(target))).mean()),
        }

    return {"unchanged_state": metrics(state), "trained_transition": metrics(prediction)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state_dir = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=state_dir / "agentnet_adjacent_world_v1.jsonl")
    parser.add_argument("--report", type=Path, default=state_dir / "world_latent_transition_v1.json")
    parser.add_argument("--weights", type=Path, default=state_dir / "world_latent_transition_v1.pt")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--selection-metric", choices=("retrieval", "cosine"), default="retrieval")
    args = parser.parse_args()
    if (
        not args.rows.is_file()
        or args.report.exists()
        or args.weights.exists()
        or args.dimension < 32
        or args.epochs < 1
    ):
        parser.error("Use existing grouped rows, fresh outputs, and positive training settings")

    import torch
    from torch import nn
    from torch.nn import functional as F

    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 1024**3):
        raise RuntimeError("A CUDA GPU with at least 1 GiB free is required")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    partitions = load_rows(args.rows)
    fit, validation = task_group_split(partitions["train"], seed=args.seed)
    encoded = {
        split: encode_rows(rows, args.dimension)
        for split, rows in (("fit", fit), ("validation", validation), ("eval", partitions["eval"]))
    }
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    tensors = {
        split: tuple(torch.from_numpy(array).to(device) for array in arrays) for split, arrays in encoded.items()
    }

    class Transition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action = nn.Linear(args.dimension, args.dimension, bias=False)
            self.state = nn.Linear(args.dimension, args.dimension)

        def forward(self, current, action):
            return F.normalize(current + self.state(current) + self.action(action), dim=-1)

    model = Transition().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    fit_state, fit_action, fit_target = tensors["fit"]
    validation_baseline = score_predictions(
        encoded["validation"][0], encoded["validation"][0], encoded["validation"][2]
    )["unchanged_state"]
    best_validation = (float("-inf"), float("-inf"))
    best_epoch = 0
    best_weights = None
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(fit_state), device=device).split(64):
            prediction = model(fit_state[indices], fit_action[indices])
            # Distinct tasks may share a valid next state; avoid false-negative
            # in-batch contrastive labels for those duplicate observations.
            targets = fit_target[indices]
            loss = (1 - (prediction * targets).sum(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            current, action, target = tensors["validation"]
            validation_prediction = model(current, action).cpu().numpy()
            validation_metrics = score_predictions(
                validation_prediction, encoded["validation"][0], encoded["validation"][2]
            )["trained_transition"]
            if args.selection_metric == "cosine":
                validation_score = (validation_metrics["mean_target_cosine"], validation_metrics["retrieval_at_1"])
            else:
                validation_score = (validation_metrics["retrieval_at_1"], validation_metrics["mean_target_cosine"])
        if validation_score > best_validation:
            best_validation = validation_score
            best_epoch = epoch + 1
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        current, action, target = tensors["eval"]
        prediction = model(current, action).cpu().numpy()
    report = {
        "purpose": "research_only_action_conditioned_latent_world_transition",
        "source_rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "device": str(device),
        "dimension": args.dimension,
        "seed": args.seed,
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "eval_rows": len(partitions["eval"]),
        "fit_groups": len({row["group_sha256"] for row in fit}),
        "validation_groups": len({row["group_sha256"] for row in validation}),
        "eval_groups": len({row["group_sha256"] for row in partitions["eval"]}),
        "selected_epoch": best_epoch,
        "selection_metric": f"validation_{args.selection_metric}_then_other",
        "validation_baseline": validation_baseline,
        "validation_selected": {
            "retrieval_at_1": best_validation[1 if args.selection_metric == "cosine" else 0],
            "mean_target_cosine": best_validation[0 if args.selection_metric == "cosine" else 1],
        },
        "heldout": score_predictions(prediction, encoded["eval"][0], encoded["eval"][2]),
        "limitation": "Text-hash latent features and screenshot-derived descriptions, not images "
        "or Heliox UI latents. Retrieval is defined only against candidates in "
        "each partition; duplicate/near-equivalent targets and partition size "
        "affect its value. Compare models within the same partition only. "
        "No runtime loading, action authorization, safety claim, or Jev parity.",
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, args.weights)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

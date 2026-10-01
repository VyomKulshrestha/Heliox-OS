"""Licensed data mixture, per-kind calibration, and soft ordinal targets for decision training."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from benchmarks.general_decision import load_split, probabilities
from scripts.train_general_decision import (
    apply_kind_temperatures,
    checkpoint_signature,
    fit_kind_temperatures,
    noul_safety_by_source,
    parse_extra_config,
    soft_ordinal_target,
)


def test_soft_ordinal_targets_keep_the_mode_and_spread_to_neighbours() -> None:
    soft = soft_ordinal_target([0.0, 1.0, 0.0, 0.0], 0.75)
    assert sum(soft) == pytest.approx(1.0)
    assert int(np.argmax(soft)) == 1
    assert soft[0] == pytest.approx(soft[2]) and soft[2] > soft[3] > 0
    with pytest.raises(ValueError, match="sigma"):
        soft_ordinal_target([1.0, 0.0], 0.0)


def test_kind_temperatures_calibrate_each_decision_type_separately() -> None:
    rng = np.random.default_rng(7)
    rows, logits = [], []
    for _ in range(60):
        # Overconfident Noul scores that are right only 70% of the time.
        yes = bool(rng.random() < 0.5)
        correct = rng.random() < 0.7
        rows.append({"kind": "noul", "target": [0.0, 1.0] if yes else [1.0, 0.0]})
        logits.append(np.array([0.0, 6.0 if yes == correct else -6.0]))
        # Under-confident but always-correct choice scores.
        gold = int(rng.integers(3))
        rows.append({"kind": "choice", "target": [float(index == gold) for index in range(3)]})
        logits.append(np.array([1.0 if index == gold else 0.0 for index in range(3)]))
    temperatures = fit_kind_temperatures(logits, rows)
    assert set(temperatures) == {"noul", "choice"}
    assert temperatures["noul"] > 2.0 > temperatures["choice"]
    scaled = apply_kind_temperatures(logits, rows, temperatures, fallback=1.0)
    assert probabilities(scaled[0], 1.0).max() < probabilities(logits[0], 1.0).max()
    with pytest.raises(ValueError, match="align"):
        apply_kind_temperatures(logits[:1], rows, temperatures, fallback=1.0)


def test_extra_configs_are_allowlisted_and_bounded() -> None:
    assert parse_extra_config("email-selection-control-v1:128:16:32") == ("email-selection-control-v1", 128, 16, 32)
    for spec in ("release-v2-redistributable:1:1:1", "mailroom-control-v1:1:1", "mailroom-control-v1:0:1:1"):
        with pytest.raises(ValueError):
            parse_extra_config(spec)
    with pytest.raises(ValueError, match="Unsupported Open-Jev config"):
        load_split("train", 1, 1, config="customer-support-typesafe-docs")


def test_noul_safety_counts_false_allowances_per_source() -> None:
    rows = [
        {"kind": "noul", "source": "desktop", "target": [1.0, 0.0]},
        {"kind": "noul", "source": "desktop", "target": [0.0, 1.0]},
        {"kind": "choice", "source": "desktop", "target": [1.0, 0.0]},
    ]
    logits = [np.array([0.0, 2.0]), np.array([0.0, -2.0]), np.array([0.0, 1.0])]
    assert noul_safety_by_source(rows, logits) == {"desktop": {"rows": 2, "false_allowances": 1, "false_warnings": 1}}


def test_checkpoint_signature_binds_the_new_mixture_options() -> None:
    base = argparse.Namespace(
        backbone="qwen3.5-2b",
        initial_openjev=False,
        seed=1,
        split_seed=1,
        max_length=512,
        lora_rank=8,
        attention="eager",
        accumulation=4,
        candidate_microbatch_size=1,
        lr=2e-5,
        head_lr=5e-5,
        brier_weight=0.1,
        ordinal_weight=0.1,
        prompt_style="full_options_v2",
        dtype="bfloat16",
    )
    original = checkpoint_signature(base, {}, {}, None)
    variants = [
        {"extra_config": ["mailroom-control-v1:8:2:2"]},
        {"desktop_corpus": True, "desktop_corpus_seed": 3, "desktop_train_scenarios": 40},
        {"desktop_corpus": True, "desktop_corpus_seed": 3, "desktop_train_scenarios": 2},
        {"score_label_sigma": 0.75},
    ]
    signatures = {original}
    for change in variants:
        signatures.add(checkpoint_signature(argparse.Namespace(**{**vars(base), **change}), {}, {}, None))
    assert len(signatures) == len(variants) + 1
    disabled = argparse.Namespace(**{**vars(base), "extra_config": None, "desktop_corpus": False})
    assert checkpoint_signature(disabled, {}, {}, None) == original

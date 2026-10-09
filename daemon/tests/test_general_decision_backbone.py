"""Base-model selection must be explicit and bound to resumable checkpoints."""

from argparse import Namespace

import pytest

from scripts.train_general_decision import (
    MODEL,
    OPENJEV_BASE_MODEL,
    OPENJEV_BASE_REVISION,
    REVISION,
    checkpoint_signature,
    model_identity,
)


def test_backbone_selection_preserves_legacy_default_and_warm_start():
    assert model_identity(Namespace()) == (MODEL, REVISION)
    assert model_identity(Namespace(backbone="qwen3.5-2b")) == (OPENJEV_BASE_MODEL, OPENJEV_BASE_REVISION)
    assert model_identity(Namespace(initial_openjev=True)) == (OPENJEV_BASE_MODEL, OPENJEV_BASE_REVISION)
    with pytest.raises(ValueError, match="Unknown pinned"):
        model_identity(Namespace(backbone="unpinned"))


def test_clean_base_4b_backbone_is_pinned_and_has_no_warm_start():
    assert model_identity(Namespace(backbone="qwen3.5-4b")) == (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
    with pytest.raises(ValueError, match="only for the 2B"):
        model_identity(Namespace(backbone="qwen3.5-4b", initial_openjev=True))
    args = Namespace(
        seed=1,
        max_length=128,
        lora_rank=8,
        attention="sdpa",
        accumulation=4,
        lr=2e-5,
        head_lr=5e-5,
        brier_weight=0.1,
        ordinal_weight=0.1,
        prompt_style="full_options_v2",
        dtype="float16",
    )
    encoded = {"train": [{"row": {"id": "case"}, "sequences": [[1, 2]]}]}
    signatures = {
        checkpoint_signature(Namespace(**vars(args), backbone=backbone), encoded, {}, None)
        for backbone in ("qwen3.5-2b", "qwen3.5-4b")
    }
    assert len(signatures) == 2


def test_checkpoint_binds_base_selection_separately_from_warm_start():
    args = Namespace(
        seed=1,
        max_length=128,
        lora_rank=8,
        attention="eager",
        accumulation=4,
        lr=2e-5,
        head_lr=5e-5,
        brier_weight=0.1,
        ordinal_weight=0.1,
        prompt_style="candidate_v1",
    )
    encoded = {"train": [{"row": {"id": "case"}, "sequences": [[1, 2]]}]}
    small = checkpoint_signature(args, encoded, {}, None)
    args.backbone = "qwen3.5-2b"
    clean_2b = checkpoint_signature(args, encoded, {}, None)
    assert clean_2b != small
    args.initial_openjev = True
    assert checkpoint_signature(args, encoded, {}, None) != clean_2b

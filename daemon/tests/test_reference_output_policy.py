from __future__ import annotations

import importlib
import sys

import pytest

from scripts.evaluate_refscope import evaluate as evaluate_refscope
from scripts.reference_output_policy import block_jev_output_use


def test_historical_output_use_is_blocked_without_verified_authorization():
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        block_jev_output_use()


def test_live_provider_evaluator_is_blocked_before_network_access():
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        evaluate_refscope(key="not-a-real-key", repeats=1)


@pytest.mark.parametrize("module_name", [
    "scripts.compare_frozen_general_reference",
    "scripts.compare_scopejudge_reference",
    "scripts.evaluate_joint_opendec_general",
    "scripts.evaluate_opendec_scope_fresh",
    "scripts.evaluate_qwen_scope_baseline",
    "scripts.evaluate_world_general_qlora",
])
def test_archived_output_clis_fail_closed_before_argument_parsing(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name])
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        module.main()


def test_cached_output_functions_fail_closed_before_reading_files():
    from pathlib import Path

    from scripts.evaluate_opendec_scope_fresh import evaluate as evaluate_scope
    from scripts.evaluate_world_general_qlora import evaluate as evaluate_general

    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        evaluate_scope(Path("missing-model"), Path("missing-teacher"), Path("unused-output"))
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        evaluate_general(Path("missing-adapter"), Path("missing-training-report"), Path("missing-reference"))

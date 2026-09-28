from __future__ import annotations

import re

import pytest

from scripts.openjev_training_runtime import validate_training_versions


def test_accepts_pinned_openjev_training_stack_with_cuda_torch_suffix():
    validate_training_versions({
        "torch": "2.8.0+cu128",
        "transformers": "5.10.2",
        "peft": "0.19.1",
        "accelerate": "1.13.0",
    })


@pytest.mark.parametrize(
    ("package", "actual", "expected"),
    [
        ("torch", "2.6.0+cu124", "torch 2.6.0+cu124"),
        ("transformers", "5.2.0", "transformers 5.2.0"),
        ("peft", "0.18.1", "peft 0.18.1"),
        ("accelerate", "1.12.0", "accelerate 1.12.0"),
    ],
)
def test_rejects_unsupported_openjev_training_package(package, actual, expected):
    versions = {
        "torch": "2.8.0+cu128",
        "transformers": "5.10.2",
        "peft": "0.19.1",
        "accelerate": "1.13.0",
    }
    versions[package] = actual

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        validate_training_versions(versions)


def test_openjev_scoring_entry_points_fail_before_loading_a_model(monkeypatch, tmp_path):
    from scripts import evaluate_released_openjev_general, evaluate_world_general_qlora
    from scripts import openjev_training_runtime, score_general_decision

    def reject_runtime():
        raise RuntimeError("unsupported research runtime")

    monkeypatch.setattr(openjev_training_runtime, "require_training_runtime", reject_runtime)
    with pytest.raises(RuntimeError, match="unsupported research runtime"):
        evaluate_released_openjev_general.run(tmp_path / "protocol.json", tmp_path / "output", resume=False)

    monkeypatch.setattr(evaluate_world_general_qlora, "block_jev_output_use", lambda: None)
    with pytest.raises(RuntimeError, match="unsupported research runtime"):
        evaluate_world_general_qlora.evaluate(tmp_path / "adapter", tmp_path / "report.json",
                                              tmp_path / "reference.json")

    request = {"state": {}, "kind": "noul", "question": "Is this safe?", "options": ["no", "yes"]}
    with pytest.raises(RuntimeError, match="unsupported research runtime"):
        score_general_decision.score_requests([request], tmp_path / "model", tmp_path / "report.json")

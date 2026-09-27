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

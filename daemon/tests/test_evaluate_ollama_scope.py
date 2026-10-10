"""Unit checks for local, GPU-only Ollama scope comparison."""

import pytest

from scripts.evaluate_ollama_scope import gpu_residency, model_digest, possibly_truncated


def test_model_digest_requires_one_exact_name():
    tags = {"models": [{"name": "model:a", "digest": "one"}, {"name": "model:b", "digest": "two"}]}
    assert model_digest(tags, "model:a") == "one"
    with pytest.raises(ValueError, match="not uniquely installed"):
        model_digest(tags, "model:c")


def test_gpu_residency_rejects_partial_offload_and_changed_digest():
    process = {"name": "model:a", "digest": "one", "size": 100, "size_vram": 100, "context_length": 2048}
    assert gpu_residency({"models": [process]}, "model:a", "one")["size_vram_bytes"] == 100
    with pytest.raises(RuntimeError, match="fully GPU-resident"):
        gpu_residency({"models": [{**process, "size_vram": 80}]}, "model:a", "one")
    with pytest.raises(RuntimeError, match="loaded once"):
        gpu_residency({"models": [process]}, "model:a", "changed")


def test_near_context_limit_is_excluded_from_quality_metrics():
    assert not possibly_truncated(2031, 2048, 16)
    assert possibly_truncated(2032, 2048, 16)

"""No held-out task domain is included in a GPU transfer fold."""

import pytest

from scripts.evaluate_world_cross_domain import training_domains


def test_training_domains_excludes_held_out():
    assert training_domains(("agentic", "scope", "when2call"), "scope") == (
        "agentic",
        "when2call",
    )


def test_training_domains_rejects_unknown_or_duplicate_domain():
    with pytest.raises(ValueError, match="distinct"):
        training_domains(("agentic", "scope"), "when2call")
    with pytest.raises(ValueError, match="distinct"):
        training_domains(("agentic", "agentic"), "agentic")

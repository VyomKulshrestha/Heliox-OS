"""Scope and synthetic preference require different output heads."""

import pytest

from scripts.train_world_multitask_head import task_for_domain


@pytest.mark.parametrize("domain", ("agentic", "tbac", "scope"))
def test_scope_domains_share_scope_head(domain):
    assert task_for_domain(domain) == "scope"


def test_when2call_uses_preference_head():
    assert task_for_domain("when2call") == "preference"


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError, match="Unknown training domain"):
        task_for_domain("untrusted")

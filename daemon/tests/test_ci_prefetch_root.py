"""ci_prefetch fetches gigabytes of speech models into HELIOX_BENCH_ROOT; without it, it stops before fetching."""

from __future__ import annotations

import pytest

from benchmarks.conversation_bench import ci_prefetch


@pytest.mark.parametrize("value", [None, "", "relative/folder"])
def test_without_an_absolute_bench_root_nothing_is_fetched(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("HELIOX_BENCH_ROOT", raising=False)
    else:
        monkeypatch.setenv("HELIOX_BENCH_ROOT", value)
    fetched = []
    monkeypatch.setattr(ci_prefetch, "fetch_catalogue", lambda: fetched.append("catalogue") or [])
    monkeypatch.setattr(ci_prefetch, "fetch_voice", lambda: fetched.append("voice"))

    with pytest.raises(SystemExit, match="HELIOX_BENCH_ROOT"):
        ci_prefetch.main([])
    assert fetched == []

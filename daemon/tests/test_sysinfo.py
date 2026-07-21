"""Tests for pilot.system.sysinfo's CPU info collection.

psutil.cpu_freq() raises AttributeError/NotImplementedError on some
platforms (e.g. Apple Silicon macOS) instead of returning None -- this
covers _collect_cpu_info's defensive handling of that.
"""

import psutil

from pilot.system.sysinfo import _collect_cpu_info


def test_cpu_freq_attribute_error_does_not_raise(monkeypatch):
    def _raise(*args, **kwargs):
        raise AttributeError("module 'psutil' has no attribute 'cpu_freq'")

    monkeypatch.setattr(psutil, "cpu_freq", _raise)

    result = _collect_cpu_info(sample_interval=0.01)

    assert "=== CPU ===" in result
    assert "Frequency" not in result


def test_cpu_freq_not_implemented_error_does_not_raise(monkeypatch):
    def _raise(*args, **kwargs):
        raise NotImplementedError("cpu_freq not implemented on this platform")

    monkeypatch.setattr(psutil, "cpu_freq", _raise)

    result = _collect_cpu_info(sample_interval=0.01)

    assert "=== CPU ===" in result
    assert "Frequency" not in result


def test_cpu_freq_none_is_handled_like_before():
    """cpu_freq() returning None (rather than raising) was already handled;
    guard against a regression in that existing behavior."""
    result = _collect_cpu_info(sample_interval=0.01)
    assert "=== CPU ===" in result
    assert "Physical cores" in result

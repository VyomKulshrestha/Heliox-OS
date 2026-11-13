"""Two copies of Heliox starting at once do not take each other's write probe for an unwritable data folder.

CI on 5a3a9b3 (Real faults, ubuntu-24.04, two-copies/one-stops): the second copy ended with "DATA_DIR is not
writable" and a traceback instead of the sentence about the copy already running. Both copies wrote and removed
the same ".write_test" in the data folder; one removed it while the other was about to. Here the two probes are
interleaved on purpose: the second runs whole between the first's write and its removal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pilot import config


def _interleaved(monkeypatch, folder: Path) -> list[str]:
    """Run one probe, and a whole second probe (another copy) right after the first one's write."""
    real_write = Path.write_text
    order: list[str] = []
    nested = {"done": False}

    def write_text(self, *args, **kwargs):
        result = real_write(self, *args, **kwargs)
        order.append(f"write {self.name}")
        if not nested["done"] and self.parent == folder:
            nested["done"] = True
            monkeypatch.setattr(config.os, "getpid", lambda: 99999)  # the other copy
            config._probe_writable(folder)
            order.append("second probe finished")
        return result

    monkeypatch.setattr(Path, "write_text", write_text)
    config._probe_writable(folder)
    return order


def test_a_second_copy_probing_in_the_middle_does_not_fail_the_first(tmp_path, monkeypatch):
    order = _interleaved(monkeypatch, tmp_path)

    assert "second probe finished" in order
    assert len({line for line in order if line.startswith("write ")}) == 2  # two different probe files
    assert not list(tmp_path.glob(".write_test*"))  # both removed


def test_a_probe_removed_by_someone_else_still_counts_as_writable(tmp_path, monkeypatch):
    real_write = Path.write_text

    def write_then_lose_it(self, *args, **kwargs):
        result = real_write(self, *args, **kwargs)
        self.unlink()  # another process removed it
        return result

    monkeypatch.setattr(Path, "write_text", write_then_lose_it)
    config._probe_writable(tmp_path)  # no error


def test_a_folder_that_cannot_be_written_still_raises(tmp_path, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "write_text", refuse)
    with pytest.raises(PermissionError):
        config._probe_writable(tmp_path)

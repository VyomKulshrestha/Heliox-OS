"""The general benchmark must replay the same rows as archived Jev evidence."""

import pytest

from scripts import evaluate_world_general_qlora as probe


def test_selected_rows_rejects_selection_drift(monkeypatch):
    monkeypatch.setattr(probe, "load_split", lambda *_args: pytest.fail("dataset loaded"))
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        probe.selected_rows({})

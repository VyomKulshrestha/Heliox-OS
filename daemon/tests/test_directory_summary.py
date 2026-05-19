"""Tests for the directory_summary filesystem action."""

from __future__ import annotations

import pytest

from pilot.system.filesystem import directory_summary


@pytest.mark.asyncio
async def test_directory_summary_includes_tree_and_file_sizes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Project readme", encoding="utf-8")

    summary = await directory_summary(str(tmp_path))

    assert "src/" in summary
    assert "main.py (16 B)" in summary
    assert "README.md (14 B)" in summary


@pytest.mark.asyncio
async def test_directory_summary_ignores_git_and_node_modules_by_default(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('ok')", encoding="utf-8")

    summary = await directory_summary(str(tmp_path))

    assert "app.py" in summary
    assert ".git" not in summary
    assert "node_modules" not in summary


@pytest.mark.asyncio
async def test_directory_summary_respects_depth_and_entry_limits(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.txt").write_text("deep", encoding="utf-8")
    for index in range(3):
        (tmp_path / f"file_{index}.txt").write_text("x", encoding="utf-8")

    summary = await directory_summary(str(tmp_path), max_depth=1, max_entries=2)

    assert "a/" in summary
    assert "deep.txt" not in summary
    assert "more entries omitted" in summary

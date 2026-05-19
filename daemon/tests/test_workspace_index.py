"""Tests for WorkspaceIndex â€” local RAG semantic search."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pilot.memory.workspace_index import WorkspaceIndex


@pytest.fixture
def temp_index_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_workspace():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        # Create sample Python files
        (folder / "auth.py").write_text(
            "def login(username, password):\n"
            '    """Authenticate user with username and password."""\n'
            "    return check_credentials(username, password)\n"
        )
        (folder / "database.py").write_text(
            "import sqlite3\n\n"
            "def get_connection():\n"
            '    """Return a database connection."""\n'
            "    return sqlite3.connect('app.db')\n"
        )
        (folder / "utils.py").write_text(
            "def format_date(date):\n"
            '    """Format a date object to string."""\n'
            "    return date.strftime('%Y-%m-%d')\n"
        )
        yield folder


def test_index_workspace_success(temp_index_dir, sample_workspace):
    """Test that indexing a workspace returns success with correct counts."""
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    idx = WorkspaceIndex(temp_index_dir)
    result = idx.index_workspace(str(sample_workspace))

    assert result["success"] is True
    assert result["files_indexed"] == 3
    assert result["total_chunks"] > 0


def test_index_workspace_missing_folder(temp_index_dir):
    """Test that indexing a non-existent folder returns an error."""
    idx = WorkspaceIndex(temp_index_dir)
    result = idx.index_workspace("/nonexistent/path/12345")

    assert result["success"] is False
    assert "error" in result


def test_search_returns_results(temp_index_dir, sample_workspace):
    """Test that search returns relevant results after indexing."""
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    idx = WorkspaceIndex(temp_index_dir)
    idx.index_workspace(str(sample_workspace))

    results = idx.search("database connection", n_results=3)

    assert len(results) > 0
    assert "file" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]


def test_search_without_index_returns_empty(temp_index_dir):
    """Test that searching without an index returns empty list."""
    idx = WorkspaceIndex(temp_index_dir)
    results = idx.search("anything")
    assert results == []


def test_cache_avoids_reindexing(temp_index_dir, sample_workspace):
    """Test that unchanged files are not re-indexed on second run."""
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    idx1 = WorkspaceIndex(temp_index_dir)
    idx1.index_workspace(str(sample_workspace))

    idx2 = WorkspaceIndex(temp_index_dir)
    result = idx2.index_workspace(str(sample_workspace))

    assert result["success"] is True
    assert result["files_indexed"] == 0
    assert result.get("files_unchanged", 0) == 3


def test_is_ready_after_indexing(temp_index_dir, sample_workspace):
    """Test that is_ready() returns True after indexing."""
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    idx = WorkspaceIndex(temp_index_dir)
    assert idx.is_ready() is False

    idx.index_workspace(str(sample_workspace))
    assert idx.is_ready() is True

"""Tests for startup DATA_DIR write-permission validation (issue #194).

Covers four scenarios:
  1. All directories writable      → validate_data_dir_writable() returns None
  2. One directory not writable    → PermissionError raised, all bad dirs listed
  3. Multiple directories bad      → PermissionError lists every failure
  4. ensure_dirs() integration     → calls validate_data_dir_writable() internally
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pilot.config as cfg_module
from pilot.config import ensure_dirs, validate_data_dir_writable


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_readonly_dir(tmp_path: Path) -> Path:
    """Create a directory and remove all write bits (POSIX only)."""
    ro = tmp_path / "readonly"
    ro.mkdir()
    # Remove write permission for owner, group, and others
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return ro


# ── tests ─────────────────────────────────────────────────────────────────────


class TestValidateDataDirWritable:
    """Unit tests for validate_data_dir_writable()."""

    def test_all_writable_dirs_pass(self, tmp_path: Path) -> None:
        """validate_data_dir_writable() must return None when all dirs are writable."""
        dir_a = tmp_path / "config"
        dir_b = tmp_path / "data"
        dir_c = tmp_path / "state"
        for d in (dir_a, dir_b, dir_c):
            d.mkdir()

        with (
            patch.object(cfg_module, "CONFIG_DIR", dir_a),
            patch.object(cfg_module, "DATA_DIR", dir_b),
            patch.object(cfg_module, "STATE_DIR", dir_c),
        ):
            # Should complete without raising
            result = validate_data_dir_writable()
        assert result is None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod read-only directories behave differently on Windows",
    )
    def test_single_unwritable_dir_raises(self, tmp_path: Path) -> None:
        """A single unwritable directory must raise PermissionError."""
        writable = tmp_path / "writable"
        writable.mkdir()
        readonly = _make_readonly_dir(tmp_path)

        try:
            with (
                patch.object(cfg_module, "CONFIG_DIR", writable),
                patch.object(cfg_module, "DATA_DIR", readonly),
                patch.object(cfg_module, "STATE_DIR", writable),
            ):
                with pytest.raises(PermissionError) as exc_info:
                    validate_data_dir_writable()

            error_msg = str(exc_info.value)
            assert "DATA_DIR" in error_msg, "Error must name the failing directory key"
            assert str(readonly) in error_msg, "Error must include the failing path"
        finally:
            # Restore write permission so tmp_path cleanup can delete the dir
            readonly.chmod(stat.S_IRWXU)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod read-only directories behave differently on Windows",
    )
    def test_multiple_unwritable_dirs_all_listed(self, tmp_path: Path) -> None:
        """All unwritable directories must appear in the single PermissionError."""
        writable = tmp_path / "writable"
        writable.mkdir()
        ro1 = _make_readonly_dir(tmp_path / "ro1".as_posix() if False else tmp_path)
        ro2 = tmp_path / "ro2"
        ro2.mkdir()
        ro2.chmod(stat.S_IRUSR | stat.S_IXUSR)

        # Give ro1 a unique sub-path to distinguish it from ro2
        ro1_real = tmp_path / "ro1_real"
        ro1_real.mkdir()
        ro1_real.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            with (
                patch.object(cfg_module, "CONFIG_DIR", ro1_real),
                patch.object(cfg_module, "DATA_DIR", ro2),
                patch.object(cfg_module, "STATE_DIR", writable),
            ):
                with pytest.raises(PermissionError) as exc_info:
                    validate_data_dir_writable()

            error_msg = str(exc_info.value)
            assert "CONFIG_DIR" in error_msg
            assert "DATA_DIR" in error_msg
            # STATE_DIR was writable and must NOT appear
            assert "STATE_DIR" not in error_msg
        finally:
            ro1_real.chmod(stat.S_IRWXU)
            ro2.chmod(stat.S_IRWXU)

    def test_error_message_contains_actionable_hint(self, tmp_path: Path) -> None:
        """The PermissionError message must include an actionable recovery hint."""
        writable = tmp_path / "w"
        writable.mkdir()

        # Simulate an OSError by patching tempfile.NamedTemporaryFile
        def _raise_permission(*args, **kwargs):  # noqa: ANN001
            raise PermissionError(13, "Permission denied")

        with (
            patch.object(cfg_module, "CONFIG_DIR", writable),
            patch.object(cfg_module, "DATA_DIR", writable),
            patch.object(cfg_module, "STATE_DIR", writable),
            patch("tempfile.NamedTemporaryFile", side_effect=_raise_permission),
        ):
            with pytest.raises(PermissionError) as exc_info:
                validate_data_dir_writable()

        error_msg = str(exc_info.value)
        # The hint should mention a concrete recovery path
        assert "chown" in error_msg or "XDG_DATA_HOME" in error_msg, (
            "Error message should contain an actionable recovery hint"
        )

    def test_probe_file_is_cleaned_up(self, tmp_path: Path) -> None:
        """The temporary probe file must not be left behind after the check."""
        dir_a = tmp_path / "config"
        dir_b = tmp_path / "data"
        dir_c = tmp_path / "state"
        for d in (dir_a, dir_b, dir_c):
            d.mkdir()

        with (
            patch.object(cfg_module, "CONFIG_DIR", dir_a),
            patch.object(cfg_module, "DATA_DIR", dir_b),
            patch.object(cfg_module, "STATE_DIR", dir_c),
        ):
            validate_data_dir_writable()

        # After the check, no probe file remnants should exist in any directory
        for d in (dir_a, dir_b, dir_c):
            probe_files = list(d.glob(".pilot_write_probe_*"))
            assert not probe_files, (
                f"Probe file was not cleaned up in {d}: {probe_files}"
            )


class TestEnsureDirsIntegration:
    """Integration tests for ensure_dirs() calling validate_data_dir_writable()."""

    def test_ensure_dirs_calls_validation(self, tmp_path: Path) -> None:
        """ensure_dirs() must invoke validate_data_dir_writable() after creating dirs."""
        with (
            patch.object(cfg_module, "CONFIG_DIR", tmp_path / "config"),
            patch.object(cfg_module, "DATA_DIR", tmp_path / "data"),
            patch.object(cfg_module, "STATE_DIR", tmp_path / "state"),
            patch.object(cfg_module, "RUNTIME_DIR", tmp_path / "runtime"),
            patch.object(cfg_module, "validate_data_dir_writable") as mock_validate,
        ):
            ensure_dirs()

        mock_validate.assert_called_once(), (
            "ensure_dirs() must call validate_data_dir_writable() exactly once"
        )

    def test_ensure_dirs_creates_directories_before_validation(self, tmp_path: Path) -> None:
        """Directories are created before the write-permission check runs."""
        config_dir = tmp_path / "config"
        data_dir = tmp_path / "data"
        state_dir = tmp_path / "state"
        runtime_dir = tmp_path / "runtime"

        with (
            patch.object(cfg_module, "CONFIG_DIR", config_dir),
            patch.object(cfg_module, "DATA_DIR", data_dir),
            patch.object(cfg_module, "STATE_DIR", state_dir),
            patch.object(cfg_module, "RUNTIME_DIR", runtime_dir),
        ):
            ensure_dirs()

        for d in (config_dir, data_dir, state_dir, runtime_dir):
            assert d.exists() and d.is_dir(), f"{d} should have been created by ensure_dirs()"

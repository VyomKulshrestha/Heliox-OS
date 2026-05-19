from __future__ import annotations

import subprocess
import sys

import pytest

from pilot.system import linux_syscall_guard as guard


def test_guard_command_noops_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard.sys, "platform", "darwin")

    assert guard.guard_command(["python", "script.py"]) == ["python", "script.py"]


def test_guard_command_wraps_supported_linux_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard.sys, "platform", "linux")
    monkeypatch.setattr(guard.platform, "machine", lambda: "x86_64")

    wrapped = guard.guard_command(["bash", "-c", "python script.py"])

    assert wrapped[:5] == [
        sys.executable,
        "-m",
        "pilot.system.linux_syscall_guard",
        "--block",
        "unlink,unlinkat",
    ]
    assert wrapped[5:] == ["--", "bash", "-c", "python script.py"]


def test_supported_syscalls_filters_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard.platform, "machine", lambda: "x86_64")

    assert guard.supported_syscalls(["unlink", "unknown", "unlinkat"]) == (87, 263)


def test_parse_blocked_syscalls_trims_empty_values() -> None:
    assert guard.parse_blocked_syscalls(" unlink, ,unlinkat ") == ("unlink", "unlinkat")


def test_build_filter_denies_each_syscall_then_allows_rest() -> None:
    filters = guard._build_filter([87, 263])

    assert len(filters) == 6
    assert filters[0].k == guard.SECCOMP_DATA_NR_OFFSET
    assert filters[1].k == 87
    assert filters[2].k == guard.SECCOMP_RET_ERRNO | guard.errno.EPERM
    assert filters[3].k == 263
    assert filters[4].k == guard.SECCOMP_RET_ERRNO | guard.errno.EPERM
    assert filters[5].k == guard.SECCOMP_RET_ALLOW


@pytest.mark.skipif(
    not guard.is_supported(["unlink", "unlinkat"]),
    reason="seccomp-BPF syscall guard is Linux-architecture specific",
)
def test_guard_blocks_unlink_syscall_on_supported_linux(tmp_path) -> None:
    target = tmp_path / "protected.txt"
    target.write_text("keep me", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot.system.linux_syscall_guard",
            "--block",
            "unlink,unlinkat",
            "--",
            sys.executable,
            "-c",
            "import os,sys; os.unlink(sys.argv[1])",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert target.exists()
    assert "PermissionError" in result.stderr

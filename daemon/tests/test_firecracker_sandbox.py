from __future__ import annotations

import pytest

from pilot.system import sandbox_exec
from pilot.system.sandbox_exec import SandboxConfig, SecureExecutionSandbox


def test_firecracker_mode_falls_back_when_host_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_exec.sys, "platform", "darwin")
    monkeypatch.setattr(sandbox_exec, "_docker_available", lambda: False)

    sandbox = SecureExecutionSandbox(SandboxConfig(mode="firecracker"))

    assert sandbox.active_mode == "restricted"


@pytest.mark.asyncio
async def test_firecracker_mode_can_fail_closed_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_exec.sys, "platform", "darwin")

    sandbox = SecureExecutionSandbox(SandboxConfig(mode="firecracker", firecracker_fallback=False))

    assert sandbox.active_mode == "firecracker"
    result = await sandbox.run("print('hello')", "python")

    assert result is not None
    assert result.startswith("ERROR: Firecracker sandbox unavailable:")
    assert "Linux hosts" in result


@pytest.mark.asyncio
async def test_firecracker_mode_requires_guest_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_exec.sys, "platform", "linux")
    monkeypatch.setattr(sandbox_exec.shutil, "which", lambda _: "/usr/bin/firecracker")
    monkeypatch.setattr(sandbox_exec.os.path, "exists", lambda _: False)

    sandbox = SecureExecutionSandbox(SandboxConfig(mode="firecracker", firecracker_fallback=False))
    result = await sandbox.run("print('hello')", "python")

    assert result is not None
    assert "missing kernel image and rootfs image" in result


@pytest.mark.asyncio
async def test_firecracker_mode_resolves_when_preflight_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_exec.sys, "platform", "linux")
    monkeypatch.setattr(sandbox_exec.shutil, "which", lambda _: "/usr/bin/firecracker")
    monkeypatch.setattr(sandbox_exec.os.path, "exists", lambda _: True)

    sandbox = SecureExecutionSandbox(
        SandboxConfig(
            mode="firecracker",
            firecracker_kernel_image="/var/lib/heliox/vmlinux",
            firecracker_rootfs_path="/var/lib/heliox/rootfs.ext4",
        )
    )

    assert sandbox.active_mode == "firecracker"
    result = await sandbox.run("print('hello')", "python")

    assert result is not None
    assert "preflight passed" in result

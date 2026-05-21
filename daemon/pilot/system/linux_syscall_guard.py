"""Linux syscall guard for restricted sandbox subprocesses.

The guard installs a small seccomp-BPF filter in a launcher process before it
execs the sandbox command. The filter is inherited across exec and blocks the
configured denylisted syscalls inside generated code while leaving macOS,
Windows, and unsupported Linux architectures as no-ops.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import logging
import os
import platform
import sys
from collections.abc import Iterable, Sequence

logger = logging.getLogger("pilot.system.linux_syscall_guard")

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_K = 0x00
BPF_RET = 0x06

SECCOMP_DATA_NR_OFFSET = 0

_ARCH_SYSCALLS: dict[str, dict[str, int]] = {
    "x86_64": {
        "unlink": 87,
        "unlinkat": 263,
    },
    "amd64": {
        "unlink": 87,
        "unlinkat": 263,
    },
    "aarch64": {
        "unlinkat": 35,
    },
    "arm64": {
        "unlinkat": 35,
    },
}


class SeccompInstallError(RuntimeError):
    """Raised when the Linux kernel refuses to install the seccomp filter."""


class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(SockFilter)),
    ]


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def supported_syscalls(blocked_syscalls: Iterable[str]) -> tuple[int, ...]:
    """Return syscall numbers supported by the current Linux architecture."""
    machine = platform.machine().lower()
    arch_map = _ARCH_SYSCALLS.get(machine, {})
    numbers = []

    for syscall_name in blocked_syscalls:
        syscall_number = arch_map.get(syscall_name)
        if syscall_number is not None and syscall_number not in numbers:
            numbers.append(syscall_number)

    return tuple(numbers)


def is_supported(blocked_syscalls: Iterable[str] = ("unlink", "unlinkat")) -> bool:
    return is_linux() and bool(supported_syscalls(blocked_syscalls))


def guard_command(
    cmd: Sequence[str],
    *,
    blocked_syscalls: Iterable[str] = ("unlink", "unlinkat"),
) -> list[str]:
    """Wrap *cmd* with the Linux guard launcher when the host supports it."""
    syscall_names = tuple(blocked_syscalls)
    if not is_supported(syscall_names):
        return list(cmd)

    return [
        sys.executable,
        "-m",
        "pilot.system.linux_syscall_guard",
        "--block",
        ",".join(syscall_names),
        "--",
        *cmd,
    ]


def _stmt(code: int, k: int, jt: int = 0, jf: int = 0) -> SockFilter:
    return SockFilter(code=code, jt=jt, jf=jf, k=k)


def _build_filter(syscall_numbers: Sequence[int]) -> ctypes.Array[SockFilter]:
    deny_action = SECCOMP_RET_ERRNO | errno.EPERM
    instructions: list[SockFilter] = [
        _stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_DATA_NR_OFFSET),
    ]

    for syscall_number in syscall_numbers:
        instructions.extend(
            [
                _stmt(BPF_JMP | BPF_JEQ | BPF_K, syscall_number, jt=0, jf=1),
                _stmt(BPF_RET | BPF_K, deny_action),
            ]
        )

    instructions.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW))
    return (SockFilter * len(instructions))(*instructions)


def install_seccomp_filter(blocked_syscalls: Iterable[str]) -> None:
    """Install a denylist seccomp-BPF filter for the current process."""
    syscall_numbers = supported_syscalls(blocked_syscalls)
    if not syscall_numbers:
        return

    libc = ctypes.CDLL(None, use_errno=True)

    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise SeccompInstallError(f"PR_SET_NO_NEW_PRIVS failed: {os.strerror(err)}")

    filters = _build_filter(syscall_numbers)
    program = SockFprog(len=len(filters), filter=filters)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        err = ctypes.get_errno()
        raise SeccompInstallError(f"PR_SET_SECCOMP failed: {os.strerror(err)}")


def parse_blocked_syscalls(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a command behind a Linux seccomp syscall guard.")
    parser.add_argument("--block", default="unlink,unlinkat", help="Comma-separated syscall names to block.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    blocked_syscalls = parse_blocked_syscalls(args.block)
    if is_supported(blocked_syscalls):
        try:
            install_seccomp_filter(blocked_syscalls)
        except SeccompInstallError as exc:
            logger.warning("Linux syscall guard unavailable; continuing without kernel filter: %s", exc)

    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

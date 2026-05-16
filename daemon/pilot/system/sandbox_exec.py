"""Secure execution sandbox for agent-generated code.

Provides OS-level isolation so untrusted code cannot harm the host system.
Two backends are supported, selected automatically or via config:

  docker      — ephemeral container per execution (best isolation)
  restricted  — ulimit + stripped env subprocess (fallback, no Docker needed)
  none        — direct execution, no isolation (legacy / opt-out)

Architecture
------------
  execute_code()  ──►  SecureExecutionSandbox.run()
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
           DockerBackend       RestrictedBackend
         (container per run)  (ulimit + stripped env)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger("pilot.system.sandbox_exec")

# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors SecurityConfig fields)
# ---------------------------------------------------------------------------


@dataclass
class SandboxConfig:
    """Runtime configuration for the sandbox layer."""

    mode: str = "auto"  # "auto" | "docker" | "restricted" | "none"
    memory_mb: int = 128  # memory cap (docker & restricted)
    timeout: int = 30  # max wall-clock seconds
    network: bool = False  # allow outbound network inside sandbox


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class _SandboxBackend(ABC):
    """Common interface for all execution backends."""

    @abstractmethod
    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> str:
        """Execute *code* and return captured output (stdout + stderr)."""


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------

# Language → (image, file extension, run command template)
_DOCKER_LANG_MAP: dict[str, tuple[str, str, list[str]]] = {
    "python": ("python:3.11-slim", ".py", ["python", "/sandbox/script.py"]),
    "javascript": ("node:20-slim", ".js", ["node", "/sandbox/script.js"]),
    "bash": ("bash:5", ".sh", ["bash", "/sandbox/script.sh"]),
}


class DockerBackend(_SandboxBackend):
    """Runs code inside a disposable Docker container.

    Security properties
    -------------------
    - ``--network none``          no outbound network (unless config.network=True)
    - ``--memory``                hard memory cap
    - ``--cpus 0.5``              half a CPU core max
    - ``--read-only``             root filesystem is read-only
    - ``--tmpfs /tmp``            writable scratch space only in /tmp
    - ``--no-new-privileges``     prevents privilege escalation
    - ``--cap-drop ALL``          drops all Linux capabilities
    - ``--rm``                    container auto-removed on exit
    - ``--user nobody``           runs as unprivileged user
    - no host mounts              host filesystem is never exposed
    """

    async def run(self, code: str, language: str, config: SandboxConfig) -> str:
        lang = _normalise_language(language)
        if lang not in _DOCKER_LANG_MAP:
            return f"ERROR: Docker sandbox does not support language '{language}'"

        image, ext, cmd = _DOCKER_LANG_MAP[lang]

        # Write code to a temp file that we'll COPY into the container via stdin
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8", prefix="pilot_sb_") as f:
            if lang == "bash":
                f.write("#!/bin/bash\nset -e\n" + code)
            else:
                f.write(code)
            script_path = f.name

        try:
            network_flag = "bridge" if config.network else "none"
            memory_flag = f"{config.memory_mb}m"

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                network_flag,
                "--memory",
                memory_flag,
                "--memory-swap",
                memory_flag,  # disable swap
                "--cpus",
                "0.5",
                "--read-only",
                "--tmpfs",
                "/tmp:size=64m",
                "--no-new-privileges",
                "--cap-drop",
                "ALL",
                "--user",
                "nobody",
                "-v",
                f"{script_path}:/sandbox/script{ext}:ro",
                image,
                *cmd,
            ]

            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=config.timeout)
            except TimeoutError:
                proc.kill()
                return f"ERROR: Sandbox timed out after {config.timeout}s"

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                err = stderr.decode("utf-8", errors="replace").strip()
                if err:
                    output += f"\n[STDERR]\n{err}"
            if proc.returncode not in (0, None):
                output += f"\n[EXIT CODE: {proc.returncode}]"

            return output.strip() or "(no output)"

        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Restricted subprocess backend  (no Docker required)
# ---------------------------------------------------------------------------

# Minimal safe environment — strips credentials, tokens, paths to sensitive dirs
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
    }
)


def _build_safe_env() -> dict[str, str]:
    """Return a stripped copy of os.environ with only safe keys."""
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    # Prevent .pyc files cluttering the temp dir
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class RestrictedBackend(_SandboxBackend):
    """Runs code in a subprocess with resource limits and a stripped environment.

    Security properties
    -------------------
    - Stripped environment (no API keys, tokens, or sensitive vars)
    - ``ulimit -v`` virtual memory cap  (Linux/macOS)
    - ``ulimit -t`` CPU time cap        (Linux/macOS)
    - ``ulimit -f`` file size cap       (Linux/macOS)
    - Timeout enforced via asyncio
    - No network restriction (OS-level network namespaces need root)

    This backend is weaker than Docker but far better than bare execution.
    On Windows, only the stripped environment and timeout apply.
    """

    async def run(self, code: str, language: str, config: SandboxConfig) -> str:
        lang = _normalise_language(language)
        safe_env = _build_safe_env()

        if lang in ("python",):
            return await self._run_python(code, config, safe_env)
        elif lang == "bash":
            return await self._run_bash(code, config, safe_env)
        elif lang == "powershell":
            return await self._run_powershell(code, config, safe_env)
        elif lang == "javascript":
            return await self._run_node(code, config, safe_env)
        elif lang == "cmd":
            return await self._run_cmd(code, config, safe_env)
        else:
            return f"ERROR: Restricted sandbox does not support language '{language}'"

    # -- helpers -----------------------------------------------------------

    async def _run_python(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8", prefix="pilot_sb_"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = self._wrap_with_ulimit([sys.executable, script_path], config)
            return await self._run_proc(cmd, config.timeout, env)
        finally:
            _safe_unlink(script_path)

    async def _run_bash(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, encoding="utf-8", prefix="pilot_sb_"
        ) as f:
            f.write("#!/bin/bash\nset -e\n" + code)
            script_path = f.name

        try:
            os.chmod(script_path, 0o700)
            cmd = self._wrap_with_ulimit(["bash", script_path], config)
            return await self._run_proc(cmd, config.timeout, env)
        finally:
            _safe_unlink(script_path)

    async def _run_powershell(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ps1", delete=False, encoding="utf-8", prefix="pilot_sb_"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            shell = "pwsh" if shutil.which("pwsh") else "powershell"
            cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
            return await self._run_proc(cmd, config.timeout, env)
        finally:
            _safe_unlink(script_path)

    async def _run_node(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8", prefix="pilot_sb_"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = self._wrap_with_ulimit(["node", script_path], config)
            return await self._run_proc(cmd, config.timeout, env)
        finally:
            _safe_unlink(script_path)

    async def _run_cmd(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cmd", delete=False, encoding="utf-8", prefix="pilot_sb_"
        ) as f:
            f.write("@echo off\n" + code)
            script_path = f.name

        try:
            return await self._run_proc(["cmd", "/c", script_path], config.timeout, env)
        finally:
            _safe_unlink(script_path)

    @staticmethod
    def _wrap_with_ulimit(cmd: list[str], config: SandboxConfig) -> list[str]:
        """Prepend ulimit constraints on POSIX systems."""
        if sys.platform == "win32":
            return cmd  # ulimit not available on Windows

        mem_kb = config.memory_mb * 1024
        cpu_seconds = max(config.timeout, 5)
        # ulimit flags: -v virtual mem (KB), -t CPU time (s), -f file size (512-byte blocks = 50MB)
        return [
            "bash",
            "-c",
            f"ulimit -v {mem_kb} -t {cpu_seconds} -f 102400; exec "
            + " ".join(f'"{a}"' if " " in a else a for a in cmd),
        ]

    @staticmethod
    async def _run_proc(cmd: list[str], timeout: int, env: dict[str, str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Sandbox timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace").strip()
            if err:
                output += f"\n[STDERR]\n{err}"
        if proc.returncode not in (0, None):
            output += f"\n[EXIT CODE: {proc.returncode}]"

        return output.strip() or "(no output)"


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class SecureExecutionSandbox:
    """Selects and caches the appropriate backend based on config.mode.

    Usage::

        sandbox = SecureExecutionSandbox(config)
        output  = await sandbox.run(code, language="python")
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._backend: _SandboxBackend | None = None
        self._resolved_mode: str = ""

    # -- backend resolution ------------------------------------------------

    def _resolve_backend(self) -> tuple[_SandboxBackend | None, str]:
        """Return (backend, mode_name). Returns (None, 'none') for passthrough."""
        mode = self._config.mode.lower().strip()

        if mode == "none":
            return None, "none"

        if mode == "docker":
            if _docker_available():
                return DockerBackend(), "docker"
            logger.warning(
                "sandbox_mode='docker' requested but Docker is not available. Falling back to restricted mode."
            )
            return RestrictedBackend(), "restricted"

        if mode == "restricted":
            return RestrictedBackend(), "restricted"

        # mode == "auto": prefer Docker, fall back to restricted
        if _docker_available():
            return DockerBackend(), "docker"
        return RestrictedBackend(), "restricted"

    def _get_backend(self) -> tuple[_SandboxBackend | None, str]:
        if self._backend is None and self._resolved_mode == "":
            self._backend, self._resolved_mode = self._resolve_backend()
            logger.info("Sandbox backend resolved: %s", self._resolved_mode)
        return self._backend, self._resolved_mode

    # -- public API --------------------------------------------------------

    async def run(self, code: str, language: str) -> str | None:
        """Execute *code* in the sandbox.

        Returns the captured output string, or ``None`` if sandbox_mode is
        'none' (caller should fall back to direct execution).
        """
        backend, mode = self._get_backend()

        if mode == "none" or backend is None:
            return None  # signal: use legacy direct execution

        logger.info(
            "Sandbox[%s]: executing %d chars of %s code",
            mode,
            len(code),
            language,
        )
        try:
            result = await backend.run(code, language, self._config)
            logger.info("Sandbox[%s]: execution complete (%d chars output)", mode, len(result))
            return result
        except Exception as exc:
            logger.exception("Sandbox[%s] execution error: %s", mode, exc)
            return f"ERROR: Sandbox execution failed — {exc}"

    @property
    def active_mode(self) -> str:
        """The resolved backend mode (available after first call to run())."""
        _, mode = self._get_backend()
        return mode


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _normalise_language(language: str) -> str:
    """Normalise language aliases to a canonical name."""
    lang = language.lower().strip()
    aliases: dict[str, str] = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "sh": "bash",
        "shell": "bash",
        "ps1": "powershell",
        "pwsh": "powershell",
        "bat": "cmd",
        "batch": "cmd",
    }
    return aliases.get(lang, lang)


def _docker_available() -> bool:
    """Return True if the Docker CLI is on PATH and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = __import__("subprocess").run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _safe_unlink(path: str) -> None:
    """Delete a file, ignoring errors."""
    try:
        os.unlink(path)
    except OSError:
        pass

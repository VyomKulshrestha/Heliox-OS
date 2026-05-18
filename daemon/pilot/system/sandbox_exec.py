"""Secure execution sandbox for agent-generated code.

Provides OS-level isolation so untrusted code cannot harm the host system.
Two backends are supported, selected automatically or via config:

  docker     — ephemeral container per execution (best isolation)
  restricted  — ulimit + stripped env subprocess (fallback, no Docker needed)
  none        — direct execution, no isolation (legacy / opt-out)

Architecture
------------
  execute_code()  ──►  SecureExecutionSandbox.run()
                           │
                  ┌─────────┴──────────┐
                  ▼                    ▼
               DockerBackend        RestrictedBackend
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
    kernel_guard: bool = True  # apply Linux seccomp-BPF syscall denylist when available
    blocked_syscalls: tuple[str, ...] = ("unlink", "unlinkat")


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
    """Runs code inside a disposable Docker container."""

    async def run(self, code: str, language: str, config: SandboxConfig) -> str:
        lang = _normalise_language(language)
        if lang not in _DOCKER_LANG_MAP:
            logger.warning("Unsupported language requested for Docker sandbox: %s", language)
            return f"ERROR: Docker sandbox does not support language '{language}'"

        image, ext, cmd = _DOCKER_LANG_MAP[lang]
        script_path = None

        # Robust creation of temporary script files with graceful fallbacks
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                if lang == "bash":
                    f.write("#!/bin/bash\nset -e\n" + code)
                else:
                    f.write(code)
                script_path = f.name
        except PermissionError as exc:
            logger.error("🚫 Disk write permission denied when generating sandbox script container: %s", exc)
            return f"ERROR: Sandbox execution failed due to host filesystem permission restrictions."
        except OSError as exc:
            logger.error("💻 OS System IO Error encountered during script generation: %s", exc)
            return f"ERROR: Sandbox environment encountered an underlying storage subsystem error."

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
                memory_flag,
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
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                logger.warning("⚠️ Docker sandbox execution timed out after %ds", config.timeout)
                return f"ERROR: Sandbox timed out after {config.timeout}s"

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                err = stderr.decode("utf-8", errors="replace").strip()
                if err:
                    output += f"\n[STDERR]\n{err}"
            if proc.returncode not in (0, None):
                logger.info("Sandbox container exited with non-zero code: %d", proc.returncode)
                output += f"\n[EXIT CODE: {proc.returncode}]"

            return output.strip() or "(no output)"

        except Exception as exc:
            logger.exception("Unexpected error inside Docker sandbox engine execution loop: %s", exc)
            return f"ERROR: Internal sandbox error occurred during runtime lifecycle setup."

        finally:
            if script_path:
                _safe_unlink(script_path)


# ---------------------------------------------------------------------------
# Restricted subprocess backend  (no Docker required)
# ---------------------------------------------------------------------------

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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class RestrictedBackend(_SandboxBackend):
    """Runs code in a subprocess with resource limits and a stripped environment."""

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
            logger.warning("Unsupported language requested for Restricted sandbox: %s", language)
            return f"ERROR: Restricted sandbox does not support language '{language}'"

    # -- helpers -----------------------------------------------------------

    async def _run_python(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                f.write(code)
                script_path = f.name
        except (PermissionError, OSError) as exc:
            logger.error("🚫 Python script staging failed due to filesystem access boundaries: %s", exc)
            return "ERROR: Subprocess staging blocked by filesystem host constraints."

        try:
            cmd = self._wrap_with_ulimit([sys.executable, script_path], config)
            return await self._run_proc(cmd, config.timeout, env, config)
        finally:
            if script_path:
                _safe_unlink(script_path)

    async def _run_bash(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                f.write("#!/bin/bash\nset -e\n" + code)
                script_path = f.name
        except (PermissionError, OSError) as exc:
            logger.error("🚫 Bash script staging failed due to filesystem access boundaries: %s", exc)
            return "ERROR: Subprocess staging blocked by filesystem host constraints."

        try:
            try:
                os.chmod(script_path, 0o700)
            except PermissionError as exc:
                logger.warning(
                    "⚠️ Failed to adjust execute bits on script %s: %s. Attempting raw execute fallback.",
                    script_path,
                    exc,
                )

            cmd = self._wrap_with_ulimit(["bash", script_path], config)
            return await self._run_proc(cmd, config.timeout, env, config)
        finally:
            if script_path:
                _safe_unlink(script_path)

    async def _run_powershell(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ps1", delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                f.write(code)
                script_path = f.name
        except (PermissionError, OSError) as exc:
            logger.error("🚫 PowerShell script staging failed due to filesystem access boundaries: %s", exc)
            return "ERROR: Subprocess staging blocked by filesystem host constraints."

        try:
            shell = "pwsh" if shutil.which("pwsh") else "powershell"
            cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
            return await self._run_proc(cmd, config.timeout, env, config)
        finally:
            if script_path:
                _safe_unlink(script_path)

    async def _run_node(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                f.write(code)
                script_path = f.name
        except (PermissionError, OSError) as exc:
            logger.error("🚫 Node script staging failed due to filesystem access boundaries: %s", exc)
            return "ERROR: Subprocess staging blocked by filesystem host constraints."

        try:
            cmd = self._wrap_with_ulimit(["node", script_path], config)
            return await self._run_proc(cmd, config.timeout, env, config)
        finally:
            if script_path:
                _safe_unlink(script_path)

    async def _run_cmd(self, code: str, config: SandboxConfig, env: dict[str, str]) -> str:
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cmd", delete=False, encoding="utf-8", prefix="pilot_sb_"
            ) as f:
                f.write("@echo off\n" + code)
                script_path = f.name
        except (PermissionError, OSError) as exc:
            logger.error("🚫 Batch script staging failed due to filesystem access boundaries: %s", exc)
            return "ERROR: Subprocess staging blocked by filesystem host constraints."

        try:
            return await self._run_proc(["cmd", "/c", script_path], config.timeout, env, config)
        finally:
            if script_path:
                _safe_unlink(script_path)

    @staticmethod
    def _wrap_with_ulimit(cmd: list[str], config: SandboxConfig) -> list[str]:
        """Prepend ulimit constraints on POSIX systems."""
        if sys.platform == "win32":
            return cmd

        mem_kb = config.memory_mb * 1024
        cpu_seconds = max(config.timeout, 5)
        return [
            "bash",
            "-c",
            f"ulimit -v {mem_kb} -t {cpu_seconds} -f 102400; exec "
            + " ".join(f'"{a}"' if " " in a else a for a in cmd),
        ]

    @staticmethod
    async def _run_proc(
        cmd: list[str],
        timeout: int,
        env: dict[str, str],
        config: SandboxConfig,
    ) -> str:
        from pilot.system.linux_syscall_guard import guard_command

        if config.kernel_guard:
            cmd = guard_command(cmd, blocked_syscalls=config.blocked_syscalls)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError) as exc:
            logger.error("🚫 Subprocess initialization blocked by target execution runtime boundary: %s", exc)
            return f"ERROR: Failed to initialize environment runtime context."

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            logger.warning("⚠️ Restricted process isolation lifecycle exceeded wall-clock timeout of %ds", timeout)
            return f"ERROR: Sandbox timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace").strip()
            if err:
                output += f"\n[STDERR]\n{err}"
        if proc.returncode not in (0, None):
            logger.info("Restricted isolated subprocess exited with code: %s", proc.returncode)
            output += f"\n[EXIT CODE: {proc.returncode}]"

        return output.strip() or "(no output)"


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class SecureExecutionSandbox:
    """Selects and caches the appropriate backend based on config.mode."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._backend: _SandboxBackend | None = None
        self._resolved_mode: str = ""

    # -- backend resolution ------------------------------------------------

    def _resolve_backend(self) -> tuple[_SandboxBackend | None, str]:
        """Return (backend, mode_name). Returns (None, 'none') for passthrough."""
        try:
            mode = self._config.mode.lower().strip()
        except AttributeError:
            logger.error("Invalid configuration schema detected inside mode mapping properties.")
            mode = "auto"

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
        """Execute *code* in the sandbox safely."""
        backend, mode = self._get_backend()

        if mode == "none" or backend is None:
            return None

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
            logger.exception("Critical unexpected recovery trace inside Sandbox[%s]: %s", mode, exc)
            return f"ERROR: Sandbox execution failed — {exc}"

    @property
    def active_mode(self) -> str:
        """The resolved backend mode."""
        _, mode = self._get_backend()
        return mode


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _normalise_language(language: str) -> str:
    """Normalise language aliases to a canonical name."""
    try:
        lang = language.lower().strip()
    except AttributeError:
        return ""
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
    except Exception as exc:
        logger.debug("Docker daemon status check failed: %s", exc)
        return False


def _safe_unlink(path: str) -> None:
    """Delete a file, logging anomalies while catching permissions and system blocks."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        logger.warning("⚠️ Restricted disk access denied cleanup tracking for workspace file descriptor: %s", path)
    except OSError as e:
        logger.debug("Minor context IO block during scratch space unlinking: %s", e)

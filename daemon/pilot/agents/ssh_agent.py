"""SSH Agent — executes remote bash commands over SSH using Paramiko.

This agent is intentionally constrained:
  - It only connects to host aliases configured in config.ssh.allowed_hosts
  - Private keys (and optional passphrases) are fetched from the KeyVault at runtime
  - No SSH secrets are ever logged or returned in ActionResult output
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import (
    ActionPlan,
    ActionResult,
    ActionType,
    SshCommandParams,
    SshScriptParams,
)
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride

logger = logging.getLogger("pilot.agents.ssh_agent")

PROBE_OUTPUT_LIMIT = 64 * 1024

SSH_ACTION_TYPES: set[ActionType] = {
    ActionType.SSH_COMMAND,
    ActionType.SSH_SCRIPT,
}


def _load_private_key(paramiko: Any, key_pem: str, passphrase: str | None) -> Any:
    """Best-effort private key loader supporting common key types."""

    key_file = io.StringIO(key_pem)
    # Order matters a bit: OpenSSH keys are often Ed25519 these days, but RSA is common too.
    key_types = (
        getattr(paramiko, "Ed25519Key", None),
        getattr(paramiko, "ECDSAKey", None),
        getattr(paramiko, "RSAKey", None),
        getattr(paramiko, "DSSKey", None),
    )
    last_exc: Exception | None = None
    for key_cls in key_types:
        if key_cls is None:
            continue
        try:
            key_file.seek(0)
            return key_cls.from_private_key(key_file, password=passphrase)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            continue
    raise ValueError("Unsupported or invalid private key format") from last_exc


class SshAgent(BaseAgent):
    """Specialist agent for executing commands on remote hosts via SSH."""

    def __init__(self, model_router: ModelRouter) -> None:
        super().__init__(role=AgentRole.SSH, model_router=model_router)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.SSH_COMMAND,
                description="Execute a single shell command on a configured remote host via SSH",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.SSH_SCRIPT,
                description="Execute a multi-line bash script on a configured remote host via SSH",
                requires_confirmation=True,
                estimated_duration_ms=10_000,
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the SSH AGENT for Heliox OS. "
            "You execute remote shell commands on pre-configured hosts via SSH. "
            "You MUST only use host aliases that exist in config.ssh.allowed_hosts. "
            "SSH private keys and passphrases are retrieved from the encrypted KeyVault at runtime "
            "and must never be logged or included in outputs. "
            "Prefer idempotent commands and verify remote state when possible."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in SSH_ACTION_TYPES

    async def test_connection(self, host_alias: str) -> dict[str, Any]:
        """Authenticate to an allowlisted host without executing a command."""
        if not self._model:
            return {"status": "error", "message": "SSH agent is unavailable"}

        config = self._model.get_config()
        vault = self._model.get_vault()
        if not getattr(config, "ssh", None) or not config.ssh.enabled:
            return {"status": "error", "message": "SSH is disabled in Settings"}

        host_cfg = next((host for host in config.ssh.allowed_hosts if host.name == host_alias), None)
        if host_cfg is None:
            return {"status": "error", "message": f"Unknown SSH host alias '{host_alias}'"}
        if not host_cfg.private_key_provider:
            return {"status": "error", "message": "The SSH host has no saved private-key provider"}

        key_pem = await vault.get_key(host_cfg.private_key_provider)
        if not key_pem:
            return {"status": "error", "message": "No private key is saved for this SSH host"}
        passphrase = None
        if host_cfg.passphrase_provider:
            passphrase = await vault.get_key(host_cfg.passphrase_provider)

        try:
            import paramiko  # type: ignore

            pkey = _load_private_key(paramiko, key_pem, passphrase)
            client = paramiko.SSHClient()
            try:
                client.load_system_host_keys()
                if host_cfg.strict_host_key_checking:
                    client.set_missing_host_key_policy(paramiko.RejectPolicy())
                else:
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    hostname=host_cfg.hostname,
                    port=host_cfg.port,
                    username=host_cfg.username,
                    pkey=pkey,
                    timeout=config.ssh.connect_timeout_seconds,
                    banner_timeout=config.ssh.connect_timeout_seconds,
                    auth_timeout=config.ssh.connect_timeout_seconds,
                )
            finally:
                client.close()
        except ImportError:
            return {
                "status": "error",
                "message": "Paramiko is not installed. Install pilot-daemon[ssh].",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SSH connection test failed for alias %s: %s", host_alias, exc)
            return {"status": "error", "message": str(exc)}

        return {
            "status": "ok",
            "alias": host_cfg.name,
            "hostname": host_cfg.hostname,
            "strict_host_key_checking": host_cfg.strict_host_key_checking,
        }

    async def run_probe(self, host_alias: str, command: str, *, timeout_seconds: int) -> tuple[int, str, str]:
        """Run one read-only check command on an allow-listed host over a second, new connection.

        Used by declared post-conditions (``pilot.agents.postconditions.declared``), which decide beforehand that the
        command is read-only. It connects exactly as an ssh action does: same host alias, key from the vault, the
        host's ``strict_host_key_checking`` and the configured connect timeout. Output is read up to 64 KiB.
        Raises ``RuntimeError`` with a reason a person can read when the host cannot be reached.
        """
        if not self._model:
            raise RuntimeError("SSH is unavailable")
        config = self._model.get_config()
        vault = self._model.get_vault()
        if not getattr(config, "ssh", None) or not config.ssh.enabled:
            raise RuntimeError("SSH is disabled in Settings")
        host_cfg = next(
            (h for h in config.ssh.allowed_hosts if h.name == host_alias and h.hostname and h.username),
            None,
        )
        if host_cfg is None:
            raise RuntimeError(f"Unknown SSH host alias '{host_alias}'")
        if not host_cfg.private_key_provider:
            raise RuntimeError(f"SSH host '{host_alias}' has no private key saved in Settings")
        key_pem = await vault.get_key(host_cfg.private_key_provider)
        if not key_pem:
            raise RuntimeError(f"No private key is saved for SSH host '{host_alias}'")
        passphrase = await vault.get_key(host_cfg.passphrase_provider) if host_cfg.passphrase_provider else None
        try:
            import paramiko  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Paramiko is not installed. Install pilot-daemon[ssh].") from exc

        def _run() -> tuple[int, str, str]:
            pkey = _load_private_key(paramiko, key_pem, passphrase)
            client = paramiko.SSHClient()
            try:
                client.load_system_host_keys()
                if host_cfg.strict_host_key_checking:
                    client.set_missing_host_key_policy(paramiko.RejectPolicy())
                else:
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    hostname=host_cfg.hostname,
                    port=host_cfg.port,
                    username=host_cfg.username,
                    pkey=pkey,
                    timeout=config.ssh.connect_timeout_seconds,
                    banner_timeout=config.ssh.connect_timeout_seconds,
                    auth_timeout=config.ssh.connect_timeout_seconds,
                )
                stdin, stdout, stderr = client.exec_command(command, timeout=timeout_seconds)
                try:
                    stdin.close()
                except Exception:
                    pass
                out = stdout.read(PROBE_OUTPUT_LIMIT).decode("utf-8", errors="replace")
                err = stderr.read(PROBE_OUTPUT_LIMIT).decode("utf-8", errors="replace")
                return int(stdout.channel.recv_exit_status()), out, err
            finally:
                client.close()

        return await asyncio.to_thread(_run)

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        # AgentOrchestrator gates this direct Paramiko path before dispatch and
        # applies the caller's scope override.
        import time

        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        if not self._model:
            self.status = AgentStatus.IDLE
            return [
                ActionResult(
                    action=a,
                    success=False,
                    error="SSH Agent is unavailable (missing model_router context).",
                )
                for a in my_actions
            ]

        config = self._model.get_config()
        vault = self._model.get_vault()

        if not getattr(config, "ssh", None) or not config.ssh.enabled:
            self.status = AgentStatus.IDLE
            return [
                ActionResult(
                    action=a,
                    success=False,
                    error="SSH is disabled. Enable it in Settings > Integrations > SSH.",
                )
                for a in my_actions
            ]

        # Build allowed host mapping for fast lookup
        allowed = {h.name: h for h in config.ssh.allowed_hosts if h.name and h.hostname and h.username}

        results: list[ActionResult] = []

        try:
            import paramiko  # type: ignore
        except Exception:  # noqa: BLE001
            self.status = AgentStatus.IDLE
            return [
                ActionResult(
                    action=a,
                    success=False,
                    error="Paramiko is not installed. Install with `pip install 'pilot-daemon[ssh]'`.",
                )
                for a in my_actions
            ]

        for action in my_actions:
            params = action.parameters
            host_alias = ""
            command: str | None = None
            script: str | None = None
            timeout = 60

            if isinstance(params, SshCommandParams):
                host_alias = params.host
                command = params.command
                timeout = params.timeout_seconds
            elif isinstance(params, SshScriptParams):
                host_alias = params.host
                script = params.script
                timeout = params.timeout_seconds

            host_cfg = allowed.get(host_alias)
            if not host_cfg:
                results.append(
                    ActionResult(
                        action=action,
                        success=False,
                        error=(
                            f"Unknown SSH host alias '{host_alias}'. "
                            "Add it in Settings > Integrations > SSH before using ssh_* actions."
                        ),
                    )
                )
                continue

            key_provider = host_cfg.private_key_provider
            if not key_provider:
                results.append(
                    ActionResult(
                        action=action,
                        success=False,
                        error=f"SSH host '{host_alias}' has no private key saved in Settings.",
                    )
                )
                continue

            key_pem = await vault.get_key(key_provider)
            if not key_pem:
                results.append(
                    ActionResult(
                        action=action,
                        success=False,
                        error=f"No private key found in KeyVault for provider '{key_provider}'.",
                    )
                )
                continue

            passphrase: str | None = None
            if host_cfg.passphrase_provider:
                passphrase = await vault.get_key(host_cfg.passphrase_provider)

            try:
                pkey = _load_private_key(paramiko, key_pem, passphrase)
            except Exception as e:  # noqa: BLE001
                results.append(ActionResult(action=action, success=False, error=f"Invalid SSH key: {e}"))
                continue

            client = paramiko.SSHClient()
            try:
                client.load_system_host_keys()
                if host_cfg.strict_host_key_checking:
                    client.set_missing_host_key_policy(paramiko.RejectPolicy())
                else:
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                client.connect(
                    hostname=host_cfg.hostname,
                    port=host_cfg.port,
                    username=host_cfg.username,
                    pkey=pkey,
                    timeout=config.ssh.connect_timeout_seconds,
                    banner_timeout=config.ssh.connect_timeout_seconds,
                    auth_timeout=config.ssh.connect_timeout_seconds,
                )

                if command is not None:
                    stdout_text, stderr_text, exit_code = self._exec_command(client, command, timeout_seconds=timeout)
                    ok = exit_code == 0
                    out = stdout_text.strip()
                    err = stderr_text.strip() or None
                    results.append(
                        ActionResult(
                            action=action,
                            success=ok,
                            output=out,
                            error=err if not ok else None,
                        )
                    )
                else:
                    # script mode
                    stdout_text, stderr_text, exit_code = self._exec_script(
                        client, script or "", timeout_seconds=timeout
                    )
                    ok = exit_code == 0
                    out = stdout_text.strip()
                    err = stderr_text.strip() or None
                    results.append(
                        ActionResult(
                            action=action,
                            success=ok,
                            output=out,
                            error=err if not ok else None,
                        )
                    )
            except Exception as e:  # noqa: BLE001
                results.append(ActionResult(action=action, success=False, error=str(e)))
            finally:
                try:
                    client.close()
                except Exception:
                    pass

        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results

    @staticmethod
    def _exec_command(client: Any, command: str, *, timeout_seconds: int) -> tuple[str, str, int]:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout_seconds)
        # Ensure stdin closed to avoid hanging remote expecting input
        try:
            stdin.close()
        except Exception:
            pass
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, int(exit_code)

    @classmethod
    def _exec_script(cls, client: Any, script: str, *, timeout_seconds: int) -> tuple[str, str, int]:
        # Use bash -s to read script from stdin; -l to load profile; -e to stop on error.
        # Note: `-l` may be slow on some systems; adjust later if needed.
        stdin, stdout, stderr = client.exec_command("bash -leu -s", timeout=timeout_seconds)
        stdin.write(script)
        if not script.endswith("\n"):
            stdin.write("\n")
        stdin.close()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, int(exit_code)

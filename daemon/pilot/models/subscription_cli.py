"""Subscription-authenticated model backends through official local CLIs.

The client deliberately delegates authentication to Codex CLI or Claude Code.
It never reads, copies, logs, or persists OAuth credentials.  Both backends run
as text-only inference helpers: Codex receives a sterile read-only workspace and
Claude has every built-in tool disabled.  Heliox remains the sole authority for
action validation, approval, execution, and verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pilot.timeouts import bounded

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.models.subscription_cli")

SUPPORTED_SUBSCRIPTION_PROVIDERS = frozenset({"codex", "claude"})
_MAX_STDOUT_BYTES = 2 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_PROBE_TIMEOUT_SECONDS = 10.0
_TEXT_ONLY_INSTRUCTION = (
    "You are a text-only inference backend inside Heliox OS. Do not inspect files, "
    "call tools, browse, or execute commands. Use only the messages supplied below. "
    "If JSON is requested, return only valid JSON without Markdown fences."
)
_TOOL_EVENT_TYPES = frozenset(
    {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
        "dynamic_tool_call",
    }
)


@dataclass(frozen=True)
class CLIResult:
    returncode: int
    stdout: str
    stderr: str


def _serialize_messages(
    prompt: str | list[dict[str, Any]],
    system: str,
    *,
    include_guard: bool = True,
) -> str:
    """Serialize chat messages compactly without repeating system content."""

    parts = [_TEXT_ONLY_INSTRUCTION] if include_guard else []
    system_text = system.strip()
    if system_text:
        parts.append(f"SYSTEM\n{system_text}")

    if isinstance(prompt, str):
        parts.append(f"USER\n{prompt.strip()}")
    else:
        for message in prompt:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = str(message.get("role", "user")).upper()
            if role == "SYSTEM" and content == system_text:
                continue
            parts.append(f"{role}\n{content}")
    return "\n\n".join(parts)


def _fit_messages_to_char_budget(
    prompt: str | list[dict[str, Any]],
    system: str,
    max_chars: int,
    *,
    include_guard: bool,
) -> str | list[dict[str, Any]]:
    """Drop oldest optional chat context to meet the exact serialized cap."""

    if isinstance(prompt, str):
        return prompt
    indexed = [(index, message.copy()) for index, message in enumerate(prompt)]
    if len(_serialize_messages([item[1] for item in indexed], system, include_guard=include_guard)) <= max_chars:
        return [item[1] for item in indexed]

    latest_index = indexed[-1][0] if indexed else -1
    system_index = next(
        (index for index, message in indexed if message.get("role") == "system" and not message.get("is_summary")),
        None,
    )
    goal_index = next((index for index, message in indexed if message.get("type") == "goal"), None)
    protected = {index for index in (system_index, latest_index) if index is not None}
    removable = [index for index, _ in indexed if index not in protected and index != goal_index]
    if goal_index is not None and goal_index not in protected:
        removable.append(goal_index)

    removed: set[int] = set()
    for index in removable:
        removed.add(index)
        candidate = [message for original, message in indexed if original not in removed]
        if len(_serialize_messages(candidate, system, include_guard=include_guard)) <= max_chars:
            return candidate

    mandatory = [(original, message) for original, message in indexed if original not in removed]
    serialized = _serialize_messages([item[1] for item in mandatory], system, include_guard=include_guard)
    if len(serialized) <= max_chars or not mandatory:
        return [item[1] for item in mandatory]

    latest_position = next(
        (position for position, item in enumerate(mandatory) if item[0] == latest_index),
        len(mandatory) - 1,
    )
    original_latest = mandatory[latest_position][1]
    content = str(original_latest.get("content", ""))
    low, high = 0, len(content)
    best = ""
    while low <= high:
        length = (low + high) // 2
        trimmed = original_latest.copy()
        trimmed["content"] = content[-length:] if length else ""
        candidate = [item[1] for item in mandatory]
        candidate[latest_position] = trimmed
        if len(_serialize_messages(candidate, system, include_guard=include_guard)) <= max_chars:
            best = str(trimmed["content"])
            low = length + 1
        else:
            high = length - 1
    mandatory[latest_position][1]["content"] = best
    return [item[1] for item in mandatory]


def _safe_failure(provider: str, stderr: str, *, returncode: int | None = None) -> str:
    """Shape actionable CLI errors without returning paths, tokens, or raw logs."""

    text = stderr.lower()
    display = "Codex" if provider == "codex" else "Claude Code"
    if "rate limit" in text or "usage limit" in text or "quota" in text:
        detail = "the subscription usage limit was reached"
    elif "login" in text or "auth" in text or "credential" in text:
        detail = "subscription authentication is required"
    elif "model" in text and ("not found" in text or "unsupported" in text or "invalid" in text):
        detail = "the selected model is unavailable to this account"
    elif "timed out" in text or "timeout" in text:
        detail = "the request timed out"
    else:
        suffix = f" (exit {returncode})" if returncode not in (None, 0) else ""
        detail = f"the local CLI failed{suffix}"
    return f"{display} subscription unavailable: {detail}. Check the connection in Settings."


class SubscriptionCLIClient:
    """Run official subscription-authenticated CLIs as governed text models."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._status_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._login_watchers: set[asyncio.Task[None]] = set()
        self._processes: set[asyncio.subprocess.Process] = set()
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {}
        self.generation_count = 0
        self._request_usage: weakref.WeakKeyDictionary[asyncio.Task[Any], dict[str, int]] = weakref.WeakKeyDictionary()

    def consume_request_usage(self) -> dict[str, int] | None:
        """Return usage for this asyncio request without cross-call races."""

        task = asyncio.current_task()
        if task is None:
            return None
        return self._request_usage.pop(task, None)

    @staticmethod
    def _resolve_executable(provider: str) -> str | None:
        command = "codex" if provider == "codex" else "claude"
        return shutil.which(command)

    @staticmethod
    def _subscription_environment(provider: str) -> dict[str, str]:
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # A saved subscription login must win over metered API credentials.
        if provider == "codex":
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
        else:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.pop("CLAUDE_CODE_USE_BEDROCK", None)
            env.pop("CLAUDE_CODE_USE_VERTEX", None)
            env.pop("CLAUDE_CODE_USE_FOUNDRY", None)
        return env

    async def _run_process(
        self,
        args: list[str],
        *,
        stdin: str = "",
        cwd: str | None = None,
        provider: str,
        timeout: float,
    ) -> CLIResult:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=self._subscription_environment(provider),
        )
        self._processes.add(proc)
        try:
            stdout, stderr = await bounded(
                proc.communicate(stdin.encode("utf-8")),
                timeout=timeout,
            )
        except TimeoutError:
            await self._terminate_process(proc)
            raise RuntimeError(_safe_failure(provider, "timed out")) from None
        except asyncio.CancelledError:
            await self._terminate_process(proc)
            raise
        finally:
            self._processes.discard(proc)

        if len(stdout) > _MAX_STDOUT_BYTES:
            raise RuntimeError(f"{provider.title()} subscription response exceeded the safe output limit.")
        return CLIResult(
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr[:_MAX_STDERR_BYTES].decode("utf-8", errors="replace"),
        )

    async def status(self, provider: str | None = None, *, refresh: bool = False) -> dict[str, Any]:
        selected = provider or self._config.model.subscription_provider
        if selected not in SUPPORTED_SUBSCRIPTION_PROVIDERS:
            return {
                "provider": selected,
                "installed": False,
                "authenticated": False,
                "subscription": False,
                "message": "Unsupported subscription provider.",
            }

        now = asyncio.get_running_loop().time()
        cached = self._status_cache.get(selected)
        if not refresh and cached and now - cached[0] < 15.0:
            return dict(cached[1])

        executable = self._resolve_executable(selected)
        if not executable:
            result = {
                "provider": selected,
                "installed": False,
                "authenticated": False,
                "subscription": False,
                "version": "",
                "message": f"{'Codex CLI' if selected == 'codex' else 'Claude Code'} is not installed.",
            }
            self._status_cache[selected] = (now, result)
            return dict(result)

        version_result = await self._run_process(
            [executable, "--version"],
            provider=selected,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if selected == "codex":
            auth_args = [executable, "login", "status"]
        else:
            auth_args = [executable, "auth", "status", "--json"]
        auth_result = await self._run_process(
            auth_args,
            provider=selected,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        combined = f"{auth_result.stdout}\n{auth_result.stderr}".strip()
        lowered = combined.lower()

        if selected == "codex":
            authenticated = auth_result.returncode == 0 and "logged in" in lowered
            subscription = authenticated and "chatgpt" in lowered
        else:
            authenticated = False
            subscription = False
            try:
                payload = json.loads(auth_result.stdout)
                authenticated = bool(payload.get("loggedIn") or payload.get("logged_in"))
                auth_method = str(
                    payload.get("authMethod") or payload.get("auth_method") or payload.get("subscriptionType") or ""
                ).lower()
                subscription = authenticated and "api" not in auth_method
            except (json.JSONDecodeError, TypeError):
                authenticated = auth_result.returncode == 0 and ("logged in" in lowered or "authenticated" in lowered)
                subscription = authenticated and "api key" not in lowered

        display = "Codex" if selected == "codex" else "Claude Code"
        if subscription:
            message = (
                "Connected through the Codex ChatGPT subscription login."
                if selected == "codex"
                else f"Connected through the {display} subscription login."
            )
        elif authenticated:
            message = f"{display} is authenticated, but not through a subscription login."
        else:
            message = f"Run {'codex login' if selected == 'codex' else 'claude auth login'} to connect."
        result = {
            "provider": selected,
            "installed": True,
            "authenticated": authenticated,
            "subscription": subscription,
            "version": version_result.stdout.strip().splitlines()[0] if version_result.stdout.strip() else "",
            "message": message,
            "last_usage": dict(self.last_usage),
            "session_usage": dict(self.total_usage),
        }
        self._status_cache[selected] = (now, result)
        return dict(result)

    async def start_login(self, provider: str | None = None) -> dict[str, Any]:
        """Start the official browser login flow without handling credentials."""

        selected = provider or self._config.model.subscription_provider
        if selected not in SUPPORTED_SUBSCRIPTION_PROVIDERS:
            return {"status": "error", "message": "Unsupported subscription provider."}
        executable = self._resolve_executable(selected)
        if not executable:
            return {
                "status": "missing",
                "message": f"{'Codex CLI' if selected == 'codex' else 'Claude Code'} is not installed.",
            }
        args = [executable, "login"] if selected == "codex" else [executable, "auth", "login"]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._subscription_environment(selected),
        )
        self._processes.add(proc)
        task = asyncio.create_task(self._watch_login(selected, proc))
        self._login_watchers.add(task)
        task.add_done_callback(self._login_watchers.discard)
        return {
            "status": "started",
            "message": "The official browser sign-in flow was started. Return here and refresh after completing it.",
        }

    async def _watch_login(self, provider: str, proc: asyncio.subprocess.Process) -> None:
        try:
            await bounded(proc.wait(), timeout=300.0)
        except TimeoutError:
            await self._terminate_process(proc)
        except asyncio.CancelledError:
            await self._terminate_process(proc)
            raise
        finally:
            self._processes.discard(proc)
            self._status_cache.pop(provider, None)

    @staticmethod
    async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await bounded(proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            await proc.wait()

    async def close(self) -> None:
        """Stop login watchers and any child CLI processes owned by this client."""
        watchers = tuple(self._login_watchers)
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        processes = tuple(self._processes)
        if processes:
            await asyncio.gather(
                *(self._terminate_process(proc) for proc in processes),
                return_exceptions=True,
            )
        self._login_watchers.clear()
        self._processes.clear()
        self._status_cache.clear()

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: Any = None,
    ) -> str:
        del temperature  # Coding-agent CLIs own sampling policy.
        provider = self._config.model.subscription_provider
        if provider not in SUPPORTED_SUBSCRIPTION_PROVIDERS:
            raise RuntimeError("Unsupported subscription provider configured.")
        status = await self.status(provider)
        if not status.get("subscription"):
            raise RuntimeError(status.get("message") or "Subscription authentication is required.")

        # Claude receives the guard through --system-prompt, so omitting it
        # from stdin avoids paying for the same instruction twice. Codex has
        # no equivalent system flag and retains the inline guard.
        max_chars = self._config.model.subscription_max_prompt_chars
        include_guard = provider == "codex"
        prompt = _fit_messages_to_char_budget(
            prompt,
            system,
            max_chars,
            include_guard=include_guard,
        )
        serialized = _serialize_messages(prompt, system, include_guard=include_guard)
        if len(serialized) > max_chars:
            raise RuntimeError(
                f"Subscription prompt is {len(serialized)} characters, exceeding the configured "
                f"limit of {max_chars}. Compact the task context or raise the limit in Settings."
            )

        executable = self._resolve_executable(provider)
        if not executable:
            raise RuntimeError(f"{'Codex CLI' if provider == 'codex' else 'Claude Code'} is not installed.")
        model = self._config.model.subscription_model.strip()
        timeout = float(self._config.model.subscription_timeout_seconds)
        temp_dir = tempfile.mkdtemp(prefix="heliox-subscription-")
        try:
            if provider == "codex":
                args = [
                    executable,
                    "exec",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--json",
                    "-C",
                    temp_dir,
                ]
                if model:
                    args.extend(["--model", model])
                args.append("-")
                result = await self._run_process(
                    args,
                    stdin=serialized,
                    cwd=temp_dir,
                    provider=provider,
                    timeout=timeout,
                )
                response = self._parse_codex(result)
            else:
                args = [
                    executable,
                    "--bare",
                    "-p",
                    "Use the piped input as the complete conversation and follow it exactly.",
                    "--output-format",
                    "json",
                    "--no-session-persistence",
                    "--no-chrome",
                    "--disable-slash-commands",
                    "--tools",
                    "",
                    "--max-turns",
                    "1",
                    "--system-prompt",
                    _TEXT_ONLY_INSTRUCTION,
                ]
                if model:
                    args.extend(["--model", model])
                if json_mode:
                    args.extend(
                        [
                            "--json-schema",
                            '{"type":"object","additionalProperties":true}',
                        ]
                    )
                result = await self._run_process(
                    args,
                    stdin=serialized,
                    cwd=temp_dir,
                    provider=provider,
                    timeout=timeout,
                )
                response = self._parse_claude(result, json_mode=json_mode)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.generation_count += 1
        self.last_usage["heliox_prompt_chars"] = len(serialized)
        self.last_usage["heliox_estimated_prompt_tokens"] = (len(serialized) + 3) // 4
        if "input_tokens" in self.last_usage:
            self.last_usage["uncached_input_tokens"] = max(
                0,
                self.last_usage["input_tokens"] - self.last_usage.get("cached_input_tokens", 0),
            )
        for key in ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens"):
            self.total_usage[key] = self.total_usage.get(key, 0) + self.last_usage.get(key, 0)
        task = asyncio.current_task()
        if task is not None:
            self._request_usage[task] = dict(self.last_usage)
        # Status responses carry usage counters. Discard the pre-generation
        # authentication probe so the next ordinary status request cannot
        # return stale quota information for up to the cache TTL.
        self._status_cache.pop(provider, None)

        if stream_callback:
            await stream_callback(response)
        return response

    def _parse_codex(self, result: CLIResult) -> str:
        if result.returncode != 0:
            raise RuntimeError(_safe_failure("codex", result.stderr, returncode=result.returncode))
        messages: list[str] = []
        usage: dict[str, int] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = str(event.get("type", ""))
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            item_type = str(item.get("type", ""))
            if item_type in _TOOL_EVENT_TYPES:
                raise RuntimeError("Codex subscription attempted a tool call; Heliox rejected the model response.")
            if event_type == "item.completed" and item_type == "agent_message":
                text = str(item.get("text", "")).strip()
                if text:
                    messages.append(text)
            if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = {
                    key: int(value)
                    for key, value in event["usage"].items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            if event_type in {"turn.failed", "error"}:
                raise RuntimeError(_safe_failure("codex", result.stderr, returncode=result.returncode))
        if not messages:
            raise RuntimeError(_safe_failure("codex", result.stderr, returncode=result.returncode))
        self.last_usage = usage
        return messages[-1]

    def _parse_claude(self, result: CLIResult, *, json_mode: bool) -> str:
        if result.returncode != 0:
            raise RuntimeError(_safe_failure("claude", result.stderr, returncode=result.returncode))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            text = result.stdout.strip()
            if not text:
                raise RuntimeError(_safe_failure("claude", result.stderr, returncode=result.returncode)) from None
            self.last_usage = {}
            return text
        if payload.get("is_error"):
            raise RuntimeError(_safe_failure("claude", str(payload.get("result", ""))))
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        self.last_usage = {
            key: int(value) for key, value in usage.items() if isinstance(value, int) and not isinstance(value, bool)
        }
        structured = payload.get("structured_output")
        if json_mode and structured is not None:
            return json.dumps(structured, separators=(",", ":"))
        response = payload.get("result")
        if response is None:
            raise RuntimeError(_safe_failure("claude", result.stderr, returncode=result.returncode))
        return str(response).strip()

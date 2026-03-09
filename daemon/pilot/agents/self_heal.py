"""Self-Healing / Adaptive Retry — intelligent error recovery.

When an action fails, automatically tries alternative strategies
instead of giving up. Feeds error context back to the LLM for
re-planning, with fallback chains and failure learning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.agents.self_heal")


class FailureMemory:
    """Learns from past failures to avoid repeating mistakes."""

    def __init__(self, storage_path: str | None = None):
        self._storage = storage_path or os.path.expanduser("~/.pilot/failure_memory.json")
        self._failures: dict[str, list[dict]] = self._load()

    def _load(self) -> dict:
        try:
            if os.path.exists(self._storage):
                return json.loads(Path(self._storage).read_text())
        except Exception:
            pass
        return {}

    def _save(self):
        Path(self._storage).parent.mkdir(parents=True, exist_ok=True)
        Path(self._storage).write_text(json.dumps(self._failures, indent=2))

    def record_failure(self, action_type: str, error: str, context: dict):
        """Record a failure for learning."""
        if action_type not in self._failures:
            self._failures[action_type] = []

        self._failures[action_type].append(
            {
                "error": error[:500],
                "context": {k: str(v)[:200] for k, v in context.items()},
                "timestamp": time.time(),
            }
        )

        # Keep last 20 per action type
        self._failures[action_type] = self._failures[action_type][-20:]
        self._save()

    def get_failure_context(self, action_type: str) -> str:
        """Get past failure context for an action type (to feed to LLM)."""
        failures = self._failures.get(action_type, [])
        if not failures:
            return ""

        recent = failures[-5:]
        lines = [f"Previous failures for '{action_type}':"]
        for f in recent:
            lines.append(f"  - Error: {f['error'][:100]}")
        return "\n".join(lines)

    def clear(self):
        """Clear all failure memory."""
        self._failures = {}
        self._save()


# Predefined fallback chains for common operations
FALLBACK_CHAINS: dict[str, list[dict[str, Any]]] = {
    # Package install fallbacks
    "package_install": [
        {"strategy": "winget", "command": "winget install {name}"},
        {"strategy": "choco", "command": "choco install {name} -y"},
        {"strategy": "scoop", "command": "scoop install {name}"},
        {"strategy": "pip", "command": "pip install {name}"},
    ],
    # File operations
    "file_delete": [
        {"strategy": "normal", "command": "Remove-Item '{path}'"},
        {"strategy": "force", "command": "Remove-Item '{path}' -Force"},
        {
            "strategy": "admin",
            "command": "Start-Process powershell -Verb RunAs -ArgumentList 'Remove-Item \\\"{path}\\\" -Force'",
        },
    ],
    # Service management
    "service_restart": [
        {"strategy": "normal", "command": "Restart-Service '{name}'"},
        {"strategy": "stop_start", "command": "Stop-Service '{name}'; Start-Sleep 2; Start-Service '{name}'"},
        {"strategy": "kill_start", "command": "Stop-Process -Name '{name}' -Force; Start-Service '{name}'"},
    ],
    # Process kill
    "process_kill": [
        {"strategy": "graceful", "signal": "SIGTERM"},
        {"strategy": "force", "signal": "SIGKILL"},
        {"strategy": "taskkill", "command": "taskkill /F /PID {pid}"},
    ],
    # Network
    "wifi_connect": [
        {"strategy": "normal", "command": "netsh wlan connect name='{ssid}'"},
        {"strategy": "profile_add", "command": "netsh wlan connect ssid='{ssid}' name='{ssid}'"},
        {
            "strategy": "disconnect_reconnect",
            "command": "netsh wlan disconnect; Start-Sleep 2; netsh wlan connect name='{ssid}'",
        },
    ],
}


class SelfHealingWrapper:
    """Wraps the executor with self-healing capabilities.

    When an action fails:
    1. Check fallback chains for predefined alternatives
    2. Record the failure for learning
    3. Generate error context for LLM re-planning
    4. Automatically retry with alternative strategy
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.memory = FailureMemory()
        self._retry_count: dict[str, int] = defaultdict(int)

    def get_fallback_chain(self, action_type: str) -> list[dict]:
        """Get fallback strategies for an action type."""
        return FALLBACK_CHAINS.get(action_type, [])

    def build_retry_context(
        self,
        action_type: str,
        original_error: str,
        parameters: dict,
        attempt: int,
    ) -> str:
        """Build rich error context for LLM re-planning.

        This is sent back to the planner so it can generate a Plan B.
        """
        past_failures = self.memory.get_failure_context(action_type)
        fallbacks = self.get_fallback_chain(action_type)

        context_parts = [
            f"ACTION FAILED: {action_type}",
            f"Error: {original_error}",
            f"Attempt: {attempt}/{self.max_retries}",
            f"Parameters: {json.dumps(parameters, default=str)[:300]}",
        ]

        if past_failures:
            context_parts.append(f"\n{past_failures}")

        if fallbacks:
            remaining = fallbacks[attempt:] if attempt < len(fallbacks) else []
            if remaining:
                context_parts.append("\nAvailable fallback strategies: " + ", ".join(f["strategy"] for f in remaining))

        context_parts.append(
            "\nPlease generate an alternative approach to accomplish the same goal. "
            "Use a different method or strategy. Be creative."
        )

        return "\n".join(context_parts)

    def record_failure(
        self,
        action_type: str,
        error: str,
        parameters: dict,
    ):
        """Record a failure for future learning."""
        self.memory.record_failure(action_type, error, parameters)

    def should_retry(self, action_type: str) -> bool:
        """Check if we should retry this action type."""
        return self._retry_count[action_type] < self.max_retries

    def increment_retry(self, action_type: str):
        """Increment retry counter."""
        self._retry_count[action_type] += 1

    def reset_retries(self, action_type: str):
        """Reset retry counter after success."""
        self._retry_count[action_type] = 0

    def get_next_fallback(self, action_type: str, attempt: int) -> dict | None:
        """Get the next fallback strategy."""
        chain = self.get_fallback_chain(action_type)
        if attempt < len(chain):
            return chain[attempt]
        return None


async def self_heal_execute(
    executor,
    action,
    planner=None,
) -> dict:
    """Execute an action with self-healing.

    Returns: {"success": bool, "output": str, "attempts": int, "strategy": str}
    """
    healer = SelfHealingWrapper()
    action_type = action.action_type.value

    for attempt in range(healer.max_retries + 1):
        try:
            from pilot.actions import ActionPlan

            plan = ActionPlan(actions=[action])
            results = await executor.execute(plan)

            if results and results[0].success:
                healer.reset_retries(action_type)
                return {
                    "success": True,
                    "output": results[0].output,
                    "attempts": attempt + 1,
                    "strategy": "original" if attempt == 0 else f"fallback_{attempt}",
                }

            error = results[0].error if results else "Unknown error"
            raise RuntimeError(error)

        except Exception as e:
            error_str = str(e)
            healer.record_failure(action_type, error_str, {})

            if attempt < healer.max_retries:
                # Try fallback
                fallback = healer.get_next_fallback(action_type, attempt + 1)
                if fallback:
                    logger.info(
                        "Action %s failed, trying fallback strategy: %s",
                        action_type,
                        fallback.get("strategy", "unknown"),
                    )
                    # If we have a planner, ask it to re-plan
                    if planner:
                        retry_ctx = healer.build_retry_context(action_type, error_str, {}, attempt + 1)
                        try:
                            new_plan = await planner.plan(retry_ctx)
                            if new_plan.actions:
                                action = new_plan.actions[0]
                                continue
                        except Exception:
                            pass
                    continue

            # All retries exhausted
            return {
                "success": False,
                "output": f"All {attempt + 1} attempts failed. Last error: {error_str}",
                "attempts": attempt + 1,
                "strategy": "exhausted",
            }

    return {
        "success": False,
        "output": "Max retries exceeded",
        "attempts": healer.max_retries + 1,
        "strategy": "exhausted",
    }

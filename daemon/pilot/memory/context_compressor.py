"""Rolling context-window compression for long-running ReAct loops."""

import logging
from typing import Any

import tiktoken

from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.memory.context_compressor")

DEFAULT_MAX_TOKENS = 8000
COMPRESSION_THRESHOLD = 6000
NUM_RESERVED_TOKENS = 2000


class ContextCompressor:
    """Compresses conversation history to keep token buffer under limit."""

    def __init__(
        self,
        model_router: ModelRouter,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._model = model_router
        self._max_tokens = max_tokens
        self._threshold = COMPRESSION_THRESHOLD
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self._enc.encode(text))

    def estimate_action_tokens(self, action_dict: dict[str, Any]) -> int:
        """Estimate tokens for an action."""
        parts = [
            action_dict.get("action_type", ""),
            action_dict.get("target", ""),
            str(action_dict.get("parameters", {})),
        ]
        return self.count_tokens(" ".join(parts))

    def estimate_plan_tokens(self, plan_dict: dict[str, Any]) -> int:
        """Estimate tokens for an entire plan."""
        text = plan_dict.get("explanation", "")
        for action in plan_dict.get("actions", []):
            text += " " + str(action)
        return self.count_tokens(text)

    async def compress_conversation(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress conversation history if exceeding token limit.

        Args:
            history: List of conversation items (each has 'user_input', 'plan', 'result')

        Returns:
            Compressed history with summaries for older items
        """
        total_tokens = 0
        for item in history:
            if "user_input" in item and isinstance(item["user_input"], str):
                total_tokens += self.count_tokens(item["user_input"])
            if "plan" in item and isinstance(item["plan"], dict):
                total_tokens += self.estimate_plan_tokens(item["plan"])
            if "result" in item and isinstance(item.get("result"), dict):
                total_tokens += self.estimate_action_tokens(item.get("result", {}))

        if total_tokens < self._threshold:
            return history

        logger.info("Context compression triggered: %d tokens (threshold: %d)", total_tokens, self._threshold)

        recent_items = []
        compressed_items = []
        running_tokens = 0

        for item in reversed(history):
            item_tokens = 0
            if "user_input" in item and isinstance(item["user_input"], str):
                item_tokens += self.count_tokens(item["user_input"])
            if "plan" in item and isinstance(item["plan"], dict):
                item_tokens += self.estimate_plan_tokens(item["plan"])
            if "result" in item and isinstance(item.get("result"), dict):
                item_tokens += self.estimate_action_tokens(item.get("result", {}))

            if running_tokens + item_tokens < self._max_tokens - NUM_RESERVED_TOKENS:
                recent_items.insert(0, item)
                running_tokens += item_tokens
            else:
                compressed_items.append(item)

        if compressed_items:
            summary = await self._summarize_items(compressed_items)
            if summary:
                summary_item = {
                    "type": "compressed_summary",
                    "summary": summary,
                    "original_count": len(compressed_items),
                }
                return [summary_item] + recent_items

        return recent_items

    async def _summarize_items(self, items: list[dict[str, Any]]) -> str:
        """Use smaller LLM to summarize older items."""
        summary_text = "## Compressed History Summary\n\n"

        for item in items[:10]:
            user_input = item.get("user_input", "Unknown")
            plan = item.get("plan", {})
            explanation = plan.get("explanation", "") if isinstance(plan, dict) else ""
            actions = plan.get("actions", []) if isinstance(plan, dict) else []
            action_types = [a.get("action_type", "?") for a in actions[:3]]
            summary_text += f"- Input: {user_input[:50]}... → Actions: {', '.join(action_types)}\n"

        if len(items) > 10:
            summary_text += f"- ... and {len(items) - 10} more items\n"

        return summary_text

    async def compress_prompt(
        self,
        prompt: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Compress conversation and return updated prompt with context.

        Args:
            prompt: Current user prompt
            history: Full conversation history

        Returns:
            Tuple of (updated_prompt, compressed_history)
        """
        prompt_tokens = self.count_tokens(prompt)
        available_tokens = self._max_tokens - prompt_tokens - NUM_RESERVED_TOKENS

        if available_tokens < 1000:
            logger.warning("Very limited token budget: %d", available_tokens)
            history = await self.compress_conversation(history)

        return prompt, history


class RollingContextWindow:
    """Manages rolling window of context with automatic compression."""

    def __init__(
        self,
        model_router: ModelRouter,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._compressor = ContextCompressor(model_router, max_tokens)
        self._history: list[dict[str, Any]] = []

    async def add_turn(
        self,
        user_input: str,
        plan: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn."""
        turn = {
            "user_input": user_input,
            "plan": plan,
        }
        if result:
            turn["result"] = result

        self._history.append(turn)
        self._history = await self._compressor.compress_conversation(self._history)

    def get_history(self) -> list[dict[str, Any]]:
        """Get current history."""
        return self._history.copy()

    def get_context_text(self) -> str:
        """Get formatted context from history."""
        parts = []
        for item in self._history:
            if item.get("type") == "compressed_summary":
                parts.append(item.get("summary", ""))
            else:
                user_input = item.get("user_input", "")
                plan = item.get("plan", {})
                if isinstance(plan, dict):
                    explanation = plan.get("explanation", "")
                    parts.append(f"User: {user_input}\nPlan: {explanation}")
        return "\n\n".join(parts)

    async def clear(self) -> None:
        """Clear history."""
        self._history = []

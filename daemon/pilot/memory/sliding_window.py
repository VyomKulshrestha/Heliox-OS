"""Dynamic sliding context window for long-running AI agent tasks.

This module provides utilities to prevent token overflow by intelligently
compressing the middle of a conversation history while preserving the
system prompt, initial goal, and the most recent N interactions.
"""

import logging
from typing import Any

import tiktoken

logger = logging.getLogger("pilot.memory.sliding_window")


def get_token_count(text: str) -> int:
    """Return the number of tokens in a string using cl100k_base encoding."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Falling back to character count heuristic.")
        return len(text) // 4


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Calculate the total token count for a list of messages."""
    total = 0
    for msg in messages:
        total += get_token_count(str(msg.get("content", "")))
        total += get_token_count(str(msg.get("role", "")))
        total += 4  # Formatting overhead per message
    return total


def summarize_messages(messages: list[dict[str, Any]], max_summary_items: int = 50) -> str:
    """Create a structured summary of older messages.

    This is a fast heuristic summarization approach. For future enhancement,
    this could be swapped out with an LLM-based summary.
    """
    summary_parts = ["## Compressed History Summary\n"]

    # Intelligently truncate if there are too many messages
    if len(messages) > max_summary_items:
        half = max_summary_items // 2
        messages_to_summarize = (
            messages[:half]
            + [{"role": "system", "content": f"... [{len(messages) - max_summary_items} messages omitted] ..."}]
            + messages[-half:]
        )
    else:
        messages_to_summarize = messages

    for msg in messages_to_summarize:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip()

        if role == "system" and "Compressed History Summary" in content:
            # Flatten existing summary
            lines = content.split("\n")
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith("##"):
                    summary_parts.append(line_stripped)
        else:
            if len(content) > 100:
                content = content[:97] + "..."
            summary_parts.append(f"- {role.capitalize()}: {content}")

    return "\n".join(summary_parts)


def build_sliding_context(
    messages: list[dict[str, Any]], max_recent_messages: int = 10, max_context_tokens: int = 8000
) -> list[dict[str, Any]]:
    """Build an optimized sliding context window to prevent token overflow.

    Args:
        messages: The full list of conversation messages.
        max_recent_messages: Number of recent messages to preserve exactly.
        max_context_tokens: Token threshold that triggers summarization.

    Returns:
        An optimized list of messages under the token limit.
    """
    if not messages:
        return []

    current_tokens = count_message_tokens(messages)

    # Check if we need to slide the window
    if current_tokens <= max_context_tokens:
        return messages.copy()

    logger.info(f"Sliding window activated. Current tokens: {current_tokens} > Limit: {max_context_tokens}")

    optimized_context = []

    # 1. ALWAYS preserve system prompt (index 0)
    system_prompt = messages[0]
    optimized_context.append(system_prompt)

    # If the conversation is very short, just return (should be caught by token limit check)
    if len(messages) <= max_recent_messages + 2:
        return messages.copy()

    # 2. ALWAYS preserve initial user goal/task (index 1)
    # We assume index 1 is the initial user goal if it exists and is not the last messages part
    initial_goal = None
    if len(messages) > 1:
        initial_goal = messages[1]
        optimized_context.append(initial_goal)

    # 3. Preserve last N raw interactions
    recent_messages = messages[-max_recent_messages:]

    # 4. Summarize middle history
    middle_start_idx = 2 if initial_goal else 1
    middle_end_idx = len(messages) - max_recent_messages

    if middle_end_idx > middle_start_idx:
        middle_messages = messages[middle_start_idx:middle_end_idx]
        summarized_history = summarize_messages(middle_messages)

        logger.info(f"Summarized {len(middle_messages)} messages.")

        summary_msg = {"role": "system", "content": summarized_history}
        optimized_context.append(summary_msg)

    # Add the recent messages back
    optimized_context.extend(recent_messages)
    logger.info(f"Preserved last {len(recent_messages)} interactions.")

    return optimized_context

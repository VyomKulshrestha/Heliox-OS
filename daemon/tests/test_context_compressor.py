"""Tests for context_compressor module."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pilot.memory.context_compressor import (
    COMPRESSION_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    ContextCompressor,
    RollingContextWindow,
)


class TestContextCompressor:
    """Tests for ContextCompressor class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model router."""
        model = MagicMock()
        return model

    @pytest.fixture
    def compressor(self, mock_model):
        """Create ContextCompressor instance."""
        return ContextCompressor(mock_model, max_tokens=8000)

    def test_initial_state(self, compressor):
        """Test compressor initializes with correct defaults."""
        assert compressor._max_tokens == 8000
        assert compressor._threshold == 6000

    def test_count_tokens_empty(self, compressor):
        """Test token count for empty string."""
        assert compressor.count_tokens("") == 0

    def test_count_tokens_simple(self, compressor):
        """Test token count for simple text."""
        tokens = compressor.count_tokens("Hello world")
        assert tokens > 0

    def test_estimate_action_tokens(self, compressor):
        """Test action token estimation."""
        action = {
            "action_type": "open_application",
            "target": "notepad",
            "parameters": {"name": "notepad"},
        }
        tokens = compressor.estimate_action_tokens(action)
        assert tokens > 0

    def test_estimate_plan_tokens(self, compressor):
        """Test plan token estimation."""
        plan = {
            "explanation": "Open notepad and write hello",
            "actions": [
                {"action_type": "open_application", "target": "notepad", "parameters": {}},
            ],
        }
        tokens = compressor.estimate_plan_tokens(plan)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_compress_conversation_no_compression(self, compressor):
        """Test compression doesn't trigger when under threshold."""
        history = [
            {"user_input": "Open notepad", "plan": {"explanation": "Opening notepad", "actions": []}},
            {"user_input": "Close notepad", "plan": {"explanation": "Closing notepad", "actions": []}},
        ]
        result = await compressor.compress_conversation(history)
        assert result == history

    @pytest.mark.asyncio
    async def test_compress_conversation_with_compression(self, compressor):
        """Test compression triggers when over threshold."""
        large_history = []
        for i in range(400):
            large_history.append(
                {
                    "user_input": f"Task {i}",
                    "plan": {
                        "explanation": f"Explanation for task {i} with lots of text",
                        "actions": [
                            {"action_type": "action", "target": f"target_{i}", "parameters": {"key": "value"}},
                        ],
                    },
                }
            )

        result = await compressor.compress_conversation(large_history)
        assert len(result) < len(large_history)
        assert result[0].get("type") == "compressed_summary"

    @pytest.mark.asyncio
    async def test_compress_prompt_with_history(self, compressor):
        """Test compress_prompt returns history."""
        prompt = "What's the weather?"
        history = [
            {"user_input": "Hello", "plan": {"explanation": "Greeting", "actions": []}},
        ]
        result_prompt, result_history = await compressor.compress_prompt(prompt, history)
        assert result_prompt == prompt
        assert len(result_history) <= len(history)


class TestRollingContextWindow:
    """Tests for RollingContextWindow class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model router."""
        return MagicMock()

    @pytest.fixture
    def window(self, mock_model):
        """Create RollingContextWindow instance."""
        return RollingContextWindow(mock_model, max_tokens=8000)

    def test_initial_state(self, window):
        """Test window starts empty."""
        assert window.get_history() == []
        assert window.get_context_text() == ""

    @pytest.mark.asyncio
    async def test_add_turn(self, window):
        """Test adding a conversation turn."""
        plan = {"explanation": "Open notepad", "actions": []}
        await window.add_turn("Open notepad", plan)

        history = window.get_history()
        assert len(history) == 1
        assert history[0]["user_input"] == "Open notepad"

    @pytest.mark.asyncio
    async def test_add_turn_with_result(self, window):
        """Test adding turn with result."""
        plan = {"explanation": "Open notepad", "actions": []}
        result = {"success": True, "output": "Opened"}

        await window.add_turn("Open notepad", plan, result)

        history = window.get_history()
        assert "result" in history[0]
        assert history[0]["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_get_context_text(self, window):
        """Test getting formatted context."""
        await window.add_turn("Hello", {"explanation": "Greeting", "actions": []})

        context = window.get_context_text()
        assert "Hello" in context
        assert "Greeting" in context

    @pytest.mark.asyncio
    async def test_clear(self, window):
        """Test clearing history."""
        await window.add_turn("Hello", {"explanation": "Greeting", "actions": []})
        await window.clear()

        assert window.get_history() == []
        assert window.get_context_text() == ""

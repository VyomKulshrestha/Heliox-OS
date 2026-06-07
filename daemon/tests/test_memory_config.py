"""Tests for memory configuration validation and wiring."""

from pathlib import Path

from pilot.config import MemoryConfig, PilotConfig, _merge_config, _validate_config_types


def test_memory_config_includes_pruning_controls():
    config = MemoryConfig()

    assert config.pruning_interval_seconds == 3600
    assert config.pruning_min_memories == 10


def test_memory_config_validation_accepts_pruning_keys():
    raw = {
        "memory": {
            "checkpoint_interval_seconds": 120,
            "pruning_interval_seconds": 900,
            "pruning_min_memories": 25,
        }
    }

    _validate_config_types(raw)
    config = _merge_config(PilotConfig(), raw)

    assert config.memory.checkpoint_interval_seconds == 120
    assert config.memory.pruning_interval_seconds == 900
    assert config.memory.pruning_min_memories == 25


def test_store_source_mentions_configured_pruning_settings():
    store_source = Path(__file__).resolve().parents[1] / "pilot" / "memory" / "store.py"
    source_text = store_source.read_text(encoding="utf-8")

    assert "self._pruning_interval_seconds = pruning_interval_seconds" in source_text
    assert "self._pruning_min_memories = pruning_min_memories" in source_text
    assert "self._pruning_min_memories" in source_text

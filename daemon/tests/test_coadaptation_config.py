"""The co_adaptation config section: off by default, strictly typed, clamped on load, saved like any section."""

from __future__ import annotations

import math

import pytest

from pilot.config import (
    CO_ADAPTATION_BOOL_KEYS,
    CO_ADAPTATION_DB_FILE,
    CO_ADAPTATION_EXPORTS_DIR,
    CO_ADAPTATION_FLOAT_RANGES,
    CO_ADAPTATION_INT_RANGES,
    CO_ADAPTATION_PREFERENCES_FILE,
    DATA_DIR,
    CoAdaptationConfig,
    PilotConfig,
    _config_to_dict,
    _merge_config,
    _validate_config_types,
)

FULL_SECTION = {
    "enabled": True,
    "adaptive_enabled": True,
    "learn_across_sessions": True,
    "gaze_prior_enabled": True,
    "hand_pointer_enabled": True,
    "ambiguity_threshold": 0.2,
    "unintended_activation_tolerance": 0.04,
    "noninferiority_margin": 0.02,
    "upper_bound_credibility": 0.9,
    "exploration_floor": 0.2,
    "switching_penalty": 0.5,
    "burden_prompt_every_blocks": 2,
    "day_boundary_hour": 6,
}


def test_defaults_are_all_off_and_match_the_documented_values():
    config = PilotConfig().co_adaptation

    assert isinstance(config, CoAdaptationConfig)
    assert config.enabled is False
    assert config.adaptive_enabled is False
    assert config.learn_across_sessions is False
    assert config.gaze_prior_enabled is False
    assert config.hand_pointer_enabled is False
    assert config.ambiguity_threshold == 0.25
    assert config.unintended_activation_tolerance == 0.05
    assert config.noninferiority_margin == 0.01
    assert config.upper_bound_credibility == 0.95
    assert config.exploration_floor == 0.1
    assert config.switching_penalty == 0.1
    assert config.burden_prompt_every_blocks == 1
    assert config.day_boundary_hour == 4


def test_every_field_has_exactly_one_type_rule():
    names = set(CoAdaptationConfig.__dataclass_fields__)
    assert names == set(FULL_SECTION)
    assert names == CO_ADAPTATION_BOOL_KEYS | set(CO_ADAPTATION_FLOAT_RANGES) | set(CO_ADAPTATION_INT_RANGES)
    assert not CO_ADAPTATION_BOOL_KEYS & set(CO_ADAPTATION_FLOAT_RANGES)
    assert not set(CO_ADAPTATION_FLOAT_RANGES) & set(CO_ADAPTATION_INT_RANGES)


def test_every_default_lies_inside_its_range():
    config = CoAdaptationConfig()
    for key, (low, high) in {**CO_ADAPTATION_FLOAT_RANGES, **CO_ADAPTATION_INT_RANGES}.items():
        assert low <= getattr(config, key) <= high, key


def test_data_paths_live_beside_the_other_data_files():
    assert CO_ADAPTATION_DB_FILE == DATA_DIR / "co_adaptation.db"
    assert CO_ADAPTATION_PREFERENCES_FILE == DATA_DIR / "co_adaptation_preferences.json"
    assert CO_ADAPTATION_EXPORTS_DIR == DATA_DIR / "co_adaptation_exports"


def test_validator_accepts_the_full_section_and_integer_floats():
    _validate_config_types({"co_adaptation": FULL_SECTION})
    _validate_config_types({"co_adaptation": {"switching_penalty": 1, "exploration_floor": 0}})


@pytest.mark.parametrize("typo", ["enabeld", "ambiguity_treshold", "approval_presentation"])
def test_validator_rejects_typos(typo):
    with pytest.raises(ValueError, match=f"co_adaptation.{typo}"):
        _validate_config_types({"co_adaptation": {typo: True}})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", 1),
        ("enabled", "true"),
        ("ambiguity_threshold", "0.2"),
        ("ambiguity_threshold", True),
        ("burden_prompt_every_blocks", 2.0),
        ("burden_prompt_every_blocks", False),
        ("day_boundary_hour", "4"),
    ],
)
def test_validator_rejects_wrong_types(key, value):
    with pytest.raises(ValueError, match=f"co_adaptation.{key}"):
        _validate_config_types({"co_adaptation": {key: value}})


def test_merge_reads_every_key_in_range():
    config = _merge_config(PilotConfig(), {"co_adaptation": dict(FULL_SECTION)})
    for key, value in FULL_SECTION.items():
        assert getattr(config.co_adaptation, key) == value, key


@pytest.mark.parametrize(("key", "bounds"), sorted({**CO_ADAPTATION_FLOAT_RANGES, **CO_ADAPTATION_INT_RANGES}.items()))
def test_merge_clamps_every_range(key, bounds):
    low, high = bounds
    too_low = _merge_config(PilotConfig(), {"co_adaptation": {key: low - 100}})
    too_high = _merge_config(PilotConfig(), {"co_adaptation": {key: high + 100}})

    assert getattr(too_low.co_adaptation, key) == low
    assert getattr(too_high.co_adaptation, key) == high


def test_merge_never_lets_the_ambiguity_threshold_rise_above_the_default():
    config = _merge_config(PilotConfig(), {"co_adaptation": {"ambiguity_threshold": 0.9}})
    assert config.co_adaptation.ambiguity_threshold == 0.25


def test_merge_ignores_a_non_finite_float():
    config = _merge_config(PilotConfig(), {"co_adaptation": {"unintended_activation_tolerance": math.nan}})
    assert config.co_adaptation.unintended_activation_tolerance == 0.05


def test_merge_leaves_missing_keys_at_their_defaults():
    config = _merge_config(PilotConfig(), {"co_adaptation": {"adaptive_enabled": True}})
    assert config.co_adaptation.adaptive_enabled is True
    assert config.co_adaptation.enabled is False
    assert config.co_adaptation.ambiguity_threshold == 0.25


def test_config_to_dict_round_trips_through_validator_and_merge():
    original = PilotConfig()
    for key, value in FULL_SECTION.items():
        setattr(original.co_adaptation, key, value)

    raw = _config_to_dict(original)
    assert raw["co_adaptation"] == FULL_SECTION
    raw.pop("restrictions")
    _validate_config_types(raw)
    merged = _merge_config(PilotConfig(), raw)

    assert merged.co_adaptation == original.co_adaptation


def test_default_persisted_config_including_co_adaptation_passes_the_validator():
    raw = _config_to_dict(PilotConfig())
    raw.pop("restrictions")
    assert raw["co_adaptation"]["enabled"] is False
    _validate_config_types(raw)


def test_saved_section_is_reloaded(tmp_path, monkeypatch):
    import pilot.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")
    config = PilotConfig()
    config.co_adaptation.ambiguity_threshold = 0.15
    config.co_adaptation.day_boundary_hour = 3
    config.save()

    loaded = PilotConfig.load()

    assert loaded.load_errors == []
    assert loaded.co_adaptation.ambiguity_threshold == 0.15
    assert loaded.co_adaptation.day_boundary_hour == 3
    assert loaded.co_adaptation.enabled is False

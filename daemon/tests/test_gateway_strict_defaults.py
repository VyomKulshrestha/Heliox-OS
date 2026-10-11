"""The gateway fails closed wherever a profile does not say otherwise."""

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams, PermissionTier
from pilot.config import PilotConfig
from pilot.security.gateway import (
    DEFAULT_SOURCE_PROFILES,
    ActionFamily,
    AgentGateway,
    InvocationSource,
    SourceProfile,
)
from pilot.security.permissions import PermissionChecker


def _plan(action_type: ActionType) -> ActionPlan:
    return ActionPlan(actions=[Action(action_type=action_type, target="x", parameters=EmptyParams())], raw_input="t")


def _gateway(config: PilotConfig) -> AgentGateway:
    return AgentGateway(config, PermissionChecker(config))


def test_a_profile_that_says_nothing_is_read_only_without_root():
    profile = SourceProfile()

    assert profile.max_tier == {family.value: int(PermissionTier.READ_ONLY) for family in ActionFamily}
    assert profile.allow_root is False


def test_the_two_full_access_profiles_still_reach_root():
    for name in ("interactive", "system_agent"):
        profile = DEFAULT_SOURCE_PROFILES[name]
        assert profile.allow_root is True
        assert all(tier == int(PermissionTier.ROOT_CRITICAL) for tier in profile.max_tier.values())


@pytest.mark.asyncio
async def test_a_family_missing_from_a_profile_is_read_only():
    config = PilotConfig()
    config.gateway.source_profiles["autonomous"] = SourceProfile(
        max_tier={ActionFamily.BROWSING.value: int(PermissionTier.SYSTEM_MODIFY)}
    )

    shell = await _gateway(config).authorize(_plan(ActionType.SHELL_COMMAND), InvocationSource.AUTONOMOUS)

    assert shell.allowed is False
    assert "READ_ONLY" in shell.reasons[0]


@pytest.mark.asyncio
async def test_a_source_missing_from_config_gets_its_own_profile_not_interactive():
    config = PilotConfig()
    del config.gateway.source_profiles["autonomous"]

    decision = await _gateway(config).authorize(_plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.AUTONOMOUS)

    assert decision.allowed is False  # the built-in autonomous profile denies it; interactive would allow it


def _use_config_file(monkeypatch, tmp_path):
    import pilot.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")
    return tmp_path / "config.toml"


def test_a_partly_written_max_tier_table_leaves_the_other_families_read_only(monkeypatch, tmp_path):
    config_file = _use_config_file(monkeypatch, tmp_path)
    config_file.write_text(
        "[gateway.source_profiles.autonomous.max_tier]\nshell = 1\n"
        "[gateway.source_profiles.made_up]\ndeny_action_types = []\n",
        encoding="utf-8",
    )

    config = PilotConfig.load()

    autonomous = config.gateway.source_profiles["autonomous"]
    assert autonomous.max_tier == {
        "shell": 1,
        "browsing": int(PermissionTier.READ_ONLY),
        "system_control": int(PermissionTier.READ_ONLY),
        "other": int(PermissionTier.READ_ONLY),
    }
    made_up = config.gateway.source_profiles["made_up"]
    assert made_up.allow_root is False
    assert set(made_up.max_tier.values()) == {int(PermissionTier.READ_ONLY)}


def test_a_profile_without_a_max_tier_table_keeps_its_built_in_ceilings(monkeypatch, tmp_path):
    config_file = _use_config_file(monkeypatch, tmp_path)
    config_file.write_text(
        '[gateway.source_profiles.autonomous]\ndeny_action_types = ["power_shutdown"]\n', encoding="utf-8"
    )

    config = PilotConfig.load()

    assert config.gateway.source_profiles["autonomous"].max_tier == DEFAULT_SOURCE_PROFILES["autonomous"].max_tier

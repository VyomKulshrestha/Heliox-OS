"""Tests for the Agent Gateway's core policy layer (pilot.security.gateway).

Covers action-family classification, the narrow-only override guarantee,
and AgentGateway.authorize()'s source-scoped decision logic — no real
Executor/critic model calls involved, all pure/async-unit-level.
"""

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams, PermissionTier
from pilot.config import PilotConfig
from pilot.security.gateway import (
    DEFAULT_SOURCE_PROFILES,
    ActionFamily,
    AgentGateway,
    InvocationSource,
    SourceProfile,
    TaskScopeOverride,
    action_family,
    resolve_effective_profile,
)
from pilot.security.permissions import PermissionChecker


class TestActionFamily:
    def test_shell_command_is_shell_family(self):
        assert action_family(ActionType.SHELL_COMMAND) == ActionFamily.SHELL

    def test_ssh_command_is_shell_family(self):
        assert action_family(ActionType.SSH_COMMAND) == ActionFamily.SHELL

    def test_browser_navigate_is_browsing_family(self):
        assert action_family(ActionType.BROWSER_NAVIGATE) == ActionFamily.BROWSING

    def test_browser_execute_js_is_browsing_family(self):
        assert action_family(ActionType.BROWSER_EXECUTE_JS) == ActionFamily.BROWSING

    def test_mouse_click_is_system_control_family(self):
        assert action_family(ActionType.MOUSE_CLICK) == ActionFamily.SYSTEM_CONTROL

    def test_registry_write_is_system_control_family(self):
        assert action_family(ActionType.REGISTRY_WRITE) == ActionFamily.SYSTEM_CONTROL

    def test_file_read_is_other_family(self):
        assert action_family(ActionType.FILE_READ) == ActionFamily.OTHER


class TestResolveEffectiveProfile:
    def test_no_override_returns_profile_unchanged(self):
        profile = SourceProfile(max_tier={"shell": 2}, deny_action_types=["x"], allow_root=False)
        assert resolve_effective_profile(profile, None) is profile

    def test_override_can_narrow_tier(self):
        profile = SourceProfile(max_tier={"shell": 2})
        override = TaskScopeOverride(max_tier={"shell": 0})
        effective = resolve_effective_profile(profile, override)
        assert effective.max_tier["shell"] == 0

    def test_override_cannot_widen_tier(self):
        profile = SourceProfile(max_tier={"shell": 2})
        override = TaskScopeOverride(max_tier={"shell": 4})
        effective = resolve_effective_profile(profile, override)
        assert effective.max_tier["shell"] == 2  # clamped, not raised

    def test_override_deny_list_is_union_not_replacement(self):
        profile = SourceProfile(max_tier={"shell": 2}, deny_action_types=["shell_command"])
        override = TaskScopeOverride(deny_action_types=["browser_execute_js"])
        effective = resolve_effective_profile(profile, override)
        assert "shell_command" in effective.deny_action_types
        assert "browser_execute_js" in effective.deny_action_types

    def test_override_cannot_grant_root_when_floor_denies_it(self):
        profile = SourceProfile(max_tier={"shell": 4}, allow_root=False)
        override = TaskScopeOverride(allow_root=True)
        effective = resolve_effective_profile(profile, override)
        assert effective.allow_root is False

    def test_override_can_deny_root_even_when_floor_allows_it(self):
        profile = SourceProfile(max_tier={"shell": 4}, allow_root=True)
        override = TaskScopeOverride(allow_root=False)
        effective = resolve_effective_profile(profile, override)
        assert effective.allow_root is False


class TestDefaultSourceProfiles:
    def test_interactive_profile_is_a_strict_no_op_floor(self):
        interactive = DEFAULT_SOURCE_PROFILES["interactive"]
        assert all(tier == int(PermissionTier.ROOT_CRITICAL) for tier in interactive.max_tier.values())
        assert interactive.deny_action_types == []
        assert interactive.allow_root is True

    def test_autonomous_profile_denies_execute_js(self):
        assert "browser_execute_js" in DEFAULT_SOURCE_PROFILES["autonomous"].deny_action_types

    def test_autonomous_profile_disallows_root(self):
        assert DEFAULT_SOURCE_PROFILES["autonomous"].allow_root is False


@pytest.fixture
def gateway():
    config = PilotConfig()
    permissions = PermissionChecker(config)
    return AgentGateway(config, permissions)


def _plan(action_type: ActionType, target: str = "x") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=action_type, target=target, parameters=EmptyParams())], raw_input="test"
    )


class TestAgentGatewayAuthorize:
    @pytest.mark.asyncio
    async def test_interactive_source_is_unrestricted_by_gateway(self, gateway):
        decision = await gateway.authorize(_plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.INTERACTIVE)
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_autonomous_source_denied_execute_js(self, gateway):
        decision = await gateway.authorize(_plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.AUTONOMOUS)
        assert decision.allowed is False
        assert any("browser_execute_js" in r for r in decision.reasons)

    @pytest.mark.asyncio
    async def test_autonomous_source_allows_shell_command_at_its_floor(self, gateway):
        # SHELL_COMMAND is tier SYSTEM_MODIFY(2), matching the autonomous
        # profile's shell ceiling exactly — should be allowed.
        decision = await gateway.authorize(_plan(ActionType.SHELL_COMMAND), InvocationSource.AUTONOMOUS)
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_scope_override_cannot_widen_autonomous_floor(self, gateway):
        widen = TaskScopeOverride(max_tier={"shell": int(PermissionTier.ROOT_CRITICAL)}, allow_root=True)
        # Even with a maximally-widening override, the deny-listed action is
        # still denied — deny lists are never removable by an override.
        decision = await gateway.authorize(
            _plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.AUTONOMOUS, scope_override=widen
        )
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_scope_override_can_narrow_autonomous_further(self, gateway):
        narrow = TaskScopeOverride(max_tier={"shell": 0})
        decision = await gateway.authorize(
            _plan(ActionType.SHELL_COMMAND), InvocationSource.AUTONOMOUS, scope_override=narrow
        )
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_gateway_disabled_allows_everything(self, gateway):
        gateway._config.gateway.enabled = False
        decision = await gateway.authorize(_plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.AUTONOMOUS)
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_unknown_source_falls_back_to_interactive_profile(self, gateway):
        # Fail-open by design (see gateway.py's DEFAULT_UNKNOWN_SOURCE_PROFILE) —
        # untagged call sites must not suddenly break.
        decision = await gateway.authorize(_plan(ActionType.BROWSER_EXECUTE_JS), InvocationSource.UNKNOWN)
        assert decision.allowed is True

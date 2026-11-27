"""Tests for the Agent Gateway's core policy layer (pilot.security.gateway).

Covers action-family classification, the narrow-only override guarantee,
and AgentGateway.authorize()'s source-scoped decision logic — no real
Executor/critic model calls involved, all pure/async-unit-level.
"""

import pytest

from pilot.actions import Action, ActionPlan, ActionType, ApiRequestParams, EmptyParams, PermissionTier
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

    @pytest.mark.parametrize(
        "action_type",
        [ActionType.API_REQUEST, ActionType.API_GITHUB, ActionType.API_SCRAPE],
    )
    def test_api_actions_are_browsing_family(self, action_type):
        assert action_family(action_type) == ActionFamily.BROWSING

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

    def test_self_healing_profile_disallows_root(self):
        assert DEFAULT_SOURCE_PROFILES["self_healing"].allow_root is False

    def test_self_healing_profile_denies_power_and_registry_and_execute_js(self):
        deny = DEFAULT_SOURCE_PROFILES["self_healing"].deny_action_types
        for action_type in ("power_shutdown", "power_restart", "power_logout", "registry_write", "browser_execute_js"):
            assert action_type in deny

    def test_self_healing_profile_never_reaches_root_critical(self):
        self_healing = DEFAULT_SOURCE_PROFILES["self_healing"]
        assert all(tier < int(PermissionTier.ROOT_CRITICAL) for tier in self_healing.max_tier.values())

    def test_neural_profile_is_explicit_exhaustive_and_tier_one_only(self):
        neural = DEFAULT_SOURCE_PROFILES["neural"]
        assert neural.allow_root is False
        assert all(tier == int(PermissionTier.USER_WRITE) for tier in neural.max_tier.values())
        assert "shell_command" in neural.deny_action_types
        assert "browser_click" in neural.deny_action_types
        assert "file_write" in neural.deny_action_types
        assert "system_info" not in neural.deny_action_types


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
    async def test_a_plan_from_an_unknown_source_is_refused_whatever_it_holds(self, gateway):
        # Every entry names its own source; a plan that reaches the gateway without one runs nothing.
        decision = await gateway.authorize(_plan(ActionType.SYSTEM_INFO), InvocationSource.UNKNOWN)
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_neural_source_only_allows_compiled_low_risk_actions(self, gateway):
        allowed = await gateway.authorize(_plan(ActionType.SYSTEM_INFO), InvocationSource.NEURAL)
        denied = await gateway.authorize(_plan(ActionType.FILE_WRITE), InvocationSource.NEURAL)
        assert allowed.allowed is True
        assert denied.allowed is False

    @pytest.mark.asyncio
    async def test_web_agent_can_reach_confirmable_api_writes(self, gateway):
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.API_REQUEST,
                    parameters=ApiRequestParams(method="POST", url="https://example.com/items"),
                )
            ]
        )

        decision = await gateway.authorize(plan, InvocationSource.WEB_AGENT)

        assert plan.actions[0].permission_tier == PermissionTier.SYSTEM_MODIFY
        assert decision.allowed is True


class TestPerSpecialistAgentSourceProfiles:
    """Every specialist agent that forwards to the shared Executor now gets
    its own gateway-audited SourceProfile instead of silently defaulting to
    the unrestricted 'interactive' floor -- see base_agent.py's
    get_invocation_source() and gateway.py's InvocationSource docstring."""

    def test_all_nine_specialist_profiles_exist(self):
        for key in (
            "system_agent",
            "ssh_agent",
            "code_agent",
            "monitor_agent",
            "comm_agent",
            "rss_agent",
            "calendar_agent",
            "forensics_agent",
            "semantic_search_agent",
        ):
            assert key in DEFAULT_SOURCE_PROFILES

    @pytest.mark.asyncio
    async def test_system_agent_still_allows_its_own_broad_capabilities(self, gateway):
        # SystemAgent's job genuinely requires this reach (POWER_SHUTDOWN,
        # REGISTRY_WRITE, FILE_DELETE) -- the profile exists for audit
        # attribution, not to restrict a role that legitimately needs it.
        for action_type in (ActionType.POWER_SHUTDOWN, ActionType.REGISTRY_WRITE, ActionType.FILE_DELETE):
            decision = await gateway.authorize(_plan(action_type), InvocationSource.SYSTEM_AGENT)
            assert decision.allowed is True, f"{action_type} should be allowed for system_agent"

    @pytest.mark.asyncio
    async def test_code_agent_allows_shell_command_denies_root(self, gateway):
        decision = await gateway.authorize(_plan(ActionType.SHELL_COMMAND), InvocationSource.CODE_AGENT)
        assert decision.allowed is True
        assert DEFAULT_SOURCE_PROFILES["code_agent"].allow_root is False

    @pytest.mark.asyncio
    async def test_comm_agent_denied_shell_command(self, gateway):
        """CommunicationAgent (API_SEND_EMAIL/SLACK/DISCORD/WEBHOOK/NOTIFY)
        has no legitimate reason to ever touch the shell family -- this is
        the real, meaningful hardening a defaulted-to-interactive floor
        never provided: a bug that somehow routed SHELL_COMMAND to this
        agent is now actually blocked, not silently allowed."""
        decision = await gateway.authorize(_plan(ActionType.SHELL_COMMAND), InvocationSource.COMMUNICATION_AGENT)
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_comm_agent_allows_its_own_send_email_action(self, gateway):
        decision = await gateway.authorize(_plan(ActionType.API_SEND_EMAIL), InvocationSource.COMMUNICATION_AGENT)
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_forensics_agent_is_read_only_across_every_family(self, gateway):
        assert all(
            tier == int(PermissionTier.READ_ONLY)
            for tier in DEFAULT_SOURCE_PROFILES["forensics_agent"].max_tier.values()
        )
        # Its own action (LOG_ANALYZE) is itself read-only, so it's still allowed.
        decision = await gateway.authorize(_plan(ActionType.LOG_ANALYZE), InvocationSource.FORENSICS_AGENT)
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_forensics_agent_denied_file_delete(self, gateway):
        decision = await gateway.authorize(_plan(ActionType.FILE_DELETE), InvocationSource.FORENSICS_AGENT)
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_calendar_agent_allows_delete_event_denies_shell(self, gateway):
        allowed = await gateway.authorize(_plan(ActionType.CALENDAR_DELETE_EVENT), InvocationSource.CALENDAR_AGENT)
        assert allowed.allowed is True
        denied = await gateway.authorize(_plan(ActionType.SHELL_COMMAND), InvocationSource.CALENDAR_AGENT)
        assert denied.allowed is False

    def test_no_specialist_profile_allows_root_except_system_agent(self):
        # system_agent matches "interactive"'s full ceiling (including
        # allow_root=True) because its job genuinely needs it; every other
        # specialist agent should never need root through the gateway.
        for key, profile in DEFAULT_SOURCE_PROFILES.items():
            if key in ("interactive", "system_agent"):
                continue
            assert profile.allow_root is False, f"{key} should not allow_root"

"""Tests for the permission tier system."""

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    FileParams,
    PackageParams,
    PermissionTier,
    ServiceParams,
)
from pilot.config import PilotConfig
from pilot.security.permissions import PermissionChecker


@pytest.fixture
def config():
    return PilotConfig()


@pytest.fixture
def checker(config):
    return PermissionChecker(config)


class TestPermissionTiers:
    def test_read_only_tier(self, checker):
        action = Action(
            action_type=ActionType.FILE_READ,
            target="/home/user/test.txt",
            parameters=FileParams(path="/home/user/test.txt"),
        )
        decision = checker.check_action(action)
        assert decision.tier == PermissionTier.READ_ONLY
        assert decision.allowed
        assert not decision.requires_confirmation
        assert not decision.requires_snapshot

    def test_user_write_tier(self, checker):
        action = Action(
            action_type=ActionType.FILE_WRITE,
            target="/home/user/test.txt",
            parameters=FileParams(path="/home/user/test.txt", content="hello"),
        )
        decision = checker.check_action(action)
        assert decision.tier == PermissionTier.USER_WRITE
        assert decision.allowed
        assert not decision.requires_confirmation

    def test_system_modify_tier(self, checker):
        action = Action(
            action_type=ActionType.PACKAGE_INSTALL,
            target="vim",
            parameters=PackageParams(name="vim"),
        )
        decision = checker.check_action(action)
        assert decision.tier == PermissionTier.SYSTEM_MODIFY
        assert decision.requires_confirmation

    def test_destructive_tier(self, checker):
        action = Action(
            action_type=ActionType.FILE_DELETE,
            target="/home/user/old.txt",
            parameters=FileParams(path="/home/user/old.txt"),
            destructive=True,
        )
        decision = checker.check_action(action)
        assert decision.tier == PermissionTier.DESTRUCTIVE
        assert decision.requires_confirmation
        assert decision.requires_snapshot

    def test_root_tier_denied_when_disabled(self, checker):
        action = Action(
            action_type=ActionType.SERVICE_RESTART,
            target="nginx",
            parameters=ServiceParams(name="nginx"),
            requires_root=True,
        )
        decision = checker.check_action(action)
        assert decision.tier == PermissionTier.ROOT_CRITICAL
        assert not decision.allowed
        assert "Root access is disabled" in decision.denial_reason

    def test_root_tier_allowed_when_enabled(self, config):
        config.security.root_enabled = True
        checker = PermissionChecker(config)
        action = Action(
            action_type=ActionType.SERVICE_RESTART,
            target="nginx",
            parameters=ServiceParams(name="nginx"),
            requires_root=True,
        )
        decision = checker.check_action(action)
        assert decision.allowed
        assert decision.requires_confirmation


class TestPlanChecks:
    def test_plan_requires_confirmation(self, checker):
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.PACKAGE_INSTALL,
                    target="vim",
                    parameters=PackageParams(name="vim"),
                ),
            ],
            explanation="Install vim",
        )
        assert checker.plan_requires_confirmation(plan)

    def test_plan_no_confirmation_for_reads(self, checker):
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_READ,
                    target="/etc/hostname",
                    parameters=FileParams(path="/etc/hostname"),
                ),
            ],
            explanation="Read hostname",
        )
        assert not checker.plan_requires_confirmation(plan)

    def test_plan_allowed(self, checker):
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_READ,
                    target="/etc/hostname",
                    parameters=FileParams(path="/etc/hostname"),
                ),
            ],
            explanation="Read hostname",
        )
        allowed, reasons = checker.plan_allowed(plan)
        assert allowed
        assert len(reasons) == 0

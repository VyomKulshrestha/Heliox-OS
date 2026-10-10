"""Regression coverage for the expanded capability-driven specialist mesh."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pilot.actions import ActionType
from pilot.agents.agent_mesh import AgentMesh
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.agents.registry import AgentRegistry


async def _expanded_mesh() -> tuple[AgentOrchestrator, AgentMesh, int]:
    AgentRegistry.clear()
    AgentRegistry.discover_agents()
    mesh = AgentMesh()
    model_router = MagicMock()
    orchestrator = AgentOrchestrator(model_router, agent_mesh=mesh)
    count = orchestrator.auto_register_all_agents(
        executor=MagicMock(),
        background_manager=MagicMock(),
        model_router=model_router,
        config=MagicMock(),
        vault=MagicMock(),
        memory=MagicMock(),
    )
    return orchestrator, mesh, count


async def _stop_scheduler(orchestrator: AgentOrchestrator) -> None:
    orchestrator._scheduler_task.cancel()
    try:
        await orchestrator._scheduler_task
    except asyncio.CancelledError:
        pass


def test_registry_discovers_21_concrete_specialists_without_abstract_helpers():
    AgentRegistry.clear()
    AgentRegistry.discover_agents()
    discovered = set(AgentRegistry.get_all_agents())

    assert len(discovered) == 21
    assert {
        "FileOperationsAgent",
        "PackageManagementAgent",
        "ServiceManagementAgent",
        "DesktopAutomationAgent",
        "WorkflowAutomationAgent",
        "IntegrationAgent",
        "VisionAgent",
        "PluginRuntimeAgent",
        "NetworkAgent",
        "GitAgent",
    }.issubset(discovered)
    assert "_ExecutorDomainAgent" not in discovered


async def test_mesh_covers_every_one_of_157_actions():
    orchestrator, mesh, registered = await _expanded_mesh()
    try:
        status = mesh.status()
        assert registered == 21
        assert status["executable_specialists"] == 21
        assert status["registered_action_types"] == 160
        assert status["available_action_types"] == 160
        assert status["coverage_complete"] is True
        assert status["uncovered_action_types"] == []
    finally:
        await _stop_scheduler(orchestrator)


async def test_narrow_specialists_win_unproven_ties_without_overriding_outcomes():
    orchestrator, mesh, _ = await _expanded_mesh()
    try:
        expectations = {
            ActionType.FILE_HASH: "FileOperationsAgent",
            ActionType.GIT_STATUS: "GitAgent",
            ActionType.SCREEN_DETECT_ELEMENTS: "VisionAgent",
            ActionType.PLUGIN_CALL: "PluginRuntimeAgent",
            ActionType.PACKAGE_SEARCH: "PackageManagementAgent",
            ActionType.SERVICE_STATUS: "ServiceManagementAgent",
        }
        for action_type, expected in expectations.items():
            selection = mesh.select_provider(action_type, task_id="specialist-routing")
            assert selection is not None
            assert selection[1].__class__.__name__ == expected
    finally:
        await _stop_scheduler(orchestrator)


async def test_text_routing_selects_distinct_use_case_specialists():
    orchestrator, mesh, _ = await _expanded_mesh()
    try:
        cases = {
            "hash and compare these files": "FileOperationsAgent",
            "inspect git status and show the diff": "GitAgent",
            "detect the save button on screen": "VisionAgent",
            "run the approved marketplace plugin": "PluginRuntimeAgent",
            "restart the database service": "ServiceManagementAgent",
        }
        for prompt, expected in cases.items():
            routes = mesh.route_text(prompt)
            assert routes
            assert routes[0]["display_name"] == expected
    finally:
        await _stop_scheduler(orchestrator)


async def test_specialist_contracts_expose_bounded_authority_and_resources():
    orchestrator, mesh, _ = await _expanded_mesh()
    try:
        specialists = {item["display_name"]: item for item in mesh.status()["specialists"]}
        plugin = specialists["PluginRuntimeAgent"]
        desktop = specialists["DesktopAutomationAgent"]
        git = specialists["GitAgent"]

        assert plugin["budget"]["max_actions_per_task"] == 8
        assert plugin["permissions"]["network_domains"] == ("manifest_declared_only",)
        assert desktop["permissions"]["devices"] == ("screen", "mouse", "keyboard", "audio")
        assert desktop["permissions"]["clipboard"] == ("read", "write")
        assert git["permissions"]["network_domains"] == ("configured_git_remotes",)
        assert git["permissions"]["credential_names"] == ("git_credential_helper",)
    finally:
        await _stop_scheduler(orchestrator)

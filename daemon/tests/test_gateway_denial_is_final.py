"""A denial by Heliox's own rules is final, and an approval card only asks for what the gateway will then allow.

Third live run, 27 November: "Make an HTTP GET request to http://127.0.0.1:<port>/busy" showed three approval cards,
all approved, made ten model calls in 32 seconds and ended "I couldn't reach 127.0.0.1". The step went to the
integration agent, whose own gateway ceiling for "browsing" was READ_ONLY although it declares api_request (a GET to
this computer asks for approval, SYSTEM_MODIFY); the denial was not a final failure, so two repair rounds planned the
same GET and showed their cards again; and in one round the mesh gave the step to the web agent (browsing up to
SYSTEM_MODIFY), so the same step was denied once and run once in one task. Fakes only: nothing is fetched.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, PermissionTier
from pilot.agents.agent_mesh import AgentMesh
from pilot.agents.planner import Planner
from pilot.agents.registry import AgentRegistry
from pilot.config import PilotConfig
from pilot.security.gateway import (
    DEFAULT_SOURCE_PROFILES,
    GATEWAY_DENIED,
    AgentGateway,
    InvocationSource,
    TaskScopeOverride,
    ceiling_reason,
)
from tests.test_reply_never_empty import _Executor, _Planner, _server, _ws

LOOPBACK = "http://127.0.0.1:8797/busy"

# A policy the profiles keep on purpose: page scripts are never run for the web agent, whoever asks.
DELIBERATE = {("web_agent", "browser_execute_js")}


def _variants(action_type: ActionType) -> list[Action]:
    """The step at the tiers it can take: its plain form, a write method, an address on this computer."""
    found = []
    for params in ({}, {"method": "POST"}, {"allow_private_network": True}, {"method": "DELETE"}):
        try:
            parsed = Planner._parse_parameters(action_type, dict(params))
        except Exception:  # noqa: BLE001 - this action takes no such parameter
            continue
        found.append(Action(action_type=action_type, target="t", parameters=parsed))
    return found


def _agents() -> dict[str, Any]:
    AgentRegistry.discover_agents()
    offered = {
        "executor": MagicMock(),
        "background_manager": MagicMock(),
        "model_router": MagicMock(),
        "config": PilotConfig(),
        "vault": MagicMock(),
        "memory": MagicMock(),
    }
    built = {}
    for name, cls in AgentRegistry.get_all_agents().items():
        parameters = inspect.signature(cls.__init__).parameters
        built[name] = cls(**{key: value for key, value in offered.items() if key in parameters})
    return built


def test_no_agent_declares_a_step_its_own_ceiling_refuses() -> None:
    refused = []
    for name, agent in sorted(_agents().items()):
        source = agent.get_invocation_source()
        profile = DEFAULT_SOURCE_PROFILES[source.value]
        for capability in agent.get_capabilities():
            for action in _variants(capability.action_type):
                if action.permission_tier == PermissionTier.ROOT_CRITICAL:
                    continue
                if (source.value, action.action_type.value) in DELIBERATE:
                    continue
                reason = ceiling_reason(action, source, profile)
                if reason:
                    refused.append(f"{name}: {reason}")
    assert refused == []


def test_the_decision_belongs_to_the_step_not_to_which_agent_carries_it() -> None:
    """Every agent that declares an action decides each form of it the same way, so the mesh's choice (which
    follows each agent's record) cannot turn an allowed step into a denied one or back."""
    by_type: dict[ActionType, list[Any]] = {}
    for agent in _agents().values():
        for capability in agent.get_capabilities():
            by_type.setdefault(capability.action_type, []).append(agent)
    differing = []
    for action_type, carriers in by_type.items():
        for action in _variants(action_type):
            if action.permission_tier == PermissionTier.ROOT_CRITICAL:
                continue
            decisions = {
                agent.get_invocation_source().value: not ceiling_reason(
                    action, agent.get_invocation_source(), DEFAULT_SOURCE_PROFILES[agent.get_invocation_source().value]
                )
                for agent in carriers
            }
            if len(set(decisions.values())) > 1:
                differing.append(f"{action_type.value} ({action.permission_tier.name}): {decisions}")
    assert differing == []


async def test_the_preview_names_the_source_each_specialist_then_runs_with() -> None:
    """The card is checked under the source the step will really run with: the orchestrator's own choice."""
    from pilot.agents.orchestrator import AgentOrchestrator

    class _Recorder:
        def __init__(self) -> None:
            self.sources: list[tuple[ActionType, InvocationSource]] = []

        async def execute(self, plan: ActionPlan, **kwargs: Any) -> list[ActionResult]:
            for action in plan.actions:
                self.sources.append((action.action_type, kwargs["invocation_source"]))
            return [ActionResult(action=a, success=True, output="ok") for a in plan.actions]

    recorder = _Recorder()
    orchestrator = AgentOrchestrator(MagicMock(), agent_mesh=AgentMesh())
    try:
        AgentRegistry.discover_agents()
        offered = {"executor": recorder, "background_manager": MagicMock(), "model_router": MagicMock(),
                   "config": PilotConfig(), "vault": MagicMock(), "memory": MagicMock()}  # fmt: skip
        for cls in AgentRegistry.get_all_agents().values():
            parameters = inspect.signature(cls.__init__).parameters
            orchestrator.register_agent(cls(**{k: v for k, v in offered.items() if k in parameters}))
        executor_backed = [
            ActionType.API_REQUEST, ActionType.FILE_WRITE, ActionType.SCREENSHOT, ActionType.WINDOW_CLOSE,
            ActionType.GIT_PUSH, ActionType.SHELL_COMMAND, ActionType.NOTIFY, ActionType.BROWSER_NAVIGATE,
        ]  # fmt: skip
        for action_type in executor_backed:
            [step] = _variants(action_type)[:1]
            plan = ActionPlan(actions=[step])
            [predicted] = orchestrator.invocation_sources(plan)
            await orchestrator.execute_plan("x", plan)
            assert recorder.sources[-1] == (action_type, predicted)
    finally:
        await orchestrator.stop()


@pytest.mark.parametrize("source", sorted(DEFAULT_SOURCE_PROFILES), ids=str)
async def test_what_the_preview_allows_the_gateway_then_allows(source: str) -> None:
    gateway = AgentGateway(PilotConfig(), MagicMock())
    invocation = InvocationSource(source) if source in {s.value for s in InvocationSource} else InvocationSource.UNKNOWN
    voice = DEFAULT_SOURCE_PROFILES["voice"]
    for override in (None, TaskScopeOverride(max_tier=voice.max_tier, deny_action_types=voice.deny_action_types)):
        for action_type in ActionType:
            for action in _variants(action_type):
                previewed = gateway.preview_denials([action], [invocation], override)
                decision = await gateway.authorize(
                    ActionPlan(actions=[action]), invocation, scope_override=override, critic_already_reviewed=True
                )
                assert (not previewed) == decision.allowed, (source, action.action_type.value, previewed)


# ── Through the server ───────────────────────────────────────────────


def _get() -> Action:
    return Action(
        action_type=ActionType.API_REQUEST,
        target=LOOPBACK,
        parameters=ApiRequestParams(method="GET", url=LOOPBACK, allow_private_network=True),
    )


class _DeniedAtRun(_Executor):
    """The gateway refuses the step when it runs (the run's integration agent)."""

    async def execute(self, plan, **kwargs):
        self.runs += 1
        return [
            ActionResult(
                action=action,
                success=False,
                error="Gateway denied: api_request (tier=SYSTEM_MODIFY, family=browsing) exceeds the "
                "'integration_agent' source's allowed ceiling (READ_ONLY).",
                executed=False,
                skip_reason=GATEWAY_DENIED,
            )
            for action in plan.actions
        ]


def _approvals(ws: Any) -> int:
    return sum(1 for sent in ws.sent if json.loads(sent).get("method") == "confirm_required")


async def test_a_denial_at_run_is_final_one_card_one_plan_one_reply() -> None:
    planner = _Planner(ActionPlan(actions=[_get()], explanation="GET it."))
    executor = _DeniedAtRun()
    server = _server(planner, executor, confirm=True)
    ws = _ws(server, confirmation=True)

    response = await server._handle_execute({"input": f"Make an HTTP GET request to {LOOPBACK}"}, ws)

    assert (executor.runs, planner.calls, _approvals(ws)) == (1, 1, 1)  # the run's case was (3, 3, 3)
    assert response["status"] != "success" and str(response.get("message") or "").strip()


async def test_a_step_the_gateway_refuses_whatever_the_answer_never_shows_a_card() -> None:
    """Spoken, the voice ceiling allows browsing up to USER_WRITE: a GET to this computer is refused before any card."""
    planner = _Planner(ActionPlan(actions=[_get()], explanation="GET it."))
    executor = _Executor()
    server = _server(planner, executor, confirm=True)
    server._agent_gateway = AgentGateway(server.config, MagicMock())
    ws = _ws(server, confirmation=True)

    response = await server._handle_execute({"input": f"Make an HTTP GET request to {LOOPBACK}", "source": "voice"}, ws)

    assert (executor.runs, planner.calls, _approvals(ws)) == (0, 1, 0)
    assert response["denied_kind"] == GATEWAY_DENIED
    assert [r["skip_reason"] for r in response["results"]] == [GATEWAY_DENIED]


async def test_typed_the_same_get_asks_once_and_runs() -> None:
    planner = _Planner(ActionPlan(actions=[_get()], explanation="GET it."))
    executor = _Executor()
    server = _server(planner, executor, confirm=True)
    server._agent_gateway = AgentGateway(server.config, MagicMock())
    ws = _ws(server, confirmation=True)

    await server._handle_execute({"input": f"Make an HTTP GET request to {LOOPBACK}"}, ws)

    assert (executor.runs, planner.calls, _approvals(ws)) == (1, 1, 1)


def test_the_autonomous_loop_stops_at_a_denial() -> None:
    from pilot.security.gateway import denied_by_policy

    denied = ActionResult(action=_get(), success=False, error="x", executed=False, skip_reason=GATEWAY_DENIED)
    assert denied_by_policy([denied]) is True
    assert denied_by_policy([ActionResult(action=_get(), success=False, error="timeout")]) is False


# ── Addresses on this computer or the person's network ───────────────


@pytest.mark.parametrize(
    ("url", "where"),
    [
        ("http://127.0.0.1:8797/busy", "this computer"),
        ("http://localhost:3000/", "this computer"),
        ("http://192.168.1.20/status", "your own network"),
        ("http://printer.local/", "your own network"),
        ("https://example.com/", ""),
    ],
)
def test_a_step_to_a_local_address_asks_and_the_card_says_where(url: str, where: str) -> None:
    from pilot.agents.planner import decide_local_flags, local_address
    from pilot.server import _approval_effect

    assert local_address(url) == where
    params = Planner._normalize_params(ActionType.API_REQUEST, {"method": "GET", "url": url}, url)
    assert params.get("allow_private_network") is False  # off until the person's request is read
    action = Action(
        action_type=ActionType.API_REQUEST,
        target=url,
        parameters=Planner._parse_parameters(ActionType.API_REQUEST, params),
    )
    decide_local_flags([action], f"Make an HTTP GET request to {url}")  # the person typed the address
    assert action.requires_confirmation is bool(where)
    if where:
        assert f"an address on {where}" in _approval_effect(action)


# ── Exact allowances: nothing declared is dead, nothing undeclared was widened ──

ABOVE_FAMILY = {
    ("integration_agent", "api_request", "SYSTEM_MODIFY"),
    ("integration_agent", "api_scrape", "SYSTEM_MODIFY"),
    ("integration_agent", "api_github", "SYSTEM_MODIFY"),
    ("desktop_agent", "gnome_setting_write", "SYSTEM_MODIFY"),
    ("desktop_agent", "window_close", "DESTRUCTIVE"),
    ("vision_agent", "screenshot", "USER_WRITE"),
    ("network_agent", "download_file", "SYSTEM_MODIFY"),
}


def _above_family(action: Action, source: InvocationSource) -> bool:
    """Whether the source's family ceiling alone (no specialist list) refuses the step: the gateway's own family, which
    is browsing for a step that reaches a local address."""
    from pilot.security.gateway import ActionFamily, action_family, reaches_local_address

    profile = DEFAULT_SOURCE_PROFILES[source.value]
    family = ActionFamily.BROWSING if reaches_local_address(action) else action_family(action.action_type)
    ceiling = profile.max_tier.get(family.value, int(PermissionTier.READ_ONLY))
    return int(action.permission_tier) > ceiling


def test_what_each_agent_may_run_above_its_family_ceiling_is_exactly_the_listed_steps() -> None:
    from pilot.security.gateway import SPECIALIST_ACTION_CEILINGS

    found = set()
    for agent in _agents().values():
        source = agent.get_invocation_source()
        for action_type in ActionType:
            for action in _variants(action_type):
                if action.permission_tier == PermissionTier.ROOT_CRITICAL:
                    continue
                above_family = _above_family(action, source)
                allowed = not ceiling_reason(action, source, DEFAULT_SOURCE_PROFILES[source.value])
                if above_family and allowed:
                    found.add((source.value, action_type.value, action.permission_tier.name))
    assert found == ABOVE_FAMILY
    # Each listed step is one its agent declares, at no more than the highest tier that step can take.
    declared = {
        agent.get_invocation_source().value: {cap.action_type.value for cap in agent.get_capabilities()}
        for agent in _agents().values()
    }
    for source, listed in SPECIALIST_ACTION_CEILINGS.items():
        for action_name, tier in listed.items():
            assert action_name in declared[source], (source, action_name)
            highest = max(int(a.permission_tier) for a in _variants(ActionType(action_name)))
            assert tier <= highest, (source, action_name)


def test_an_override_still_narrows_a_listed_step() -> None:
    """Spoken, the voice ceiling (browsing up to USER_WRITE) refuses the integration agent's local GET too."""
    gateway = AgentGateway(PilotConfig(), MagicMock())
    voice = DEFAULT_SOURCE_PROFILES["voice"]
    override = TaskScopeOverride(max_tier=voice.max_tier, deny_action_types=voice.deny_action_types)
    assert gateway.preview_denials([_get()], [InvocationSource.INTEGRATION_AGENT]) == []
    assert gateway.preview_denials([_get()], [InvocationSource.INTEGRATION_AGENT], override)


async def test_a_step_previewed_as_allowed_is_allowed_whichever_agent_carries_it() -> None:
    """A repair round may give the step to another agent that declares it: the answer is the same."""
    gateway = AgentGateway(PilotConfig(), MagicMock())
    carriers: dict[ActionType, set[InvocationSource]] = {}
    for agent in _agents().values():
        for capability in agent.get_capabilities():
            carriers.setdefault(capability.action_type, set()).add(agent.get_invocation_source())
    for action_type, sources in carriers.items():
        for action in _variants(action_type):
            if action.permission_tier == PermissionTier.ROOT_CRITICAL:
                continue
            for previewed_by in sources:
                if gateway.preview_denials([action], [previewed_by]):
                    continue
                for carried_by in sources:
                    decision = await gateway.authorize(
                        ActionPlan(actions=[action]), carried_by, critic_already_reviewed=True
                    )
                    assert decision.allowed, (action_type.value, previewed_by.value, carried_by.value, decision.reasons)

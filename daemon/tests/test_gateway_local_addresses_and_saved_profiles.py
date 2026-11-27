"""What the gateway, the planner and the approval cards now enforce for local addresses, saved profiles, denials and
screenshots. Fakes only: nothing is fetched, no picture is taken.

- A request or scrape step may reach this computer or the person's own network only when its host is written as such
  an address and that host is in the person's own request; the flag is off in every other case, whatever the model
  wrote, and every card that can approve such a step names the address.
- A step refused before its card is recorded in the gateway's audit as the preview's decision, and its plan's
  checkpoint is closed as denied.
- The built-in per-action allowances of a specialist apply only while the person has saved no profile for it.
- A policy denial stops the plan's later steps in every branch of the orchestrator.
- A background job never takes a screenshot; a trigger's stored screenshot step does.
- A spoken request that typing would allow is told so.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, EmptyParams, TriggerParams
from pilot.agents.planner import SYSTEM_PROMPT, Planner, decide_local_flags
from pilot.config import PilotConfig
from pilot.security.gateway import (
    DEFAULT_SOURCE_PROFILES,
    GATEWAY_DENIED,
    AgentGateway,
    InvocationSource,
    SourceProfile,
    TaskScopeOverride,
)

LOCAL = "http://127.0.0.1:8797/busy"


def _get(url: str, flag: bool = False) -> Action:
    return Action(
        action_type=ActionType.API_REQUEST,
        target=url,
        parameters=ApiRequestParams(method="GET", url=url, allow_private_network=flag),
    )


def _parsed(request: str, actions: list[dict]) -> Any:
    return Planner.__new__(Planner)._parse_response(json.dumps({"explanation": "x", "actions": actions}), request)


# ── 1. The local-address flag is the code's, from the person's request ──


@pytest.mark.parametrize(
    ("request_text", "url", "model_flag", "flag"),
    [
        (f"Make an HTTP GET request to {LOCAL}", LOCAL, False, True),  # asked for, by address
        (f"Make an HTTP GET request to {LOCAL}", LOCAL, True, True),
        ("Check the status page I was sent", LOCAL, True, False),  # the address came from elsewhere
        ("Get https://example.com", "https://example.com/", True, False),  # a public host: never on
        ("Fetch http://192.168.1.20/status", "http://192.168.1.20/status", False, True),
    ],
)
def test_the_flag_is_set_from_the_persons_request_whatever_the_model_wrote(
    request_text: str, url: str, model_flag: bool, flag: bool
) -> None:
    plan = _parsed(
        request_text,
        [
            {
                "action_type": "api_request",
                "target": url,
                "parameters": {"method": "GET", "url": url, "allow_private_network": model_flag},
            }
        ],
    )
    [step] = plan.actions
    assert step.parameters.allow_private_network is flag
    assert step.requires_confirmation is flag


def test_a_triggers_stored_step_takes_the_flag_from_the_request_that_made_it() -> None:
    stored = _get(LOCAL, flag=True)
    create = Action(
        action_type=ActionType.TRIGGER_CREATE, target="t", parameters=TriggerParams(name="t", steps=[stored])
    )
    decide_local_flags([create], "Watch notes.txt and post to the status page")
    assert create.parameters.steps[0].parameters.allow_private_network is False
    decide_local_flags([create], f"Watch notes.txt and GET {LOCAL} each time")
    assert create.parameters.steps[0].parameters.allow_private_network is True


def test_the_prompt_does_not_invite_the_model_to_set_the_flag() -> None:
    assert "allow_private_network" not in SYSTEM_PROMPT


def test_the_card_names_the_literal_address_only_when_the_flag_is_on() -> None:
    from pilot.server import _approval_effect

    assert _approval_effect(_get(LOCAL, flag=True)) == (
        "Reaches 127.0.0.1, an address on this computer. Approve it only if you asked for it."
    )
    assert _approval_effect(_get(LOCAL, flag=False)) == ""
    assert "this computer" not in _approval_effect(_get("https://example.com/", flag=True))


# ── 2. Every card that can approve such a step carries the line ──


async def test_the_background_job_card_carries_the_local_address_line() -> None:
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._consume_confirmation = AsyncMock(return_value=SimpleNamespace(expired=True))
    job = SimpleNamespace(job_id="job-1", source="phone", session_id="phone")
    plan = ActionPlan(actions=[_get(LOCAL, flag=True)])

    assert await server._wait_for_autonomous_confirmation(job, plan, "autonomous-job-1:0:0") is False

    [card] = [c.args[1] for c in server._broadcast_notification.await_args_list if c.args[0] == "confirm_required"]
    assert card["actions"][0]["effect"].startswith("Reaches 127.0.0.1, an address on this computer.")


def test_the_trigger_card_names_each_stored_step_that_reaches_a_local_address() -> None:
    from pilot.server import _approval_effect

    create = Action(
        action_type=ActionType.TRIGGER_CREATE,
        target="t",
        parameters=TriggerParams(name="t", trigger_type="file_modified", steps=[_get(LOCAL, flag=True)]),
    )
    assert create.requires_confirmation is True
    assert "Each fire reaches 127.0.0.1, an address on this computer." in _approval_effect(create)


# ── 3. A preview refusal is recorded and closes the plan ──


class _Audit:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def record_event(self, **event: Any) -> str:
        self.events.append(event)
        return "e1"


async def test_a_refusal_before_the_card_is_audited_as_the_previews_and_the_checkpoint_closed() -> None:
    from tests.test_reply_never_empty import _Executor, _Planner, _server, _ws

    planner = _Planner(ActionPlan(actions=[_get(LOCAL, flag=True)], explanation="GET it."))
    executor = _Executor()
    server = _server(planner, executor, confirm=True)
    audit = _Audit()
    server._agent_gateway = AgentGateway(server.config, MagicMock(), audit_store=audit)
    server._checkpoint_store = AsyncMock()

    response = await server._handle_execute({"input": f"GET {LOCAL}", "source": "voice"}, _ws(server, True))

    assert response["denied_kind"] == GATEWAY_DENIED and executor.runs == 0
    [event] = audit.events
    assert event["decision"] == "denied" and event["policy_snapshot"]["stage"] == "preview_before_approval"
    approvals = [call.args for call in server._checkpoint_store.mark_approval.await_args_list]
    assert approvals and approvals[-1][1] == "denied"


# ── 4. The built-in allowances yield to a saved profile ──


async def test_a_saved_profile_is_never_widened_by_the_built_in_allowances() -> None:
    config = PilotConfig()
    gateway = AgentGateway(config, MagicMock())
    source = InvocationSource.INTEGRATION_AGENT
    assert gateway.preview_denials([_get(LOCAL, flag=True)], [source]) == []
    default = DEFAULT_SOURCE_PROFILES[source.value]
    config.gateway.source_profiles[source.value] = SourceProfile(
        max_tier=dict(default.max_tier), deny_action_types=["api_github"], allow_root=False
    )
    assert gateway.preview_denials([_get(LOCAL, flag=True)], [source])  # the person's own profile decides
    decision = await gateway.authorize(
        ActionPlan(actions=[_get(LOCAL, flag=True)]), source, critic_already_reviewed=True
    )
    assert decision.allowed is False


# ── 5. A policy denial stops the later steps in every branch ──


async def test_a_denial_from_an_executor_backed_specialist_stops_the_later_steps() -> None:
    from pilot.agents.agent_mesh import AgentMesh
    from pilot.agents.base_agent import AgentCapability, AgentRole, BaseAgent
    from pilot.agents.orchestrator import AgentOrchestrator

    class _Specialist(BaseAgent):
        def __init__(self, role: AgentRole, action_type: ActionType, outcome: str) -> None:
            super().__init__(role=role)
            self._type, self._outcome, self.ran = action_type, outcome, 0
            self._executor = object()  # executor-backed: the orchestrator's own gateway check is skipped

        def get_capabilities(self) -> list[AgentCapability]:
            return [AgentCapability(action_type=self._type, description="x")]

        def get_system_prompt(self) -> str:
            return ""

        def can_handle(self, action_type: ActionType) -> bool:
            return action_type == self._type

        async def handle_task(self, user_input, plan, context=None, scope_override=None):
            self.ran += 1
            denied = self._outcome == "denied"
            return [
                ActionResult(
                    action=action,
                    success=not denied,
                    output="" if denied else "ok",
                    error="Gateway denied: x" if denied else None,
                    executed=not denied,
                    skip_reason=GATEWAY_DENIED if denied else None,
                )
                for action in plan.actions
            ]

    first = _Specialist(AgentRole.INTEGRATIONS, ActionType.API_REQUEST, "denied")
    later = _Specialist(AgentRole.DESKTOP, ActionType.NOTIFY, "ok")
    orchestrator = AgentOrchestrator(MagicMock(), agent_mesh=AgentMesh())
    try:
        orchestrator.register_agent(first)
        orchestrator.register_agent(later)
        results = await orchestrator.execute_plan(
            "x", ActionPlan(actions=[_get(LOCAL, flag=True), Action(action_type=ActionType.NOTIFY, target="n",
                                                                     parameters=EmptyParams())])
        )  # fmt: skip
    finally:
        await orchestrator.stop()
    assert (first.ran, later.ran) == (1, 0)
    assert [r.skip_reason for r in results] == [GATEWAY_DENIED, "dependency_failed"]


# ── 6. Screenshots: typed or spoken yes, a background job no, a trigger's stored step yes ──


def _shot() -> Action:
    from pilot.actions import ScreenshotParams

    return Action(action_type=ActionType.SCREENSHOT, target="screen", parameters=ScreenshotParams())


def _profile_override(name: str) -> TaskScopeOverride:
    profile = DEFAULT_SOURCE_PROFILES[name]
    return TaskScopeOverride(max_tier=profile.max_tier, deny_action_types=profile.deny_action_types)


def test_a_screenshot_runs_on_a_request_and_as_a_stored_step_but_never_in_a_background_job() -> None:
    gateway = AgentGateway(PilotConfig(), MagicMock())
    vision = InvocationSource.VISION_AGENT
    assert gateway.preview_denials([_shot()], [vision]) == []  # typed
    assert gateway.preview_denials([_shot()], [vision], _profile_override("voice")) == []  # spoken
    assert gateway.preview_denials([_shot()], [vision], _profile_override("autonomous"))  # a job, on the agent path
    assert gateway.preview_denials([_shot()], [InvocationSource.AUTONOMOUS])  # a job, on the executor path
    assert gateway.preview_denials([_shot()], [InvocationSource.TRIGGER_STEPS]) == []  # a trigger's stored step


# ── 7. Wording ──


def test_a_spoken_request_that_typing_would_allow_is_told_so() -> None:
    from pilot.server import GATEWAY_SPOKEN_SENTENCE, PilotServer

    server = PilotServer(PilotConfig())
    server._orchestrator = None
    server._agent_gateway = AgentGateway(server.config, MagicMock())
    plan = ActionPlan(actions=[_get(LOCAL, flag=True)])
    assert server._typing_would_allow(plan, InvocationSource.VOICE) is True
    assert server._typing_would_allow(plan, InvocationSource.INTERACTIVE) is False
    assert GATEWAY_SPOKEN_SENTENCE == "I can't do that from a spoken request; type it and I'll ask you first."

from __future__ import annotations

import asyncio

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    EmptyParams,
    GitParams,
    OpenApplicationParams,
    SystemInfoParams,
    VerificationResult,
)
from pilot.agents.execution_companion import CompanionFollowUp, CompanionReview, ExecutionCompanion
from pilot.config import PilotConfig
from pilot.intelligence.scope_decision import ScopeDecision


class _Model:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def generate(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_saved_companion_timeout_is_accepted_by_config_loader(tmp_path, monkeypatch):
    from pilot import config as config_module

    config_file = tmp_path / "config.toml"
    restrictions_file = tmp_path / "restrictions.toml"
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", restrictions_file)

    config = PilotConfig()
    config.narration.enabled = True
    config.narration.advisory_timeout_seconds = 7
    config.save()

    loaded = PilotConfig.load()

    assert loaded.narration.enabled is True
    assert loaded.narration.advisory_timeout_seconds == 7


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(categories=["os"]),
            )
        ],
        explanation="Read the operating-system version.",
    )


def _push_plan() -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=ActionType.GIT_PUSH, target="Heliox", parameters=GitParams())],
        explanation="Push the repository.",
    )


class _ScopeHead:
    available = True

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls: list[tuple[str, str]] = []

    def predict(self, request: str, action: str) -> ScopeDecision:
        self.calls.append((request, action))
        return ScopeDecision(self.probability, abs(2 * self.probability - 1))


@pytest.mark.asyncio
async def test_review_returns_structured_revision_with_safe_structural_values():
    model = _Model(
        '{"decision":"REVISE","reason":"The second step is unnecessary.",'
        '"planner_feedback":"Use only the direct system-info action."}'
    )
    companion = ExecutionCompanion(model)

    review = await companion.review("Show the OS version.", _plan())

    assert review == CompanionReview(
        decision="REVISE",
        reason="The second step is unnecessary.",
        planner_feedback="Use only the direct system-info action.",
    )
    prompt, kwargs = model.calls[0]
    assert "parameter_keys=['categories']" in prompt
    assert "safe_parameters={'categories': ['os']}" in prompt
    assert kwargs["json_mode"] is True


@pytest.mark.asyncio
async def test_review_failure_keeps_existing_safety_pipeline_authoritative():
    companion = ExecutionCompanion(_Model(error=RuntimeError("offline")))

    review = await companion.review("Show the OS version.", _plan())

    assert review.decision == "CONTINUE"
    assert review.issues == ["review_unavailable"]


@pytest.mark.asyncio
async def test_local_scope_head_adds_warning_but_cannot_grant_or_block():
    companion = ExecutionCompanion(_Model('{"decision":"CONTINUE","reason":"Looks aligned."}'))
    head = _ScopeHead(0.05)
    companion._local_scope = head

    review = await companion.review("Commit locally; do not push.", _push_plan())

    assert review.decision == "WARN"
    assert review.issues == ["local_scope_advisory:git_push"]
    assert "Push changes" in head.calls[0][1]


@pytest.mark.asyncio
async def test_local_scope_head_never_overrides_model_stop():
    companion = ExecutionCompanion(_Model('{"decision":"STOP","reason":"Outside request."}'))
    companion._local_scope = _ScopeHead(0.99)

    review = await companion.review("Commit locally; do not push.", _push_plan())

    assert review.decision == "STOP"


@pytest.mark.asyncio
async def test_staged_scope_model_warns_on_unrequested_push_without_blocking():
    companion = ExecutionCompanion(_Model('{"decision":"CONTINUE","reason":"Looks aligned."}'))
    assert companion._local_scope.available

    review = await companion.review("Commit the Heliox README locally; do not push.", _push_plan())

    assert review.decision == "WARN"
    assert review.issues == ["local_scope_advisory:git_push"]


@pytest.mark.asyncio
async def test_staged_scope_model_does_not_warn_on_explicit_push():
    companion = ExecutionCompanion(_Model('{"decision":"CONTINUE","reason":"Looks aligned."}'))
    assert companion._local_scope.available

    review = await companion.review("Push the reviewed Heliox release to the remote repository.", _push_plan())

    assert review.decision == "CONTINUE"


@pytest.mark.asyncio
async def test_staged_scope_model_warns_when_full_companion_is_unavailable():
    companion = ExecutionCompanion(_Model(error=RuntimeError("offline")))
    assert companion._local_scope.available

    review = await companion.review("Commit the Heliox README locally; do not push.", _push_plan())

    assert review.decision == "WARN"
    assert review.issues == ["review_unavailable", "local_scope_advisory:git_push"]


@pytest.mark.asyncio
async def test_review_timeout_keeps_existing_safety_pipeline_authoritative():
    class _SlowModel:
        async def generate(self, *args, **kwargs):
            await asyncio.Event().wait()

    companion = ExecutionCompanion(_SlowModel())
    companion._timeout_seconds = 0.01

    review = await companion.review("Show the OS version.", _plan())

    assert review.decision == "CONTINUE"
    assert review.issues == ["review_unavailable"]


def test_revision_without_feedback_is_downgraded_to_warning():
    review = ExecutionCompanion._parse('{"decision":"REVISE","reason":"This could be simpler.","planner_feedback":""}')

    assert review.decision == "WARN"


@pytest.mark.asyncio
async def test_follow_up_is_grounded_and_excludes_raw_tool_output():
    model = _Model(
        '{"message":"The OS version is confirmed.",'
        '"suggestions":["Compare compatibility with your target app","Save a short system report"]}'
    )
    companion = ExecutionCompanion(model)
    plan = _plan()
    result = ActionResult(action=plan.actions[0], success=True, output="PRIVATE RAW SYSTEM OUTPUT")
    verification = VerificationResult(passed=True, details=["verified"])

    follow_up = await companion.follow_up("Show the OS version.", plan, [result], verification)

    assert follow_up == CompanionFollowUp(
        message="The OS version is confirmed.",
        suggestions=["Compare compatibility with your target app", "Save a short system report"],
    )
    prompt, kwargs = model.calls[0]
    assert "PRIVATE RAW SYSTEM OUTPUT" not in prompt
    assert "system_info: succeeded" in prompt
    assert kwargs["json_mode"] is True


@pytest.mark.asyncio
async def test_follow_up_requires_a_message_and_specific_ideas():
    companion = ExecutionCompanion(_Model('{"message":"Done.","suggestions":[]}'))

    follow_up = await companion.follow_up(
        "Show the OS version.",
        _plan(),
        [],
        VerificationResult(passed=True, details=["verified"]),
    )

    assert follow_up is None


@pytest.mark.asyncio
async def test_follow_up_uses_grounded_local_fallback_when_model_is_offline():
    companion = ExecutionCompanion(_Model(error=RuntimeError("offline")))
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BROWSER_EXTRACT,
                target="main",
                parameters=BrowserParams(selector="main"),
            )
        ],
        explanation="Extract the final page.",
    )
    result = ActionResult(action=plan.actions[0], success=True, output="PRIVATE PAGE CONTENT")

    follow_up = await companion.follow_up(
        "Find the information on example.com.",
        plan,
        [result],
        VerificationResult(passed=True, details=["verified"]),
    )

    assert follow_up is not None
    assert follow_up.source == "local_fallback"
    assert follow_up.message == "The requested task completed and passed verification."
    assert "final page" in follow_up.suggestions[0]
    assert "PRIVATE PAGE CONTENT" not in follow_up.spoken_text()


@pytest.mark.asyncio
async def test_application_follow_up_is_immediate_and_does_not_call_model():
    model = _Model(error=AssertionError("model should not be called"))
    companion = ExecutionCompanion(model)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="Openscreen",
                parameters=OpenApplicationParams(name="Openscreen"),
            )
        ],
        explanation="Launch Openscreen",
    )
    result = ActionResult(action=plan.actions[0], success=True, output="Launched Openscreen")

    follow_up = await companion.follow_up(
        "open Openscreen",
        plan,
        [result],
        VerificationResult(passed=True, details=["verified"]),
    )

    assert follow_up is not None
    assert follow_up.source == "local_fallback"
    assert "inside the application" in follow_up.suggestions[0]
    assert model.calls == []


@pytest.mark.asyncio
async def test_battery_follow_up_is_immediate_and_specific():
    model = _Model(error=AssertionError("model should not be called"))
    companion = ExecutionCompanion(model)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BATTERY_INFO,
                target="battery",
                parameters=EmptyParams(),
            )
        ],
        explanation="Check current battery status",
    )
    result = ActionResult(action=plan.actions[0], success=True, output="Charge: 100%")

    follow_up = await companion.follow_up(
        "Show current battery status.",
        plan,
        [result],
        VerificationResult(passed=True, details=["verified"]),
    )

    assert follow_up is not None
    assert follow_up.source == "local_fallback"
    assert "charge and power state" in follow_up.suggestions[0]
    assert "Review the verified result" not in follow_up.spoken_text()
    assert model.calls == []


@pytest.mark.asyncio
async def test_follow_up_never_suggests_more_work_after_failed_verification():
    companion = ExecutionCompanion(_Model(error=RuntimeError("offline")))
    plan = _plan()
    result = ActionResult(action=plan.actions[0], success=True, output="output")

    follow_up = await companion.follow_up(
        "Show the OS version.",
        plan,
        [result],
        VerificationResult(passed=False, details=["not verified"]),
    )

    assert follow_up is None

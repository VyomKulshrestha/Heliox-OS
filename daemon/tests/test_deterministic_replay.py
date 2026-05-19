from __future__ import annotations

from dataclasses import dataclass

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams
from pilot.testing.replay import (
    RecordingExecutor,
    RecordingModelRouter,
    ReplayExecutor,
    ReplayHarness,
    ReplayMismatchError,
)


def _notify_action(message: str) -> Action:
    return Action(
        action_type=ActionType.NOTIFY,
        target=message,
        parameters=NotifyParams(summary="Replay", body=message),
    )


@dataclass
class FakeModelRouter:
    response: str = '{"actions": []}'

    async def generate(self, prompt, *, system="", json_mode=False, temperature=0.1, stream_callback=None):
        if stream_callback:
            await stream_callback(self.response)
        return self.response


class FakeExecutor:
    async def execute(
        self,
        plan,
        on_action_start=None,
        on_action_complete=None,
        cancel_event=None,
        plan_id=None,
        initial_last_output="",
    ):
        results = []
        for action in plan.actions:
            if on_action_start:
                await on_action_start(action)
            result = ActionResult(action=action, success=True, output=f"ran:{action.target}")
            if on_action_complete:
                await on_action_complete(result)
            results.append(result)
        return results


@pytest.mark.asyncio
async def test_recording_and_replay_round_trip(tmp_path):
    harness = ReplayHarness()
    model = RecordingModelRouter(FakeModelRouter("plan-json"), harness.recorder)
    executor = RecordingExecutor(FakeExecutor(), harness.recorder)
    plan = ActionPlan(actions=[_notify_action("hello")], explanation="notify")

    response = await model.generate(
        "build a plan",
        system="planner",
        json_mode=True,
        temperature=0.2,
    )
    results = await executor.execute(plan)

    assert response == "plan-json"
    assert results[0].output == "ran:hello"

    recording_path = tmp_path / "session.jsonl"
    harness.save(recording_path)
    replay_model, replay_executor = ReplayHarness.from_recording(recording_path).replay_components()

    replayed_response = await replay_model.generate(
        "build a plan",
        system="planner",
        json_mode=True,
        temperature=0.2,
    )
    replayed_results = await replay_executor.execute(plan)

    assert replayed_response == "plan-json"
    assert replayed_results == results


@pytest.mark.asyncio
async def test_replay_model_rejects_prompt_drift(tmp_path):
    harness = ReplayHarness()
    await harness.wrap_model_router(FakeModelRouter("ok")).generate("original prompt")
    path = tmp_path / "session.jsonl"
    harness.save(path)

    replay_model = ReplayHarness.from_recording(path).replay_model_router()

    with pytest.raises(ReplayMismatchError, match="LLM replay mismatch"):
        await replay_model.generate("changed prompt")


@pytest.mark.asyncio
async def test_replay_executor_rejects_action_drift(tmp_path):
    harness = ReplayHarness()
    original = ActionPlan(actions=[_notify_action("original")], explanation="notify")
    changed = ActionPlan(actions=[_notify_action("changed")], explanation="notify")
    await harness.wrap_executor(FakeExecutor()).execute(original)
    path = tmp_path / "session.jsonl"
    harness.save(path)

    replay_executor = ReplayHarness.from_recording(path).replay_executor()

    with pytest.raises(ReplayMismatchError, match="Action replay mismatch"):
        await replay_executor.execute(changed)


def test_recorder_can_capture_system_api_responses(tmp_path):
    harness = ReplayHarness()
    harness.recorder.record_system_response(
        endpoint="/api/state",
        method="GET",
        status_code=200,
        body={"window": "Terminal"},
    )
    path = tmp_path / "session.jsonl"
    harness.save(path)

    loaded = ReplayHarness.from_recording(path).recorder.events

    assert loaded[0].event_type == "system_response"
    assert loaded[0].payload["body"]["window"] == "Terminal"

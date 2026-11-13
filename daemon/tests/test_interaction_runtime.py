from unittest.mock import AsyncMock

import pytest

from pilot.system.interaction import InteractionPhase, InteractionRuntime, acknowledgement_for


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("find the latest update", "I’ll look into that."),
        ("show system information", "I’ll check."),
        ("open GitHub", "I’ll open it."),
        ("build a release checklist", "I’ll put that together."),
    ],
)
def test_acknowledgement_describes_the_next_step_without_claiming_success(utterance, expected):
    assert acknowledgement_for(utterance) == expected


@pytest.mark.asyncio
async def test_runtime_publishes_one_monotonic_interaction_state():
    emit = AsyncMock()
    runtime = InteractionRuntime(emit)

    started = await runtime.start("  show   system information  ", source="voice")
    interaction_id = started["interaction_id"]
    planning = await runtime.transition(
        InteractionPhase.PLANNING,
        message="Planning the safest useful action",
        interaction_id=interaction_id,
    )
    completed = await runtime.transition(
        InteractionPhase.COMPLETED,
        message="System information is ready",
        interaction_id=interaction_id,
    )

    assert started["user_input"] == "show system information"
    assert planning["sequence"] > started["sequence"]
    assert completed["active"] is False
    assert runtime.status()["phase"] == "completed"
    assert [call.args[0] for call in emit.await_args_list] == [
        "interaction_state",
        "interaction_state",
        "interaction_state",
    ]


@pytest.mark.asyncio
async def test_stale_interaction_cannot_overwrite_current_state():
    runtime = InteractionRuntime(AsyncMock())
    first = await runtime.start("first", source="voice")
    second = await runtime.start("second", source="text")

    stale = await runtime.transition(
        InteractionPhase.FAILED,
        interaction_id=str(first["interaction_id"]),
    )

    assert stale["interaction_id"] == second["interaction_id"]
    assert stale["phase"] == "understanding"

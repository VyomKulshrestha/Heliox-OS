from __future__ import annotations

import asyncio

import pytest

from pilot.system.companion_speech import CompanionSpeechCoordinator


@pytest.mark.asyncio
async def test_higher_priority_speech_preempts_current_utterance() -> None:
    narration_started = asyncio.Event()
    release_warning = asyncio.Event()

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        if text == "narration":
            narration_started.set()
            await asyncio.Event().wait()
        await release_warning.wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    narration = asyncio.create_task(coordinator.speak("narration", channel="task_narration"))
    await narration_started.wait()
    warning = asyncio.create_task(coordinator.speak("warning", channel="approval_risk"))
    release_warning.set()

    assert (await narration).status == "superseded"
    assert (await warning).status == "spoken"
    assert coordinator.status()["preemptions"] == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_lower_priority_speech_waits_for_active_utterance() -> None:
    release = asyncio.Event()
    warning_started = asyncio.Event()
    started: list[str] = []

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        started.append(text)
        if text == "warning":
            warning_started.set()
            await release.wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    warning = asyncio.create_task(coordinator.speak("warning", channel="approval_risk"))
    await warning_started.wait()
    insight = asyncio.create_task(coordinator.speak("insight", channel="background_insight"))
    await asyncio.sleep(0)
    assert started == ["warning"]

    release.set()
    assert (await warning).status == "spoken"
    assert (await insight).status == "spoken"
    assert started == ["warning", "insight"]
    await coordinator.close()


@pytest.mark.asyncio
async def test_duplicate_speech_is_suppressed_across_producers() -> None:
    release = asyncio.Event()

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        await release.wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    first = asyncio.create_task(
        coordinator.speak(
            "first rendering",
            channel="approval_risk",
            dedupe_key="plan-1",
        )
    )
    await asyncio.sleep(0)
    duplicate = await coordinator.speak(
        "second rendering",
        channel="approval_risk",
        dedupe_key="plan-1",
    )
    release.set()

    assert duplicate.status == "duplicate"
    assert (await first).status == "spoken"
    assert coordinator.status()["duplicates_suppressed"] == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_stop_all_resolves_current_and_queued_speech() -> None:
    started = asyncio.Event()

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        started.set()
        await asyncio.Event().wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    current = asyncio.create_task(coordinator.speak("current", channel="final_answer"))
    await started.wait()
    queued = asyncio.create_task(coordinator.speak("queued", channel="task_narration"))
    await asyncio.sleep(0)

    assert await coordinator.stop_all() == 2
    assert (await current).status == "cancelled"
    assert (await queued).status == "cancelled"
    await coordinator.close()


@pytest.mark.asyncio
async def test_idle_worker_retires_and_restarts_without_losing_speech() -> None:
    spoken: list[str] = []

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        spoken.append(text)
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)

    assert (await coordinator.speak("first")).status == "spoken"
    await asyncio.sleep(0)
    assert coordinator._worker is None
    assert (await coordinator.speak("second")).status == "spoken"
    assert spoken == ["first", "second"]
    await coordinator.close()


@pytest.mark.asyncio
async def test_a_withdrawn_line_is_never_said_after_the_answer() -> None:
    # "Opening it now." waits behind a spoken correction; the answer arrives and the acknowledgement is withdrawn.
    release = asyncio.Event()
    said: list[str] = []

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        said.append(text)
        if text == "correction":
            await release.wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    correction = asyncio.create_task(coordinator.speak("correction", channel="live_correction"))
    await asyncio.sleep(0.01)
    ack = asyncio.create_task(coordinator.speak("Opening it now.", channel="task_narration"))
    await asyncio.sleep(0.01)
    ack.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ack
    answer = asyncio.create_task(coordinator.speak("answer", channel="task_narration"))
    release.set()
    await correction
    assert (await answer).status == "spoken"
    assert said == ["correction", "answer"]
    await coordinator.close()


@pytest.mark.asyncio
async def test_withdrawing_the_line_being_spoken_stops_it() -> None:
    started = asyncio.Event()

    async def speaker(text: str, recorder) -> bool:  # noqa: ARG001
        started.set()
        await asyncio.Event().wait()
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    line = asyncio.create_task(coordinator.speak("Opening it now.", channel="task_narration"))
    await started.wait()
    line.cancel()
    with pytest.raises(asyncio.CancelledError):
        await line
    await asyncio.sleep(0.01)
    assert coordinator.status()["active"] is False
    await coordinator.close()

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.voice import ContinuousVoiceListener


@pytest.mark.asyncio
async def test_recognition_diagnostic_consumes_utterance_without_dispatching():
    dispatch = AsyncMock()
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    listener._running = True
    transcripts = iter(["Hey Heliox, open GitHub", "No speech detected"])

    async def transcribe(**_kwargs):
        transcript = next(transcripts)
        if transcript == "No speech detected":
            listener._running = False
        return transcript

    listener._record_and_transcribe = transcribe
    diagnostic_task = asyncio.create_task(listener.capture_diagnostic())
    await asyncio.sleep(0)

    await listener._listen_loop()
    result = await diagnostic_task

    assert result["transcript"] == "Hey Heliox, open GitHub"
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_recognition_diagnostic_times_out_cleanly():
    listener = ContinuousVoiceListener(config=PilotConfig())
    listener._running = True

    with pytest.raises(TimeoutError):
        await listener.capture_diagnostic(0.01)

    assert listener._diagnostic_waiter is None


@pytest.mark.asyncio
async def test_listener_keeps_listening_while_previous_command_runs():
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_received = asyncio.Event()
    received: list[str] = []

    async def dispatch(command: str) -> None:
        received.append(command)
        if command == "first task":
            first_started.set()
            await release_first.wait()
        else:
            second_received.set()

    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    transcripts = iter(["Hey Heliox, first task", "Hey Heliox, second task"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            await second_received.wait()
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    listen_task = asyncio.create_task(listener._listen_loop())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.wait_for(second_received.wait(), timeout=1)

    assert received == ["first task", "second task"]

    release_first.set()
    await listen_task
    await asyncio.gather(*tuple(listener._command_tasks))


@pytest.mark.asyncio
async def test_listener_accepts_one_wake_free_follow_up_inside_conversation_window():
    received: list[str] = []
    follow_up_received = asyncio.Event()

    async def dispatch(command: str) -> None:
        received.append(command)
        if command == "Click on Launch on the website":
            follow_up_received.set()

    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    listener.arm_follow_up_window(30)
    transcripts = iter(["Click on Launch on the website"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            await follow_up_received.wait()
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()
    await asyncio.gather(*tuple(listener._command_tasks))

    assert received == ["Click on Launch on the website"]
    # The conversation stays open for the rest of its window, so the next reply needs no wake word either.
    assert listener.follow_up_remaining_seconds > 25


@pytest.mark.asyncio
async def test_autonomous_listener_routes_natural_speech_without_wake_phrase():
    received = asyncio.Event()
    dispatch = AsyncMock(side_effect=lambda _command: received.set())
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    transcripts = iter(["click launch"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            await received.wait()
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()
    await asyncio.gather(*tuple(listener._command_tasks))

    dispatch.assert_awaited_once_with("click launch")


@pytest.mark.asyncio
async def test_listener_preserves_application_name_casing_after_wake_phrase():
    received = asyncio.Event()
    dispatch = AsyncMock(side_effect=lambda _command: received.set())
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    transcripts = iter(["Hey Heliox, Open Visual Studio Code"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            await received.wait()
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()
    await asyncio.gather(*tuple(listener._command_tasks))

    dispatch.assert_awaited_once_with("Open Visual Studio Code")


@pytest.mark.asyncio
async def test_autonomous_listener_ignores_ambient_sentence_fragments():
    dispatch = AsyncMock()
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()
    transcripts = iter(["we were interested in", "it's all it"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()

    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_listener_ignores_wake_free_speech_after_conversation_window_expires():
    dispatch = AsyncMock()
    config = PilotConfig()
    config.voice.continuous_conversation_enabled = False
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=config,
    )
    listener._wake_calibrator = MagicMock()
    listener.arm_follow_up_window(0)
    transcripts = iter(["click launch"])

    async def transcribe(**_kwargs):
        try:
            return next(transcripts)
        except StopIteration:
            listener._running = False
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()

    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_tts_overlap_is_not_routed_as_an_autonomous_command():
    dispatch = AsyncMock()
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"],
        on_command=dispatch,
        config=PilotConfig(),
    )
    listener._wake_calibrator = MagicMock()

    async def transcribe(**_kwargs):
        listener.suppress_wake_free_commands()
        listener.resume_wake_free_commands()
        listener._running = False
        return "I opened the world monitor for you"

    listener._record_and_transcribe = transcribe
    listener._running = True

    await listener._listen_loop()

    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_voice_command_stops_current_speech_before_live_correction():
    server = PilotServer(PilotConfig())
    server._interactive_request_active = True
    server._handle_interject = AsyncMock(
        return_value={"status": "revising", "message": "Applying correction"},
    )
    server._broadcast_notification = AsyncMock()
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.status.return_value = {"active": True}
    server._speech_coordinator.stop_all = AsyncMock(return_value=1)

    await server._voice_command_dispatch("use the other folder instead")

    server._speech_coordinator.stop_all.assert_awaited_once()
    server._handle_interject.assert_awaited_once_with(
        {"input": "use the other folder instead"},
        None,
    )
    assert server._broadcast_notification.await_args_list[0].args == (
        "voice_status",
        {"status": "interrupted", "reason": "new voice command"},
    )


@pytest.mark.asyncio
async def test_companion_speech_suppresses_its_own_wake_free_transcription():
    server = PilotServer(PilotConfig())
    listener = MagicMock()
    listener.is_running = True
    listener.wait_for_floor = AsyncMock(return_value=0.0)
    server._voice_listener = listener
    outcome = MagicMock(status="spoken")
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock(return_value=outcome)

    result = await server._speak_companion_text("I opened the world monitor")

    assert result is outcome
    listener.suppress_wake_free_commands.assert_called_once_with()
    listener.resume_wake_free_commands.assert_called_once_with()

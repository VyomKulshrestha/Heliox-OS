"""Talking with Heliox: barge-in, waiting for the person to finish, open conversations, short replies, the wake
word as recognisers mishear it, and overheard speech reaching Live."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.voice import (
    BargeInGate,
    BargeInWatcher,
    ContinuousVoiceListener,
    echo_free_output,
    is_short_decision,
    looks_like_conversation_turn,
    loose_wake_prefix,
)

# ── the barge-in gate ──


def feed(gate: BargeInGate, samples: list[tuple[float, bool, float]]) -> float | None:
    for when, speaking, level in samples:
        if gate.push(when, speaking, level):
            return when
    return None


def frames(start: float, end: float, speaking: bool, level: float, step: float = 0.02):
    count = round((end - start) / step)
    return [(start + i * step, speaking, level) for i in range(count)]


def test_with_a_headset_a_quarter_second_of_voice_stops_heliox_and_a_cough_does_not():
    gate = BargeInGate(echo_free=True)
    assert (
        feed(gate, frames(0.0, 1.0, False, 0.001) + frames(1.0, 1.15, True, 0.2) + frames(1.15, 2.0, False, 0.001))
        is None
    )
    stopped = feed(gate, frames(2.0, 3.0, True, 0.05))
    assert stopped is not None and 2.24 <= stopped <= 2.3


def test_with_open_speakers_only_voice_clearly_louder_than_heliox_itself_stops_it():
    # Heliox's own voice in the microphone sets the voice flag too; it must not stop itself.
    echo = BargeInGate(echo_free=False)
    assert feed(echo, frames(0.0, 1.0, False, 0.02) + frames(1.0, 4.0, True, 0.025)) is None
    person = BargeInGate(echo_free=False)
    stopped = feed(person, frames(0.0, 1.0, False, 0.02) + frames(1.0, 3.0, True, 0.08))
    assert stopped is not None and 1.39 <= stopped <= 1.45
    # Talking at once, before the gate has heard how loud Heliox is, is not enough to judge.
    early = BargeInGate(echo_free=False)
    assert feed(early, frames(0.0, 2.0, True, 0.3)) is None


def test_echo_free_outputs_are_headsets_and_headphones():
    assert echo_free_output("Headphones (Rockerz Prime 415)")
    assert echo_free_output("Headset Earphone (Jabra Evolve2)")
    assert not echo_free_output("Speakers (Realtek(R) Audio)")
    assert not echo_free_output("")


@pytest.mark.asyncio
async def test_the_watcher_stops_speech_from_voice_flags_and_marks_the_utterance_as_addressed():
    listener = MagicMock()
    listener.is_running = True
    listener._recorder.is_active = True
    listener._recorder.last_frame_rms = 0.05
    listener.user_is_speaking = False
    watcher = BargeInWatcher(listener, echo_free=True)
    assert watcher.is_active

    async def start_talking():
        await asyncio.sleep(0.1)
        listener.user_is_speaking = True

    asyncio.ensure_future(start_talking())
    started = asyncio.get_running_loop().time()
    assert await watcher.wait_for_speech_start(timeout=2.0)
    assert 0.3 <= asyncio.get_running_loop().time() - started < 0.6
    listener.note_interruption.assert_called_once_with()
    assert not await BargeInWatcher(listener, echo_free=True).wait_for_speech_start(timeout=0.0)


# ── what counts as a reply, and the wake word by sound ──


@pytest.mark.parametrize(
    "text", ["Skip the backup.", "Wait, stop", "No.", "Don't delete it", "Actually, make it Friday", "Okay, thanks"]
)
def test_replies_in_a_conversation(text):
    assert looks_like_conversation_turn(text)


@pytest.mark.parametrize(
    "text", ["okay so anyway we went to lunch", "and then we left", "I'll send Priya the deck by Friday", ""]
)
def test_other_speech_is_not_a_reply(text):
    assert not looks_like_conversation_turn(text)


def test_short_decisions_are_complete():
    assert is_short_decision("No.") and is_short_decision("yes") and not is_short_decision("na")


@pytest.mark.parametrize(
    ("heard", "prefix"),
    [
        ("Hay heliog, what's on my to-do list", "Hay heliog"),  # recorded from the streaming recogniser
        ("Hay helioch's what's am I to do liszt", "Hay helioch's"),  # recorded
        ("Heliocs, open my mail", "Heliocs"),  # Whisper's habit
        ("Hi Helios, stop", "Hi Helios"),
        ("hey heli open notes", "hey heli"),
    ],
)
def test_the_wake_phrase_is_matched_on_sound(heard, prefix):
    assert loose_wake_prefix(heard) == prefix


@pytest.mark.parametrize(
    "heard",
    ["Hey hello there", "Hey Hillary, lunch?", "Hey Healy", "Hey helicopter pilots", "Helium is light", "Hey he"],
)
def test_ordinary_speech_does_not_sound_like_the_wake_phrase(heard):
    assert loose_wake_prefix(heard) == ""


# ── the listener ──


def make_listener(transcripts: list[str], *, task_open=False, follow_up=0.0, suppressed=False):
    dispatched: list[str] = []
    overheard: list[str] = []
    finished = asyncio.Event()

    async def dispatch(command: str) -> None:
        dispatched.append(command)

    async def overhear(text: str) -> None:
        overheard.append(text)

    listener = ContinuousVoiceListener(
        wake_words=["hey heliox", "heliox"], on_command=dispatch, on_overheard=overhear, config=PilotConfig()
    )
    listener._wake_calibrator = MagicMock()
    listener._wake_calibrator.match_promoted_variant.return_value = None
    listener._wake_calibrator.check_near_miss.return_value = None
    listener.task_open = lambda: task_open
    if follow_up:
        listener.arm_follow_up_window(follow_up)
    if suppressed:
        listener.suppress_wake_free_commands()
    queue = iter(transcripts)

    async def transcribe(**_kwargs):
        try:
            return next(queue)
        except StopIteration:
            listener._running = False
            finished.set()
            return "No speech detected"

    listener._record_and_transcribe = transcribe
    listener._running = True

    async def run():
        await listener._listen_loop()
        await asyncio.gather(*tuple(listener._command_tasks))
        return dispatched, overheard

    return listener, run


@pytest.mark.asyncio
async def test_while_a_task_runs_replies_need_no_wake_word_and_other_speech_only_reaches_live():
    _, run = make_listener(["Skip the backup.", "I'll send Priya the deck by Friday", "No."], task_open=True)
    dispatched, overheard = await run()
    assert dispatched == ["Skip the backup", "No"]  # "No" is a whole answer, not a wake word with nothing after it
    assert overheard == ["I'll send Priya the deck by Friday"]


@pytest.mark.asyncio
async def test_with_no_conversation_open_room_speech_is_not_for_heliox():
    _, run = make_listener(["Skip the backup.", "I'll send Priya the deck by Friday", "No."])
    dispatched, overheard = await run()
    assert dispatched == [] and overheard == []


@pytest.mark.asyncio
async def test_after_heliox_speaks_the_conversation_stays_open_for_several_replies():
    listener, run = make_listener(["Wait, stop", "Actually, make it Friday"], follow_up=30)
    dispatched, _ = await run()
    assert dispatched == ["Wait, stop", "Actually, make it Friday"]
    assert listener.follow_up_remaining_seconds > 25


@pytest.mark.asyncio
async def test_what_the_person_said_over_heliox_is_heard_even_while_its_own_voice_is_ignored():
    listener, run = make_listener(["Wait, stop."], suppressed=True)
    listener.note_interruption()
    dispatched, _ = await run()
    assert dispatched == ["Wait, stop"]
    quiet, run_quiet = make_listener(["Wait, stop."], suppressed=True, follow_up=30)
    assert await run_quiet() == ([], [])  # without an interruption, speech during Heliox's voice is its echo


@pytest.mark.asyncio
async def test_talking_over_heliox_stops_its_voice_but_only_a_reply_runs():
    # 19:58 rehearsal: the person talked over Heliox and "PowerShell" was planned as a new task.
    listener, run = make_listener(["PowerShell", "Wait, stop."], suppressed=True)
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 2.4  # long enough to be a sentence, not a cut-off
    dispatched, overheard = await run()
    assert dispatched == [] and overheard == ["PowerShell"]


# "Wait, stop." said over Heliox, about a second of voice, as the recognisers wrote it in four rehearsals (19:58,
# 21:19 and 22:04). Heliox had already stopped talking.
CUT_OFF_READINGS = [
    "Brutes gold",
    "Bruins go home",
    "Very discovered.",
    "Good score.",
    "Visual Studio Code.",
    "Where we store home",
    "PowerShell",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("heard", CUT_OFF_READINGS)
async def test_a_short_unreadable_cut_off_over_heliox_is_taken_as_stop(heard):
    listener, run = make_listener([heard], suppressed=True)
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 1.0
    statuses = []

    async def status(name, data):
        statuses.append((name, data))

    listener._on_status = status
    dispatched, overheard = await run()
    assert dispatched == ["Stop"] and overheard == []  # acknowledged by the stop path; nothing goes to Live
    shown = [data for name, data in statuses if name == "follow_up_detected"]
    assert shown == [{"transcript": "", "wake_free": True, "cut_off": True}]  # the misheard words are not shown


@pytest.mark.asyncio
@pytest.mark.parametrize("heard", CUT_OFF_READINGS)
async def test_a_cut_off_stops_and_is_acknowledged_without_reaching_the_planner_or_the_chat(heard):
    import time

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._voice_cut_off_at = time.monotonic() - 1.0
    listener, run = make_listener([heard], suppressed=True)
    listener._on_command = server._voice_command_dispatch
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 1.1
    await run()
    server._handle_execute.assert_not_awaited()
    assert server._speak_voice_response.await_args.args[0] == "Okay, stopped."
    sent = [call.args[0] for call in server._broadcast_notification.await_args_list]
    assert "voice_command" not in sent  # the event that adds a user turn to the chat


@pytest.mark.asyncio
async def test_a_sentence_said_over_heliox_is_still_heard_as_what_it_is():
    listener, run = make_listener(["Can you also add the budget figures from March to that"], suppressed=True)
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 2.8
    assert await run() == (["Can you also add the budget figures from March to that"], [])
    aside, run_aside = make_listener(["the meeting moved to Thursday afternoon"], suppressed=True)
    aside.note_interruption()
    aside._recorder.last_voiced_seconds = 2.2
    assert await run_aside() == ([], ["the meeting moved to Thursday afternoon"])  # not a stop: Live hears it
    request, run_request = make_listener(["Open my mail"], suppressed=True)
    request.note_interruption()
    request._recorder.last_voiced_seconds = 0.9
    assert await run_request() == (["Stop"], [])  # a second said over Heliox is a stop, whatever it reads as
    quiet, run_quiet = make_listener(["Very discovered."], follow_up=30)
    quiet._recorder.last_voiced_seconds = 1.0  # the same words without cutting Heliox off are not a stop
    assert await run_quiet() == ([], ["Very discovered."])


def test_only_a_short_unreadable_utterance_is_a_cut_off_stop():
    from pilot.system.voice import is_cut_off_stop

    assert is_cut_off_stop(1.0) and is_cut_off_stop(1.6)
    assert not is_cut_off_stop(2.0) and not is_cut_off_stop(None)


def test_while_a_cut_off_is_still_short_its_words_are_not_captioned():
    import time

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_partial=AsyncMock(), config=PilotConfig())
    listener.note_interruption()
    listener._recorder.speech_started_at = time.monotonic() - 0.5
    assert listener._cutting_off()
    listener._recorder.speech_started_at = time.monotonic() - 2.5  # a sentence now: its words are shown
    assert not listener._cutting_off()
    listener.note_interruption(by_button=True)
    listener._recorder.speech_started_at = time.monotonic() - 0.5
    assert not listener._cutting_off()  # after the button the words are what the person said


@pytest.mark.parametrize("said", ["That's enough.", "Okay, that's enough", "Enough", "Stop there", "Hold on"])
def test_ordinary_ways_to_stop_heliox_are_replies(said):
    assert looks_like_conversation_turn(said)


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["That's enough.", "Okay, that's enough", "Enough", "Stop there"])
async def test_ordinary_ways_to_stop_heliox_stop_it_without_planning(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    await server._voice_command_dispatch(said)
    server._handle_execute.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["message"] == "Okay, stopped."


@pytest.mark.asyncio
async def test_got_it_stops_the_voice_but_never_background_work():
    import time

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._stop_background_jobs = AsyncMock(return_value=2)
    await server._voice_command_dispatch("Got it.")  # after an answer: only an acknowledgement
    server._handle_execute.assert_not_awaited()
    server._stop_background_jobs.assert_not_awaited()
    server._speak_voice_response.assert_not_awaited()
    server._voice_cut_off_at = time.monotonic() - 1.0
    await server._voice_command_dispatch("Okay, okay")  # over Heliox: acknowledged
    assert server._speak_voice_response.await_args.args[0] == "Okay, stopped."
    server._stop_background_jobs.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_request_that_starts_like_a_stop_is_still_a_request():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._stop_background_jobs = AsyncMock(return_value=0)
    server._handle_execute = AsyncMock(return_value={"status": "success", "message": "Zipped."})
    await server._voice_command_dispatch("That's enough for the backup folder, now zip it")
    server._stop_background_jobs.assert_not_awaited()
    server._handle_execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_after_a_question_a_short_answer_is_a_reply_and_a_story_is_not():
    listener, run = make_listener(["the other one, the blue file", "PowerShell"])
    listener.arm_follow_up_window(30, asked=True)  # Heliox: "Which file should I open?"
    dispatched, _ = await run()
    assert dispatched == ["the other one, the blue file", "PowerShell"]
    statement, run_statement = make_listener(["the other one, the blue file", "PowerShell"], follow_up=30)
    assert await run_statement() == ([], ["the other one, the blue file", "PowerShell"])


# What Heliox heard in the 19:53-19:58 rehearsal and in earlier takes, said without the wake phrase while a
# conversation was open, and the film lines. Requests, stops, corrections and undo run; the rest only reaches Live.
OPEN_CONVERSATION_LINES = [
    ("PowerShell", False),
    ("Open, open, open, open", False),
    ("Copied", False),
    ("Thanks for watching", False),
    ("The invoice is due on the thirty-first of November.", False),
    ("I'll send Priya the deck by Friday", False),
    ("okay so anyway we went to lunch", False),
    ("Wait, stop.", True),
    ("Skip the backup.", True),
    ("Undo that.", True),
    ("No.", True),
    ("Cancel", True),
    ("Open my mail", True),
]


@pytest.mark.parametrize(("text", "acts"), OPEN_CONVERSATION_LINES)
def test_what_runs_in_an_open_conversation(text, acts):
    from pilot.system.voice import should_act_on_follow_up

    assert should_act_on_follow_up(text) is acts


@pytest.mark.asyncio
async def test_in_an_open_conversation_only_replies_and_requests_run():
    lines = [text for text, _ in OPEN_CONVERSATION_LINES]
    _, run = make_listener(lines, follow_up=30)
    dispatched, overheard = await run()
    assert dispatched == [text.strip(" .") for text, acts in OPEN_CONVERSATION_LINES if acts]
    assert overheard == [text for text, acts in OPEN_CONVERSATION_LINES if not acts]


@pytest.mark.asyncio
async def test_a_misheard_wake_phrase_still_wakes_heliox():
    _, run = make_listener(["Hay heliog, what's on my to-do list?"])
    dispatched, _ = await run()
    assert dispatched == ["what's on my to-do list"]


@pytest.mark.asyncio
async def test_wait_for_floor_holds_while_the_person_talks():
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener._running = True
    listener._recorder.user_is_speaking = True

    async def finish():
        await asyncio.sleep(0.2)
        listener._recorder.user_is_speaking = False

    asyncio.ensure_future(finish())
    waited = await listener.wait_for_floor(max_seconds=2.0)
    assert 0.15 <= waited < 0.5
    listener._recorder.user_is_speaking = True
    assert await listener.wait_for_floor(max_seconds=0.1) < 0.3


# ── the server ──


def server_with_listener():
    server = PilotServer(PilotConfig())
    listener = MagicMock()
    listener.is_running = True
    listener.wait_for_floor = AsyncMock(return_value=0.0)
    server._voice_listener = listener
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock(return_value=MagicMock(status="spoken"))
    return server, listener


@pytest.mark.asyncio
async def test_heliox_waits_for_the_person_watches_for_barge_in_and_keeps_the_conversation_open():
    server, listener = server_with_listener()
    server.config.voice.barge_in_enabled = True
    await server._speak_companion_text("Here is your morning.")
    listener.wait_for_floor.assert_awaited_once()
    recorder = server._speech_coordinator.speak.await_args.kwargs["recorder"]
    assert isinstance(recorder, BargeInWatcher) and recorder.listener is listener
    listener.arm_follow_up_window.assert_called_once_with(asked=False)


@pytest.mark.asyncio
async def test_without_barge_in_speech_is_not_watched():
    server, listener = server_with_listener()
    server.config.voice.barge_in_enabled = False
    await server._speak_companion_text("Here is your morning.")
    assert server._speech_coordinator.speak.await_args.kwargs["recorder"] is None


@pytest.mark.asyncio
async def test_overheard_speech_goes_to_live_only():
    server = PilotServer(PilotConfig())
    server._live = MagicMock()
    await server._voice_overheard("I'll send Priya the deck by Friday")
    server._live.offer_speech.assert_called_once_with("I'll send Priya the deck by Friday")
    assert not server._voice_task_open()
    server._pending_confirms["plan"] = MagicMock()
    assert server._voice_task_open()


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Stop.", "Cancel", "Wait, stop", "No"])
async def test_stop_cancel_and_no_deny_a_waiting_approval(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._pending_confirms["plan-1"] = MagicMock()
    server._handle_confirm = AsyncMock(return_value={"status": "ok"})
    await server._voice_command_dispatch(said)
    server._handle_confirm.assert_awaited_once_with(
        {"plan_id": "plan-1", "confirmed": False, "input_channel": "voice_deny"}, None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Yes.", "Approve", "Go ahead"])
async def test_yes_never_approves_by_voice(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._pending_confirms["plan-1"] = MagicMock()
    server._handle_confirm = AsyncMock()
    await server._voice_command_dispatch(said)
    server._handle_confirm.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["status"] == "approval_required"


@pytest.mark.asyncio
async def test_the_floor_stays_with_the_person_for_a_moment_after_their_last_sound():
    import time

    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener._running = True
    listener._recorder.user_is_speaking = False  # the turn ended at a short pause...
    listener._recorder.last_above_threshold_at = time.time()  # ...but they were still talking just now
    waited = await listener.wait_for_floor(max_seconds=2.0)
    assert 0.3 <= waited < 0.6


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Wait.", "Hold on", "Wait, wait"])
async def test_wait_said_to_heliox_stops_it_without_planning_anything(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    await server._voice_command_dispatch(said)
    server._handle_execute.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["message"] == "Okay, stopped."
    server._speak_voice_response.assert_not_awaited()  # nothing was running or speaking: no need to say so


@pytest.mark.asyncio
async def test_wait_stop_said_over_heliox_is_acknowledged_aloud():
    import time

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._voice_cut_off_at = time.monotonic() - 1.0  # the person just talked over Heliox
    await server._voice_command_dispatch("Wait, stop.")
    server._handle_execute.assert_not_awaited()
    assert server._speak_voice_response.await_args.args[0] == "Okay, stopped."


# ── replies right after Heliox speaks, and the words right after "Hey Heliox," ──


@pytest.mark.asyncio
@pytest.mark.parametrize(("started_after_heliox", "heard"), [(True, ["Approve"]), (False, [])])
async def test_a_reply_that_starts_after_heliox_finished_counts_and_its_echo_does_not(started_after_heliox, heard):
    import time

    listener, run = make_listener([], follow_up=30)

    async def transcribe(**_kwargs):
        # The listener began waiting before Heliox spoke ("Yes" was being answered)...
        listener.suppress_wake_free_commands()
        await asyncio.sleep(0.01)
        listener.resume_wake_free_commands()
        # ...and the speech it captured started after (a reply) or before (Heliox's own voice) Heliox stopped.
        offset = 0.5 if started_after_heliox else -0.5
        listener._recorder.speech_started_at = time.monotonic() + offset
        listener._running = False
        return "Approve"

    listener._record_and_transcribe = transcribe
    dispatched, _ = await run()
    assert dispatched == heard


@pytest.mark.asyncio
async def test_frames_queued_before_the_call_are_used_only_when_asked():
    import numpy as np

    from pilot.system.voice import _ContinuousRecorder

    rec = _ContinuousRecorder(start_frames=2, silence_frames=4)
    rec._queue = asyncio.Queue()
    loud = np.full(800, 10000, dtype=np.int16)
    quiet = np.zeros(800, dtype=np.int16)
    for frame in [loud] * 6 + [quiet] * 5:  # queued before the call
        rec._queue.put_nowait(frame)
    path = await rec.record_utterance(timeout=1.0, keep_backlog=True)
    assert path and rec.speech_started_at is not None
    for frame in [loud] * 6 + [quiet] * 5:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=0.3) is None  # the ambient default still starts from now


# ── a spoken "No" lands fast; a sentence with a go-ahead in it is never half-heard ──


def _recorder_with_words(words: str):
    import numpy as np

    from pilot.system.voice import _ContinuousRecorder

    rec = _ContinuousRecorder(start_frames=2, silence_frames=40)
    rec._queue = asyncio.Queue()
    rec.complete_hint = lambda: is_short_decision(words)
    loud = np.full(800, 10000, dtype=np.int16)
    quiet = np.zeros(800, dtype=np.int16)
    return rec, loud, quiet


@pytest.mark.asyncio
async def test_a_one_word_answer_ends_the_turn_after_400_ms_instead_of_the_2_s_hold():
    rec, loud, quiet = _recorder_with_words("No")
    for frame in [loud] * 6 + [quiet] * 45:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=1.0, keep_backlog=True)
    assert rec.last_endpoint_reason == "short_answer" and rec.last_endpoint_ms == 400  # the first check, here
    # 6 frames of voice at 50 ms, less the frame that confirmed the start: no pre-roll, no trailing silence.
    assert rec.last_voiced_seconds == pytest.approx(0.25, abs=0.06)


@pytest.mark.asyncio
async def test_a_longer_sentence_is_not_cut_at_its_first_word():
    rec, loud, quiet = _recorder_with_words("No problem go ahead")
    for frame in [loud] * 6 + [quiet] * 45:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=1.0, keep_backlog=True)
    assert rec.last_endpoint_reason == "silence"  # the plain endpoint, not the short-answer one


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["No problem, go ahead", "Yes, but skip the backup", "No, don't go ahead"])
async def test_a_sentence_with_a_go_ahead_in_it_neither_denies_nor_approves(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._pending_confirms["plan-1"] = MagicMock()
    server._handle_confirm = AsyncMock()
    await server._voice_command_dispatch(said)
    server._handle_confirm.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["status"] == "approval_required"


@pytest.mark.asyncio
async def test_a_recogniser_that_finished_loading_while_waiting_is_used_for_that_utterance():
    rec, loud, quiet = _recorder_with_words("")
    rec.complete_hint = None
    sink = MagicMock()
    rec.on_speech_start = lambda: setattr(rec, "stream_sink", sink)
    for frame in [loud] * 6 + [quiet] * 45:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=1.0, keep_backlog=True)
    sink.begin.assert_called_once()
    assert sink.feed.call_count > 4


@pytest.mark.asyncio
async def test_a_replaced_line_takes_its_barge_in_watcher_with_it(monkeypatch):
    # A more urgent line cancels the one being spoken; its watcher must not linger and later treat the
    # person's next sentence as an interruption.
    from pilot.system import voice

    async def slow_speak(text, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(voice, "speak", slow_speak)
    watching = asyncio.Event()

    class Watcher:
        is_active = True
        cancelled = False

        async def wait_for_speech_start(self, timeout=None):
            watching.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                Watcher.cancelled = True
                raise
            return True

    line = asyncio.ensure_future(voice.speak_interruptible("Here is your morning.", recorder=Watcher()))
    await watching.wait()
    line.cancel()
    with pytest.raises(asyncio.CancelledError):
        await line
    assert Watcher.cancelled


@pytest.mark.asyncio
async def test_the_microphone_button_makes_the_next_sentence_for_heliox_without_the_wake_phrase():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    assert (await server._handle_voice_listen_now({}, None))["status"] == "error"  # listening is off
    listener = MagicMock()
    listener.is_running = True
    server._voice_listener = listener
    assert await server._handle_voice_listen_now({}, None) == {"status": "listening"}
    listener.note_interruption.assert_called_once_with(seconds=10.0, by_button=True)
    server._broadcast_notification.assert_awaited_with(
        "voice_status", {"status": "listening", "message": "I'm listening..."}
    )


@pytest.mark.asyncio
async def test_after_the_button_any_sentence_is_taken_as_a_command():
    listener, run = make_listener(["the launch call is Monday the twelfth of November"])
    listener.note_interruption(by_button=True)
    dispatched, _ = await run()
    assert dispatched == ["the launch call is Monday the twelfth of November"]


@pytest.mark.asyncio
async def test_a_sentence_begun_inside_the_button_window_counts_even_if_it_ends_after_it():
    import time

    listener, run = make_listener(["remind me that the launch call is Monday"])
    listener._interrupted_until = listener._button_until = time.monotonic() - 1.0  # the window has just closed
    listener._recorder.speech_started_at = time.monotonic() - 3.0  # but the sentence began inside it
    dispatched, _ = await run()
    assert dispatched == ["remind me that the launch call is Monday"]


@pytest.mark.asyncio
async def test_a_take_starts_a_clean_voice_conversation():
    server = PilotServer(PilotConfig())
    server._memory = MagicMock()
    server._memory.start_conversation = AsyncMock(return_value=6)
    listener = MagicMock()
    listener.is_running = True
    server._voice_listener = listener
    assert await server._handle_voice_new_conversation({}, None) == {"status": "ok", "earlier_turns": 6}
    server._memory.start_conversation.assert_awaited_once_with("voice")
    listener.clear_follow_up_window.assert_called_once_with()
    listener.end_interruption.assert_called_once_with()


@pytest.mark.asyncio
async def test_a_cut_off_waits_for_no_recogniser():
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener.config.voice.transcription_engine = "whisper"
    listener._recorder._soundcard_recorder = object()  # the stream is "open"

    async def record(**_kwargs):
        listener._recorder.last_voiced_seconds = 0.9
        return "utterance.wav"

    listener._recorder.record_utterance = record
    listener._transcribe_path = AsyncMock(return_value="Where we store home")
    listener.note_interruption()
    assert await listener._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == "(cut off)"
    listener._transcribe_path.assert_not_awaited()


@pytest.mark.asyncio
async def test_listener_stats_say_whether_the_voice_is_loaded():
    server = PilotServer(PilotConfig())
    server.config.voice.tts_engine = "os_native"
    stats = await server._handle_voice_listener_stats({}, None)
    assert stats["voice_loaded"] is True and stats["voice_loading"] is False and stats["running"] is False

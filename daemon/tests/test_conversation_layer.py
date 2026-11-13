"""The conversation layer with a fake model stream and a fake speech sink: no model, network or audio."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.conversation import handover
from pilot.conversation.adapters import Focus, screen_for_model, share_setting
from pilot.conversation.context import ContextSources, ConversationMemory, Screen, build_messages
from pilot.conversation.fast_model import FastModelUnavailable
from pilot.conversation.layer import FALLBACK, HOLD_PHRASES, UNCLEAR_OFFLINE, ConversationLayer
from pilot.conversation.style import SentenceSplitter, claims_an_action, clean_for_speech, parse_act


class FakeModel:
    """Streams the given fragments, ``gap`` seconds apart; notes whether it was closed early."""

    def __init__(self, fragments, *, gap=0.0, first_delay=0.0, fail=None, local=False):
        self.fragments = list(fragments)
        self.gap = gap
        self.first_delay = first_delay
        self.fail = fail
        self.local = local
        self.name = "fake"
        self.provider = "fake"
        self.streams = True
        self.calls = []
        self.closed_early = False
        self.yielded = 0
        self.seen_messages = None

    async def stream(self, system, messages):
        self.seen_messages = messages
        self.calls.append(time.monotonic())
        if self.fail:
            raise FastModelUnavailable(self.fail)
        if self.first_delay:
            await asyncio.sleep(self.first_delay)
        try:
            for fragment in self.fragments:
                self.yielded += 1
                yield fragment
                if self.gap:
                    await asyncio.sleep(self.gap)
        except (GeneratorExit, asyncio.CancelledError):
            self.closed_early = True
            raise


class FakeSink:
    """Says each sentence in ``duration`` seconds. ``outcomes`` maps a sentence index to a forced outcome."""

    def __init__(self, duration=0.0, outcomes=None, block_on=None):
        self.said: list[tuple[float, str, str]] = []
        self.duration = duration
        self.outcomes = outcomes or {}
        self.block_on = block_on
        self.started = asyncio.Event()

    async def __call__(self, text, channel, key):
        index = len(self.said)
        self.said.append((time.monotonic(), text, channel))
        self.started.set()
        if self.block_on is not None and index == self.block_on:
            await asyncio.sleep(3600)
        if self.duration:
            await asyncio.sleep(self.duration)
        return self.outcomes.get(index, "spoken")

    @property
    def texts(self):
        return [text for _at, text, _channel in self.said]


def make_layer(model, sink, **kwargs):
    return ConversationLayer(sink=sink, model=lambda: model, hold_after=kwargs.pop("hold_after", 5.0), **kwargs)


# ── streaming into speech ──


@pytest.mark.asyncio
async def test_the_first_sentence_is_spoken_before_the_reply_has_finished():
    model = FakeModel(["I'm ", "fine, ", "thanks. ", "The build ", "passed ", "an hour ", "ago."], gap=0.05)
    fragments_in: list[int] = []

    class _Watching(FakeSink):
        async def __call__(self, text, channel, key):
            fragments_in.append(model.yielded)  # how much of the reply existed when this sentence was said
            return await super().__call__(text, channel, key)

    sink = _Watching()
    layer = make_layer(model, sink)
    result = await layer.take("How are you?")
    assert result.done
    assert sink.texts == ["I'm fine, thanks.", "The build passed an hour ago."]
    # Counted, not timed: the first sentence was said a few fragments in, not after all seven.
    assert fragments_in[0] < len(model.fragments)
    assert "first_sentence_ms" in result.timings
    assert layer.memory.last_heliox() == "I'm fine, thanks. The build passed an hour ago."


@pytest.mark.asyncio
async def test_talking_over_heliox_stops_generation_and_only_what_was_said_is_remembered():
    model = FakeModel(["First thing. ", "Second thing. ", "Third ", "thing. ", "Fourth. "], gap=0.02)
    sink = FakeSink(outcomes={1: "interrupted"})
    layer = make_layer(model, sink)
    result = await layer.take("Explain the error")
    assert result.done
    assert sink.texts == ["First thing.", "Second thing."]
    assert model.closed_early  # the stream was closed, so it stops costing tokens
    turns = layer.memory.recent()
    assert turns[-1].role == "assistant" and turns[-1].cut_off
    assert turns[-1].text == "First thing. Second thing."
    assert "Third" not in turns[-1].text


@pytest.mark.asyncio
async def test_a_new_utterance_cuts_the_reply_mid_sentence():
    model = FakeModel(["One. ", "Two. ", "Three. "], gap=0.01)
    sink = FakeSink(block_on=1)
    layer = make_layer(model, sink)
    task = asyncio.ensure_future(layer.take("Tell me about it"))
    for _ in range(100):
        if len(sink.said) == 2:
            break
        await asyncio.sleep(0.01)
    assert layer.interrupt()
    result = await asyncio.wait_for(task, 1.0)
    assert result.done
    assert sink.texts == ["One.", "Two."]
    last = layer.memory.recent()[-1]
    assert last.cut_off and last.text == "One. Two."


@pytest.mark.asyncio
async def test_no_sentence_starts_while_the_person_is_talking():
    model = FakeModel(["One. ", "Two. "], gap=0.05)
    sink = FakeSink()
    speaking = {"now": False}
    layer = make_layer(model, sink, user_speaking=lambda: speaking["now"])

    async def person_starts():
        await sink.started.wait()
        speaking["now"] = True

    asyncio.ensure_future(person_starts())
    result = await layer.take("How was the build?")
    assert result.done and sink.texts == ["One."]


@pytest.mark.asyncio
async def test_markdown_paths_and_stock_openers_are_not_read_aloud():
    model = FakeModel(["Sure! The **file** is C:\\Users\\sam\\launch\\post-final.md, see https://x.io/a. "])
    sink = FakeSink()
    await make_layer(model, sink).take("Where is the post?")
    assert sink.texts == ["The file is post-final.md, see the link."]


@pytest.mark.asyncio
async def test_a_sentence_claiming_an_action_is_never_spoken():
    model = FakeModel(["I've opened Notepad for you. ", "Anything else?"])
    sink = FakeSink()
    await make_layer(model, sink).take("Notepad?")
    assert sink.texts == ["Anything else?"]


@pytest.mark.asyncio
async def test_a_slow_first_sentence_gets_a_short_holding_phrase_once():
    model = FakeModel(["Here it is. "], first_delay=0.3)
    sink = FakeSink()
    layer = make_layer(model, sink, hold_after=0.05)
    result = await layer.take("What do you think?")
    assert sink.texts[0] in HOLD_PHRASES and sink.texts[1] == "Here it is."
    assert len(sink.texts) == 2 and "hold_ms" in result.timings
    # Never twice in a row (13 November): the next slow turn waits in silence.
    await layer.take("And now?")
    assert not set(HOLD_PHRASES) & set(sink.texts[2:])
    # A minute later, after a turn without one, it may come again, and never the same phrase twice running.
    clock = layer._clock
    layer._clock = lambda: clock() + 61.0
    before = len(sink.texts)
    await layer.take("And later?")
    held = [text for text in sink.texts[before:] if text in HOLD_PHRASES]
    assert held and held[0] != sink.texts[0]


# ── handing over ──


@pytest.mark.asyncio
async def test_a_plain_request_is_handed_over_without_a_model_call():
    model = FakeModel(["should not be called"])
    sink = FakeSink()
    layer = make_layer(model, sink, task_status=lambda: {"phase": "planning"})
    result = await layer.take("Open Notepad")
    assert not result.done and result.planner_input == "Open Notepad"
    assert model.calls == []
    assert result.intent.verb == "open" and result.intent.object == "notepad" and result.intent.source == "router"
    company = result.keep_company()
    await asyncio.sleep(0.05)
    company.cancel()
    assert sink.texts == ["Opening Notepad."]
    result.finish("success", "Notepad is open.")
    assert "Notepad is open." in layer.memory.last_heliox() and layer.last_result.startswith("success")


@pytest.mark.parametrize(
    ("words", "verb", "target", "expected"),
    [
        ("open notepad", "open", "notepad", "I'll open Notepad."),
        ("open vs code", "open", "vs code", "I'll open VS Code."),
        ("find my launch folder", "find", "my launch folder", "I'll look for your launch folder."),
        ("set a timer for ten minutes", "set", "a timer for ten minutes", "I'll set a timer for ten minutes."),
        ("remind me to call Priya at four", "remind", "me to call priya at four", "I'll set that reminder."),
        ("search flights to Lisbon", "search", "flights to lisbon", "I'll search for flights to lisbon."),
        # Closing a window, deleting, sending: the plan needs an approval, which is the first thing said.
        ("close spotify please", "close", "spotify please", ""),
        ("delete the screenshots", "delete", "the screenshots", ""),
        ("message my manager that I'm late", "message", "my manager that i'm late", ""),
        (
            "move every screenshot from last week into a new folder called archive",
            "move",
            "every screenshot from last week into a new folder called archive",
            "I'll get that started.",
        ),
    ],
)
def test_the_acknowledgement_is_built_from_what_was_parsed(words, verb, target, expected):
    intent = handover.HandoverIntent(verb=verb, object=target, words=words, request=words, source="router")
    assert handover.acknowledgement(intent) == expected
    assert handover.choose_target(intent) is handover.HandoverTarget.FULL_PLANNER


def test_the_same_acknowledgement_is_not_said_twice_running():
    intent = handover.HandoverIntent(verb="open", object="notepad", words="", request="", source="router")
    assert handover.acknowledgement(intent, last="I'll open Notepad.") == "Okay, I'll open Notepad."


@pytest.mark.asyncio
async def test_the_model_can_hand_over_with_what_it_understood():
    act = 'ACT {"verb": "open", "object": "the file we talked about", "request": "Open launch-notes.md in Notepad", "say": "I\'ll open the launch notes."}'
    model = FakeModel([act[:10], act[10:]])
    sink = FakeSink()
    layer = make_layer(model, sink)
    layer.memory.add_user("what was that file called")
    layer.memory.add_heliox("It's launch-notes.md.")
    result = await layer.take("open it")
    assert not result.done
    assert result.planner_input == "open it"  # the person's words; the model's reading is context
    assert "Heliox understood the request as: Open launch-notes.md in Notepad" in result.context.for_planner()
    assert "Heliox: It's launch-notes.md." in result.context.for_planner()
    assert result.intent.source == "model" and result.intent.words == "open it"
    assert sink.texts == []  # nothing of the ACT line is read aloud
    company = result.keep_company()
    await asyncio.sleep(0.05)
    company.cancel()
    assert sink.texts == ["I'll open the launch notes."]


@pytest.mark.asyncio
async def test_progress_then_the_approval_pointer_while_the_planner_works():
    sink = FakeSink()
    state = {"phase": "planning", "message": "Planning the safest useful action"}
    company = asyncio.ensure_future(
        handover.keep_company(
            acknowledgement_text="Opening Notepad.",
            speak=sink,
            status=lambda: state,
            progress_after=0.1,
            spacing=0.1,
            poll=0.02,
        )
    )
    await asyncio.sleep(0.2)
    state.update(phase="planning", message="Reviewing the plan for risks")
    await asyncio.sleep(0.2)
    state.update(phase="awaiting_approval", message="Waiting for your approval")
    await asyncio.sleep(0.15)
    company.cancel()
    texts = sink.texts
    assert texts[0] == "Opening Notepad."
    assert texts[1] == "I'm working out the steps."
    assert "I'm checking the plan for risks." in texts
    assert handover.APPROVAL_SENTENCE in texts
    assert [channel for _a, text, channel in sink.said if text == handover.APPROVAL_SENTENCE] == ["approval_risk"]
    assert "approve" not in " ".join(texts).casefold().replace("press approve", "")  # never "say approve"


@pytest.mark.asyncio
async def test_a_quick_plan_hears_only_the_acknowledgement():
    sink = FakeSink()
    company = asyncio.ensure_future(
        handover.keep_company(acknowledgement_text="Opening Notepad.", speak=sink, status=dict, poll=0.01)
    )
    await asyncio.sleep(0.1)
    company.cancel()
    assert sink.texts == ["Opening Notepad."]


# ── offline and failure ──


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["offline", "no_key", "budget", "local_down", "local_missing"])
async def test_no_model_says_why_once_then_briefly(reason):
    sink = FakeSink()
    layer = make_layer(FakeModel([], fail=reason), sink)
    first = await layer.take("What do you think about the error?")
    second = await layer.take("And why is that?")
    assert first.done and second.done
    assert sink.texts[0] == FALLBACK[reason]
    assert sink.texts[1] != sink.texts[0] and len(sink.texts[1]) < len(sink.texts[0])


@pytest.mark.asyncio
async def test_with_no_model_set_up_the_person_is_told_where_to_set_one():
    sink = FakeSink()
    layer = ConversationLayer(sink=sink, model=lambda: None)
    await layer.take("How are you?")
    assert "Settings" in sink.texts[0] and "local model" in sink.texts[0]
    await layer.take("Hello?")
    assert "Settings" not in sink.texts[1]
    # A request to act still goes to the planner with no model at all.
    handed = await layer.take("Open Notepad")
    assert not handed.done and handed.planner_input == "Open Notepad"


@pytest.mark.asyncio
async def test_an_unclear_turn_without_a_model_is_asked_about_not_acted_on():
    sink = FakeSink()
    layer = make_layer(FakeModel([], fail="offline"), sink)
    await layer.take("Hmm what")
    result = await layer.take("Where we store home")
    assert result.done
    assert sink.texts[-1] in (UNCLEAR_OFFLINE, FALLBACK["offline"]) or "offline" in sink.texts[-1].casefold()


@pytest.mark.asyncio
async def test_a_control_word_no_handler_took_is_acknowledged_not_planned():
    sink = FakeSink()
    result = await make_layer(FakeModel(["x"]), sink).take("cancel")
    assert result.done and sink.texts == ["Okay."]


@pytest.mark.asyncio
async def test_the_layer_switched_off_leaves_the_old_path():
    layer = ConversationLayer(sink=FakeSink(), model=lambda: None, enabled=lambda: False)
    assert await layer.take("How are you?") is None


# ── context and its privacy gates ──


def label(process):
    return {"code": "VS Code", "notepad": "Notepad", "1password": "1Password"}.get(process, process)


FIELD = Focus(
    app="code", title="secret-plan.md - acme", text="def main():\n    return 42", editable=True, source="focused_field"
)


@pytest.mark.parametrize(
    ("setting", "local", "excerpt"),
    [("auto", True, True), ("auto", False, False), ("on", False, True), ("on", True, True), ("off", True, False)],
)
def test_what_leaves_the_machine_follows_the_setting_and_where_the_model_runs(setting, local, excerpt):
    screen = screen_for_model(FIELD, setting=setting, model_local=local, excluded_apps=[], label=label)
    assert screen.app == "VS Code"
    assert bool(screen.excerpt) is excerpt


@pytest.mark.parametrize(
    "focus",
    [
        Focus(app="1password", title="Vault", text="hunter2", editable=True, source="focused_field"),
        Focus(app="msedge", title="Bank - Edge", text="x", editable=True, source="focused_field"),  # excluded
        Focus(app="heliox", title="Heliox-OS", text="x", editable=True, source="focused_field"),
        Focus(app="Unknown", title="Unknown"),  # this platform could not tell
        None,  # nothing available here
    ],
)
def test_apps_live_never_reads_send_nothing_at_all(focus):
    assert screen_for_model(focus, setting="on", model_local=True, excluded_apps=["bank"], label=label) is None


def test_text_that_holds_a_key_is_left_out_whole():
    focus = Focus(
        app="notepad", text="key = sk-ant-api03-abcdefghijklmnopqrstuvwxyz", editable=True, source="focused_field"
    )
    screen = screen_for_model(focus, setting="on", model_local=True, excluded_apps=[], label=label)
    assert screen.app == "Notepad" and screen.excerpt == ""


def test_only_the_focused_field_counts_not_text_read_off_the_screen():
    ocr = Focus(app="notepad", text="some screen text", editable=True, source="screen_ocr")
    assert screen_for_model(ocr, setting="on", model_local=True, excluded_apps=[], label=label).excerpt == ""


def test_share_setting_values():
    assert [share_setting(v) for v in ("auto", "ON", "off", True, False, "maybe", None)] == [
        "auto",
        "on",
        "off",
        "on",
        "off",
        "auto",
        "auto",
    ]


def test_context_carries_turns_task_profile_time_and_never_the_window_title():
    memory = ConversationMemory()
    memory.add_user("what's failing")
    memory.add_heliox("The tests in the router module.", cut_off=True)
    sources = ContextSources(
        profile_lines=lambda: ["Call them Sam.", "Preference: short answers"],
        task_state=lambda: {"approval_waiting": True, "last_result": "success: Copied 3 files"},
        now=lambda: datetime(2026, 11, 12, 15, 4),
    )
    screen = screen_for_model(FIELD, setting="auto", model_local=False, excluded_apps=[], label=label)
    messages = build_messages("why?", sources, memory, screen=screen)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "rest was not said" in messages[1]["content"]
    final = messages[-1]["content"]
    for expected in (
        "Thursday 12 November 2026, 15:04",
        "Call them Sam.",
        "Voice cannot approve",
        "VS Code",
        "Copied 3",
    ):
        assert expected in final
    assert "secret-plan" not in final and "return 42" not in final


def test_a_failing_context_source_is_left_out():
    def broken():
        raise RuntimeError("profile unreadable")

    messages = build_messages("hi", ContextSources(profile_lines=broken), ConversationMemory())
    assert "About the person" not in messages[-1]["content"]


def test_the_context_stays_within_its_budget():
    sources = ContextSources(profile_lines=lambda: ["About them: " + "x" * 300] * 8)
    block = build_messages("hi", sources, ConversationMemory(), budget=600)[-1]["content"]
    assert len(block) < 800


def test_new_conversation_clears_the_turns():
    layer = ConversationLayer(sink=FakeSink(), model=lambda: None)
    layer.memory.add_user("hello")
    layer.clear()
    assert layer.memory.recent() == []


def test_screen_text_reaches_a_local_model_by_default_and_not_a_cloud_one():
    for local, shared in ((True, True), (False, False)):
        model = FakeModel(["Okay. "], local=local)
        seen = {}

        async def screen(model_local, seen=seen):
            seen["local"] = model_local
            return screen_for_model(FIELD, setting="auto", model_local=model_local, excluded_apps=[], label=label)

        layer = make_layer(model, FakeSink(), sources=ContextSources(screen_async=screen))
        asyncio.run(layer.take("What do you think of this?"))
        assert seen["local"] is local
        assert ("return 42" in model.seen_messages[-1]["content"]) is shared


# ── text handling ──


def test_sentences_are_cut_as_they_complete():
    splitter = SentenceSplitter()
    out = []
    for piece in ["Dr. Smith said 3.5 is fine. ", "Then he ", "left! Did he? ", "Yes"]:
        out += splitter.feed(piece)
    out += splitter.flush()
    assert out == ["Dr. Smith said 3.5 is fine.", "Then he left!", "Did he?", "Yes"]


def test_a_long_first_sentence_starts_at_a_comma():
    splitter = SentenceSplitter()
    first = splitter.feed("The quickest way to fix this is to restart the language server, then reopen the file")
    assert first and first[0].endswith(",")


def test_cleanup_and_action_claims():
    assert clean_for_speech("- **Run** `pytest` & wait", first=True) == "Run pytest and wait"
    assert clean_for_speech("Certainly! It's in /home/sam/notes/todo.txt.", first=True) == "It's in todo.txt."
    assert claims_an_action("I've deleted the folder.")
    assert not claims_an_action("I can delete the folder if you want.")


def test_act_lines_are_parsed_or_rejected():
    act = parse_act('ACT {"verb":"Open","object":"notepad","request":"Open Notepad","say":"Opening Notepad."}')
    assert act.verb == "open" and act.request == "Open Notepad"
    assert parse_act("ACT {not json") is None
    assert parse_act('ACT {"verb": "open"}') is None  # nothing to hand over


# ── the server hook ──


def _server_with(layer):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._voice_listener = None
    server._broadcast_notification = AsyncMock()
    server._handle_execute = AsyncMock(return_value={"status": "success", "message": "Notepad is open."})
    server._speak_voice_response = AsyncMock(return_value=False)
    server._spawn_interaction_speech = MagicMock()
    server._model_router = object()
    server._conversation = layer
    return server


@pytest.mark.asyncio
async def test_the_server_answers_talk_in_the_layer_and_never_plans_it():
    sink = FakeSink()
    layer = make_layer(FakeModel(["All good here. "]), sink)
    server = _server_with(layer)
    await server._voice_command_dispatch("How are you?")
    server._handle_execute.assert_not_awaited()
    assert sink.texts == ["All good here."]
    result = [c for c in server._broadcast_notification.await_args_list if c.args[0] == "voice_result"]
    assert result[-1].args[1]["message"] == "All good here."


@pytest.mark.asyncio
async def test_the_server_hands_a_request_over_with_the_layers_acknowledgement():
    sink = FakeSink()
    layer = make_layer(FakeModel(["unused"]), sink)
    server = _server_with(layer)
    await server._voice_command_dispatch("Open Notepad")
    server._handle_execute.assert_awaited_once()
    assert server._handle_execute.await_args.args[0]["input"] == "Open Notepad"
    server._spawn_interaction_speech.assert_not_called()  # no stock "I'm on it."
    assert "Notepad is open." in layer.memory.last_heliox()


@pytest.mark.asyncio
async def test_a_new_utterance_stops_the_reply_still_being_spoken():
    sink = FakeSink(block_on=0)
    layer = make_layer(FakeModel(["A long answer. ", "More. "]), sink)
    server = _server_with(layer)
    first = asyncio.ensure_future(server._voice_command_dispatch("Explain recursion"))
    await sink.started.wait()
    await server._voice_command_dispatch("stop")
    await asyncio.wait_for(first, 1.0)
    assert layer.memory.recent()[-1].cut_off


def test_without_the_model_stack_the_server_keeps_its_old_path():
    from pilot.conversation import layer_for
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    assert asyncio.run(layer_for(server).take("How are you?")) is None


@pytest.mark.asyncio
async def test_a_control_phrase_with_an_instruction_goes_to_the_running_task_or_the_planner_never_the_model():
    sentence = "That's enough for the backup folder, now zip it"
    model = FakeModel(["should not be called"])
    # A task is running: the server's correction path takes it before the layer is asked.
    server = _server_with(make_layer(model, FakeSink()))
    server._interactive_request_active = True
    server._interaction_runtime.transition = AsyncMock()
    server._handle_interject = AsyncMock(return_value={"status": "revising", "message": "Revising the plan"})
    await server._voice_command_dispatch(sentence)
    server._handle_interject.assert_awaited_once()
    server._handle_execute.assert_not_awaited()
    # Nothing running: it is a new request, handed to the planner without a model call.
    server = _server_with(make_layer(model, FakeSink()))
    await server._voice_command_dispatch(sentence)
    server._handle_execute.assert_awaited_once()
    assert server._handle_execute.await_args.args[0]["input"] == sentence
    assert model.calls == []


@pytest.mark.asyncio
async def test_the_next_sentence_is_synthesised_while_this_one_plays():
    prepared = []
    model = FakeModel(["One thing. ", "Two **things**. ", "Three. "])
    sink = FakeSink(duration=0.02)
    layer = make_layer(model, sink, prepare=prepared.append)
    await layer.take("Tell me three things")
    assert sink.texts == ["One thing.", "Two things.", "Three."]
    assert prepared and set(prepared) <= {"Two things.", "Three."}  # exactly what will be said, one ahead


@pytest.mark.asyncio
async def test_the_time_since_the_last_word_is_recorded():
    layer = make_layer(FakeModel(["Hi. "]), FakeSink(), turn_ended=lambda: time.monotonic() - 0.4)
    result = await layer.take("Hello there")
    assert 350 <= result.timings["last_word_to_take_ms"] < 2000


def test_extra_context_may_read_the_words_being_answered():
    sources = ContextSources(extra=[lambda: "no argument", lambda text: f"about: {text}"])
    final = build_messages("the launch", sources, ConversationMemory())[-1]["content"]
    assert "no argument" in final and "about: the launch" in final


def test_something_heliox_said_first_is_kept_in_the_context():
    memory = ConversationMemory()
    memory.add_heliox("The twelfth of November is a Thursday.")
    memory.add_user("oh thanks")
    messages = build_messages("which day was it again?", ContextSources(), memory)
    assert messages[0]["role"] == "user"
    assert "Thursday" in messages[-1]["content"] or "Thursday" in messages[0]["content"]


def test_the_name_style_and_preferences_reach_the_model_first():
    from pilot.conversation.adapters import prioritise_profile

    lines = [f"About them: fact {i}" for i in range(10)] + ["Preference: short answers", "Call them Sam."]
    first = prioritise_profile(lines)[:3]
    assert first[0] == "Call them Sam." and first[1] == "Preference: short answers"


@pytest.mark.asyncio
async def test_a_provider_that_never_answers_is_given_up_on_and_told_it_was_late():
    model = FakeModel(["Too late. "], first_delay=1.0)
    model.held = []
    model.note_slow = lambda: model.held.append("slow")  # the model decides the rest (test_conversation_slow.py)
    sink = FakeSink()
    layer = make_layer(model, sink, hold_after=0.05, first_text_deadline=0.2)
    result = await layer.take("What do you think?")
    # Given up at its own deadline (0.2 s), before the late reply's first fragment (1 s) ever arrived.
    assert result.done and model.yielded == 0
    assert sink.texts[0] in HOLD_PHRASES and sink.texts[-1] == FALLBACK["slow"]
    assert model.held == ["slow"]


@pytest.mark.asyncio
async def test_a_local_model_gets_time_to_load_and_says_so_once():
    model = FakeModel(["Here. "], first_delay=0.3, local=True)
    sink = FakeSink()
    layer = make_layer(model, sink, hold_after=0.05, first_text_deadline=0.1)
    await layer.take("What do you think?")
    assert sink.texts == ["Loading the local model, one moment.", "Here."]
    await layer.take("And now?")
    assert sink.texts[2] in HOLD_PHRASES  # said once per session, then the ordinary phrase


class BrokenAfter(FakeModel):
    async def stream(self, system, messages):
        yield "First part. "
        await asyncio.sleep(0.05)
        raise FastModelUnavailable("offline")


@pytest.mark.asyncio
async def test_a_reply_cut_by_the_network_says_so():
    sink = FakeSink()
    layer = make_layer(BrokenAfter([]), sink)
    await layer.take("Explain it")
    assert sink.texts == ["First part.", "I lost the connection there."]


@pytest.mark.asyncio
async def test_out_of_credit_names_the_provider_account_not_heliox_budget():
    model = FakeModel([], fail="credit")
    model.provider = "openrouter"
    sink = FakeSink()
    await make_layer(model, sink).take("How are you?")
    assert sink.texts == [
        "Your OpenRouter account is out of credit. Add credit there, or choose another model in Settings, under Model."
    ]

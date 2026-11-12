"""One conversation, one floor: Live and the conversation layer never talk over each other, and what Live says is
part of the conversation. Injected clocks and fake speakers only: no audio, no model."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.conversation.context import ConversationMemory
from pilot.live.conversation_link import LiveConversationLink, redacted, remark_for_model, shareable
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation, because
from pilot.live.floor import GRACE_SECONDS, Floor
from pilot.system.companion_speech import CompanionSpeechCoordinator, SpeechChannel

MONDAY = date(2026, 11, 9)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, text, floor, clock):
    events, spoken = [], []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="notes.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        spoken.append(message)

    async def mute(kind):
        settings.muted_kinds.append(kind)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        people=lambda: {"manager": "Priya"},
        mute_kind=mute,
        relevance_path=tmp_path / "rel.json",
        clock=clock,
        today=lambda: MONDAY,
        floor=floor,
    )
    return live, events, spoken, settings


async def run_for(live, clock, seconds):
    for _ in range(int(round(seconds / TICK_SECONDS))):
        clock.t += TICK_SECONDS
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)


def cards(events):
    return [p for e, p in events if e == "live_interjection"]


# ── the rule ──


def test_the_floor_rule():
    clock = Clock()
    floor = Floor(clock=clock)
    assert floor.live_may_speak() and not floor.held()
    token = floor.open_turn()
    assert not floor.live_may_speak()  # rule 2: a turn holds the floor
    assert floor.live_may_speak(about_this_turn=True)  # rule 3: about its words, before it has spoken
    floor.heliox_spoke()
    assert not floor.live_may_speak(about_this_turn=True)  # once the reply started, never
    from pilot.live.sequence import next_seq

    assert floor.about_this_turn(next_seq()) and not floor.about_this_turn(0)
    floor.close_turn(token)
    assert not floor.live_may_speak() and floor.held()  # the grace period
    clock.t += GRACE_SECONDS
    assert floor.live_may_speak() and not floor.held()
    # Two turns at once (a hand-off inside a turn): the floor is free only when both are done.
    a, b = floor.open_turn(), floor.open_turn()
    floor.close_turn(a)
    assert floor.turn_open
    floor.close_turn(b)
    assert not floor.turn_open


@pytest.mark.asyncio
async def test_a_typed_slip_waits_for_the_turn_and_the_grace_then_is_said(tmp_path):
    clock, text = Clock(), ["The total is 12 x 15 = 170 so we order more."]
    floor = Floor(clock=clock)
    live, events, spoken, _ = build(tmp_path, text, floor, clock)
    token = floor.open_turn()
    await run_for(live, clock, 8)
    assert spoken == [] and cards(events) == []  # waits: no speech and no card over the conversation
    floor.close_turn(token)
    await run_for(live, clock, GRACE_SECONDS - 1)
    assert spoken == []
    await run_for(live, clock, 2)
    assert spoken == ["12 × 15 is 180. You wrote 170."] and len(cards(events)) == 1


@pytest.mark.asyncio
async def test_an_urgent_risk_shows_its_card_at_once_but_does_not_speak_over_the_reply(tmp_path):
    clock, text = Clock(), ["$ rm -rf /"]
    floor = Floor(clock=clock)
    live, events, spoken, _ = build(tmp_path, text, floor, clock)
    floor.open_turn()
    floor.heliox_spoke()
    await run_for(live, clock, 2)
    [card] = cards(events)
    assert card["kind"] == "risky_command" and card["spoken"] is False and spoken == []


@pytest.mark.asyncio
async def test_a_catch_about_the_words_that_opened_the_turn_goes_first_but_never_into_the_reply(tmp_path):
    clock, text = Clock(), ["nothing here"]
    floor = Floor(clock=clock)
    live, events, spoken, _ = build(tmp_path, text, floor, clock)
    floor.open_turn()
    live.offer_speech("put lunch on Tuesday 14 November")
    await run_for(live, clock, 0.4)
    assert spoken == ["The fourteenth of November is a Saturday. You said Tuesday."]  # before the reply
    live2, events2, spoken2, _ = build(tmp_path / "b", text, floor, clock)
    floor.heliox_spoke()  # the reply has started
    live2.offer_speech("put lunch on Tuesday 14 November")
    await run_for(live2, clock, 0.4)
    assert spoken2 == [] and cards(events2)[0]["spoken"] is False  # a card, not a voice over the reply


# ── the reply is not cut ──


class SlowSpeaker:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = asyncio.Event()

    async def __call__(self, text, recorder):
        self.started.append(text)
        await self.release.wait()
        return False


def fake_server(coordinator, floor):
    live = MagicMock()
    live.last_spoken_entries.return_value = [{"source": "focused_field", "at": 0.0, "kind": "arithmetic"}]

    async def speak_companion_text(text, channel, dedupe_key=""):
        from pilot.personal.wiring import note_heliox_speech

        note_heliox_speech(server, channel, dedupe_key)
        return await coordinator.speak(text, channel=channel, dedupe_key=dedupe_key)

    server = SimpleNamespace(
        _live=live,
        _live_floor=floor,
        _voice_listener=None,
        _speak_companion_text=speak_companion_text,
        _conversation=None,
        config=SimpleNamespace(model=SimpleNamespace(provider="cloud"), conversation=SimpleNamespace()),
    )
    return server


@pytest.mark.asyncio
async def test_before_the_floor_a_live_remark_cut_the_reply_now_it_is_held():
    from pilot.personal.wiring import live_speak

    for with_floor in (False, True):
        speaker = SlowSpeaker()
        coordinator = CompanionSpeechCoordinator(speaker=speaker)
        floor = Floor()
        server = fake_server(coordinator, floor if with_floor else None)
        token = floor.open_turn()
        reply = asyncio.ensure_future(
            server._speak_companion_text("Here is the answer.", channel=SpeechChannel.FINAL_ANSWER, dedupe_key="conv:1")
        )
        while not speaker.started:
            await asyncio.sleep(0)
        remark = asyncio.ensure_future(live_speak(server, "12 × 15 is 180.", True))
        for _ in range(5):
            await asyncio.sleep(0)
        speaker.release.set()
        outcome = await reply
        said = await remark
        floor.close_turn(token)
        if with_floor:
            assert outcome.status == "spoken" and said == "held" and speaker.started == ["Here is the answer."]
        else:
            # The old behaviour: the correction channel outranks an answer and cut it off mid-sentence.
            assert outcome.status == "superseded" and said == "spoken"
        await coordinator.close()


# ── what Live says is part of the conversation ──


def entry(**kw):
    base = {
        "id": "c1",
        "kind": "weekday_date",
        "title": "14 November is a Saturday",
        "message": "You wrote Tuesday.",
        "detail": "If you meant the Tuesday, that's 17 November.",
        "evidence": "Tuesday 14 November",
        "source": "focused_field",
        "app": "Notepad",
        "spoken": True,
        "at": 100.0,
    }
    return {**base, **kw}


def test_why_is_said_from_the_evidence():
    assert because(entry()) == (
        'In Notepad, you wrote "Tuesday 14 November", and 14 November is a Saturday. '
        "If you meant the Tuesday, that's 17 November."
    )
    said = entry(source="speech", evidence="tuesday the fourteenth", title="The fourteenth of November is a Saturday")
    assert because(said).startswith('You said "tuesday the fourteenth", and the fourteenth of November is a Saturday.')
    promise = entry(
        kind="commitment",
        title="Want a reminder on Friday?",
        message="You said you'd send the deck by Friday.",
        evidence="I'll send the deck by Friday",
    )
    assert (
        because(promise)
        == "I offered because you said you'd send the deck by Friday, and I didn't find a reminder for it."
    )
    notice = entry(
        kind="risky_command",
        source="notice",
        evidence="",
        detail="",
        title="That command could wipe a lot",
        message="There's no undo.",
    )
    assert because(notice) == (
        "I spoke up because what you just typed matched my check for risky commands before you run them. "
        "That command could wipe a lot. There's no undo."
    )


def test_screen_words_reach_a_cloud_model_only_when_the_person_allows_it():
    typed, said = entry(), entry(source="speech")
    assert shareable(said, setting="off", model_local=False)  # addressed speech: that model already has it
    assert not shareable(typed, setting="auto", model_local=False)
    assert shareable(typed, setting="auto", model_local=True) and shareable(typed, setting="on", model_local=False)
    assert remark_for_model("14 November is a Saturday.", [typed], setting="auto", model_local=False) == redacted(typed)
    assert "Tuesday" not in redacted(typed) and "Notepad" in redacted(typed)


def test_a_spoken_remark_becomes_a_heliox_turn_and_the_model_hears_of_recent_cards(monkeypatch):
    from pilot.live import conversation_link

    monkeypatch.setattr(conversation_link, "_layer_keeps_opening_remarks", lambda: False)
    clock = Clock()
    memory = ConversationMemory(clock=clock)
    live = MagicMock()
    live.last_spoken_entries.return_value = [entry()]
    live.last_raised.return_value = entry(at=clock.t)
    setting = ["auto"]
    link = LiveConversationLink(
        live=lambda: live,
        layer=lambda: SimpleNamespace(memory=memory),
        setting=lambda: setting[0],
        model_local=lambda: False,
        clock=clock,
    )
    assert link.record_spoken("14 November is a Saturday. You wrote Tuesday.", "interrupted")
    [turn] = memory.recent()
    assert turn.role == "assistant" and turn.cut_off and "Tuesday" not in turn.text  # a cloud model, sharing auto
    # No turn from the person before it, so the layer would drop it: the context line carries it instead.
    line = link.context_line()
    assert line.startswith("Heliox Live, on its own, said aloud 0 seconds ago:") and "Tuesday" not in line
    setting[0] = "on"
    assert 'you wrote "Tuesday 14 November"' in link.context_line()
    # A layer that keeps an opening Heliox turn already shows it to the model: no second copy.
    monkeypatch.setattr(conversation_link, "_layer_keeps_opening_remarks", lambda: True)
    assert link.context_line() == ""
    monkeypatch.setattr(conversation_link, "_layer_keeps_opening_remarks", lambda: False)
    memory.add_user("hm")
    memory.add_heliox("14 November is a Saturday.")
    assert link.context_line() == ""  # now a turn the model reads
    clock.t += 91
    memory.clear()
    assert link.context_line() == ""  # too long ago to be what they mean
    assert not link.record_spoken("x", "suppressed")


# ── the person's replies ──


@pytest.mark.asyncio
async def test_not_now_holds_that_kind_back_and_dont_mention_these_mutes_it(tmp_path):
    clock, text = Clock(), ["The total is 12 x 15 = 170 so we order more."]
    live, events, spoken, settings = build(tmp_path, text, None, clock)
    await run_for(live, clock, 6)
    assert len(cards(events)) == 1
    reply = await live.handle_reply("not now")
    assert reply.reply == "Okay. I won't bring that kind of thing up for the next half hour."
    assert "arithmetic" in live.snoozed_kinds()
    text[0] += " Also 3 x 3 = 10 so that is fine."
    await run_for(live, clock, 6)
    assert len(cards(events)) == 1  # the same kind is not raised again after "not now"
    clock.t += 1800
    await run_for(live, clock, 6)
    assert len(cards(events)) == 2  # half an hour later it is
    mute = await live.handle_reply("don't mention these again")
    assert mute.reply.startswith("Okay, I won't mention these again: arithmetic slips.")
    assert settings.muted_kinds == ["arithmetic"]


@pytest.mark.asyncio
async def test_current_observation_is_none_when_live_is_off(tmp_path):
    clock, text = Clock(), ["hello there"]
    live, _, _, settings = build(tmp_path, text, None, clock)
    await run_for(live, clock, 1)
    assert live.current_observation().text == "hello there"
    settings.enabled = False
    assert live.current_observation() is None


# ── wired into the server ──


def make_server(tmp_path):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    return server


@pytest.mark.asyncio
async def test_a_voice_turn_holds_the_floor_until_dispatch_returns_even_when_it_fails(tmp_path):
    from pilot.personal.wiring import floor_for

    server = make_server(tmp_path)
    seen = []

    async def dispatch(text):
        seen.append(floor_for(server).turn_open)
        raise RuntimeError("boom")

    server._voice_command_dispatch = dispatch
    with pytest.raises(RuntimeError):
        await server._voice_turn("what's the weather")
    assert seen == [True] and not floor_for(server).turn_open
    # The conversation layer got the Live and personal context lines once.
    from pilot.conversation import layer_for

    layer = layer_for(server)
    assert len([f for f in layer.sources.extra if getattr(f, "__name__", "") in ("context_line", "extra")]) == 2


@pytest.mark.asyncio
async def test_offer_answers_go_down_the_normal_road_and_live_wins_when_it_spoke_last(tmp_path):
    from pilot.personal.offers import Offer
    from pilot.personal.wiring import offers_for
    from pilot.workflows.user_profile import UserProfile

    server = make_server(tmp_path)
    server._user_profile = UserProfile(tmp_path / "profile.json")
    server._voice_turn = AsyncMock()
    offers = offers_for(server)
    offers.mode = lambda: "on"
    offer = Offer("habit:reading", "habit", "Want me to log it?", "log reading", "You log it daily.")
    offers.mark_offered(offer)
    assert await server._control_reply("why did you offer that") == "You log it daily."
    assert await server._control_reply("yes") == "Okay."
    await asyncio.sleep(0)
    server._voice_turn.assert_awaited_with("log reading")
    # Live raised a card after the offer: "why?" is about the card, so the offer leaves it alone.
    import time

    offers.mark_offered(offer)
    server._live = MagicMock()
    from pilot.live.sequence import next_seq

    server._live.last_raised.return_value = {"seq": next_seq()}
    server._live.handle_reply = AsyncMock(return_value=None)
    from pilot.personal.wiring import offer_reply

    assert await offer_reply(server, "why") is None
    # Remember and recall reach the profile through the control replies, typed or spoken.
    assert await server._control_reply("remember that the boiler service is in March") == (
        'Remembered: "the boiler service is in March".'
    )
    assert await server._control_reply("what did I tell you about the boiler") == (
        "You told me the boiler service is in March."
    )


@pytest.mark.asyncio
async def test_on_the_bench_live_no_longer_speaks_over_a_reply_and_still_catches_everything():
    from pilot.live import bench

    pack = [s for s in bench.load_scenarios() if s["name"].startswith("floor:")]
    [before] = await bench.run_all(pack, personas=["steady"], policies=("personal",), floor=False)
    [after] = await bench.run_all(pack, personas=["steady"], policies=("personal",), floor=True)
    assert before.over_reply >= 2 and after.over_reply == 0
    assert after.catches == after.expected == before.catches and after.false_alarms == 0


def test_order_does_not_depend_on_the_clock_moving():
    """Windows clocks move in steps of about 15 ms: two events in one step must still keep their order."""
    clock = Clock()  # never advanced
    floor = Floor(clock=clock)
    from pilot.live.sequence import next_seq

    before = next_seq()
    floor.open_turn()
    after = next_seq()
    assert not floor.about_this_turn(before) and floor.about_this_turn(after)


@pytest.mark.asyncio
async def test_with_a_frozen_clock_the_latest_remark_is_still_the_one_recorded(tmp_path):
    clock, text = Clock(), ["nothing here"]
    live, events, spoken, _ = build(tmp_path, text, None, clock)
    live.offer_speech("put lunch on Tuesday 14 November")
    await run_for(live, clock, 0.4)
    live.offer_speech("remind me on the thirty first of November to pay rent")
    for _ in range(3):
        await live._deliver_due(clock.t)  # the same clock reading as before
        await asyncio.sleep(0)
    [entry] = live.last_spoken_entries()
    assert entry["kind"] == "weekday_date" and "thirty" in entry["evidence"].casefold()

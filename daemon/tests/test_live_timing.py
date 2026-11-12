"""Heliox Live, part two: a personal timing model, and the Live Bench that measures it."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live import bench
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.live.timing import PersonalTiming, at_boundary, quantile

MONDAY = date(2026, 11, 9)


# ── the model ──


def test_boundaries_and_quantiles():
    assert at_boundary("All done.") and at_boundary("Really?  ") and at_boundary("line one\n") and at_boundary("")
    assert not at_boundary("We need eight seats and") and not at_boundary("first, second,")
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert quantile([1.0, 2.0], 0.9) == pytest.approx(1.9)


def test_only_mid_sentence_pauses_that_ended_in_more_typing_are_learned():
    timing = PersonalTiming()
    timing.observe_pause(0.6, mid_sentence=True)  # ordinary typing
    timing.observe_pause(45.0, mid_sentence=True)  # stepped away
    timing.observe_pause(2.5, mid_sentence=False)  # a finished sentence: a break, not hesitation
    assert timing.pauses == []
    timing.observe_pause(2.5, mid_sentence=True)
    assert timing.pauses == [2.5]


def test_a_slow_thinker_gets_more_room_and_a_quick_typist_is_not_slowed_down():
    default = 1.2
    thinker, quick = PersonalTiming(), PersonalTiming()
    # Nothing learned: week one is patient, 2.5 times the level's pause mid-sentence (3.0 s at "balanced").
    assert thinker.mid_sentence_pause(default) == 3.0
    assert thinker.mid_sentence_pause(default, patient=False) == default  # what a slip typed last stretches from
    for i in range(60):
        thinker.observe_pause(2.4 + (i % 5) * 0.1, mid_sentence=True)
        quick.observe_pause(0.9 + (i % 3) * 0.05, mid_sentence=True)
    assert thinker.mid_sentence_pause(default) > 2.8  # waits past their usual hesitation
    # The patience fades with the person's own pauses: a quick typist is near the level's pause after 60 of them.
    assert quick.mid_sentence_pause(default) <= 1.25
    # At the end of a sentence the shorter of the two applies, so a finished thought is not kept waiting.
    assert thinker.boundary_pause(default) == default
    assert thinker.pause_for("We need eight seats and", default) > thinker.pause_for("We need eight seats.", default)


def test_few_pauses_barely_move_it_and_the_limits_hold():
    timing = PersonalTiming()
    timing.observe_pause(6.0, mid_sentence=True)
    assert 2.5 < timing.mid_sentence_pause(1.2) < 3.2  # one long pause is not a habit (still patient)
    for _ in range(400):
        timing.observe_pause(19.0, mid_sentence=True)
    assert timing.mid_sentence_pause(1.2) == 6.0  # the ceiling


def test_interruptions_grow_the_wait_and_well_timed_cards_shrink_it(tmp_path):
    path = tmp_path / "live_timing.json"
    timing = PersonalTiming(path)
    assert timing.record_outcome("dismissed", resumed_after=0.4) == "cut_off"
    assert timing.mid_sentence_pause(1.2) == pytest.approx(3.45)  # the patient 3.0 s, grown by 15%
    assert timing.record_outcome("dismissed", resumed_after=6.0) == ""  # unwelcome, but not because of timing
    assert timing.record_outcome("dismissed", resumed_after=None) == ""
    assert timing.record_outcome("fixed", resumed_after=0.5) == "well_timed"  # they typed to fix it
    reloaded = PersonalTiming(path)
    assert reloaded.cut_offs == 1 and reloaded.well_timed == 1 and reloaded.scale == pytest.approx(1.15 * 0.98)
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"pauses", "scale", "cut_offs", "well_timed"}
    reloaded.forget()
    assert reloaded.mid_sentence_pause(1.2) == 3.0  # back to week one's patience


def test_an_unreadable_file_is_ignored(tmp_path):
    path = tmp_path / "live_timing.json"
    path.write_text("[not json", encoding="utf-8")
    assert PersonalTiming(path).pauses == []


# ── the engine ──


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, text: list[str], personal: bool = True):
    clock = Clock()
    events: list[tuple[str, dict]] = []
    settings = SimpleNamespace(
        enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[], personal_timing=personal
    )

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="notes.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        return None

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "rel.json",
        timing_path=tmp_path / "timing.json",
        clock=clock,
        today=lambda: MONDAY,
    )
    return live, clock, events, settings


async def run_for(live, clock, seconds: float) -> None:
    for _ in range(round(seconds / TICK_SECONDS)):
        clock.t = round(clock.t + TICK_SECONDS, 3)
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)


def cards(events):
    return [p for e, p in events if e == "live_interjection"]


@pytest.mark.asyncio
async def test_the_engine_learns_pauses_from_the_text_it_reads(tmp_path):
    text = ["I read the draft and"]
    live, clock, _, settings = build(tmp_path, text)
    await run_for(live, clock, 1.2)
    for word in (" the", " second", " part", " reads", " well"):
        await run_for(live, clock, 2.4)  # hesitating mid-sentence
        text[0] += word
    await run_for(live, clock, 0.6)
    # The text is read every 0.6 s, so each 2.4 s pause is measured to within a read.
    assert len(live.timing.pauses) == 5 and all(1.8 <= p <= 3.0 for p in live.timing.pauses[1:])
    settings.personal_timing = False
    text[0] += " and"
    await run_for(live, clock, 2.4)
    text[0] += " then"
    await run_for(live, clock, 0.6)
    assert len(live.timing.pauses) == 5  # off means off
    status = live.status()
    assert status["personal_timing"] is False and status["timing"]["learned_from"] == 5


@pytest.mark.asyncio
async def test_a_card_that_cut_in_and_was_waved_away_counts_as_an_interruption(tmp_path):
    text = ["Chairs: 12 x 15 = 170 for the room and"]
    live, clock, events, _ = build(tmp_path, text)
    await run_for(live, clock, 3.6)  # week one's patient mid-sentence wait (3.0 s), then the card
    [card] = cards(events)
    text[0] += " the rest"  # they were only thinking: typing resumes at once
    await run_for(live, clock, 0.6)
    await live.feedback(card["id"], "dismissed")
    assert live.timing.cut_offs == 1 and live.timing.scale > 1.0


@pytest.mark.asyncio
async def test_a_slip_that_stops_trailing_is_due_at_the_ordinary_pause(tmp_path):
    # First seen while it was the last thing typed (it needs a 3 s pause then); once they type on, 1.2 s is enough.
    text = ["We need 8 x 6 = 54"]
    live, clock, events, _ = build(tmp_path, text, personal=False)
    await run_for(live, clock, 0.6)
    text[0] += " seats."
    await run_for(live, clock, 2.0)
    assert [c["kind"] for c in cards(events)] == ["arithmetic"]


@pytest.mark.asyncio
async def test_it_looks_again_before_speaking_up(tmp_path):
    text = ["Chairs: 12 x 15 = 170 for the room and"]
    live, clock, events, _ = build(tmp_path, text, personal=False)
    await run_for(live, clock, 1.2)  # one tick short of due; the last read was up to 0.6 s ago
    text[0] += " then"  # they start typing between reads
    await run_for(live, clock, 0.2)
    assert cards(events) == []


@pytest.mark.asyncio
async def test_slips_due_at_the_same_pause_go_out_together(tmp_path):
    text = ["Kickoff is Wednesday, 3 December. We need 8 x 6 = 54 seats."]
    live, clock, events, _ = build(tmp_path, text, personal=False)
    await run_for(live, clock, 2.0)
    assert sorted(c["kind"] for c in cards(events)) == ["arithmetic", "weekday_date"]


def test_forgetting_clears_what_is_in_memory(tmp_path):
    live, _, _, _ = build(tmp_path, ["x"])
    live.relevance.update("arithmetic", "dismissed")
    live.timing.observe_pause(2.0, mid_sentence=True)
    live.forget("live")
    live.forget("live_timing")
    assert live.relevance.counts == {} and live.timing.pauses == []


# ── the bench ──


def test_scenarios_are_valid_and_bad_ones_are_rejected(tmp_path):
    scenarios = bench.load_scenarios()
    assert len(scenarios) >= 25 and any(not s["expect"] for s in scenarios)
    with pytest.raises(ValueError, match="unknown correction kind"):
        bench.validate({"name": "x", "expect": ["spelling"], "steps": [{"type": "a"}]})
    with pytest.raises(ValueError, match="bad step"):
        bench.validate({"name": "x", "steps": [{"click": "a"}]})
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps([{"name": "extra", "app": "notepad.exe", "title": "t", "steps": [{"type": "hi"}]}]))
    assert bench.load_scenarios([pack])[-1]["name"] == "extra"


# Scenarios kept in the bench although Live cannot catch them yet ("known_miss" says why), so recall stays honest.
KNOWN_MISSES = {s["name"] for s in bench.load_scenarios() if s.get("known_miss")}


@pytest.mark.asyncio
async def test_the_bench_scores_personal_timing_against_fixed_rules():
    scores = {(s.policy, s.persona): s for s in await bench.run_all()}
    for persona in bench.PERSONAS:
        personal, rules = scores["personal", persona], scores["rules", persona]
        assert personal.precision == 1.0, personal.as_dict()
        missed = {d["scenario"] for d in personal.details if d.get("missed")}
        assert missed <= KNOWN_MISSES, missed  # every miss is a case kept on purpose as a known miss
        # Week one (rules: nothing learned) is patient; learning gives the speed back without cutting in more often
        # than it did before the patient start (quick 0, steady 1, thinker 2).
        assert personal.median_wait <= rules.median_wait, (persona, personal.median_wait, rules.median_wait)
        assert personal.cut_offs <= {"quick": 0, "steady": 1, "thinker": 2}.get(persona, 2), persona
    # Week one is patient: with nothing learned (rules) a slow thinker is no longer cut off mid-sentence (16 times
    # before the patient start); their own timing keeps it as low as it was.
    assert scores["rules", "thinker"].cut_offs <= 4
    assert scores["rules", "thinker"].precision >= 0.96
    assert scores["personal", "thinker"].cut_offs <= 2
    personal = sum(s.cut_offs for (policy, _), s in scores.items() if policy == "personal")
    rules = sum(s.cut_offs for (policy, _), s in scores.items() if policy == "rules")
    assert personal <= 3 and rules <= 4, (personal, rules)
    # It is deterministic, so a change to the policy shows up as a change in these numbers.
    again = await bench.run("personal", bench.PERSONAS["thinker"], bench.load_scenarios())
    assert again.as_dict() == scores["personal", "thinker"].as_dict()


def test_the_cli_prints_a_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "scores.json"
    assert bench.main(["--persona", "quick", "--policy", "personal", "--json", str(out)]) == 0
    assert "personal  quick" in capsys.readouterr().out
    [row] = json.loads(out.read_text(encoding="utf-8"))
    assert row["details"] and {d["scenario"] for d in row["details"] if d.get("missed")} <= KNOWN_MISSES

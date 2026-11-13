"""The early-words count and the last-word-to-first-sound measurement: their arithmetic, without audio or models."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.conversation_bench import early_words, voice_latency
from benchmarks.conversation_bench import timeline as tl


def test_words_are_judged_equal_one_word_or_more_after_normalising():
    assert early_words.judge("Hey Heliox, open the deck.", "hey heliox open the deck") == "equal"
    assert early_words.judge("hey heliox open the", "Hey Heliox, open the deck.") == "one word"
    assert early_words.judge("hey heliox", "Hey Heliox, open the deck.") == "more"
    assert early_words.word_distance([], ["a", "b"]) == 2
    assert early_words.without_wake("Hey Heliocs, undo that.") == "undo that"
    assert early_words.without_wake("undo that") == "undo that"


def test_the_early_words_summary_counts_and_marks_the_load():
    rows = [
        {"usable": True, "as_is": "more", "flushed": "equal", "flushed_request": "equal", "flush_s": 0.05,
         "candidate_after_last_word_s": 0.3, "end_after_last_word_s": 0.5, "final_pass_s": 0.4, "mid_candidates": 1},
        {"usable": True, "as_is": "more", "flushed": "one word", "flushed_request": "equal", "flush_s": 0.07,
         "candidate_after_last_word_s": 0.35, "end_after_last_word_s": 0.6, "final_pass_s": 0.5, "mid_candidates": 0},
        {"usable": False, "pieces": 2},
    ]  # fmt: skip

    quiet = early_words.summarise(rows, [20.0, None])
    busy = early_words.summarise(rows, [20.0, 75.0])

    assert (quiet["n"], quiet["not_one_turn"]) == (2, 1)
    assert quiet["as_is"] == {"equal": 0, "one_word": 0, "more": 2}
    assert quiet["flushed"] == {"equal": 1, "one_word": 1, "more": 0}
    assert quiet["flushed_request"] == {"equal": 2, "one_word": 0, "more": 0}
    assert quiet["candidate_after_last_word_s"] == {"median": 0.325, "min": 0.3, "max": 0.35}
    assert quiet["final_pass_load"] == "quiet" and busy["final_pass_load"] == "under load"
    assert early_words.summarise(rows, [None])["final_pass_load"] == "under load"  # unknown counts as busy


def test_a_recording_is_named_by_a_code_only():
    code = early_words.anonymous("owner:desk/take.wav#0")
    assert code.startswith("owner-") and "take" not in code
    assert voice_latency.code_for("u-real-desk-invoice", "real").startswith("rec-")
    assert voice_latency.code_for("u-syn-x", "synthetic") == "u-syn-x"


def _event(t: float, kind: str, **data) -> tl.Event:
    return tl.Event(t=t, event=kind, detail="", data=data)


def test_each_stage_is_stamped_from_the_last_word_on_both_clocks():
    vl = voice_latency
    events = [
        _event(10.0, tl.AUDIO_STARTS, setup=True, audio_t=1.0),
        _event(20.0, tl.LAST_WORD, audio_t=12.0),
        _event(20.2, vl.CANDIDATE, audio_t=12.2),
        _event(20.2, vl.PASS_STARTED, speculative=True, audio_t=12.2),
        _event(20.4, "vad end", audio_t=12.4),
        _event(20.6, vl.PASS_DONE, speculative=True, audio_t=12.6),
        _event(20.7, tl.DISPATCH, audio_t=12.7),
        _event(20.9, vl.MODEL_FIRST, audio_t=12.9),
        _event(21.1, tl.SPEAKS, setup=False, audio_t=13.0),
        _event(21.4, tl.AUDIO_STARTS, audio_t=13.1),
    ]

    got = vl.stages(events)

    assert got["candidate end"] == {"audio": 0.2, "wall": 0.2}
    assert got["final words"] == {"audio": 0.6, "wall": 0.6}
    assert got["turn ended"]["audio"] == 0.4 and got["handed to the turn"]["audio"] == 0.7
    assert got["first sentence"]["audio"] == 1.0 and got["first sound"] == {"audio": 1.1, "wall": 1.4}
    assert got["route"] == "model answer" and got["deferred"] is False
    assert "first sound 1.10 (wall 1.40, CPU 80.0 % under load)" in vl.stage_line(got, 80.0, 50.0)
    assert "candidate end 0.20 (wall 0.20)" in vl.stage_line(got, 80.0, 50.0)


def test_a_pass_run_at_the_end_of_the_turn_is_marked_deferred_and_a_cancelled_one_is_not_the_words():
    vl = voice_latency
    events = [
        _event(20.0, tl.LAST_WORD, audio_t=12.0),
        _event(20.2, vl.CANDIDATE, audio_t=12.2),
        _event(20.2, vl.PASS_DEFERRED, audio_t=12.2),
        _event(20.4, vl.PASS_STARTED, speculative=False, audio_t=12.4),
        _event(20.5, vl.PASS_DONE, cancelled=True, audio_t=12.5),
        _event(21.0, vl.PASS_DONE, speculative=False, audio_t=13.0),
        _event(21.5, tl.AUDIO_STARTS, audio_t=13.5),
    ]

    got = vl.stages(events)

    assert got["deferred"] is True and got["route"] == "local answer"
    assert got["final words"]["audio"] == 1.0
    assert vl.stages(events[1:]) is None


def test_the_latency_summary_keeps_loaded_numbers_apart():
    rows = [
        {"route": "local answer", "seconds": 1.0, "cpu_percent": 20.0},
        {"route": "local answer", "seconds": 3.0, "cpu_percent": 90.0},
        {"route": "model answer", "seconds": 2.0, "cpu_percent": None},
    ]
    got = voice_latency.summary(rows, 50.0)

    local = got["local answer, own recordings"]
    assert (local["n"], local["quiet_n"], local["quiet_median_s"], local["under_load_n"],
            local["under_load_median_s"]) == (2, 1, 1.0, 1, 3.0)  # fmt: skip
    model = got["model answer, own recordings"]
    assert model["quiet_n"] == 0 and model["under_load_n"] == 1
    rows.append({"route": "model answer", "seconds": 4.0, "cpu_percent": 10.0, "voice": "synthetic",
                 "first sound": {"audio": 4.0, "wall": 4.0}})  # fmt: skip
    synthetic = voice_latency.summary(rows, 50.0)["model answer, synthetic"]
    assert synthetic["quiet_n"] == 1 and synthetic["stages_quiet_median"]["first sound"] == 4.0


def test_a_paid_run_states_its_bound():
    config = SimpleNamespace(
        conversation=SimpleNamespace(max_reply_tokens=400, claude_model=""),
        model=SimpleNamespace(provider="cloud", cloud_provider="claude"),
    )
    bound = voice_latency.paid_bound(config, 10)

    assert bound["calls"] == 10
    assert bound["input_tokens"] == 10 * voice_latency.PROMPT_TOKENS_AT_MOST
    assert bound["output_tokens"] == 4000
    assert bound["usd"] is not None and bound["usd"] > 0


def test_the_session_log_keeps_events_between_scenarios(tmp_path):
    import json

    from benchmarks.conversation_bench.driver import Harness
    from benchmarks.conversation_bench.models import KINDS

    harness = Harness(audio_root=str(tmp_path), data_dir=tmp_path / "root", model=KINDS["cloud"])
    harness.log(tl.SCENARIO, "first")
    harness.log(tl.TRANSCRIPT, "late words")  # after the first scenario closed, before the next began
    harness.log(tl.SCENARIO, "second")

    path = harness.save_session(tmp_path / "results")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == [tl.SCENARIO, tl.TRANSCRIPT, tl.SCENARIO]


def test_times_are_also_given_from_the_recorders_last_voiced_frame():
    got = early_words._from_last_voiced(4.05, 4.2, 4.4, 4.0)

    assert got == {"candidate_after_last_voiced_s": 0.15, "end_after_last_voiced_s": 0.35,
                   "last_voiced_minus_last_word_s": 0.05}  # fmt: skip
    assert early_words._from_last_voiced(None, 4.2, 4.4, 4.0) == {}


def test_the_layers_stamps_and_the_warm_up_are_read_into_the_turn():
    vl = voice_latency
    events = [
        _event(5.0, vl.MODEL_WARM, opened=True, audio_t=-3.0),
        _event(20.0, tl.LAST_WORD, audio_t=12.0),
        _event(20.5, tl.DISPATCH, audio_t=12.5),
        _event(20.6, vl.TRACE + "route decided", audio_t=12.6),
        _event(21.4, vl.TRACE + "model first text", audio_t=13.4),
        _event(21.5, vl.MODEL_FIRST, audio_t=13.5),
        _event(21.9, tl.AUDIO_STARTS, audio_t=13.9),
        _event(22.5, vl.TRACE + "after the sound", audio_t=14.5),
    ]
    got = vl.stages(events)

    assert [t["name"] for t in got["trace"]] == ["route decided", "model first text"]
    assert got["warm_before_handover"] is True
    assert "route decided 0.60" in vl.stage_line(got, 10.0, 50.0)
    assert vl.stages([e for e in events if e.event != vl.MODEL_WARM])["warm_before_handover"] is False


def test_the_holding_phrase_is_its_own_stamp_and_not_the_first_sentence():
    vl = voice_latency
    events = [
        _event(20.0, tl.LAST_WORD, audio_t=12.0),
        _event(20.5, tl.DISPATCH, audio_t=12.5),
        tl.Event(t=22.5, event=tl.SPEAKS, detail="One moment.", data={"setup": False, "audio_t": 14.5}),
        _event(23.6, tl.AUDIO_STARTS, audio_t=15.6),
        tl.Event(t=24.0, event=tl.SPEAKS, detail="Sure.", data={"setup": False, "audio_t": 16.0}),
        _event(25.0, tl.AUDIO_STARTS, audio_t=17.0),
    ]
    got = vl.stages(events)

    assert got["holding phrase"]["audio"] == 2.5
    assert got["first sentence"]["audio"] == 4.0
    assert got["first sound"]["audio"] == 3.6 and got["answer sound"]["audio"] == 5.0


def test_a_silent_turn_says_why_as_a_code():
    vl = voice_latency
    heard = tl.Event(t=3.0, event=tl.TRANSCRIPT, detail="Hey Heliox, open the deck.",
                     data={"wake_text": "Hey Heliox, open the deck."})  # fmt: skip
    no_wake = tl.Event(t=3.0, event=tl.TRANSCRIPT, detail="open the deck", data={"wake_text": "open the deck"})

    assert vl.drop_reason([_event(1.0, tl.AUDIO_STARTS)], False) is None
    assert vl.drop_reason([], True).startswith("expected")
    assert vl.drop_reason([], False) == "no words in the waiting time"
    assert vl.drop_reason([no_wake], False) == "no wake phrase in the wake pass"
    assert vl.drop_reason([heard], False) == "wake phrase heard, not taken as addressed"
    assert vl.drop_reason([heard, _event(3.5, tl.DISPATCH)], False) == "handed to the turn, no sound"


def test_words_that_came_between_scenarios_are_named_late():
    rows = [{"case": "a", "dropped": "no words in the waiting time", "closed_at": 10.0},
            {"case": "b", "dropped": "no words in the waiting time", "closed_at": 30.0}]  # fmt: skip
    session = [_event(1.0, tl.SCENARIO), _event(14.0, tl.TRANSCRIPT), _event(20.0, tl.SCENARIO),
               _event(40.0, tl.SCENARIO), _event(41.0, tl.TRANSCRIPT)]  # fmt: skip

    voice_latency.late_words(rows, session)

    assert rows[0]["dropped"] == "words arrived after the waiting time"
    assert rows[1]["dropped"] == "no words in the waiting time"


def test_first_calls_are_set_against_the_later_ones():
    vl = voice_latency
    events = [tl.Event(t=float(n), event=vl.RECOGNISER, detail="parakeet", data={"seconds": s})
              for n, s in enumerate((4.0, 0.3, 0.5))]  # fmt: skip
    speech = [_event(1.0, tl.SPEAKS), _event(3.0, tl.AUDIO_STARTS), _event(5.0, tl.SPEAKS),
              _event(6.0, tl.AUDIO_STARTS), _event(7.0, tl.SPEAKS), _event(8.2, tl.AUDIO_STARTS)]  # fmt: skip

    assert vl.recogniser_summary(events) == {"parakeet": {"calls": 3, "first_s": 4.0, "later_median_s": 0.4}}
    assert vl.synthesis_summary(speech) == {"calls": 3, "first_s": 2.0, "later_median_s": 1.1}


def test_the_synthetic_model_turns_go_to_the_talk_model():
    from pilot.conversation.router import route

    scenarios = voice_latency.model_turn_scenarios(14)

    assert len(scenarios) == 14 and len({s.id for s in scenarios}) == 14
    assert all(s.voice == "synthetic" and s.steps[0].say.startswith(voice_latency.MODEL_TURN_FOLDER) for s in scenarios)
    assert all(str(route(text).kind) == "conversation" for text in voice_latency.MODEL_TURNS)


def test_a_paid_run_needs_a_quiet_stub_run_first(tmp_path):
    import json

    quiet = tmp_path / "quiet.json"
    quiet.write_text(json.dumps({"model": "stub", "rows": [{"cpu_percent": 20.0}]}), encoding="utf-8")
    loaded = tmp_path / "loaded.json"
    loaded.write_text(json.dumps({"model": "stub", "rows": [{"cpu_percent": 70.0}]}), encoding="utf-8")
    paid = tmp_path / "paid.json"
    paid.write_text(json.dumps({"model": "configured", "rows": []}), encoding="utf-8")

    assert voice_latency.refusal_after_stub(str(quiet), 50.0) is None
    assert "under load" in voice_latency.refusal_after_stub(str(loaded), 50.0)
    assert "not a stub" in voice_latency.refusal_after_stub(str(paid), 50.0)
    assert "no stub run" in voice_latency.refusal_after_stub(None, 50.0)


STUCK = """
import threading
from benchmarks.conversation_bench.voice_latency import run_and_leave
threading.Thread(target=threading.Event().wait, name="stuck-worker").start()
async def main():
    return 7
raise SystemExit(run_and_leave(main(), grace_s=0.5))
"""


def test_a_run_with_a_stuck_thread_still_ends_and_names_it():
    import subprocess
    import sys
    from pathlib import Path

    ran = subprocess.run([sys.executable, "-c", STUCK], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, timeout=60)  # fmt: skip

    assert ran.returncode == 7
    assert "stuck-worker" in ran.stderr


def test_a_warm_up_still_running_at_the_request_is_neither_cold_nor_warm():
    vl = voice_latency
    start = _event(19.0, vl.MODEL_WARM_STARTED)
    done = _event(20.8, vl.MODEL_WARM, opened=True)
    request = _event(20.6, vl.TRACE + "request sent")

    assert vl.warm_state([start, request, done], request) == "in progress"
    assert vl.warm_state([start, done, request], _event(21.0, vl.TRACE + "request sent")) == "finished"
    assert vl.warm_state([request], request) == "none"
    assert vl.warm_state([], None) == "none"


def test_a_paid_run_that_fails_still_closes_the_budget_block(monkeypatch):
    import asyncio
    import contextlib
    import threading
    from types import SimpleNamespace

    from pilot import real_folders
    from pilot.conversation import fast_model
    from pilot.security import vault

    events = []

    class Block:
        """The shape of real_budget_tracker's block: opened on entry, closed on exit."""

        def __init__(self, model_config, purpose):
            self.purpose = purpose

        async def __aenter__(self):
            events.append("opened")
            return self

        async def __aexit__(self, *exc):
            events.append("closed")

    monkeypatch.setattr(real_folders, "real_budget_tracker", Block)
    monkeypatch.setattr(fast_model, "build_fast_model", lambda config, **kw: "the model")
    monkeypatch.setattr(vault, "KeyVault", lambda config: None)
    before = set(threading.enumerate())

    async def run() -> None:
        exits = contextlib.AsyncExitStack()
        try:
            assert await voice_latency._configured_model(SimpleNamespace(model=None), exits) == "the model"
            raise RuntimeError("the run failed")
        finally:
            await exits.aclose()

    try:
        asyncio.run(run())
    except RuntimeError:
        pass
    assert events == ["opened", "closed"]
    assert {t for t in threading.enumerate() if t.is_alive()} <= before | {threading.main_thread()}


def test_a_timed_method_logs_each_call_and_is_wrapped_once():
    calls = []

    class Detector:
        def probability(self, audio):
            return 0.9

    detector = Detector()

    def log(kind, detail="", **data):
        calls.append((kind, detail))

    voice_latency._time_method(log, detector, "probability", "smart turn")
    voice_latency._time_method(log, detector, "probability", "smart turn")
    assert detector.probability(b"") == 0.9

    assert calls == [(voice_latency.RECOGNISER, "smart turn")]
    voice_latency._time_method(log, None, "probability", "smart turn")  # no detector: nothing to time


def test_a_turn_carries_its_warm_up_stamps_and_its_first_sentences_length():
    vl = voice_latency
    events = [
        _event(19.5, vl.MODEL_WARM_STARTED, audio_t=11.5),
        _event(20.0, tl.LAST_WORD, audio_t=12.0),
        _event(20.6, tl.DISPATCH, audio_t=12.6),
        _event(20.6, vl.TRACE + "request sent", audio_t=12.6),
        _event(20.9, vl.MODEL_WARM, opened=True, audio_t=12.9),
        tl.Event(
            t=21.6,
            event=tl.SPEAKS,
            detail="The capital of Australia is Canberra.",
            data={"setup": False, "audio_t": 13.6},
        ),  # fmt: skip
        _event(22.7, tl.AUDIO_STARTS, audio_t=14.7),
    ]
    got = vl.stages(events)

    assert got["warm_at_request"] == "in progress"
    assert got["warm_started"]["audio"] == -0.5 and got["warm_done"]["audio"] == 0.9
    assert got["request"]["audio"] == 0.6
    assert got["first_sentence_words"] == 6 and got["synthesis_s"] == 1.1
    assert got["first_sentence_chars"] == 37 and got["first_comma_words"] is None
    events[5] = tl.Event(t=21.6, event=tl.SPEAKS, detail="Sure, it is Canberra.", data={"setup": False})
    assert voice_latency.stages(events)["first_comma_words"] == 1


def test_the_warm_up_and_synthesis_tables():
    vl = voice_latency
    rows = [
        {"route": "model answer", "warm_at_request": "in progress", "warm_started": {"audio": -0.5},
         "warm_done": {"audio": 0.9}, "request": {"audio": 0.6}, "first_sentence_words": 3, "synthesis_s": 0.8,
         "cpu_percent": 10.0},
        {"route": "model answer", "warm_at_request": "finished", "warm_started": {"audio": -1.5},
         "warm_done": {"audio": 0.1}, "request": {"audio": 0.5}, "first_sentence_words": 10, "synthesis_s": 1.2,
         "cpu_percent": 10.0},
        {"route": "local answer", "first_sentence_words": 7, "synthesis_s": 1.0, "cpu_percent": 90.0},
    ]  # fmt: skip

    warm = vl.warm_up_table(rows)
    assert (warm["turns"], warm["finished"], warm["in progress"], warm["none"]) == (2, 1, 1, 0)
    assert warm["request sent"] == 0.55

    table = vl.synthesis_table(rows, 50.0)
    assert table["1-4"] == {"n": 1, "median_s": 0.8, "min_s": 0.8, "max_s": 0.8}
    assert table["9-12"]["n"] == 1 and table["5-8"]["n"] == 0  # the loaded turn is left out
    assert vl.length_band(20) == "13+"


def test_every_mode_is_named():
    assert voice_latency.MODES == ("full", "first-utterance", "warm-up", "synthesis")

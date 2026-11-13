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

    assert got["local answer"] == {"n": 2, "quiet_n": 1, "quiet_median_s": 1.0, "under_load_n": 1,
                                   "under_load_median_s": 3.0}  # fmt: skip
    assert got["model answer"]["quiet_n"] == 0 and got["model answer"]["under_load_n"] == 1


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

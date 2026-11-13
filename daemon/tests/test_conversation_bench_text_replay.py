"""The text replay: the server, layer and provider real, the words given as text, the voice a labelled model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.conversation_bench import ci_timing, text_replay, voice_latency


def test_the_voice_model_waits_its_floor_and_per_word():
    assert text_replay.modelled_delay("Okay.") == pytest.approx(0.48)
    assert text_replay.modelled_delay("The capital of Australia is Canberra.") == pytest.approx(0.63)


def test_a_turn_is_read_from_the_hand_over_and_the_estimate_adds_the_measured_part_before_it():
    events = [
        ("warm-up done", 9.0, {"opened": True}),  # before the hand-over: not this turn's
        ("words received", 10.0, {}),
        ("request sent", 10.05, {"warm": True}),
        ("model first text", 10.95, {}),
        ("piece to the voice", 11.1, {"kind": "hold"}),
        ("first audio (modelled)", 11.6, {"kind": "hold"}),
        ("piece to the voice", 11.3, {"kind": "content"}),
        ("first audio (modelled)", 11.9, {"kind": "content"}),
    ]
    row = text_replay.turn_row(events, 10.0, 0.60)

    assert "warm-up done" not in row
    assert row["request sent"] == pytest.approx(0.05) and row["warm"] is True
    assert row["model first text"] == pytest.approx(0.95)
    assert row["first_piece"] == "hold" and row["first audio (modelled)"] == pytest.approx(1.6)
    assert row["content audio (modelled)"] == pytest.approx(1.9)
    assert row["estimate from the last word"] == pytest.approx(2.2)


def test_every_turn_line_names_the_real_first_text_and_the_modelled_voice_apart():
    row = {"turn": 3, "arm": "no warm-up", "cpu_percent": 6.0, "request sent": 0.05, "model first text": 0.95,
           "first_piece": "content", "first audio (modelled)": 1.6, "warm": False}  # fmt: skip
    line = text_replay.turn_line(row)

    assert "MODEL FIRST TEXT (real provider) 0.95" in line
    assert "MODELLED VOICE 1.60" in line and "[no warm-up]" in line


def test_arms_are_interleaved_settings_or_the_warm_up():
    assert text_replay.arms_for("") == [("as configured", "none", None)]
    assert [a[0] for a in text_replay.arms_for("warm")] == ["warm-up before the request", "no warm-up"]
    arms = text_replay.arms_for("conversation.max_reply_tokens=120,400")
    assert [a[0] for a in arms] == ["conversation.max_reply_tokens=120", "conversation.max_reply_tokens=400"]
    with pytest.raises(ValueError):
        text_replay.arms_for("max_reply_tokens=120")

    config = SimpleNamespace(conversation=SimpleNamespace(max_reply_tokens=400, enabled=True))
    text_replay.apply_arm(config, "conversation.max_reply_tokens=120,400", arms[0])
    assert config.conversation.max_reply_tokens == 120
    text_replay.apply_arm(config, "conversation.enabled=off,on", ("x", "setting", "off"))
    assert config.conversation.enabled is False


def test_the_summary_is_kept_per_arm():
    rows = [
        {"arm": "a", "model first text": 0.9, "first_piece": "content", "warm": True},
        {"arm": "b", "model first text": 1.2, "first_piece": "hold", "warm": False},
        {"arm": "a", "model first text": 1.1, "first_piece": "content", "warm": True},
    ]
    got = text_replay.summary_by_arm(rows)

    assert got["a"]["model first text"] == {"median": 1.0, "min": 0.9, "max": 1.1, "n": 2}
    assert got["b"]["first piece"] == {"hold": 1, "content": 0}
    assert got["a"]["warm at the request"]["True"] == 2


def test_voice_latency_hands_its_text_replay_mode_to_the_text_replay(monkeypatch):
    seen = []
    monkeypatch.setattr(text_replay, "main", lambda argv=None: seen.append(argv) or 0)

    assert voice_latency.main(["--root", "r", "--mode", "text-replay", "--turns", "2"]) == 0
    assert seen == [["--root", "r", "--turns", "2"]]
    assert "text-replay" in voice_latency.MODES


def test_a_comparison_summary_says_its_checks_describe_the_first_setting_only():
    note = ci_timing.comparison_note("final_pass", ["whisper", "parakeet"])

    assert "scored on the first setting only (final_pass=whisper)" in note[0]
    assert ci_timing.comparison_note("", [""]) == []


def test_a_providers_failure_is_named_in_plain_words_and_auth_or_credit_stops_its_arm():
    assert text_replay.plain("no_key") == "the key was refused (authentication)"
    assert text_replay.plain("credit") == "no credit left on the account"
    assert {"no_key", "credit"} <= text_replay.STOP_REASONS and "busy" not in text_replay.STOP_REASONS


def test_each_call_is_costed_from_its_own_tokens_and_bounded_before():
    from types import SimpleNamespace

    model = SimpleNamespace(name="claude-haiku-5-5", calls=[SimpleNamespace(input_tokens=1000, output_tokens=50,
                                                                            chars=180)])  # fmt: skip
    figures = text_replay.call_figures(model, "claude")

    assert figures["input_tokens"] == 1000 and figures["reply_chars"] == 180
    assert 0 < figures["cost_usd"] < text_replay.call_bound("claude", "claude-haiku-5-5", 400)
    assert text_replay.call_figures(SimpleNamespace(name="x", calls=[]), "claude") == {}


def test_the_first_sentences_stock_opener_is_counted():
    events = [("piece to the voice", 10.2, {"kind": "content", "words": 4, "stock": True})]
    row = text_replay.turn_row(events, 10.0, 0.6)

    assert row["first_sentence_words"] == 4 and row["first_sentence_stock_opener"] is True
    assert text_replay.stock_opener("Sure, it's Canberra.") and not text_replay.stock_opener("It's Canberra.")

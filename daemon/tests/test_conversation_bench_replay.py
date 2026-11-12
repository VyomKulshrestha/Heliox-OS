"""The transcript-level addressing replay: streams, Heliox's replies opening windows, the table and the score.

No audio, no models: the cache is built in memory.
"""

from __future__ import annotations

import pytest

from benchmarks.conversation_bench import replay


def utt(stream, t, text, label="not_addressed", speaking=False):
    return {"stream": stream, "t_start": t, "t_end": t + 2.0, "label": label, "said": text, "parakeet": text,
            "whisper": text, "heliox_speaking_at_start": speaking}  # fmt: skip


def cascading(u, ctx):
    """Today's behaviour in miniature: the wake word, or anything inside 30 s of Heliox's last reply."""
    if "heliox" in u["parakeet"].casefold():
        return {"act": "dispatch", "why": "wake phrase"}
    if ctx["since_heliox_ended_s"] is not None and ctx["since_heliox_ended_s"] < 30:
        return {"act": "dispatch", "why": "follow-up window"}
    return {"act": "ignore", "why": ""}


def test_one_misfire_opens_a_window_and_the_meeting_cascades():
    items = [
        utt("b", 0.0, "The review said Heliox is fine."),
        utt("b", 7.0, "Open the deck."),
        utt("b", 13.0, "Stop sharing."),
        utt("b", 80.0, "Lunch?"),
    ]
    rows = replay.replay(items, cascading)
    assert [r["act"] for r in rows] == ["dispatch", "dispatch", "dispatch", "ignore"]
    assert rows[1]["why"] == "follow-up window" and rows[1]["since_heliox_ended_s"] == pytest.approx(1.0)
    assert rows[1]["window_opened_by"].startswith("b@0.0")
    assert rows[2]["window_opened_by"].startswith("b@7.0")
    s = replay.score(rows)
    assert s["not_addressed_dispatched"] == 3 and s["not_addressed_utterances"] == 4


def test_streams_do_not_share_windows_and_heliox_speaking_is_seen():
    items = [utt("a", 0.0, "Hey Heliox, what can I say?", "addressed"), utt("c", 3.5, "Wait, stop.", "control", True)]
    seen = []

    def decide(u, ctx):
        seen.append((u["stream"], ctx["heliox_speaking"], ctx["since_heliox_ended_s"]))
        return {"act": "dispatch", "why": "x"}

    rows = replay.replay(items, decide)
    assert seen == [("a", False, None), ("c", True, None)]
    assert replay.summary(rows)["control"]["dispatch"] == 1
    assert replay.score(rows)["addressed_taken"] == 2


def test_a_sorry_is_short_and_an_unknown_act_is_refused():
    items = [utt("a", 0.0, "mumble"), utt("a", 3.0, "more"), utt("a", 4.0, "again")]
    acts = iter(["sorry", "ignore", "ignore"])
    seen = []

    def decide(u, ctx):
        seen.append(ctx["heliox_speaking"])
        return {"act": next(acts)}

    replay.replay(items, decide)
    assert seen == [False, True, False]  # the "Sorry?" plays from 3.0 s to 3.7 s
    with pytest.raises(ValueError):
        replay.replay(items[:1], lambda u, c: {"act": "shout"})


def test_cache_and_table_round_trip(tmp_path):
    items = [utt("b", 0.0, "Visual Studio Code.")]
    cache = tmp_path / replay.CACHE_NAME
    cache.write_text("\n".join(__import__("json").dumps(i) for i in items) + "\n", encoding="utf-8")
    rows = replay.replay(replay.load_cache(cache), lambda u, c: {"act": "ignore", "why": "no wake phrase"})
    replay.write_table(rows, tmp_path / "t.tsv")
    text = (tmp_path / "t.tsv").read_text(encoding="utf-8")
    assert "Visual Studio Code." in text and "no wake phrase" in text
    assert replay.load_decide("benchmarks.conversation_bench.replay:score") is replay.score


def test_the_real_clips_are_labelled_and_name_their_microphone():
    labels = {label for _, _, label, _, _ in replay.REAL_CLIPS}
    assert labels == {"addressed", "control", "not_addressed"}
    assert all(mic in ("bluetooth-headset-call-mode", "ptron-usb-desk") for *_, mic in replay.REAL_CLIPS)
    assert sum(1 for ref, said, *_ in replay.REAL_CLIPS if said.startswith("Hey Heliox")) == 9

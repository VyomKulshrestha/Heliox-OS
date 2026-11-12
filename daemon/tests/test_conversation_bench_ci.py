"""The CI timing run's pure parts: stage breakdown on both clocks, CHECK lines and their label, the scenario choice
(synthetic only), and the workflow's shape. No audio, no models."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.conversation_bench import ci_timing, stages
from benchmarks.conversation_bench import timeline as tl
from benchmarks.conversation_bench.scenarios import load_all, parse_scenario
from benchmarks.conversation_bench.scoring import Row

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "conversation-timing.yml"


def ev(t, kind, detail="", **data):
    return tl.Event(t, kind, detail, data)


def test_reply_stages_on_the_wall_and_the_audio_clock():
    events = [
        ev(1.0, tl.LAST_WORD, audio_t=1.0),
        ev(1.6, "vad end", audio_t=1.6),
        ev(2.9, tl.TRANSCRIPT, "hi", audio_t=2.1),  # the listener fell behind: 1.3 s of wall, 0.5 s of audio
        ev(3.0, tl.PLANNER, audio_t=2.2),
        ev(4.5, tl.SPEAKS, "On it.", audio_t=3.7),
        ev(4.9, tl.AUDIO_STARTS, audio_t=4.1),
    ]
    b = stages.breakdown(events)
    assert b["endpoint"] == {"wall": 0.6, "audio": 0.6}
    assert b["recognition"] == {"wall": 1.3, "audio": 0.5}
    assert b["decision"]["wall"] == pytest.approx(0.1) and b["planner"]["wall"] == pytest.approx(1.5)
    assert b["synthesis"]["wall"] == pytest.approx(0.4)


def test_barge_in_stages_and_missing_stages():
    events = [
        ev(0.0, tl.AUDIO_STARTS, setup=True),
        ev(1.0, tl.FIRST_WORD),
        ev(1.06, "vad start"),
        ev(1.48, tl.BARGE_IN),
        ev(1.49, tl.DONE, setup=True),
    ]
    b = stages.breakdown(events)
    assert b["onset"]["wall"] == pytest.approx(0.06) and b["gate"]["wall"] == pytest.approx(0.42)
    assert b["cut"]["wall"] == pytest.approx(0.01) and b["onset"]["audio"] is None
    assert "endpoint" not in b
    assert stages.breakdown([]) == {}


def test_medians_over_scenarios():
    per = [{"endpoint": {"wall": 0.5, "audio": 0.5}}, {"endpoint": {"wall": 0.7, "audio": None}}, {}]
    assert stages.summarise(per)["endpoint"] == {"wall": 0.6, "audio": 0.5, "n": 2}


def test_check_lines_carry_the_label_and_owe_busy_lines():
    note = stages.label(4)
    assert note == "synthetic voice, GitHub runner (4 cores), CPU only"
    rows = [
        Row("1.1a", "synthetic", "as-recorded", "cloud", 0.62, 5, True),
        Row("1.2b", "synthetic", "as-recorded", "none", 0.49, 2, False),
        Row("2.1a", "synthetic", "as-recorded", "cloud", 0.05, 9, True),  # not a timing line
        Row("4.1", "synthetic", "as-recorded", "cloud", float("inf"), 1, False),
    ]
    lines = ci_timing.check_lines(rows, note, owed={"1.2b"})
    assert (
        lines[0]
        == "CHECK 1.1a 0.62 target <=0.8s pass [synthetic voice, GitHub runner (4 cores), CPU only; cloud model; n=5]"
    )
    assert "owed" in lines[1] and "machine busy" in lines[1]
    assert lines[2].startswith("CHECK 4.1 - target <=3s FAIL")
    assert len(lines) == 3


def test_only_synthetic_scenarios_behind_the_timing_lines_run():
    picked = ci_timing.chosen(load_all())
    assert picked and all(s.voice == "synthetic" for s in picked)
    assert {"tt-syn-reply-local", "tt-syn-reply-model", "c-syn-stop"} <= {s.id for s in picked}
    real = parse_scenario({"id": "tt-syn-reply-x", "set": "turn_taking", "voice": "real",
                           "steps": [{"say": "real/a.wav"}], "expect": {}})  # fmt: skip
    with pytest.raises(ValueError):
        ci_timing.chosen([real])


def test_the_workflow_is_manual_and_weekly_never_on_push_and_never_blocking():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text and "schedule" in text
    assert "push:" not in text and "pull_request" not in text
    assert "continue-on-error: true" in text
    assert "ci_prefetch" in text and "ci_timing" in text and "upload-artifact" in text
    assert "real/" not in text  # no real recording is ever referenced

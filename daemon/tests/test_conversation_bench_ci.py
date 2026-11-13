"""The CI timing run's pure parts: stage breakdown on both clocks, CHECK lines and their label, the scenario choice
(synthetic only), and the workflow's shape. No audio, no models."""

from __future__ import annotations

import re
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
    # No real recording is ever referenced: no path part that is exactly "real" (bench_audio/real/...). A folder
    # whose name only ends in it (tests/platform_real/, where the machine checks live) is not a recording.
    assert not re.search(r"(?<![A-Za-z0-9_])real/", text)


# ── Settings compared inside one job ──


def test_settings_are_parsed_and_unknown_ones_refused():
    from benchmarks.conversation_bench import compare_settings

    assert compare_settings.parse("parakeet_threads=2, 3,4") == ("parakeet_threads", ["2", "3", "4"])
    assert compare_settings.parse("") == ("", [""])
    with pytest.raises(ValueError):
        compare_settings.parse("kokoro_speed=1,2")
    with pytest.raises(ValueError):
        compare_settings.parse("parakeet_threads=")


def test_the_order_rotates_so_each_setting_comes_first_as_often():
    from benchmarks.conversation_bench import compare_settings

    orders = [compare_settings.order(["2", "3", "4"], r) for r in range(3)]
    assert orders == [["2", "3", "4"], ["3", "4", "2"], ["4", "2", "3"]]
    assert sorted(o[0] for o in orders) == ["2", "3", "4"]


def breakdown(recognition: float, synthesis: float) -> dict:
    return {"recognition": {"wall": recognition, "audio": recognition}, "synthesis": {"wall": synthesis, "audio": None}}


def test_differences_are_taken_within_each_repeat_with_their_range():
    """A runner that slows down between repeats moves both settings: the difference stays."""
    from benchmarks.conversation_bench import compare_settings

    per = {
        "2": {0: [breakdown(0.50, 0.9)], 1: [breakdown(0.80, 1.2)], 2: [breakdown(0.60, 1.0)]},
        "4": {0: [breakdown(0.40, 0.9)], 1: [breakdown(0.66, 1.2)], 2: [breakdown(0.52, 1.0)]},
    }
    diffs = compare_settings.differences(per)
    rec = diffs["4"]["recognition"]
    assert rec["median"] == pytest.approx(-0.10) and rec["low"] == pytest.approx(-0.14)
    assert rec["high"] == pytest.approx(-0.08) and rec["repeats"] == 3
    assert diffs["4"]["synthesis"]["median"] == pytest.approx(0.0)
    table = compare_settings.difference_table("parakeet_threads", {"local reply": per})
    assert "difference from parakeet_threads=2" in table and "-0.10 s (-0.14 to -0.08)" in table
    assert compare_settings.differences({"2": per["2"]}) == {}


def test_the_calibration_line_names_the_cpu_and_both_timings():
    from benchmarks.conversation_bench import compare_settings

    line = compare_settings.calibration_line("AMD EPYC 7763 64-Core Processor", 4, 0.412, 0.231, 2)
    assert (
        line
        == "CALIBRATION cpu='AMD EPYC 7763 64-Core Processor' cores=4 python_loop=0.412 s parakeet_pass=0.231 s on 2 threads"
    )
    assert compare_settings.calibration_line("", 4, 0.4, None, None).endswith("parakeet_pass=-")
    assert compare_settings.python_loop_seconds(1000) >= 0.0


def test_the_parakeet_threads_setting_reloads_only_on_a_change():
    from types import SimpleNamespace

    from benchmarks.conversation_bench import compare_settings

    loads = []
    transcriber = SimpleNamespace(threads=2, _recognizer="loaded", load=lambda: loads.append(transcriber.threads))
    harness = SimpleNamespace(listener=SimpleNamespace(_parakeet=lambda: transcriber))
    compare_settings.SETTINGS["parakeet_threads"](harness, "2")
    assert loads == []
    compare_settings.SETTINGS["parakeet_threads"](harness, "4")
    assert transcriber.threads == 4 and transcriber._recognizer is None and loads == [4]
    with pytest.raises(RuntimeError):
        compare_settings.SETTINGS["parakeet_threads"](SimpleNamespace(listener=None), "2")


# ── A comparison that fits the job, and reports what it finished ──


def args_for(settings="", repeats=0, only="", limit=0.0):
    from types import SimpleNamespace

    return SimpleNamespace(settings=settings, repeats=repeats, only=only, limit_minutes=limit)


def test_a_comparison_runs_only_the_scenarios_its_setting_can_move():
    name, values, repeats, wanted, todo = ci_timing.plan(args_for("parakeet_threads=2,3,4"))
    assert (name, values, repeats) == ("parakeet_threads", ["2", "3", "4"], 3)
    assert wanted == ["u-syn-hey_heliox", "tt-syn-reply-model"]
    assert todo and all(s.id.startswith(("u-syn-hey_heliox", "tt-syn-reply-model")) for s, _model in todo)
    assert not any(s.id.startswith(("tt-syn-bargein-", "c-syn-")) for s, _model in todo)
    # --only names others; with no setting the usual run takes every timing scenario
    assert {s.id for s, _m in ci_timing.plan(args_for("parakeet_threads=2,3", only="c-syn-stop"))[4]} == {"c-syn-stop"}
    usual = ci_timing.plan(args_for())
    assert usual[3] == [] and len(usual[4]) == len([(s, m) for s in ci_timing.chosen(load_all()) for m in s.models])


def test_a_comparison_that_cannot_finish_in_the_limit_is_refused_with_what_would():
    from benchmarks.conversation_bench import compare_settings

    # run 43290465258: 42 scenario passes x 3 settings x 3 repeats on a 150-minute job
    sentence = compare_settings.refusal(42, 3, 3, 125)
    assert sentence.startswith("Refused: 378 scenario passes")
    assert "2 repeats of these 42 scenarios" in sentence and "at most 28 scenarios at 3 repeats" in sentence
    assert compare_settings.refusal(25, 3, 3, 125) == ""  # the scoped comparison: about 100 minutes
    assert compare_settings.refusal(42, 3, 3, 0) == ""  # no limit given: never refused
    assert 98 < compare_settings.estimate_minutes(225) < 101


def test_main_refuses_before_building_anything_and_says_so_in_the_summary(tmp_path, monkeypatch):
    def no_run(*_a, **_k):
        raise AssertionError("nothing may be built for a refused run")

    monkeypatch.setattr(ci_timing, "run", no_run)
    summary = tmp_path / "step_summary.md"
    code = ci_timing.main([
        "--audio", str(tmp_path), "--results", str(tmp_path / "out"), "--summary", str(summary),
        "--settings", "parakeet_threads=2,3,4", "--only", "-", "--limit-minutes", "125",
    ])  # fmt: skip
    assert code == 2
    assert "Refused:" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "Refused:" in summary.read_text(encoding="utf-8")


def test_the_summary_and_comparison_are_written_after_every_repeat(tmp_path):
    from types import SimpleNamespace

    per = {"2": {0: [{"final pass": 0.5}]}, "4": {0: [{"final pass": 0.8}]}}
    args = SimpleNamespace(settings="parakeet_threads=2,4")
    text = ci_timing.save(tmp_path, args, [], {g: [] for g in ci_timing.GROUPS}, set(), {"spoken requests": per},
                          "CALIBRATION x", "u-syn-hey_heliox", 22, 1, 3, 0.0)  # fmt: skip
    assert "Finished 1 of 3 repeats so far" in text and "Scenarios run: u-syn-hey_heliox (22 scenario passes" in text
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").strip() == text.strip()
    assert "spoken requests" in (tmp_path / "compared.json").read_text(encoding="utf-8")


def test_the_workflow_saves_its_caches_before_the_long_step_and_always_shows_the_summary():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/cache/save" in text and "actions/cache/restore" in text and "uses: actions/cache@" not in text
    assert text.index("actions/cache/save") < text.index("Timing lines (load-checked")
    assert text.index("timeout-minutes: 125") > text.index("Timing lines (load-checked")
    summary = text[text.index("- name: Summary") :]
    assert "if: always()" in summary.split("- name:")[1] and "summary.md" in summary
    assert '--only "$COMPARE_ONLY"' in text and "COMPARE_ONLY: ${{ inputs.only }}" in text


def _step(text: str, name: str) -> str:
    """One step's text in the workflow, from its "- name:" line to the next step's."""
    start = text.index(f"      - name: {name}\n")
    end = text.find("\n      - ", start + 1)
    return text[start : end if end != -1 else len(text)]


def test_a_latency_mode_runs_instead_of_the_timing_lines_on_synthetic_voices_only():
    # Read as text (PyYAML is not a dependency of the test run); the YAML itself is parsed before every hand-over.
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'options: ["", "full", "first-utterance", "warm-up", "synthesis", "voice-released"]' in text
    timing = _step(text, "Timing lines (load-checked before every scenario)")
    latency = _step(text, "Voice latency mode (synthetic voices, stub model)")
    assert "if: inputs.latency_mode == ''" in timing and "if: inputs.latency_mode != ''" in latency
    run = latency[latency.index("run: |") :]
    assert "${{" not in run  # the typed inputs reach the command through the environment, never pasted into it
    assert "LATENCY_ONLY: ${{ inputs.only || 'u-syn' }}" in latency
    assert "--model configured" not in run and "--paid" not in run
    assert "--recommended" in run  # the timing lines' set-up: Parakeet as the final pass when fetched
    assert text.index("- name: Voice latency mode") < text.index("- name: The local voice on this runner")


def test_the_final_pass_setting_takes_whisper_or_parakeet_and_nothing_else():
    from benchmarks.conversation_bench import compare_settings

    assert compare_settings.parse("final_pass=whisper,parakeet") == ("final_pass", ["whisper", "parakeet"])
    with pytest.raises(ValueError, match="takes whisper, parakeet; not vosk"):
        compare_settings.parse("final_pass=whisper,vosk")
    with pytest.raises(ValueError, match="unknown setting"):
        compare_settings.parse("final_pas=whisper")
    assert compare_settings.SCOPE["final_pass"] == ("u-syn-hey_heliox",)
    assert compare_settings.CONDITIONS["final_pass"] == ("bt-call",)


def test_a_wrong_spec_is_refused_by_the_check_with_the_reason(capsys):
    from benchmarks.conversation_bench import compare_settings

    assert compare_settings.main(["--check", "final_pass=whisper,parakeet"]) == 0
    assert compare_settings.main(["--check", ""]) == 0
    assert compare_settings.main(["--check", "final_pass=parakeet,vosk"]) == 2
    assert "Refused" in capsys.readouterr().out


def test_word_errors_are_counted_per_setting_and_condition_with_the_difference_in_points():
    from benchmarks.conversation_bench import compare_settings

    assert compare_settings.word_errors("Hey Heliox, open the deck.", "hey heliox open a deck") == (1, 5)
    tallies = {
        "whisper": {"as-recorded": [2, 100, 10], "bt-call": [20, 100, 10]},
        "parakeet": {"as-recorded": [1, 100, 10], "bt-call": [12, 100, 10]},
    }
    table = compare_settings.word_error_table("final_pass", tallies)

    assert "| as-recorded | whisper | 10 | 100 | 2.0 % | - |" in table
    assert "| as-recorded | parakeet | 10 | 100 | 1.0 % | -1.0 points |" in table
    assert "| bt-call | parakeet | 10 | 100 | 12.0 % | -8.0 points |" in table
    assert table.index("as-recorded") < table.index("bt-call")
    assert compare_settings.word_error_table("final_pass", {}) == ""


def test_a_sentence_is_tallied_into_its_setting_and_condition():
    from types import SimpleNamespace

    words = {"whisper": {}, "parakeet": {}}
    scenario = SimpleNamespace(said="Hey Heliox, open the deck.")
    ci_timing.tally(words, "parakeet", "bt-call", scenario, SimpleNamespace(heard="hey heliox open the deck"))
    ci_timing.tally(words, "parakeet", "bt-call", scenario, SimpleNamespace(heard="hey helix open deck"))

    assert words == {"whisper": {}, "parakeet": {"bt-call": [2, 10, 2]}}


def test_the_settings_are_checked_before_the_install_and_the_fetch():
    text = WORKFLOW.read_text(encoding="utf-8")
    check = _step(text, "Check the inputs")
    assert "python3 -m benchmarks.conversation_bench.compare_settings --check" in check
    assert "COMPARE_SETTINGS: ${{ inputs.settings }}" in check
    assert "${{" not in check[check.index("run:") :]
    assert text.index("- name: Check the inputs") < text.index("- name: Setup Python")
    assert text.index("- name: Check the inputs") < text.index("- name: Install the daemon")


def test_switching_the_final_pass_sets_the_listeners_setting_and_loads_parakeet_outside_timing():
    from types import SimpleNamespace

    from benchmarks.conversation_bench import compare_settings

    loads = []

    class Parakeet:
        def load(self):
            loads.append(1)
            return True

    voice = SimpleNamespace(final_pass="whisper", parakeet_model_dir="/models/parakeet")
    listener = SimpleNamespace(config=SimpleNamespace(voice=voice), _parakeet=lambda: Parakeet())
    harness = SimpleNamespace(listener=listener)

    compare_settings.SETTINGS["final_pass"](harness, "parakeet")
    assert voice.final_pass == "parakeet" and loads == [1]
    compare_settings.SETTINGS["final_pass"](harness, "whisper")
    assert voice.final_pass == "whisper" and loads == [1]

    listener._parakeet = lambda: None
    with pytest.raises(RuntimeError, match="Parakeet is not on this machine"):
        compare_settings.SETTINGS["final_pass"](harness, "parakeet")


def test_the_final_pass_comparison_fits_the_steps_limit_with_its_call_mode_copies():
    name, values, repeats, wanted, todo = ci_timing.plan(args_for("final_pass=whisper,parakeet"))
    extra = len(ci_timing.compare_settings.CONDITIONS[name])

    assert wanted == ["u-syn-hey_heliox"] and todo and all(s.id.startswith("u-syn-hey_heliox") for s, _m in todo)
    assert ci_timing.compare_settings.refusal(len(todo), len(values), repeats + extra, 125.0) == ""

"""The Conversation Bench's own logic: scenario files, audio references, timelines, scoring, rubric, conditions.

Runs in CI on Windows, macOS and Linux with no audio, speech or model package: nothing here plays, records or
recognises sound. numpy and scipy are core dependencies, so the derived listening conditions are tested too.
"""

from __future__ import annotations

import json
import struct
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from benchmarks.conversation_bench import audio, conditions, corpus, models, rubric, scoring, soak, wavio
from benchmarks.conversation_bench import timeline as tl
from benchmarks.conversation_bench.platform_scan import platform_branches, unavailable_messages
from benchmarks.conversation_bench.scenarios import ScenarioError, load_all, parse_scenario
from benchmarks.conversation_bench.targets import BY_ID, passes


def ev(t, kind, detail="", **data):
    return {"t": t, "event": kind, "detail": detail, "data": data}


def scenario(**over):
    raw = {
        "id": "s1",
        "set": "understanding",
        "voice": "real",
        "steps": [{"say": "real/a.wav", "text": "Hey Heliox, what can I say?"}],
        "expect": {"woken": True, "reply": True},
    }
    raw.update(over)
    return parse_scenario(raw)


# Scenario files -------------------------------------------------------------------------------------------


def test_the_shipped_scenario_files_load_and_keep_real_and_synthetic_apart():
    all_ = load_all()
    assert len(all_) >= 80
    sets = {s.set for s in all_}
    assert sets == {"understanding", "turn_taking", "not_addressed", "control", "conversation_quality"}
    for s in all_:
        for step in s.steps:
            if step.say:
                assert step.say.startswith("real/" if s.voice == "real" else "synthetic/"), s.id
    assert all(s.pending for s in all_ if s.set == "conversation_quality")


def test_the_scenarios_hold_no_audio_and_no_absolute_paths():
    for s in load_all():
        for step in s.steps:
            parsed = audio.parse_ref(step.say) if step.say else None
            assert parsed is None or not PureWindowsPath(parsed.path).drive


@pytest.mark.parametrize(
    "bad",
    [
        {"set": "nonsense"},
        {"voice": "robot"},
        {"expect": {"route": "teleport"}},
        {"models": ["gpt"]},
        {"steps": [{"say": "../secret.wav"}]},
        {"steps": [{"say": "real/a.wav", "when": "later"}]},
        {"steps": [{"say": "real/a.wav", "loud": True}]},
        {"steps": []},
    ],
)
def test_malformed_scenarios_are_refused(bad):
    with pytest.raises((ScenarioError, audio.BadAudioRef)):
        scenario(**bad)


# Audio references and paths, under both rule sets -----------------------------------------------------------


def test_an_audio_reference_names_a_file_and_a_span():
    ref = audio.parse_ref("real/desk/take.wav#24.0-26.5")
    assert (ref.path, ref.start, ref.end) == ("real/desk/take.wav", 24.0, 26.5)
    assert ref.label == "real/desk/take.wav#24-26.5"
    assert audio.parse_ref("synthetic/x.wav").start is None


@pytest.mark.parametrize(
    "ref", ["/etc/passwd", "C:/Users/x.wav", "real/../../x.wav", "real\\x.wav", "a.wav#5-2", "#1-2"]
)
def test_references_cannot_leave_the_audio_folder(ref):
    with pytest.raises(audio.BadAudioRef):
        audio.parse_ref(ref)


def test_references_resolve_under_windows_and_posix_rules():
    ref = audio.parse_ref("real/desk/take.wav")
    assert audio.resolve(r"C:\bench audio", ref, PureWindowsPath) == r"C:\bench audio\real\desk\take.wav"
    assert audio.resolve("/home/a/bench", ref, PurePosixPath) == "/home/a/bench/real/desk/take.wav"


def test_the_audio_folder_comes_from_the_flag_then_the_environment_then_home():
    assert audio.audio_root("x", env={audio.ENV_VAR: "y"}) == "x"
    assert audio.audio_root(None, env={audio.ENV_VAR: "y"}) == "y"
    assert audio.audio_root(None, env={}, home="h").endswith(audio.DEFAULT_DIR_NAME)


# WAV reading without OS audio libraries ----------------------------------------------------------------------


def _wav(fmt_tag, bits, rate, payload, channels=1):
    fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_round_trip_and_formats():
    np = pytest.importorskip("numpy")
    x = np.sin(np.linspace(0, 20, 1600)).astype(np.float32) * 0.5
    back, rate = wavio.decode(wavio.encode(x))
    assert rate == 16000 and np.allclose(back, x, atol=1e-3)
    floats, _ = wavio.decode(_wav(3, 32, 16000, x.astype("<f4").tobytes()))
    assert np.allclose(floats, x)
    pcm24 = b"".join(int(v * 8388607).to_bytes(3, "little", signed=True) for v in x[:10])
    p24, _ = wavio.decode(_wav(1, 24, 16000, pcm24))
    assert np.allclose(p24, x[:10], atol=1e-5)
    stereo = np.stack([x, -x], axis=1).astype("<f4").tobytes()
    first, _ = wavio.decode(_wav(3, 32, 16000, stereo, channels=2))
    assert np.allclose(first, x)
    with pytest.raises(wavio.WavError):
        wavio.decode(b"not a wav file at all")


def test_wav_spans_and_resampling(tmp_path):
    np = pytest.importorskip("numpy")
    x = np.zeros(48000, dtype=np.float32)
    x[24000:24480] = 0.5
    path = tmp_path / "a.wav"
    path.write_bytes(_wav(1, 16, 48000, (x * 32767).astype("<i2").tobytes()))
    clip = wavio.read(path, 0.4, 0.6)
    assert len(clip) == 3200 and float(np.abs(clip).max()) > 0.3


# Timelines -------------------------------------------------------------------------------------------------


def run_events():
    return [
        ev(0.0, tl.SCENARIO, "s1"),
        ev(0.1, tl.PERSON_STARTS, "real/a.wav"),
        ev(0.3, tl.FIRST_WORD),
        ev(2.0, tl.LAST_WORD),
        ev(2.6, tl.TRANSCRIPT, "Hey Heliox, what can I say?"),
        ev(2.6, tl.WAKE, "Hey Heliox, what can I say?"),
        ev(2.7, tl.DISPATCH, "what can I say"),
        ev(2.8, tl.RESULT, "answered", status="answered", message="Besides doing things..."),
        ev(2.8, tl.SPEAKS, "Besides doing things on your computer, I handle undo and reminders."),
        ev(3.4, tl.AUDIO_STARTS),
        ev(6.0, tl.DONE),
    ]


def test_a_local_answer_is_measured_from_the_last_word():
    m = tl.measure(tl.parse({"events": run_events()})[1:])
    assert m.reply_start == pytest.approx(1.4)
    assert m.under_way == pytest.approx(0.8)
    assert m.route == "local" and m.woken and not m.used_model
    assert m.talk_over == 0 and m.silent_gap == pytest.approx(1.4)


def test_the_probes_older_timeline_names_parse():
    probe = {
        "events": [
            {"t": 1.0, "event": "person's last word ends", "detail": ""},
            {"t": 2.0, "event": "planner (stubbed)", "detail": "open notepad"},
            {"t": 5.0, "event": "heliox audio starts", "detail": ""},
        ]
    }
    m = tl.measure(tl.parse(json.dumps(probe)))
    assert m.route == "planner" and m.reply_start == pytest.approx(4.0) and m.used_model


def test_starting_to_speak_over_the_person_is_counted_but_barge_in_is_not():
    events = tl.parse(
        [
            ev(0.0, tl.SETUP_SPEECH, "long answer", setup=True),
            ev(0.2, tl.AUDIO_STARTS, setup=True),
            ev(1.0, tl.FIRST_WORD),
            ev(1.2, tl.BARGE_IN),
            ev(1.3, tl.CUT_OFF, setup=True),
            ev(1.6, tl.AUDIO_STARTS),  # Heliox starts again while the person is still talking
            ev(2.5, tl.LAST_WORD),
        ]
    )
    m = tl.measure(events)
    assert m.stop_time == pytest.approx(0.3) and m.interrupted_heard
    assert m.talk_over == 1


def test_routes_read_from_results():
    deny = [ev(1, tl.DISPATCH, "No"), ev(1.1, tl.RESULT, "answered", message="Okay, I won't run that.", plan_id="p")]
    refuse = [ev(1, tl.DISPATCH, "Yes"), ev(1.1, tl.RESULT, "approval_required", status="approval_required")]
    stop = [ev(1, tl.DISPATCH, "Stop"), ev(1.1, tl.RESULT, "answered", message="Okay, stopped.")]
    assert tl.measure(tl.parse(deny)).route == "deny"
    assert tl.measure(tl.parse(refuse)).route == "refuse_approval"
    assert tl.measure(tl.parse(stop)).route == "control"
    assert tl.measure(tl.parse([ev(1, tl.OVERHEARD, "x")])).route == "overheard"
    assert tl.measure(tl.parse([ev(1, tl.INTERJECT, "skip it")])).route == "interject"
    assert tl.measure([]).route == "ignored"


def test_a_long_silence_before_the_answer_is_the_silent_gap():
    events = tl.parse([ev(0, tl.LAST_WORD), ev(0.5, tl.PLANNER), ev(9.0, tl.RESULT), ev(9.4, tl.AUDIO_STARTS),
                       ev(10.0, tl.DONE)])  # fmt: skip
    m = tl.measure(events)
    assert m.silent_gap == pytest.approx(9.4) and m.reply_start == pytest.approx(9.4)


def test_a_run_is_cut_into_scenarios():
    events = tl.parse([ev(0, tl.SCENARIO, "a"), ev(1, tl.DISPATCH, "x"), ev(2, tl.SCENARIO, "b"), ev(3, tl.CARD, "c")])
    cut = tl.slice_scenarios(events)
    assert [e.event for e in cut["a"]] == [tl.DISPATCH] and [e.event for e in cut["b"]] == [tl.CARD]
    assert tl.to_json(cut["b"])[0]["event"] == tl.CARD


# Scoring ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("Monday the twelfth of November at ten", "Monday, the 12th of November at 10.", 0.0),
        ("the thirty-first of November", "the 31st of November", 0.0),
        ("Hey Heliox, what can I say?", "Here he was what can I say?", 3 / 6),
        ("Wait, stop.", "", 1.0),
        ("", "anything", 1.0),
    ],
)
def test_word_error_rate_with_numbers_written_either_way(ref, hyp, expected):
    assert scoring.wer(ref, hyp) == pytest.approx(expected)


def test_tolerance_matching():
    assert scoring.within("what can I say", "what can I say", 0.0)
    assert scoring.within("remind me to call the bank", "remind me to call a bank", 0.2)
    assert not scoring.within("remind me to call the bank", "remind me to call a bank", 0.1)
    assert scoring.number_words(31, ordinal=True) == "thirty first"
    assert scoring.number_words(20, ordinal=True) == "twentieth"


def test_a_scenario_is_scored_against_its_expectations():
    m = tl.measure(tl.parse(run_events()))
    m.transcripts = ["Hey Heliox, what can I say?"]
    outcome = scoring.score(scenario(), m)
    assert outcome.passed, outcome.failures
    lines = dict(outcome.contributions)
    assert lines["2.1a"] == (0, 6) and lines["2.1b"] is True and lines["1.1a"] == pytest.approx(1.4)
    assert lines["1.3"] is False


def test_a_split_sentence_and_a_wrong_route_fail():
    events = run_events() + [ev(3.5, tl.DISPATCH, "say")]  # the second half went on as a turn of its own
    m = tl.measure(tl.parse(events))
    outcome = scoring.score(scenario(expect={"woken": True, "route": "planner"}), m)
    assert not outcome.passed
    assert ("1.3", True) in outcome.contributions


def test_a_split_is_judged_by_what_ran_not_by_transcripts():
    two_transcripts = tl.measure(tl.parse([
        ev(1, tl.TRANSCRIPT, "Hey Heliox, remind me to"), ev(3, tl.TRANSCRIPT, "call the bank."),
        ev(3.1, tl.DISPATCH, "remind me to call the bank."), ev(3.2, tl.PLANNER, "remind me to call the bank"),
    ]))  # fmt: skip
    assert len(two_transcripts.transcripts) == 2 and not scoring.split(two_transcripts, 1, "remind me to call the bank")
    withdrawn = tl.measure(tl.parse([
        ev(1, tl.DISPATCH, "remind me too"), ev(1.1, tl.PLANNER, "remind me too"),
        ev(3, tl.WITHDRAW, "remind me too call the bank", status="withdrawn"),
        ev(3.05, tl.PLANNER_CANCELLED, "remind me too"),
        ev(3.1, tl.DISPATCH, "remind me too call the bank", joined=True), ev(3.2, tl.PLANNER, "remind me too call the bank"),
    ]))  # fmt: skip
    assert (withdrawn.turns, withdrawn.kept_actions, withdrawn.withdrawn) == (1, 1, 1)
    assert not scoring.split(withdrawn, 1, "remind me to call the bank")
    too_late = tl.measure(tl.parse([
        ev(1, tl.DISPATCH, "close Chrome"), ev(1.1, tl.PLANNER, "close Chrome"),
        ev(3, tl.WITHDRAW, "close Chrome it's slow", status="too_late"), ev(3.1, tl.DISPATCH, "it's slow"),
    ]))  # fmt: skip
    assert scoring.split(too_late, 1, "close Chrome, it's slow")
    stop = tl.measure(tl.parse([ev(1, tl.DISPATCH, "the radio,"), ev(2, tl.DISPATCH, "Stop")]))
    assert scoring.split(stop, 1, "He kept the radio, where it played") and not scoring.split(
        tl.measure(tl.parse([ev(1, tl.DISPATCH, "Stop")])), 1, "Wait, stop."
    )


def test_per_model_expectations_override():
    s = scenario(expect={"spoken": True, "by_model": {"none": {"spoken": "provider"}}})
    assert scoring.expectations_for(s, "none")["spoken"] == "provider"
    assert scoring.expectations_for(s, "cloud")["spoken"] is True


def test_not_addressed_speech_that_is_acted_on_fails():
    s = scenario(set="not_addressed", voice="synthetic", steps=[{"say": "synthetic/b.wav"}], expect={})
    m = tl.measure(tl.parse([ev(1, tl.DISPATCH, "open the deck"), ev(2, tl.SPEAKS, "Opening it.")]))
    outcome = scoring.score(s, m)
    assert not outcome.passed and ("2.2", 2) in outcome.contributions


def test_approval_by_voice_always_fails():
    s = scenario(set="control", voice="synthetic", steps=[{"say": "synthetic/y.wav", "text": "Yes."}],
                 expect={"approved": False})  # fmt: skip
    m = tl.measure(tl.parse([ev(1, tl.APPROVAL, "confirmed=True resolved=True")]))
    assert not scoring.score(s, m).passed
    m = tl.measure(tl.parse([ev(1, tl.APPROVAL, "confirmed=False resolved=False")]))
    assert scoring.score(s, m).passed


def test_the_scorecard_keeps_voices_conditions_and_models_apart():
    m = tl.measure(tl.parse(run_events()))
    m.transcripts = ["Hey Heliox, what can I say?"]
    real = scoring.score(scenario(), m, condition="as-recorded", model="cloud")
    derived = scoring.score(scenario(), m, condition="bt-call", model="cloud")
    synth = scoring.score(scenario(voice="synthetic", steps=[{"say": "synthetic/a.wav", "text": "Hey Heliox, hi"}]), m)
    rows = scoring.scorecard([real, derived, synth])
    wer_rows = [(r.voice, r.condition) for r in rows if r.line == "2.1a"]
    assert (
        ("real", "as-recorded") in wer_rows
        and ("real", "bt-call") in wer_rows
        and ("synthetic", "as-recorded") in wer_rows
    )
    local = next(r for r in rows if r.line == "1.1a" and r.voice == "real" and r.condition == "as-recorded")
    assert local.value == pytest.approx(1.4) and local.ok is False


def test_targets():
    assert passes(BY_ID["2.1a"], 0.08) and not passes(BY_ID["2.1a"], 0.081)
    assert passes(BY_ID["2.1b"], 0.95) and not passes(BY_ID["2.1b"], 0.9)
    assert passes(BY_ID["2.2"], 0) and not passes(BY_ID["2.2"], 1)
    assert passes(BY_ID["1.2b"], None) is None


# Rubric ----------------------------------------------------------------------------------------------------


def test_the_plain_speech_rubric():
    assert rubric.check("Your next meeting is the launch call at ten.", required_facts=("ten",)).passed
    assert "markdown or symbols" in rubric.check("**Sure.** Here it is.").failures
    assert "reads out a list" in rubric.check("Here:\n- one\n- two").failures
    assert rubric.check("I've sent the email to Priya.").failures == ["claims an action"]
    assert rubric.check("I've sent the email to Priya.", action_happened=True).passed
    assert rubric.check(" ".join(["word"] * 50)).failures[0].startswith("too long")
    assert "missing facts: four" in rubric.check("Soon.", required_facts=("four",)).failures
    assert rubric.repeats_running(["Okay, I'm on it now.", "Okay, I'm on it again.", "Done here."]) == 1
    assert "Thursday" in rubric.judge_prompt([{"who": "person", "text": "why?"}], "Because it's Thursday.", "")


# Conditions, corpus, models, soak, platform scan ----------------------------------------------------------------


def test_derived_conditions_keep_length_level_and_band():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    t = np.arange(16000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    bt = conditions.apply("bt-call", x)
    spectrum = np.abs(np.fft.rfft(bt))
    assert spectrum[5000] < spectrum[300] / 50  # nothing above 3.4 kHz survives the call-mode headset
    noisy = conditions.apply("room-20db", x)
    noise = noisy - x
    assert 18 < 20 * np.log10(conditions.speech_rms(x) / conditions.rms(noise)) < 22
    assert conditions.rms(conditions.apply("quiet-12db", x)) == pytest.approx(
        conditions.rms(x) * 10 ** (-0.6), rel=1e-3
    )
    assert float(np.abs(conditions.apply("loud+10db", x)).max()) <= 1.0
    assert len(conditions.apply("slow-0.9", x)) == pytest.approx(len(x) / 0.9, rel=0.01)
    assert len(conditions.apply("fast-1.1", x)) == pytest.approx(len(x) / 1.1, rel=0.01)
    with pytest.raises(ValueError):
        conditions.apply("voice-10db", x)
    assert len(conditions.apply("voice-5db", x, babble=np.ones(40000, np.float32) * 0.01)) == len(x)
    assert "voice-10db" not in conditions.available(False) and len(conditions.available(True)) == 10
    assert all(c.derived for c in conditions.CONDITIONS if c.name != "as-recorded")


def test_the_not_addressed_corpus_is_deterministic_and_covers_known_misfires():
    a, b = corpus.lines(300), corpus.lines(300)
    assert a == b
    kinds = {line.kind for line in corpus.lines()}
    assert {"meeting", "tutorial", "reading", "misfire", "mention"} <= kinds
    texts = " ".join(line.text for line in corpus.lines())
    for phrase in ("PowerShell", "Where we store home", "Visual Studio Code"):
        assert phrase in texts
    unaddressed = [line for line in corpus.lines() if line.kind != "mention"]
    assert not any("heliox" in line.text.casefold() for line in unaddressed)
    assert sum(len(block) for block in corpus.blocks(corpus.lines())) == 975


def test_model_kinds_and_their_delays():
    assert models.KINDS["cloud"].plan_seconds() < models.KINDS["local"].plan_seconds()
    assert models.KINDS["none"].plan_seconds() < 0.1 and not models.KINDS["none"].available
    assert models.KINDS["cloud"].reply_seconds() < 1.0


def test_soak_judging_and_trend():
    s = scenario()
    missed, duplicated = soak.judge([], s)
    assert missed and not duplicated
    two = tl.parse([ev(1, tl.DISPATCH, "a"), ev(2, tl.DISPATCH, "a")])
    assert soak.judge(two, s) == (False, True)
    samples = [soak.Sample(t * 360.0, 100 + t * 10, 0, 50, 10, 5, 1, 0.0) for t in range(20)]
    assert soak.slope_per_hour(samples, "rss_mb") == pytest.approx(100.0)
    assert len(soak.plan([s, scenario(id="c", set="control")])) == 4000
    sample = soak.process_sample(1.0, 1, 0.0)
    assert sample.threads >= 1


def test_the_platform_scan_finds_branches_and_unavailable_messages():
    src = "import sys\n\ndef a():\n    if sys.platform != 'win32':\n        return None\n\ndef b():\n    return 1\n"
    assert [name for name, _, _ in platform_branches(src)] == ["a"]
    said = ["Call detection is not available on this system.", "Okay, stopped."]
    assert unavailable_messages(said) == ["Call detection is not available on this system."]


def test_the_run_log_takes_any_data_field_name():
    from benchmarks.conversation_bench.driver import Log, stored_delta, voiced_span

    log = Log()
    log(tl.DISPATCH, "open it", kind="meeting", detail_extra=1)  # "kind" once broke the dispatch hook
    assert log.events[0].data["kind"] == "meeting" and log.since(0)[0].detail == "open it"
    assert stored_delta({"a.db:t": 1}, {"a.db:t": 3, "crashes.db:c": 9}) == {"a.db:t": 2}
    np = pytest.importorskip("numpy")
    x = np.zeros(16000, np.float32)
    x[4000:8000] = 0.5
    assert voiced_span(x) == pytest.approx((0.25, 0.5))


def test_derived_conditions_apply_to_real_sentences_and_models_to_their_scenarios():
    from benchmarks.conversation_bench.__main__ import jobs

    todo = jobs(
        load_all(), profile="baseline", sets=["understanding", "control"], conditions=["as-recorded", "bt-call"],
        models=[], only="u-real-desk-usb-cue-a,c-syn-stop", babble=False, na_blocks=None,
    )  # fmt: skip
    got = {(s.id, c, m) for s, c, m in todo}
    assert ("u-real-desk-usb-cue-a", "bt-call", "cloud") in got
    assert {m for s, c, m in todo if s.id == "c-syn-stop"} == {"cloud", "local", "none"}
    assert {c for s, c, m in todo if s.id == "c-syn-stop"} == {"as-recorded"}


class _Recorder:
    frame_size = 320


def test_a_stalled_loop_does_not_move_a_clip_on_the_audio_clock():
    np = pytest.importorskip("numpy")
    from benchmarks.conversation_bench.driver import FilePump, Log

    pump = FilePump(_Recorder(), Log())
    first = np.full(16000, 0.5, dtype=np.float32)  # one second of "speech"
    end = pump.play(first, "a", first=0.0, last=1.0)
    assert end == 16000
    stream, marks = [], []
    for _ in range(80):  # the pump keeps running while the event loop is stalled: 1.6 s of audio go by
        block, due = pump.next_block()
        stream.append(block)
        marks += due
    second = np.full(8000, -0.5, dtype=np.float32)
    start = end + int(1.5 * 16000)  # a 1.5 s pause, queued late (after 0.6 s of it had already played)
    assert pump.play(second, "b", first=0.0, last=0.5, start_at=start) == start + 8000
    while pump.busy():
        block, due = pump.next_block()
        stream.append(block)
        marks += due
    audio = np.concatenate(stream)
    assert int(np.argmax(audio < -10000)) == start  # the second clip begins exactly 1.5 s after the first ends
    starts = [index for index, kind, data in marks if kind == tl.PERSON_STARTS]
    assert starts == [0, start]
    late = pump.play(second, "c", first=0.0, last=0.5, start_at=pump.played - 100)
    assert late == pump.played + 8000  # a start already in the past begins at once, never before the clock


def test_the_bench_wires_its_listener_exactly_as_the_server_does():
    import inspect

    from benchmarks.conversation_bench import driver

    source = driver.SERVER_SOURCE.read_text(encoding="utf-8")
    entry, attrs = driver.server_listener_wiring(source)
    # The server today: turns enter through _voice_turn; a split sentence can be withdrawn. If the server's wiring
    # changes, this fails, and the bench (which reads the same source) must be checked against the new wiring.
    assert entry == "_voice_turn"
    assert attrs["withdraw_turn"] == "withdraw_voice_turn" and attrs["task_open"] == "_voice_task_open"

    class Server:
        def withdraw_voice_turn(self):
            return "withdrawn"

        def _voice_task_open(self):
            return True

    class Listener:
        pass

    listener, server = Listener(), Server()
    assert driver.wire_like_server(listener, server, source) == attrs
    assert listener.withdraw_turn() == "withdrawn" and listener.task_open() is True
    build = inspect.getsource(driver.Harness.build)
    assert "wire_like_server(listener, server)" in build and "getattr(server, turn_entry)" in build
    teardown = inspect.getsource(driver.Harness._teardown)
    assert "_handle_voice_new_conversation" in teardown  # every scenario starts from a clean conversation
